"""
SAC Market Making — Training Script.

Usage:
    python3 -m agents.strategies.train_sac

Input:  training_data/BTCUSDT-aggTrades-*.csv
Output: models/sac_agent.pth   (best checkpoint by cumulative reward)

Simulation
──────────
Each episode is a 4-hour window of real 1-minute BTC bars.
At each step the agent quotes (bid_spread, ask_spread).
An order is filled if the bar price touched the quoted level.

Reward per step
───────────────
  r = spread_captured − λ_inv × (inventory/max_inv)² × σ − λ_adverse × adverse_cost

  spread_captured:  (mid − bid) × qty  if bid filled
                  + (ask − mid) × qty  if ask filled
  inventory penalty: quadratic, scaled by volatility (Avellaneda-Stoikov style)
  adverse selection: price moved against a fill during the same bar

Terminal reward
───────────────
  Liquidation at mid price with 0.1% cost applied to remaining inventory.

Best model saved whenever validation reward improves.
Early stop after PATIENCE episodes without improvement.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "training_data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sac_agent.pth"

sys.path.insert(0, str(ROOT))
from agents.strategies.sac_agent import SACAgent, STATE_DIM, ACTION_DIM, MAX_SPREAD_PCT

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPISODE_LEN = 240        # steps (4 hours of 1-min bars)
HISTORY_LEN = 10         # bars of price history in state
TOTAL_EPISODES = 2000
WARMUP_EPISODES = 20     # random policy before training starts
UPDATES_PER_STEP = 2     # gradient steps per env step
BATCH_SIZE = 256
REPLAY_CAPACITY = 500_000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
PATIENCE = 80            # early stopping
VAL_EPISODES = 10        # episodes used to evaluate best reward

# Market making parameters
QTY = 0.1                # units per order
MAX_INVENTORY = 5.0      # position cap (units)
LAMBDA_INV = 0.1         # inventory penalty weight
LAMBDA_ADVERSE = 0.05    # adverse selection penalty weight
LIQD_COST_PCT = 0.001    # liquidation cost on terminal inventory


# ── Data loading (same pipeline as train_lstm) ────────────────────────────────

COLS = [
    "agg_id", "price", "qty",
    "first_trade_id", "last_trade_id",
    "timestamp", "is_buyer_maker", "is_best_match",
]


def load_bars(data_dir: Path) -> np.ndarray:
    """
    Load aggTrades CSVs → 1-minute bars.
    Returns float32 array with columns:
      [open, high, low, close, volume, buyer_vol, log_ret, volatility, buyer_frac, vol_ratio]
    """
    csv_files = sorted(data_dir.glob("BTCUSDT-aggTrades-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    print(f"Loading {len(csv_files)} file(s)...")
    all_bars = []

    for f in csv_files:
        print(f"  {f.name} ...", end="", flush=True)
        chunks = []
        for chunk in pd.read_csv(
            f, header=None, names=COLS,
            usecols=["timestamp", "price", "qty", "is_buyer_maker"],
            dtype={"timestamp": "int64", "price": "float64",
                   "qty": "float64", "is_buyer_maker": "str"},
            chunksize=5_000_000,
        ):
            chunk["ts"] = pd.to_datetime(chunk["timestamp"], unit="us", utc=True)
            chunk["is_buyer_maker"] = chunk["is_buyer_maker"].map(
                {"True": True, "False": False}
            ).fillna(False)
            chunk["buyer_vol"] = chunk["qty"] * (~chunk["is_buyer_maker"]).astype(float)
            chunk = chunk.set_index("ts").sort_index()
            bar = chunk.resample("1min").agg(
                open=("price", "first"), high=("price", "max"),
                low=("price", "min"),   close=("price", "last"),
                volume=("qty", "sum"),  buyer_vol=("buyer_vol", "sum"),
            ).dropna(subset=["open"])
            chunks.append(bar)

        df = pd.concat(chunks).resample("1min").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"),     close=("close", "last"),
            volume=("volume", "sum"), buyer_vol=("buyer_vol", "sum"),
        ).dropna(subset=["open"])
        all_bars.append(df)
        print(f" {len(df):,} bars")

    df = pd.concat(all_bars).sort_index()
    df = df.resample("1min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
        volume=("volume", "sum"), buyer_vol=("buyer_vol", "sum"),
    ).dropna(subset=["open"])

    eps = 1e-10
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    bv = df["buyer_vol"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)

    log_ret = np.concatenate([[0.0], np.log(c[1:] / (c[:-1] + eps))])
    volatility = pd.Series(log_ret).rolling(10).std().fillna(0.0).values
    buyer_frac = bv / (v + eps)
    vol_ratio = v / (pd.Series(v).rolling(20).mean().fillna(v.mean()).values + eps)

    bars = np.stack([
        df["open"].values, h, lo, c, v, bv,
        log_ret, volatility, buyer_frac, vol_ratio,
    ], axis=1).astype(np.float32)

    print(f"\nTotal: {len(bars):,} bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return bars


# ── Market making simulation environment ──────────────────────────────────────

class MarketMakingEnv:
    """
    Simulates a market making agent in a 1-minute bar environment
    built from real Binance aggTrades data.

    State (STATE_DIM = 16):
      [inventory_norm, log_ret×10, volatility, buyer_frac,
       market_spread_bps, vol_ratio]

    Action (ACTION_DIM = 2):
      [bid_spread_pct, ask_spread_pct] ∈ [0, MAX_SPREAD_PCT]
    """

    BAR_OPEN, BAR_HIGH, BAR_LOW, BAR_CLOSE = 0, 1, 2, 3
    BAR_VOL, BAR_BVOL = 4, 5
    BAR_LOG_RET, BAR_VOLATILITY, BAR_BUYER_FRAC, BAR_VOL_RATIO = 6, 7, 8, 9

    def __init__(self, bars: np.ndarray, episode_len: int = EPISODE_LEN,
                 history_len: int = HISTORY_LEN):
        self.bars = bars
        self.episode_len = episode_len
        self.history_len = history_len
        self.max_start = len(bars) - episode_len - history_len - 1

        self.step_idx = 0
        self.start = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.initial_mid = 1.0

    def reset(self, start: Optional[int] = None) -> np.ndarray:
        self.start = start if start is not None else np.random.randint(
            self.history_len + 1, self.max_start
        )
        self.step_idx = 0
        self.inventory = 0.0
        mid = float(self.bars[self.start + self.history_len, self.BAR_CLOSE])
        self.cash = mid * MAX_INVENTORY  # start with enough cash to buy max inventory
        self.initial_mid = mid
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        idx = self.start + self.history_len + self.step_idx
        bar = self.bars[idx]
        # Recent log returns
        log_rets = self.bars[idx - self.history_len:idx + 1, self.BAR_LOG_RET][-self.history_len:]
        inv_norm = np.clip(self.inventory / MAX_INVENTORY, -1.0, 1.0)
        vol = float(bar[self.BAR_VOLATILITY])
        buyer_frac = float(bar[self.BAR_BUYER_FRAC])
        # Approximate market spread in bps from bar range
        mid = float(bar[self.BAR_CLOSE])
        rng_bps = (float(bar[self.BAR_HIGH]) - float(bar[self.BAR_LOW])) / (mid + 1e-10) * 10000
        vol_ratio = float(bar[self.BAR_VOL_RATIO])

        state = np.array([
            inv_norm,
            *np.clip(log_rets * 100, -5.0, 5.0),
            np.clip(vol * 100, 0.0, 5.0),
            np.clip(buyer_frac, 0.0, 1.0),
            np.clip(rng_bps / 10.0, 0.0, 5.0),
            np.clip(vol_ratio, 0.0, 5.0),
        ], dtype=np.float32)
        return state

    def step(self, action: np.ndarray):
        """
        action: [bid_spread_pct, ask_spread_pct] (already in [0, MAX_SPREAD_PCT])
        Returns: (next_state, reward, done, info)
        """
        idx = self.start + self.history_len + self.step_idx
        bar = self.bars[idx]

        mid = float(bar[self.BAR_CLOSE])
        bar_high = float(bar[self.BAR_HIGH])
        bar_low = float(bar[self.BAR_LOW])
        volatility = float(bar[self.BAR_VOLATILITY])
        buyer_frac = float(bar[self.BAR_BUYER_FRAC])

        bid_spread = float(np.clip(action[0], 0.0, MAX_SPREAD_PCT))
        ask_spread = float(np.clip(action[1], 0.0, MAX_SPREAD_PCT))
        bid_price = mid * (1.0 - bid_spread)
        ask_price = mid * (1.0 + ask_spread)

        # ── Order fill simulation ──────────────────────────────────────────
        # Bid fills if bar touched the bid level (sellers came to us)
        bid_filled = (bar_low <= bid_price) and (self.inventory < MAX_INVENTORY)
        # Ask fills if bar touched the ask level (buyers came to us)
        ask_filled = (bar_high >= ask_price) and (self.inventory > -MAX_INVENTORY)

        reward = 0.0
        next_mid = float(self.bars[idx + 1, self.BAR_CLOSE]) if idx + 1 < len(self.bars) else mid

        if bid_filled:
            self.inventory += QTY
            self.cash -= bid_price * QTY
            spread_captured = (mid - bid_price) * QTY
            # Adverse selection: if price fell after we bought, we overpaid
            adverse = max(0.0, (bid_price - next_mid) * QTY)
            reward += spread_captured - LAMBDA_ADVERSE * adverse

        if ask_filled:
            self.inventory -= QTY
            self.cash += ask_price * QTY
            spread_captured = (ask_price - mid) * QTY
            # Adverse selection: if price rose after we sold, we undersold
            adverse = max(0.0, (next_mid - ask_price) * QTY)
            reward += spread_captured - LAMBDA_ADVERSE * adverse

        # Inventory penalty — quadratic, scaled by volatility (A-S style)
        inv_norm = self.inventory / MAX_INVENTORY
        reward -= LAMBDA_INV * inv_norm ** 2 * max(volatility * 100, 0.001)

        self.step_idx += 1
        done = self.step_idx >= self.episode_len

        if done:
            # Terminal: liquidate remaining inventory at mid price with friction
            liq_cost = abs(self.inventory) * next_mid * LIQD_COST_PCT
            reward -= liq_cost
            self.cash += self.inventory * next_mid
            self.inventory = 0.0

        next_state = self._get_state() if not done else np.zeros(STATE_DIM, np.float32)
        info = {
            "bid_filled": bid_filled, "ask_filled": ask_filled,
            "inventory": self.inventory, "mid": mid,
        }
        return next_state, reward, done, info


# ── Training ───────────────────────────────────────────────────────────────────

def evaluate(agent: SACAgent, bars: np.ndarray, n_episodes: int = VAL_EPISODES
             ) -> float:
    """Run n_episodes with deterministic policy, return mean cumulative reward."""
    env = MarketMakingEnv(bars)
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, deterministic=True)
            state, r, done, _ = env.step(action)
            ep_reward += r
        total += ep_reward
    return total / n_episodes


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print("  SAC Market Maker — Training")
    print(f"  Device: {device}")
    print(f"  Episode length: {EPISODE_LEN} steps (4h of simulated market time per episode)")
    print(f"{'='*60}\n")
    train_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    bars = load_bars(DATA_DIR)

    # Chronological split: 70% train, 15% val, 15% test
    n = len(bars)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    train_bars = bars[:n_train]
    val_bars = bars[n_train:n_val]
    print(f"Split — train: {n_train:,}  val: {n_val - n_train:,}  "
          f"test: {n - n_val:,} bars\n")

    if len(train_bars) < EPISODE_LEN + HISTORY_LEN + 10:
        print("❌ Not enough data for training.")
        sys.exit(1)

    # ── Agent & environment ────────────────────────────────────────────────
    agent = SACAgent(
        lr=LR, gamma=GAMMA, tau=TAU,
        batch_size=BATCH_SIZE, replay_capacity=REPLAY_CAPACITY,
        device=device,
    )
    env = MarketMakingEnv(train_bars)

    n_params = sum(p.numel() for p in agent.actor.parameters()) + \
               sum(p.numel() for p in agent.critic.parameters())
    print(f"Actor + Critic parameters: {n_params:,}")
    print(f"Warmup episodes (random): {WARMUP_EPISODES}")
    print(f"Training episodes:        {TOTAL_EPISODES}")
    print(f"Early stop patience:      {PATIENCE}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_reward = float("-inf")
    patience_counter = 0
    episode_rewards = []

    print(f"{'─'*60}")
    print(f"{'Ep':>5}  {'Ep Reward':>10}  {'Avg(20)':>8}  "
          f"{'Val Reward':>10}  {'Updates':>8}  Best")
    print(f"{'─'*60}")

    for episode in range(1, TOTAL_EPISODES + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False
        metrics_sum = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0}
        n_updates = 0

        while not done:
            # Random policy during warmup
            if episode <= WARMUP_EPISODES:
                action = np.random.uniform(0, MAX_SPREAD_PCT, size=ACTION_DIM).astype(np.float32)
            else:
                action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            agent.buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            # Train
            if episode > WARMUP_EPISODES:
                for _ in range(UPDATES_PER_STEP):
                    m = agent.update()
                    if m:
                        for k in metrics_sum:
                            metrics_sum[k] += m.get(k, 0.0)
                        n_updates += 1

        episode_rewards.append(ep_reward)
        avg20 = np.mean(episode_rewards[-20:])

        # Validate every 10 episodes after warmup
        val_str = ""
        is_best = False
        if episode > WARMUP_EPISODES and episode % 10 == 0:
            val_reward = evaluate(agent, val_bars)
            val_str = f"{val_reward:>10.2f}"
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                patience_counter = 0
                is_best = True
                torch.save(
                    {
                        "episode": episode,
                        "actor_state_dict": agent.actor.state_dict(),
                        "critic_state_dict": agent.critic.state_dict(),
                        "log_alpha": agent.log_alpha.item(),
                        "best_reward": round(best_val_reward, 4),
                        "config": {
                            "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
                            "max_spread_pct": MAX_SPREAD_PCT,
                            "history_len": HISTORY_LEN,
                        },
                    },
                    MODEL_PATH,
                )
            else:
                patience_counter += 10
        else:
            val_str = f"{'':>10}"

        best_mark = " ✓" if is_best else ""
        print(
            f"{episode:>5}  {ep_reward:>10.2f}  {avg20:>8.2f}  "
            f"{val_str}  {agent.updates:>8,}  {best_mark}"
        )

        if episode > WARMUP_EPISODES and patience_counter >= PATIENCE:
            print(f"\nEarly stop — no improvement for {PATIENCE} episodes.")
            break

    # ── Final test evaluation ──────────────────────────────────────────────
    if MODEL_PATH.exists():
        print(f"\n{'─'*70}")
        print(f"Best val reward: {best_val_reward:.4f}")
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        agent.actor.load_state_dict(ckpt["actor_state_dict"])
        test_bars = bars[n_val:]
        if len(test_bars) >= EPISODE_LEN + HISTORY_LEN + 10:
            test_reward = evaluate(agent, test_bars, n_episodes=20)
            print(f"Test reward (20 episodes): {test_reward:.4f}")
        print(f"\n✅ Model saved to {MODEL_PATH}")
    else:
        print("\n⚠️  No checkpoint saved (training may need more episodes).")

    elapsed = time.time() - train_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n⏱  Training time: {h}h {m}m {s}s")


if __name__ == "__main__":
    from typing import Optional
    train()
