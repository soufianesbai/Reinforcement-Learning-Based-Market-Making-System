"""
SAC Market Maker — Backtest & Evaluation Script.

Usage:
    python3 agents/strategies/eval_sac.py

Compares SAC against two baselines on the held-out test set:
  - Random policy   : random spreads in [0, MAX_SPREAD_PCT]
  - Fixed symmetric : constant 0.2% spread on both sides

Outputs:
  - Printed summary table (PnL, Sharpe, drawdown, fill rate)
  - Matplotlib figure saved to models/sac_eval.png
"""

import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.strategies.sac_agent import Actor, STATE_DIM, ACTION_DIM, MAX_SPREAD_PCT
from agents.strategies.train_sac import (
    load_bars, MarketMakingEnv,
    EPISODE_LEN, HISTORY_LEN, QTY, MAX_INVENTORY,
    LAMBDA_INV, LAMBDA_ADVERSE, LIQD_COST_PCT, DATA_DIR,
)

N_EPISODES   = 50
FIXED_SPREAD = 0.002   # 0.2% each side for the symmetric baseline


# ── Policies ──────────────────────────────────────────────────────────────────

def run_policy(policy_fn, bars, n_episodes=N_EPISODES, seed=42):
    """
    Roll out a policy on n_episodes random windows.
    Returns dict of per-episode metrics.
    """
    rng = np.random.default_rng(seed)
    env = MarketMakingEnv(bars, episode_len=EPISODE_LEN, history_len=HISTORY_LEN)

    ep_rewards, ep_pnl = [], []
    bid_fills, ask_fills, total_steps = 0, 0, 0
    max_inventories = []
    step_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_r = 0.0
        ep_inv_max = 0.0
        ep_step_r = []

        while not done:
            action = policy_fn(state, rng)
            state, r, done, info = env.step(action)
            ep_r += r
            ep_step_r.append(r)
            if info["bid_filled"]:
                bid_fills += 1
            if info["ask_filled"]:
                ask_fills += 1
            total_steps += 1
            ep_inv_max = max(ep_inv_max, abs(info["inventory"]))

        ep_rewards.append(ep_r)
        ep_pnl.append(ep_r)
        max_inventories.append(ep_inv_max)
        step_rewards.extend(ep_step_r)

    rewards = np.array(ep_rewards)
    sr = rewards.mean() / (rewards.std() + 1e-8) * np.sqrt(N_EPISODES)
    cumulative = np.cumsum(rewards)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(drawdowns.max())

    return {
        "rewards": rewards,
        "cumulative": cumulative,
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "sharpe": float(sr),
        "max_drawdown": max_dd,
        "fill_rate": (bid_fills + ask_fills) / (total_steps * 2) * 100,
        "bid_fills": bid_fills,
        "ask_fills": ask_fills,
        "max_inv": float(np.mean(max_inventories)),
        "step_rewards": np.array(step_rewards),
    }


# ── Load model ─────────────────────────────────────────────────────────────────

def make_sac_policy(model_path, device):
    actor = Actor(STATE_DIM, ACTION_DIM).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    ep = ckpt.get("episode", "?")
    reward = ckpt.get("best_reward", "?")
    print(f"  SAC checkpoint: episode={ep}, best_val_reward={reward}")

    def policy(state, rng):
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return actor.deterministic(s).squeeze(0).cpu().numpy()
    return policy


def random_policy(state, rng):
    return rng.uniform(0, MAX_SPREAD_PCT, size=ACTION_DIM).astype(np.float32)


def fixed_policy(state, rng):
    return np.array([FIXED_SPREAD, FIXED_SPREAD], dtype=np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = ROOT / "models" / "sac_agent.pth"

    if not model_path.exists():
        print("❌ models/sac_agent.pth not found. Train first.")
        sys.exit(1)

    print("Loading data...")
    bars = load_bars(DATA_DIR)
    n = len(bars)
    test_bars = bars[int(n * 0.85):]
    print(f"Test set: {len(test_bars):,} bars\n")

    policies = {
        "SAC (trained)": make_sac_policy(model_path, device),
        "Random":        random_policy,
        f"Fixed {FIXED_SPREAD*100:.1f}%": fixed_policy,
    }

    results = {}
    for name, policy_fn in policies.items():
        print(f"Evaluating {name}...", end=" ", flush=True)
        results[name] = run_policy(policy_fn, test_bars)
        print("done")

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"{'Policy':<22} {'Mean R':>8} {'Std R':>7} {'Sharpe':>7} "
          f"{'MaxDD':>8} {'Fill%':>7} {'MaxInv':>7}")
    print(f"{'─'*72}")
    for name, r in results.items():
        print(f"{name:<22} {r['mean']:>8.2f} {r['std']:>7.2f} {r['sharpe']:>7.3f} "
              f"{r['max_drawdown']:>8.2f} {r['fill_rate']:>6.1f}% {r['max_inv']:>7.2f}")
    print(f"{'─'*72}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("SAC Market Maker — Backtest Evaluation", fontsize=14, fontweight="bold")
        colors = {"SAC (trained)": "#4aa3ff", "Random": "#ff7043", f"Fixed {FIXED_SPREAD*100:.1f}%": "#66bb6a"}

        # 1. Cumulative reward
        ax = axes[0, 0]
        for name, r in results.items():
            ax.plot(r["cumulative"], label=name, color=colors[name], linewidth=2)
        ax.set_title("Cumulative Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Per-episode reward distribution
        ax = axes[0, 1]
        data = [r["rewards"] for r in results.values()]
        bp = ax.boxplot(data, labels=list(results.keys()), patch_artist=True)
        for patch, color in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title("Episode Reward Distribution")
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.3, axis="y")

        # 3. Fill rate comparison
        ax = axes[1, 0]
        names = list(results.keys())
        bid_rates = [r["bid_fills"] / (r["bid_fills"] + r["ask_fills"] + 1e-8) * 100 for r in results.values()]
        ask_rates = [100 - b for b in bid_rates]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, bid_rates, w, label="Bid fills", color="#4aa3ff", alpha=0.8)
        ax.bar(x + w/2, ask_rates, w, label="Ask fills", color="#ff7043", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title("Fill Rate Split (Bid vs Ask)")
        ax.set_ylabel("%")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # 4. Step reward histogram for SAC
        ax = axes[1, 1]
        sac_steps = results["SAC (trained)"]["step_rewards"]
        ax.hist(sac_steps, bins=60, color="#4aa3ff", alpha=0.8, edgecolor="none")
        ax.axvline(0, color="white", linewidth=1, linestyle="--")
        ax.set_title("SAC Step Reward Distribution")
        ax.set_xlabel("Step reward")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = ROOT / "models" / "sac_eval.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n📊 Chart saved to {out}")

    except ImportError:
        print("\nmatplotlib not installed — skipping chart (pip install matplotlib)")


if __name__ == "__main__":
    from typing import Optional
    main()
