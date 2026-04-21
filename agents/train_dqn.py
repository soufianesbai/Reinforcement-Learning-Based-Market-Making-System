import argparse
import csv
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from agents.strategies.dqn_agent import DQNStrategy as NormalDQNStrategy
from agents.strategies.dqn_agent_faster import DQNStrategy as FastDQNStrategy


@dataclass
class TradeRow:
    price: float
    quantity: float
    timestamp_us: int
    is_buyer_maker: bool


class RollingWindowFeatures:
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.prices: Deque[float] = deque(maxlen=window_size)
        self.quantities: Deque[float] = deque(maxlen=window_size)
        self.signed_volumes: Deque[float] = deque(maxlen=window_size)
        self.timestamps: Deque[int] = deque(maxlen=window_size)

    def update(self, row: TradeRow) -> None:
        self.prices.append(row.price)
        self.quantities.append(row.quantity)
        taker_sign = -1.0 if row.is_buyer_maker else 1.0
        self.signed_volumes.append(taker_sign * row.quantity)
        self.timestamps.append(row.timestamp_us)

    def ready(self) -> bool:
        return len(self.prices) >= 20

    def build_state(self, inventory: float, max_inventory: float) -> Optional[np.ndarray]:
        if not self.ready():
            return None

        prices = np.asarray(self.prices, dtype=np.float64)
        quantities = np.asarray(self.quantities, dtype=np.float64)
        signed = np.asarray(self.signed_volumes, dtype=np.float64)

        if len(prices) < 3:
            return None

        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-8)
        short_returns = returns[-10:] if len(returns) >= 10 else returns
        momentum = float(np.mean(short_returns) * 150.0)
        volatility = float(np.std(short_returns) * 200.0)

        buy_vol = float(np.sum(np.clip(signed, 0.0, None)))
        sell_vol = float(np.sum(np.clip(-signed, 0.0, None)))
        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)

        qty_mean = float(np.mean(quantities))
        qty_std = float(np.std(quantities)) + 1e-8
        qty_z = float((quantities[-1] - qty_mean) / qty_std)

        sma = float(np.mean(prices[-20:]))
        deviation = float((prices[-1] - sma) / (sma + 1e-8) * 100.0 / 2.0)

        if len(self.timestamps) >= 2:
            dt_sec = max((self.timestamps[-1] - self.timestamps[0]) / 1_000_000.0, 1e-6)
            trade_rate = len(self.timestamps) / dt_sec
        else:
            trade_rate = 0.0
        trade_rate_feature = (math.log1p(trade_rate) - 3.0) / 3.0

        inventory_norm = inventory / max(1e-8, max_inventory)

        state = np.array(
            [
                np.clip(imbalance, -1.0, 1.0),
                np.clip(deviation, -1.0, 1.0),
                np.clip(momentum, -1.0, 1.0),
                np.clip(volatility, -1.0, 1.0),
                np.clip(qty_z / 3.0, -1.0, 1.0),
                np.clip((trade_rate_feature + inventory_norm) / 2.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return state


def parse_trade_row(raw: list[str]) -> Optional[TradeRow]:
    if len(raw) < 8:
        return None
    try:
        return TradeRow(
            price=float(raw[1]),
            quantity=float(raw[2]),
            timestamp_us=int(raw[5]),
            is_buyer_maker=raw[6].strip().lower() == "true",
        )
    except Exception:
        return None


def stream_trades(csv_path: str) -> Iterator[TradeRow]:
    with open(csv_path, "r", newline="") as file:
        reader = csv.reader(file)
        for raw in reader:
            row = parse_trade_row(raw)
            if row is not None:
                yield row


def analyze_dataset(csv_path: str, sample_rows: int = 2_000_000) -> Dict[str, float]:
    print("\n[1/3] Dataset analysis (sample-based)")
    file_size_bytes = os.path.getsize(csv_path)

    min_price = float("inf")
    max_price = 0.0
    min_qty = float("inf")
    max_qty = 0.0
    sum_qty = 0.0
    buy_count = 0
    sell_count = 0
    first_ts = None
    last_ts = None
    bytes_acc = 0
    rows = 0

    with open(csv_path, "r", newline="") as file:
        reader = csv.reader(file)
        for raw in reader:
            row = parse_trade_row(raw)
            if row is None:
                continue

            rows += 1
            bytes_acc += sum(len(x) for x in raw) + max(0, len(raw) - 1) + 1
            min_price = min(min_price, row.price)
            max_price = max(max_price, row.price)
            min_qty = min(min_qty, row.quantity)
            max_qty = max(max_qty, row.quantity)
            sum_qty += row.quantity
            if row.is_buyer_maker:
                sell_count += 1
            else:
                buy_count += 1

            if first_ts is None:
                first_ts = row.timestamp_us
            last_ts = row.timestamp_us

            if rows >= sample_rows:
                break

    avg_row_bytes = bytes_acc / max(1, rows)
    estimated_total_rows = int(file_size_bytes / max(1e-8, avg_row_bytes))

    first_dt = datetime.fromtimestamp((first_ts or 0) / 1_000_000.0, tz=timezone.utc)
    last_dt = datetime.fromtimestamp((last_ts or 0) / 1_000_000.0, tz=timezone.utc)

    print(f"  File: {csv_path}")
    print(f"  Size: {file_size_bytes / (1024**3):.2f} GB")
    print(f"  Sample rows: {rows:,}")
    print(f"  Estimated total rows: {estimated_total_rows:,}")
    print(f"  Price range (sample): {min_price:.2f} -> {max_price:.2f}")
    print(f"  Qty range (sample): {min_qty:.6f} -> {max_qty:.6f}")
    print(f"  Mean qty (sample): {sum_qty / max(1, rows):.6f}")
    print(f"  Taker buy ratio (sample): {buy_count / max(1, rows):.3f}")
    print(f"  Taker sell ratio (sample): {sell_count / max(1, rows):.3f}")
    print(f"  Time range (sample): {first_dt.isoformat()} -> {last_dt.isoformat()}")

    return {
        "sample_rows": float(rows),
        "estimated_total_rows": float(estimated_total_rows),
        "first_ts": float(first_ts or 0),
        "last_ts": float(last_ts or 0),
    }


def gpu_summary() -> str:
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    total_mb = torch.cuda.get_device_properties(index).total_memory / (1024**2)
    return f"GPU {index}: {name} | VRAM {total_mb:.0f} MB"


def format_metrics_line(metrics: Dict[str, float]) -> str:
    return (
        f"step={int(metrics['step']):>8} | "
        f"loss={metrics['loss']:+.6f} | reward={metrics['reward']:+.6f} | "
        f"eps={metrics['epsilon']:.4f} | q={metrics['q_mean']:+.5f} | "
        f"pnl={metrics['pnl']:+.2f} | inv={metrics['inventory']:+.2f} | "
        f"hold/buy/sell={metrics['hold_ratio']:.2f}/{metrics['buy_ratio']:.2f}/{metrics['sell_ratio']:.2f} | "
        f"sharpe≈{metrics['sharpe']:+.3f} | maxDD={metrics['max_drawdown']:.3f} | "
        f"samples/s={metrics['samples_per_sec']:.0f} | gpuMB={metrics['gpu_mem_mb']:.0f}"
    )


def train_from_aggtrades(
    csv_path: str,
    epochs: int,
    train_rows_per_epoch: int,
    warmup_rows: int,
    log_every_steps: int,
    train_every_steps: int,
    gradient_steps: int,
    model_out: str,
    metrics_out: str,
    max_inventory: float,
    qty_per_trade: float,
    strategy_variant: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run. No GPU detected by PyTorch.")

    torch.backends.cudnn.benchmark = True
    print("\n[2/3] GPU setup")
    print(f"  Using {gpu_summary()}")

    strategy_cls = FastDQNStrategy if strategy_variant == "fast" else NormalDQNStrategy

    if strategy_variant == "fast":
        batch_size = 1024
        replay_capacity = 300_000
        target_update_steps = 2_000
        epsilon_decay = 0.99995
        train_every_steps = max(1, train_every_steps)
        gradient_steps = max(1, gradient_steps)
    else:
        batch_size = 256
        replay_capacity = 50_000
        target_update_steps = 1_000
        epsilon_decay = 0.9998
        train_every_steps = 1
        gradient_steps = 1

    strategy = strategy_cls(
        symbol="BTCUSDT",
        model_path=model_out,
        quantity_per_trade=qty_per_trade,
        max_inventory=max_inventory,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=epsilon_decay,
        gamma=0.995,
        learning_rate=3e-4,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        target_update_steps=target_update_steps,
        device="cuda",
    )
    strategy.set_training_mode(True)
    strategy.model_loaded = True

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)

    feature_builder = RollingWindowFeatures(window_size=200)
    action_counts = np.zeros(3, dtype=np.int64)
    rewards_window = deque(maxlen=5_000)
    losses_window = deque(maxlen=5_000)

    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    cash = 0.0
    inventory = 0.0
    trade_fee = 0.0002

    global_step = 0
    updates = 0
    last_q_mean = 0.0

    with open(metrics_out, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "timestamp",
                "epoch",
                "step",
                "loss",
                "reward",
                "epsilon",
                "q_mean",
                "pnl",
                "inventory",
                "hold_ratio",
                "buy_ratio",
                "sell_ratio",
                "sharpe_proxy",
                "max_drawdown",
                "samples_per_sec",
                "gpu_mem_mb",
            ]
        )

        print("\n[3/3] Training (real-time metrics)")
        start_train = time.time()

        for epoch in range(1, epochs + 1):
            feature_builder = RollingWindowFeatures(window_size=200)
            inventory = 0.0
            cash = 0.0
            equity = 0.0
            peak_equity = 0.0
            max_drawdown = 0.0

            pending_state = None
            pending_action = None
            pending_price = None

            prev_row = None
            consumed_rows = 0

            for row in stream_trades(csv_path):
                consumed_rows += 1
                feature_builder.update(row)

                if prev_row is None:
                    prev_row = row
                    continue

                if consumed_rows < warmup_rows:
                    prev_row = row
                    continue

                state = feature_builder.build_state(inventory=inventory, max_inventory=max_inventory)
                if state is None:
                    prev_row = row
                    continue

                strategy.current_inventory = inventory
                if random.random() < strategy.epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.inference_mode():
                        state_t = torch.from_numpy(state).to(strategy.device, non_blocking=True).unsqueeze(0)
                        q_values = strategy.policy_net(state_t)[0]
                    action = int(torch.argmax(q_values).item())
                    last_q_mean = float(q_values.mean().item())

                price = row.price
                if action == 1 and inventory < max_inventory:
                    inventory += qty_per_trade
                    cash -= price * qty_per_trade * (1.0 + trade_fee)
                elif action == 2 and inventory > 0.0:
                    inventory -= qty_per_trade
                    cash += price * qty_per_trade * (1.0 - trade_fee)
                else:
                    action = 0

                action_counts[action] += 1

                next_state = feature_builder.build_state(inventory=inventory, max_inventory=max_inventory)
                if next_state is None:
                    prev_row = row
                    continue

                next_price = row.price
                pnl_now = cash + inventory * next_price
                reward = (pnl_now - equity) / (abs(prev_row.price) + 1e-8)
                reward -= 0.0002 * abs(inventory / max_inventory)

                equity = pnl_now
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / (abs(peak_equity) + 1e-8) if peak_equity > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

                strategy.remember(state, action, float(reward), next_state, False)
                if global_step % max(1, train_every_steps) == 0:
                    for _ in range(max(1, gradient_steps)):
                        loss = strategy.optimize_model()
                        if loss is not None:
                            losses_window.append(loss)
                            updates += 1

                rewards_window.append(float(reward))

                global_step += 1
                if global_step % log_every_steps == 0 and len(losses_window) > 0:
                    elapsed = max(1e-6, time.time() - start_train)
                    samples_per_sec = global_step / elapsed
                    rewards_np = np.asarray(rewards_window, dtype=np.float64)
                    sharpe_proxy = 0.0
                    if rewards_np.size > 5 and rewards_np.std() > 1e-9:
                        sharpe_proxy = float(np.sqrt(60.0) * rewards_np.mean() / rewards_np.std())

                    total_actions = max(1, int(action_counts.sum()))
                    gpu_mem_mb = torch.cuda.memory_allocated() / (1024**2)
                    metrics = {
                        "step": float(global_step),
                        "loss": float(np.mean(losses_window)),
                        "reward": float(np.mean(rewards_window)),
                        "epsilon": float(strategy.epsilon),
                        "q_mean": last_q_mean,
                        "pnl": equity,
                        "inventory": inventory,
                        "hold_ratio": float(action_counts[0] / total_actions),
                        "buy_ratio": float(action_counts[1] / total_actions),
                        "sell_ratio": float(action_counts[2] / total_actions),
                        "sharpe": sharpe_proxy,
                        "max_drawdown": float(max_drawdown),
                        "samples_per_sec": float(samples_per_sec),
                        "gpu_mem_mb": float(gpu_mem_mb),
                    }

                    print(format_metrics_line(metrics), flush=True)
                    writer.writerow(
                        [
                            datetime.now(timezone.utc).isoformat(),
                            epoch,
                            global_step,
                            metrics["loss"],
                            metrics["reward"],
                            metrics["epsilon"],
                            metrics["q_mean"],
                            metrics["pnl"],
                            metrics["inventory"],
                            metrics["hold_ratio"],
                            metrics["buy_ratio"],
                            metrics["sell_ratio"],
                            metrics["sharpe"],
                            metrics["max_drawdown"],
                            metrics["samples_per_sec"],
                            metrics["gpu_mem_mb"],
                        ]
                    )
                    mf.flush()

                prev_row = row
                if consumed_rows >= train_rows_per_epoch:
                    break

            strategy.save_model(model_out)
            print(
                f"Epoch {epoch}/{epochs} finished | rows={consumed_rows:,} | updates={updates:,} | "
                f"epsilon={strategy.epsilon:.4f} | pnl={equity:+.2f}"
            )

    print(f"\nTraining completed. Model saved to: {model_out}")
    print(f"Metrics log saved to: {metrics_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-first DQN trainer for Binance aggTrades CSV")
    parser.add_argument("--csv", type=str, default="BTCUSDT-aggTrades-2026-03.csv", help="Path to aggTrades CSV")
    parser.add_argument("--variant", type=str, choices=["normal", "fast"], default="fast", help="Choose the DQN implementation")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rows-per-epoch", type=int, default=8_000_000)
    parser.add_argument("--warmup-rows", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=2_000)
    parser.add_argument("--train-every-steps", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--sample-analysis-rows", type=int, default=1_000_000)
    parser.add_argument("--model-out", type=str, default="models/dqn_agent.pth")
    parser.add_argument("--metrics-out", type=str, default="data/dqn_training_metrics.csv")
    parser.add_argument("--max-inventory", type=float, default=50.0)
    parser.add_argument("--qty-per-trade", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    analyze_dataset(args.csv, sample_rows=args.sample_analysis_rows)
    train_from_aggtrades(
        csv_path=args.csv,
        epochs=args.epochs,
        train_rows_per_epoch=args.rows_per_epoch,
        warmup_rows=args.warmup_rows,
        log_every_steps=args.log_every,
        train_every_steps=args.train_every_steps,
        gradient_steps=args.gradient_steps,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        max_inventory=args.max_inventory,
        qty_per_trade=args.qty_per_trade,
        strategy_variant=args.variant,
    )


if __name__ == "__main__":
    main()