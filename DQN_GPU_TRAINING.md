# GPU DQN Training on `BTCUSDT-aggTrades-2026-03.csv`

## Run

```bash
python3 -m agents.train_dqn \
  --csv BTCUSDT-aggTrades-2026-03.csv \
  --epochs 3 \
  --rows-per-epoch 12000000 \
  --log-every 2000 \
   --train-every-steps 4 \
   --gradient-steps 1 \
  --model-out models/dqn_agent.pth \
  --metrics-out data/dqn_training_metrics.csv
```

## What happens

1. **Dataset analysis** (sample-based):
   - file size
   - estimated number of rows
   - sampled price/quantity ranges
   - taker buy/sell ratio
   - sampled time range

2. **GPU check**:
   - training aborts if CUDA is not available
   - prints GPU name and VRAM

3. **Streaming DQN training**:
   - reads the 3GB CSV progressively (no full load in RAM)
   - updates replay buffer and trains continuously
   - saves model after each epoch

## Real-time terminal metrics

At each logging interval, the trainer prints:

- `loss`: running SmoothL1 TD loss
- `reward`: running average normalized reward
- `eps`: current epsilon (exploration)
- `q`: mean Q-value from policy network
- `pnl`: mark-to-market strategy PnL (simulated)
- `inv`: current inventory
- `hold/buy/sell`: action distribution ratios
- `sharpe≈`: rolling Sharpe proxy from recent rewards
- `maxDD`: max drawdown from current epoch equity
- `samples/s`: training throughput
- `gpuMB`: allocated GPU memory in MB

## Speed tuning (recommended)

- Keep `--train-every-steps 4` to reduce per-row optimization overhead.
- Increase `--gradient-steps` only if learning becomes too slow (start with `1`).
- Start with a smaller pilot run first, e.g.:

```bash
python3 -m agents.train_dqn \
   --csv BTCUSDT-aggTrades-2026-03.csv \
   --epochs 1 \
   --rows-per-epoch 500000 \
   --log-every 2000 \
   --train-every-steps 4 \
   --gradient-steps 1
```

## Output artifacts

- Model checkpoint: `models/dqn_agent.pth`
- Metrics CSV: `data/dqn_training_metrics.csv`


# Commands to run the training : 
Use these commands:

Normal DQN:
python3 -m agents.train_dqn --variant normal --csv BTCUSDT-aggTrades-2026-03.csv --epochs 1 --rows-per-epoch 500000 --warmup-rows 2000 --log-every 2000

Fast DQN:
python3 -m agents.train_dqn --variant fast --csv BTCUSDT-aggTrades-2026-03.csv --epochs 1 --rows-per-epoch 500000 --warmup-rows 2000 --log-every 2000