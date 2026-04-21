# DQN Strategy Implementation

## What was implemented
- A PyTorch-based `DQNStrategy` that follows `BaseStrategy`.
- Automatic CUDA selection when a GPU is available, with CPU fallback.
- A dueling Q-network with LayerNorm and SiLU activations for more stable trading signals.
- A replay buffer, target network, and Double DQN target computation.
- Smooth L1 loss and gradient clipping for robustness on noisy market data.
- Live inference with inventory guards so the agent does not SELL without inventory.

## Why these parameters
- `state_dim = 6`: adds realized volatility on top of imbalance, spread, deviation, momentum, and inventory.
- Hidden layers `256 -> 256 -> 128`: larger than the old NumPy model so the network can learn richer trade patterns from large datasets, while still being efficient on GPU.
- Dueling head: helps separate the value of staying flat from the advantage of BUY/SELL actions, which is useful in markets where HOLD is often the best action.
- `gamma = 0.995`: keeps a slightly longer horizon than the classic 0.99, which is useful when market impact unfolds over several steps.
- `learning_rate = 3e-4`: a safer default for AdamW on a deep network and noisy rewards.
- `batch_size = 256`: better GPU utilization and more stable gradient estimates on large replay buffers.
- `replay_capacity = 200_000`: enough variety for large training runs without becoming memory-heavy.
- `target_update_steps = 1_000`: reduces target churn while still refreshing the bootstrap target regularly.
- `epsilon` schedule (`1.0 -> 0.01` with slow decay): supports exploration during training but remains mostly greedy in production.

## Validation
The manual validation script confirms:
- BUY selection works
- SELL selection works
- SELL is blocked when inventory is empty

Result: the strategy now runs with PyTorch tensors, supports CUDA training/inference, and keeps the same trading interface.
