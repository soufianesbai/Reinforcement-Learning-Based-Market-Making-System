"""
DQN Training Script for Cryptocurrency Trading

This script:
1. Loads and preprocesses SOLUSDT market data
2. Creates a trading environment with discrete actions
3. Trains a DQN agent using experience replay
4. Evaluates performance and generates visualizations
"""

import numpy as np
import pandas as pd
import os
import sys
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict, TYPE_CHECKING

# Add workspace root to path to import agents module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

if TYPE_CHECKING:
    # Import for type checking only (avoids requiring TensorFlow at runtime)
    from agents.strategies.dqn_agent import DQNAgent


def build_volume_bars(csv_path: str, volume_threshold: float = 1000) -> pd.DataFrame:
    """
    Transform raw transactions into volume-based candles (bars).
    
    Instead of fixed time intervals, bars are created when accumulated volume
    reaches a threshold. This provides more consistent sampling of market activity.
    
    Args:
        csv_path: Path to the aggregated trades CSV file
        volume_threshold: Volume threshold in SOL for bar creation
    
    Returns:
        DataFrame with OHLCV (Open, High, Low, Close, Volume) data
    """
    print(f"📖 Reading file: {csv_path}...")
    
    # Load raw trade data
    df = pd.read_csv(
        csv_path,
        header=None,
        names=['id', 'price', 'qty', 'f', 'l', 'timestamp', 'm', 'M'],
        low_memory=False
    )
    
    # Convert to numeric types and remove corrupted rows
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df = df.dropna(subset=['price', 'qty'])
    
    if len(df) == 0:
        raise ValueError("❌ ERROR: CSV file is empty!")
    
    print(f"✓ Loaded {len(df):,} trades")
    print(f"📊 Creating Volume Bars (threshold: {volume_threshold} SOL)...")
    
    # Accumulate volume and assign to bars
    df['cum_vol'] = df['qty'].cumsum()
    df['bar_id'] = (df['cum_vol'] // volume_threshold).astype(int)
    
    # Aggregate trades into bars
    ohlcv = df.groupby('bar_id').agg(
        open=('price', 'first'),      # First price
        high=('price', 'max'),        # Max price
        low=('price', 'min'),         # Min price
        close=('price', 'last'),      # Last price
        volume=('qty', 'sum')         # Total volume
    ).reset_index(drop=True)
    
    # Remove incomplete last bar (< 90% of threshold)
    if len(ohlcv) > 0 and ohlcv.iloc[-1]['volume'] < (volume_threshold * 0.9):
        ohlcv = ohlcv.iloc[:-1]
    
    print(f"✓ Generated {len(ohlcv):,} volume bars")
    return ohlcv


class CryptoTradingEnv:
    """
    Simple trading environment for DQN agent.
    
    State representation: A sliding window of price features normalized by the current price
    Actions: 0=SELL, 1=HOLD, 2=BUY
    Reward: Profit from closed positions + penalty for holding inventory
    
    Args:
        csv_path: Path to market data CSV
        window_size: Number of historical bars to include in state
        volume_threshold: Volume threshold for bar creation
        max_inventory: Maximum position size (in coin units)
    """
    
    def __init__(self, csv_path: str, window_size: int = 30, 
                 volume_threshold: float = 1000, max_inventory: int = 3):
        # Load market data
        self.ohlcv = build_volume_bars(csv_path, volume_threshold)
        self.closes = self.ohlcv['close'].values.astype(np.float32)
        self.highs = self.ohlcv['high'].values.astype(np.float32)
        self.lows = self.ohlcv['low'].values.astype(np.float32)
        self.volumes = self.ohlcv['volume'].values.astype(np.float32)
        
        self.window_size = window_size
        self.max_inventory = max_inventory
        
        # Episode configuration
        total_candles = len(self.closes)
        if total_candles < 250:
            self.max_steps = max(10, total_candles - window_size - 5)
        else:
            self.max_steps = 200
        
        # State dimension: expanded features × window_size + 1 inventory
        # Features per candle: normalized close, high, low, volume, log_return, sma_fast, sma_slow
        self.features_per_candle = 7
        self.state_dim = window_size * self.features_per_candle + 1
        
        print(f"📈 Environment created:")
        print(f"   State dimension: {self.state_dim}")
        print(f"   Max steps per episode: {self.max_steps}")
        print(f"   Total candles available: {total_candles}")
    
    def reset(self, seed: int = None) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial observation (state)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.steps_this_episode = 0
        
        # Random starting position
        max_start = len(self.closes) - self.max_steps - 2
        if max_start <= self.window_size:
            self.current_step = self.window_size
        else:
            self.current_step = np.random.randint(self.window_size, max_start)
        
        # Reset trading state
        self.inventory = 0
        self.cash = 0.0
        self.last_price = self.closes[self.current_step]
        self.episode_trades = 0
        self.episode_realized_pnl = 0.0
        self.cost_basis = deque()
        self.cost_basis_short = deque()
        self.trade_history = []
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """
        Construct the state observation from market data.
        
        Normalized features:
        - Price changes relative to current price
        - Volume relative to average volume
        - Current inventory position
        
        Returns:
            State vector as 1D numpy array
        """
        i = self.current_step
        w = self.window_size
        
        # Extract window of price data
        close_w = self.closes[i-w:i]
        high_w = self.highs[i-w:i]
        low_w = self.lows[i-w:i]
        volume_w = self.volumes[i-w:i]
        
        # Normalize by current price
        current_price = close_w[-1]
        base = current_price + 1e-8
        norm_close = (close_w / base) - 1.0
        norm_high = (high_w / base) - 1.0
        norm_low = (low_w / base) - 1.0

        # Normalize volume by average
        vol_mean = volume_w.mean() + 1e-8
        norm_volume = (volume_w / vol_mean) - 1.0

        # Additional features: log returns and simple moving averages
        # Log returns (length w): pad first element with 0
        returns = np.zeros_like(close_w)
        if len(close_w) > 1:
            returns[1:] = (close_w[1:] - close_w[:-1]) / (close_w[:-1] + 1e-8)
        norm_returns = returns  # already relative

        # SMA fast/slow
        try:
            sma_fast = pd.Series(close_w).rolling(window=5, min_periods=1).mean().values
            sma_slow = pd.Series(close_w).rolling(window=15, min_periods=1).mean().values
        except Exception:
            # Fallback to simple mean windows if pandas rolling fails
            sma_fast = np.repeat(close_w.mean(), len(close_w))
            sma_slow = np.repeat(close_w.mean(), len(close_w))

        norm_sma_fast = (sma_fast / base) - 1.0
        norm_sma_slow = (sma_slow / base) - 1.0
        
        # Normalize inventory position
        norm_inventory = self.inventory / self.max_inventory
        
        # Concatenate all features: order is important and matches state_dim
        obs = np.concatenate([
            norm_close,
            norm_high,
            norm_low,
            norm_volume,
            norm_returns,
            norm_sma_fast,
            norm_sma_slow,
            [norm_inventory]
        ])
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one trading step.
        
        Actions:
        - 0: SELL (reduce position)
        - 1: HOLD (no action)
        - 2: BUY (increase position)
        
        Args:
            action: Action index (0, 1, or 2)
        
        Returns:
            Tuple of (observation, reward, done)
        """
        current_price = self.closes[self.current_step]
        self.current_step += 1
        next_price = self.closes[self.current_step]
        
        # Execute action
        realized_pnl = 0.0
        
        if action == 2:  # BUY
            if self.inventory < 0 and len(self.cost_basis_short) > 0:
                # Close one short position (FIFO)
                short_entry = self.cost_basis_short.popleft()
                realized_pnl = short_entry - current_price
                self.inventory += 1
                self.episode_trades += 1
                self.episode_realized_pnl += realized_pnl
                self.trade_history.append(('COVER', current_price, self.inventory))
            elif self.inventory < self.max_inventory:
                # Open one long position
                self.inventory += 1
                self.cost_basis.append(current_price)
                self.episode_trades += 1
                self.trade_history.append(('BUY', current_price, self.inventory))
        
        elif action == 0:  # SELL
            if self.inventory > 0 and len(self.cost_basis) > 0:
                # Close one long position (FIFO)
                entry = self.cost_basis.popleft()
                realized_pnl = current_price - entry
                self.inventory -= 1
                self.episode_trades += 1
                self.episode_realized_pnl += realized_pnl
                self.trade_history.append(('SELL', current_price, self.inventory))
            elif self.inventory > -self.max_inventory:
                # Open one short position
                self.inventory -= 1
                self.cost_basis_short.append(current_price)
                self.episode_trades += 1
                self.trade_history.append(('SHORT', current_price, self.inventory))
        
        # HOLD (action == 1): do nothing
        
        # Compute reward
        # Main reward: realized P&L from closed positions
        reward = realized_pnl
        
        # Penalty for holding inventory (opportunity cost)
        inventory_penalty = 0.01 * (self.inventory ** 2)
        reward -= inventory_penalty
        
        # Encourage only profitable closes; penalize pointless non-HOLD actions
        if realized_pnl > 0:
            reward += 0.05
        elif action != 1 and realized_pnl == 0.0:
            reward -= 0.02
        
        # Clip reward to prevent explosive gradients
        reward = np.clip(reward, -5.0, 5.0)
        
        self.steps_this_episode += 1
        done = self.steps_this_episode >= self.max_steps
        
        return self._get_obs(), float(reward), done
    
    def render(self) -> Dict[str, float]:
        """Get current episode statistics."""
        return {
            'trades': self.episode_trades,
            'inventory': self.inventory,
            'realized_pnl': self.episode_realized_pnl,
            'step': self.steps_this_episode
        }


def train_dqn(csv_path: str, num_episodes: int = 50, save_checkpoint_interval: int = 5) -> Tuple['DQNAgent', 'CryptoTradingEnv', Dict]:
    """
    Train a DQN agent on cryptocurrency trading data.
    
    Args:
        csv_path: Path to market data CSV
        num_episodes: Number of training episodes
        save_checkpoint_interval: Save model every N episodes
    
    Returns:
        Tuple of (agent, environment, training metrics)
    """
    # Create environment
    env = CryptoTradingEnv(csv_path, window_size=30, volume_threshold=1000)
    
    # Use the full TensorFlow DQNAgent implementation only (full model)
    # Import the full TensorFlow-backed DQNAgent at runtime.
    try:
        from agents.strategies.dqn_agent import DQNAgent
    except Exception as e:
        print("❌ Could not import the TensorFlow DQNAgent:", e)
        print("   Please install TensorFlow in your environment, e.g.:")
        print("     pip install tensorflow")
        print("   Or activate the venv that has TensorFlow available.")
        sys.exit(1)

    print("ℹ Using TensorFlow DQNAgent (full implementation)")
    agent = DQNAgent(
        state_dim=env.state_dim,
        num_actions=3,  # SELL, HOLD, BUY
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_freq=1000,
        replay_buffer_size=50000,
        batch_size=32
    )
    policy_ext = ".keras"
    
    # Training metrics
    episode_rewards = []
    episode_trades = []
    episode_inventory = []
    episode_realized_pnl = []
    episode_epsilons = []
    episode_losses = []
    best_episode_reward = -np.inf
    best_model_path = os.path.join("models_checkpoints", "dqn_policy_best.keras")
    
    print("\n" + "="*60)
    print("🚀 Starting DQN Training")
    print("="*60)
    
    # Create checkpoint directory
    os.makedirs("models_checkpoints", exist_ok=True)
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        done = False
        episode_step_losses = []
        
        while not done:
            # Select action using epsilon-greedy
            action = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Train on batch
            loss = agent.replay()
            if loss is not None:
                episode_step_losses.append(float(loss))
            
            episode_reward += reward
            state = next_state
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Collect metrics
        stats = env.render()
        episode_rewards.append(episode_reward)
        episode_trades.append(stats['trades'])
        episode_inventory.append(abs(stats['inventory']))
        episode_realized_pnl.append(stats['realized_pnl'])
        episode_epsilons.append(agent.get_epsilon())
        episode_loss = float(np.mean(episode_step_losses)) if episode_step_losses else 0.0
        episode_losses.append(episode_loss)

        # Save the best model by episode reward
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            agent.save(best_model_path)
        
        # Print progress
        print(f"📌 Episode {episode+1:3d}/{num_episodes} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Trades: {stats['trades']:3d} | "
              f"PnL: {stats['realized_pnl']:7.2f} | "
              f"Loss: {episode_loss:7.4f} | "
              f"ε: {agent.get_epsilon():.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_checkpoint_interval == 0:
            agent.save(f"models_checkpoints/dqn_policy_ep{episode+1}{policy_ext}")
    
    # Save final model
    agent.save(f"models_checkpoints/dqn_policy_FINAL{policy_ext}")
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    
    return agent, env, {
        'episode_rewards': episode_rewards,
        'episode_trades': episode_trades,
        'episode_inventory': episode_inventory,
        'episode_realized_pnl': episode_realized_pnl,
        'episode_epsilons': episode_epsilons,
        'episode_losses': episode_losses,
        'best_model_path': best_model_path,
        'agent_losses': agent.losses,
        'agent_q_values': agent.q_values_mean
    }


def evaluate_dqn(agent: 'DQNAgent', env: 'CryptoTradingEnv', num_episodes: int = 5) -> Dict[str, float]:
    """
    Evaluate a trained DQN agent without exploration.
    
    Args:
        agent: Trained DQN agent
        env: Trading environment
        num_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("🧪 Evaluating DQN Agent")
    print("="*60)
    
    eval_rewards = []
    eval_trades = []
    eval_pnl = []
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            # Always exploit (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        
        stats = env.render()
        eval_rewards.append(episode_reward)
        eval_trades.append(stats['trades'])
        eval_pnl.append(stats['realized_pnl'])
        
        print(f"📊 Eval Episode {ep+1}/{num_episodes} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Trades: {stats['trades']:3d} | "
              f"PnL: {stats['realized_pnl']:7.2f}")
    
    metrics = {
        'avg_reward': np.mean(eval_rewards),
        'avg_trades': np.mean(eval_trades),
        'avg_pnl': np.mean(eval_pnl),
        'std_reward': np.std(eval_rewards),
    }
    
    print(f"\n📈 Average Reward: {metrics['avg_reward']:.2f}")
    print(f"📈 Average Trades/Episode: {metrics['avg_trades']:.1f}")
    print(f"📈 Average PnL: {metrics['avg_pnl']:.2f}")
    
    return metrics


def plot_training_results(metrics: Dict, output_dir: str = ".") -> None:
    """
    Generate comprehensive training visualization plots.
    
    Args:
        metrics: Dictionary with training metrics
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("⚠️  Matplotlib not available or incompatible with current NumPy; skipping plots.")
        return

    print("\n📊 Generating plots...")

    # Smoothing function for better visualization
    smooth = lambda data, w=20: pd.Series(data).rolling(window=w, min_periods=1).mean()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("DQN Training Metrics — Cryptocurrency Trading", fontsize=16, fontweight="bold")
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    ax.plot(metrics['episode_rewards'], alpha=0.3, color="blue", label="Raw reward")
    ax.plot(smooth(metrics['episode_rewards'], 20), color="blue", linewidth=2, label="Smoothed (20 ep)")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Episode Rewards", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss per Episode
    ax = axes[0, 1]
    if metrics.get('episode_losses'):
        ax.plot(metrics['episode_losses'], alpha=0.5, color="red", label="Episode loss")
        ax.plot(smooth(metrics['episode_losses'], 5), color="red", linewidth=2, label="Loss (smoothed)")
        ax.set_title("DQN Loss Per Episode", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Plot 3: Epsilon Decay
    ax = axes[1, 0]
    ax.plot(metrics['episode_epsilons'], color="green", linewidth=2, label="Exploration rate (ε)")
    ax.set_title("Epsilon Decay (Exploration → Exploitation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Plot 4: Number of Trades
    ax = axes[1, 1]
    ax.plot(metrics['episode_trades'], alpha=0.3, color="teal")
    ax.plot(smooth(metrics['episode_trades'], 20), color="teal", linewidth=2, label="Trades/episode")
    ax.set_title("Trading Activity", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of Trades")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Average Inventory
    ax = axes[2, 0]
    ax.plot(metrics['episode_inventory'], alpha=0.3, color="brown")
    ax.plot(smooth(metrics['episode_inventory'], 20), color="brown", linewidth=2, label="Avg position (|inv|)")
    ax.set_title("Position Management", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Inventory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Realized P&L
    ax = axes[2, 1]
    ax.plot(metrics['episode_realized_pnl'], alpha=0.3, color="purple")
    ax.plot(smooth(metrics['episode_realized_pnl'], 20), color="purple", linewidth=2, label="Realized P&L")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Cumulative Realized Profit/Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Realized P&L")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "dqn_training_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Path to market data
    csv_file = "SOLUSDT-aggTrades-2026-02_1.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("   Please ensure the CSV file is in the project root")
        exit(1)
    
    # Train agent
    agent, env, metrics = train_dqn(csv_file, num_episodes=50, save_checkpoint_interval=5)
    
    # Evaluate agent
    eval_metrics = evaluate_dqn(agent, env, num_episodes=5)
    
    # Generate plots
    plot_training_results(metrics)
    
    print("\n" + "="*60)
    print("🎉 DQN Training and Evaluation Complete!")
    print("="*60)
    print("Model saved to: models_checkpoints/dqn_policy_FINAL.keras")
    print(f"Best model saved to: {metrics['best_model_path']}")
    print(f"Plots saved to: dqn_training_metrics.png")
