"""
Deep Q-Network (DQN) Agent for Cryptocurrency Trading

This module implements a DQN agent for making trading decisions (BUY, SELL, HOLD)
based on cryptocurrency market data. It includes:
- Neural network for Q-value approximation
- Experience replay buffer for decorrelated learning
- Target network for stable training
- Epsilon-greedy exploration strategy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from collections import deque
from typing import Tuple, Dict, Optional


class DQNNetwork(keras.Model):
    """
    Neural network for Q-value estimation in DQN.
    
    This is a simple MLP (Multi-Layer Perceptron) that learns to estimate
    the Q-value (expected future reward) for each action given a state.
    
    Architecture: Input -> Dense(128, ReLU) -> Dense(128, ReLU) -> Output (3 actions)
    
    Args:
        state_dim (int): Dimension of the input state
        num_actions (int): Number of possible actions (3 for BUY, SELL, HOLD)
    """
    def __init__(self, state_dim: int, num_actions: int = 3, hidden_sizes: tuple = (256, 256, 128, 64), **kwargs):
        super().__init__(**kwargs)
        # Store constructor args so the model can be serialized
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        # Ensure hidden sizes are tuples of ints
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        # Build dense layers according to hidden_sizes
        self.hidden_layers = []
        for i, h in enumerate(self.hidden_sizes):
            layer = layers.Dense(h, activation='relu', name=f'dense_{i+1}')
            setattr(self, f'dense_{i+1}', layer)
            self.hidden_layers.append(layer)

        self.output_layer = layers.Dense(self.num_actions, name='q_output')
    
    def call(self, state: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass to compute Q-values for all actions.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            training: Boolean flag for training mode (used by BatchNorm, Dropout, etc.)
        
        Returns:
            Q-values of shape (batch_size, num_actions)
        """
        x = state
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'hidden_sizes': list(self.hidden_sizes),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct the model from its config."""
        # Convert hidden_sizes back to tuple if needed
        if 'hidden_sizes' in config and isinstance(config['hidden_sizes'], list):
            config['hidden_sizes'] = tuple(config['hidden_sizes'])
        return cls(**config)


class ReplayBuffer:
    """
    Experience Replay Buffer for decorrelated training data.
    
    Stores transitions (state, action, reward, next_state, done) and samples them
    in random batches to break temporal correlations in the data. This is crucial
    for stable training in DQN.
    
    Args:
        max_size (int): Maximum number of transitions to store (default: 50000)
        state_dim (int): Dimension of each state
    """
    def __init__(self, max_size: int = 50000, state_dim: int = 121):
        self.max_size = max_size
        self.state_dim = state_dim
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken (0=SELL, 1=HOLD, 2=BUY)
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Trading.
    
    This agent learns to make trading decisions by estimating Q-values (expected
    cumulative future rewards) for each possible action. It uses:
    - A main network to select actions
    - A target network for stable learning (updated periodically)
    - Experience replay to decorrelate training samples
    - Epsilon-greedy exploration to balance exploration vs exploitation
    
    Key Parameters:
    - learning_rate: Controls how fast the network learns (typically 1e-4 to 1e-3)
    - gamma: Discount factor (0.99) - how much future rewards matter
    - epsilon: Initial exploration rate (1.0) - probability of random action
    - epsilon_min: Minimum exploration rate (0.01)
    - epsilon_decay: How fast exploration decreases per episode
    - target_update_freq: How often to update target network
    - batch_size: Number of transitions per training step
    """
    
    def __init__(self, state_dim: int, num_actions: int = 3,
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, target_update_freq: int = 1000,
                 replay_buffer_size: int = 50000, batch_size: int = 32):
        """
        Initialize the DQN Agent.
        
        Args:
            state_dim: Dimension of the state space
            num_actions: Number of possible actions (3: BUY, SELL, HOLD)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon per episode
            target_update_freq: Update target network every N training steps
            replay_buffer_size: Maximum size of replay buffer
            batch_size: Batch size for training
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Create main and target networks (larger/deeper by default)
        hidden = (512, 512, 256, 128)
        self.q_network = DQNNetwork(state_dim, num_actions, hidden_sizes=hidden)
        self.target_network = DQNNetwork(state_dim, num_actions, hidden_sizes=hidden)
        
        # Copy initial weights to target network
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Optimizer and loss function
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size, state_dim=state_dim)
        
        # Tracking metrics
        self.training_steps = 0
        self.losses = []
        self.q_values_mean = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        With probability epsilon: select a random action (exploration)
        Otherwise: select the action with highest Q-value (exploitation)
        
        Args:
            state: Current state (1D array)
            training: If True, use epsilon-greedy; if False, always exploit
        
        Returns:
            Action index (0=SELL, 1=HOLD, 2=BUY)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploitation: best action according to Q-network
            state_t = tf.convert_to_tensor(np.expand_dims(state, 0), dtype=tf.float32)
            q_values = self.q_network(state_t, training=False)
            return np.argmax(q_values.numpy()[0])
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """
        Train on a batch of transitions from the replay buffer.
        
        This performs one gradient descent step using the Bellman equation:
        Q(s,a) = r + gamma * max_a' Q(s', a')
        
        The target network (slowly-updated copy) is used to compute target Q-values,
        which provides stable learning targets.
        
        Returns:
            Loss value for this training step, or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        with tf.GradientTape() as tape:
            # Compute Q-values for current states
            q_values = self.q_network(states, training=True)
            
            # Compute target Q-values for next states using target network
            next_q_values = self.target_network(next_states, training=False)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            
            # Compute target: r + gamma * max_a' Q(s', a')
            # (if done=1, target is just the reward)
            targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)
            
            # Create target Q-values: copy q_values and replace with targets for taken actions
            target_q_values = q_values.numpy()
            for i in range(self.batch_size):
                target_q_values[i, actions[i]] = targets[i].numpy()
            target_q_values = tf.convert_to_tensor(target_q_values, dtype=tf.float32)
            
            # Compute loss
            loss = self.loss_fn(target_q_values, q_values)
        
        # Apply gradients
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
        # Track metrics
        self.losses.append(float(loss))
        mean_q = tf.reduce_mean(q_values).numpy()
        self.q_values_mean.append(float(mean_q))
        
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        
        return float(loss)
    
    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate (epsilon) after each episode.
        
        This transitions the agent from exploration to exploitation over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, policy_path: str) -> None:
        """
        Save the Q-network to disk.
        
        Args:
            policy_path: Path where to save the model
        """
        self.q_network.save(policy_path)
        print(f"✓ Model saved to {policy_path}")
    
    def load(self, policy_path: str) -> None:
        """
        Load a pre-trained Q-network from disk.
        
        Args:
            policy_path: Path to load the model from
        """
        self.q_network = keras.models.load_model(policy_path)
        self.target_network.set_weights(self.q_network.get_weights())
        print(f"✓ Model loaded from {policy_path}")
    
    def get_epsilon(self) -> float:
        """Return current exploration rate."""
        return self.epsilon
    
    def get_training_steps(self) -> int:
        """Return total number of training steps performed."""
        return self.training_steps
