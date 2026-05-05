"""
Dreamer v3 training on Historical Binance Kline Data.
Run with: python3 -m agents.train_dreamer
"""

import os
import random
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Bootstrap: pre-register sub-modules so relative imports work without
# triggering agents/__init__.py (which has an unrelated circular import).
# ---------------------------------------------------------------------------
import sys, types, importlib.util as _ilu, pathlib as _pl

_PKG_ROOT = _pl.Path(__file__).resolve().parent          # agents/

def _bootstrap(mod_name: str, file_path: _pl.Path):
    """Register a module from its file, skip if already loaded."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = _ilu.spec_from_file_location(mod_name, str(file_path))
    mod = _ilu.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

# 1. Ensure 'agents' package exists as a namespace (no __init__ execution)
if "agents" not in sys.modules:
    _pkg = types.ModuleType("agents")
    _pkg.__path__ = [str(_PKG_ROOT)]
    _pkg.__package__ = "agents"
    sys.modules["agents"] = _pkg

# 2. Leaf modules needed by dreamer_agent.py (relative imports)
_bootstrap("agents.types",       _PKG_ROOT / "types.py")
_bootstrap("agents.interfaces",  _PKG_ROOT / "interfaces.py")
_bootstrap("agents.market_data", _PKG_ROOT / "market_data.py")

# 3. Ensure agents.strategies sub-package exists
if "agents.strategies" not in sys.modules:
    _strat = types.ModuleType("agents.strategies")
    _strat.__path__ = [str(_PKG_ROOT / "strategies")]
    _strat.__package__ = "agents.strategies"
    sys.modules["agents.strategies"] = _strat

# 4. Now import the actual modules we need
_bootstrap("agents.strategies.a2c_rl",          _PKG_ROOT / "strategies" / "a2c_rl.py")
_bootstrap("agents.strategies.dreamer_agent",   _PKG_ROOT / "strategies" / "dreamer_agent.py")
_bootstrap("agents.train_a2c",                  _PKG_ROOT / "train_a2c.py")

from agents.train_a2c import ContinuousBinanceEnv, build_volume_bars
from agents.strategies.dreamer_agent import (
    Encoder, RSSM, Decoder, RewardPredictor, ContinuePredictor,
    Actor, Critic, symlog, symexp,
)


# ---------------------------------------------------------------------------
# Discrete action wrapper
# ---------------------------------------------------------------------------

ACTION_MAP = {
    0: np.array([-0.5, -0.5], dtype=np.float32),  # HOLD
    1: np.array([-0.8, 0.2], dtype=np.float32),    # BUY
    2: np.array([0.2, -0.8], dtype=np.float32),     # SELL
}


class DiscreteEnvWrapper:
    """Wraps ContinuousBinanceEnv to accept discrete actions (0,1,2)."""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action_idx: int):
        continuous_action = ACTION_MAP[action_idx]
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        done = terminated or truncated
        return obs, reward, done


# ---------------------------------------------------------------------------
# Sequential replay buffer
# ---------------------------------------------------------------------------

class SequentialReplayBuffer:
    """Stores full episodes, samples sub-sequences."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes = []

    def add_episode(self, obs, actions, rewards, continues):
        """All inputs are numpy arrays. actions: one-hot [T, 3]."""
        ep = {
            "obs": np.array(obs, dtype=np.float32),
            "action": np.array(actions, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "continue": np.array(continues, dtype=np.float32),
        }
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def can_sample(self, seq_len: int) -> bool:
        return any(len(ep["obs"]) >= seq_len for ep in self.episodes)

    def sample(self, batch_size: int, seq_len: int, device: torch.device):
        valid_eps = [ep for ep in self.episodes if len(ep["obs"]) >= seq_len]
        batch_obs, batch_act, batch_rew, batch_cont = [], [], [], []

        for _ in range(batch_size):
            ep = random.choice(valid_eps)
            T = len(ep["obs"])
            start = random.randint(0, T - seq_len)
            batch_obs.append(ep["obs"][start:start + seq_len])
            batch_act.append(ep["action"][start:start + seq_len])
            batch_rew.append(ep["reward"][start:start + seq_len])
            batch_cont.append(ep["continue"][start:start + seq_len])

        return (
            torch.tensor(np.stack(batch_obs), device=device),    # [B, T, 121]
            torch.tensor(np.stack(batch_act), device=device),    # [B, T, 3]
            torch.tensor(np.stack(batch_rew), device=device),    # [B, T]
            torch.tensor(np.stack(batch_cont), device=device),   # [B, T]
        )


# ---------------------------------------------------------------------------
# KL divergence for categoricals
# ---------------------------------------------------------------------------

def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(Cat(p) || Cat(q)) summed over categories. logits: [B, 32, 32]"""
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    kl = (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=-1)  # [B, 32]
    return kl.sum(dim=-1)  # [B] — sum over 32 categories


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dreamer(
    n_epochs: int = 2000,
    batch_size: int = 16,
    seq_len: int = 64,
    imagination_horizon: int = 15,
    grad_steps: int = 4,
    gamma: float = 0.997,
    lam: float = 0.95,
    free_nats: float = 1.0,
    kl_alpha: float = 0.8,
    unimix: float = 0.01,
    ema_decay: float = 0.98,
    wm_lr: float = 1e-4,
    actor_lr: float = 3e-5,
    critic_lr: float = 3e-5,
    grad_clip: float = 100.0,
    entropy_coef: float = 3e-4,
    live_plot: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Dreamer v3] Training on {device}")
    start_time = time.time()

    # --- Live plot setup ---
    live_fig = None
    if live_plot:
        plt.ion()
        live_fig, live_axes = plt.subplots(2, 2, figsize=(14, 10))
        titles = ["Episode Rewards", "World Model Loss", "Actor Loss", "KL Divergence"]
        ylabels = ["Total Reward", "Loss", "Loss", "KL (nats)"]
        for ax, title, ylabel in zip(live_axes.flat, titles, ylabels):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.text(0.5, 0.5, "Waiting for data...", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")
        live_fig.suptitle("Dreamer v3 Training — collecting first episodes...", fontsize=16)
        live_fig.tight_layout()
        live_fig.canvas.draw()
        live_fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(0.5)

    # --- Environment ---
    csv_file = "./data/SOLUSDT-aggTrades-2026-02.csv"
    base_env = ContinuousBinanceEnv(csv_file, window_size=30, volume_threshold=1000)
    env = DiscreteEnvWrapper(base_env)
    obs_dim = base_env.observation_space.shape[0]  # 121

    # --- Models ---
    encoder = Encoder(obs_dim=obs_dim).to(device)
    rssm = RSSM().to(device)
    decoder = Decoder().to(device)
    reward_pred = RewardPredictor().to(device)
    continue_pred = ContinuePredictor().to(device)
    actor = Actor().to(device)
    critic = Critic().to(device)
    critic_target = copy.deepcopy(critic)
    critic_target.requires_grad_(False)

    # --- Optimizers ---
    wm_params = (
        list(encoder.parameters()) + list(rssm.parameters()) +
        list(decoder.parameters()) + list(reward_pred.parameters()) +
        list(continue_pred.parameters())
    )
    wm_optim = optim.AdamW(wm_params, lr=wm_lr)
    actor_optim = optim.AdamW(actor.parameters(), lr=actor_lr)
    critic_optim = optim.AdamW(critic.parameters(), lr=critic_lr)

    # --- Replay buffer ---
    buffer = SequentialReplayBuffer(capacity=1000)

    # --- Logging ---
    history = {"wm_loss": [], "actor_loss": [], "critic_loss": [],
               "reward": [], "kl": [], "inventory": []}

    log_every = max(1, n_epochs // 100)  # ~1% progress ticks, at least every epoch

    for epoch in range(1, n_epochs + 1):
        ep_t0 = time.time()
        # =====================================================================
        # 1. Collect one episode
        # =====================================================================
        obs = env.reset()
        done = False
        ep_obs, ep_actions, ep_rewards, ep_continues = [obs.copy()], [], [], []
        ep_reward_sum = 0.0

        # RSSM state for collection
        h, z = rssm.initial_state(1, device)
        prev_action = torch.zeros(1, 3, device=device)

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                embed = encoder(obs_t)
                h, z, _, _ = rssm.observe_step(h, z, prev_action, embed)
                state = torch.cat([h, z], dim=-1)
                probs = actor(state)
                action_idx = torch.multinomial(probs, 1).item()

                action_onehot = np.zeros(3, dtype=np.float32)
                action_onehot[action_idx] = 1.0
                prev_action = torch.tensor(action_onehot, device=device).unsqueeze(0)

            next_obs, reward, done = env.step(action_idx)
            ep_actions.append(action_onehot)
            ep_rewards.append(reward)
            ep_continues.append(0.0 if done else 1.0)
            ep_obs.append(next_obs.copy())
            ep_reward_sum += reward
            obs = next_obs

        # Remove the extra trailing obs (we want obs[0..T-1] aligned with actions)
        ep_obs = ep_obs[:-1]

        buffer.add_episode(ep_obs, ep_actions, ep_rewards, ep_continues)
        history["reward"].append(ep_reward_sum)
        history["inventory"].append(float(base_env.inventory))

        # =====================================================================
        # 2. Train if buffer ready
        # =====================================================================
        if not buffer.can_sample(seq_len):
            continue

        for _grad_step in range(grad_steps):
            # -----------------------------------------------------------------
            # 2a. World model training
            # -----------------------------------------------------------------
            obs_b, act_b, rew_b, cont_b = buffer.sample(batch_size, seq_len, device)

            # Encode all observations
            B, T = obs_b.shape[:2]
            embed_b = encoder(obs_b.reshape(B * T, -1)).reshape(B, T, -1)

            # Roll RSSM. The GRU expects the action that LED to the current
            # observation (matches collect/inference convention), not the one
            # taken FROM it — so feed prev_act and shift each iteration.
            h, z = rssm.initial_state(B, device)
            states_list = []
            prior_logits_list = []
            post_logits_list = []

            prev_act_seq = torch.zeros(B, 3, device=device)
            for t in range(T):
                embed_t = embed_b[:, t]
                h, z, prior_logits, post_logits = rssm.observe_step(h, z, prev_act_seq, embed_t)
                states_list.append(torch.cat([h, z], dim=-1))
                prior_logits_list.append(prior_logits)
                post_logits_list.append(post_logits)
                prev_act_seq = act_b[:, t]

            states = torch.stack(states_list, dim=1)          # [B, T, 1536]
            prior_logits = torch.stack(prior_logits_list, 1)  # [B, T, 32, 32]
            post_logits = torch.stack(post_logits_list, 1)    # [B, T, 32, 32]

            states_flat = states.reshape(B * T, -1)

            # Losses
            obs_pred = decoder(states_flat).reshape(B, T, -1)
            l_recon = F.mse_loss(obs_pred, obs_b)

            rew_pred = reward_pred(states_flat).reshape(B, T)
            l_reward = F.mse_loss(rew_pred, symlog(rew_b))

            cont_pred_logit = continue_pred(states_flat).reshape(B, T)
            l_continue = F.binary_cross_entropy_with_logits(cont_pred_logit, cont_b)

            # KL with free bits and balancing
            kl_per_step = []
            for t in range(T):
                kl_fwd = categorical_kl(post_logits[:, t].detach(), prior_logits[:, t])
                kl_rev = categorical_kl(post_logits[:, t], prior_logits[:, t].detach())
                kl_per_step.append(
                    kl_alpha * torch.clamp(kl_fwd, min=free_nats).mean() +
                    (1.0 - kl_alpha) * torch.clamp(kl_rev, min=free_nats).mean()
                )
            l_kl = torch.stack(kl_per_step).mean()

            wm_loss = l_recon + l_reward + l_continue + l_kl

            wm_optim.zero_grad()
            wm_loss.backward()
            nn.utils.clip_grad_norm_(wm_params, grad_clip)
            wm_optim.step()

            # -----------------------------------------------------------------
            # 2b. Actor-critic training in imagination
            # -----------------------------------------------------------------
            # Detach starting states from world model graph
            init_states = states_flat.detach()

            # Split h, z from init_states
            init_h = init_states[:, :512]
            init_z = init_states[:, 512:]

            # Freeze world model for imagination
            for p in wm_params:
                p.requires_grad_(False)

            # Imagine forward. Reward/continue are predicted from the CURRENT
            # state (matches WM training: reward_pred(state[t]) → rew_b[t]).
            # Track the actions actually sampled so REINFORCE log_probs match
            # the advantages.
            imag_states = []
            imag_actions = []
            imag_rewards = []
            imag_continues = []
            cur_h, cur_z = init_h, init_z

            for _t in range(imagination_horizon):
                state_t = torch.cat([cur_h, cur_z], dim=-1)
                probs_t = actor(state_t)
                action_idx_t = torch.multinomial(probs_t, 1).squeeze(-1)
                action_oh = F.one_hot(action_idx_t, 3).float()

                imag_states.append(state_t)
                imag_actions.append(action_idx_t)
                imag_rewards.append(symexp(reward_pred(state_t).squeeze(-1)))
                imag_continues.append(torch.sigmoid(continue_pred(state_t).squeeze(-1)))

                cur_h, cur_z = rssm.imagine_step(cur_h, cur_z, action_oh)

            # Final state for value bootstrap
            final_state = torch.cat([cur_h, cur_z], dim=-1)
            imag_states.append(final_state)

            # Re-enable world model gradients
            for p in wm_params:
                p.requires_grad_(True)

            imag_states_t = torch.stack(imag_states, dim=0)        # [H+1, N, 1536]
            imag_actions_t = torch.stack(imag_actions, dim=0)      # [H, N]
            imag_rewards_t = torch.stack(imag_rewards, dim=0)      # [H, N]
            imag_continues_t = torch.stack(imag_continues, dim=0)  # [H, N]

            H = imagination_horizon
            N = init_states.shape[0]

            # Values from EMA target critic
            with torch.no_grad():
                values_ema = critic_target(imag_states_t.reshape(-1, 1536)).reshape(H + 1, N)
                values_ema = symexp(values_ema)

            # Lambda-returns: returns[t] = r[t] + γ·c[t]·((1-λ)·V(s[t+1]) + λ·G[t+1])
            returns = torch.zeros(H, N, device=device)
            returns[-1] = imag_rewards_t[-1] + gamma * imag_continues_t[-1] * values_ema[-1]
            for t in reversed(range(H - 1)):
                returns[t] = imag_rewards_t[t] + gamma * imag_continues_t[t] * (
                    (1.0 - lam) * values_ema[t + 1] + lam * returns[t + 1]
                )

            # Percentile normalization
            ret_flat = returns.detach().flatten()
            per5 = torch.quantile(ret_flat, 0.05)
            per95 = torch.quantile(ret_flat, 0.95)
            scale = torch.clamp(per95 - per5, min=1.0)

            # Critic loss
            critic_values = critic(imag_states_t[:-1].reshape(-1, 1536).detach()).reshape(H, N)
            critic_loss = F.mse_loss(critic_values, symlog(returns).detach())

            critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            critic_optim.step()

            # Actor loss
            with torch.no_grad():
                values_for_adv = symexp(critic_target(
                    imag_states_t[:-1].reshape(-1, 1536)
                ).reshape(H, N))
                norm_ret = (returns - per5) / scale
                norm_val = (values_for_adv - per5) / scale
                advantages = norm_ret - norm_val

            # REINFORCE: log_probs MUST be of the actions actually taken during
            # imagination (which produced these advantages), not fresh samples.
            actor_states = imag_states_t[:-1].reshape(-1, 1536).detach()
            actor_probs = actor(actor_states).reshape(H, N, 3)

            actor_dist = torch.distributions.Categorical(probs=actor_probs.reshape(-1, 3))
            log_probs = actor_dist.log_prob(imag_actions_t.reshape(-1)).reshape(H, N)
            entropy = actor_dist.entropy().reshape(H, N)

            actor_loss = -(advantages.detach() * log_probs + entropy_coef * entropy).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            actor_optim.step()

            # EMA update
            with torch.no_grad():
                for p_online, p_target in zip(critic.parameters(), critic_target.parameters()):
                    p_target.data.mul_(ema_decay).add_(p_online.data, alpha=1.0 - ema_decay)

        # Store losses from last grad step
        history["wm_loss"].append(wm_loss.item())
        history["actor_loss"].append(actor_loss.item())
        history["critic_loss"].append(critic_loss.item())
        history["kl"].append(l_kl.item())

        # =====================================================================
        # 3. Logging & checkpoints
        # =====================================================================
        ep_elapsed = time.time() - ep_t0
        elapsed = time.time() - start_time

        # Short status every epoch so the user sees progress
        if epoch % log_every == 0 or epoch == 1:
            wm_str = f"{history['wm_loss'][-1]:.4f}" if history["wm_loss"] else "n/a"
            print(
                f"Ep {epoch:4d}/{n_epochs} | Reward: {ep_reward_sum:+.2f} | "
                f"WM: {wm_str} | {ep_elapsed:.1f}s/ep | Total: {elapsed:.0f}s",
                flush=True,
            )

        # Detailed metrics + plot every 10 epochs (or first epoch with data)
        if (epoch % 10 == 0 or epoch <= 2) and history["wm_loss"]:
            n = min(50, len(history["reward"]))
            avg_rew = np.mean(history["reward"][-n:])
            avg_inv = np.mean(history["inventory"][-n:])
            avg_kl = np.mean(history["kl"][-n:]) if history["kl"] else 0
            print(
                f"  >> Avg Reward: {avg_rew:+.3f} | Avg Inv: {avg_inv:.1f} | "
                f"Actor: {history['actor_loss'][-1]:.4f} | Critic: {history['critic_loss'][-1]:.4f} | "
                f"KL: {avg_kl:.2f}",
                flush=True,
            )

            # Update live plot + save PNG
            if live_plot and live_fig is not None:
                _update_live_plot(live_fig, live_axes, history, epoch)
            _plot_metrics(history, epoch)

        if epoch % 200 == 0 or epoch == n_epochs:
            _save_checkpoint(encoder, rssm, decoder, reward_pred,
                             continue_pred, actor, critic, critic_target)

    elapsed = time.time() - start_time
    print(f"[Dreamer v3] Training complete. Total time: {elapsed:.1f}s")

    if live_plot:
        plt.ioff()
        print("[Dreamer v3] Training finished. Close the plot window to exit.")
        plt.show()


def _update_live_plot(fig, axes, history, epoch):
    """Refresh the live matplotlib window in-place."""
    (ax1, ax2), (ax3, ax4) = axes
    window = min(50, max(1, len(history["wm_loss"])))

    for ax in [ax1, ax2, ax3, ax4]:
        ax.cla()

    # --- Rewards ---
    ax1.plot(history["reward"], alpha=0.3, color="blue", label="Raw")
    if len(history["reward"]) >= window and window >= 2:
        ma = np.convolve(history["reward"], np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(history["reward"])), ma,
                 color="blue", linewidth=3, label="Moving Avg")
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- World model loss ---
    ax2.plot(history["wm_loss"], alpha=0.3, color="red")
    if len(history["wm_loss"]) >= window and window >= 2:
        ma = np.convolve(history["wm_loss"], np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(history["wm_loss"])), ma,
                 color="red", linewidth=3)
    ax2.set_title("World Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    if history["wm_loss"] and min(history["wm_loss"]) > 0:
        ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # --- Actor loss ---
    ax3.plot(history["actor_loss"], alpha=0.3, color="green")
    if len(history["actor_loss"]) >= window and window >= 2:
        ma = np.convolve(history["actor_loss"], np.ones(window) / window, mode="valid")
        ax3.plot(range(window - 1, len(history["actor_loss"])), ma,
                 color="green", linewidth=3)
    ax3.set_title("Actor Loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)

    # --- KL divergence ---
    ax4.plot(history["kl"], alpha=0.3, color="purple")
    if len(history["kl"]) >= window and window >= 2:
        ma = np.convolve(history["kl"], np.ones(window) / window, mode="valid")
        ax4.plot(range(window - 1, len(history["kl"])), ma,
                 color="purple", linewidth=3)
    ax4.set_title("KL Divergence")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("KL (nats)")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"Dreamer v3 Training — Epoch {epoch}", fontsize=16)
    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)


def _save_checkpoint(encoder, rssm, decoder, reward_pred,
                     continue_pred, actor, critic, critic_target):
    os.makedirs("models", exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "rssm": rssm.state_dict(),
        "decoder": decoder.state_dict(),
        "reward_pred": reward_pred.state_dict(),
        "continue_pred": continue_pred.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
    }, "models/dreamer_agent.pth")
    print("[Dreamer] Checkpoint saved to models/dreamer_agent.pth")


def _plot_metrics(history, epoch):
    window = min(50, len(history["wm_loss"]))
    if window < 2:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Rewards
    ax1.plot(history["reward"], alpha=0.3, color="blue", label="Raw")
    if len(history["reward"]) >= window:
        ma = np.convolve(history["reward"], np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(history["reward"])), ma,
                 color="blue", linewidth=3, label="Moving Avg")
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # World model loss
    ax2.plot(history["wm_loss"], alpha=0.3, color="red")
    if len(history["wm_loss"]) >= window:
        ma = np.convolve(history["wm_loss"], np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(history["wm_loss"])), ma,
                 color="red", linewidth=3)
    ax2.set_title("World Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Actor loss
    ax3.plot(history["actor_loss"], alpha=0.3, color="green")
    if len(history["actor_loss"]) >= window:
        ma = np.convolve(history["actor_loss"], np.ones(window) / window, mode="valid")
        ax3.plot(range(window - 1, len(history["actor_loss"])), ma,
                 color="green", linewidth=3)
    ax3.set_title("Actor Loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)

    # KL divergence
    ax4.plot(history["kl"], alpha=0.3, color="purple")
    if len(history["kl"]) >= window:
        ma = np.convolve(history["kl"], np.ones(window) / window, mode="valid")
        ax4.plot(range(window - 1, len(history["kl"])), ma,
                 color="purple", linewidth=3)
    ax4.set_title("KL Divergence")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("KL (nats)")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Dreamer v3 Training Progress (Epoch {epoch})", fontsize=16)
    plt.tight_layout()
    plt.savefig("dreamer_training_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_dreamer()
