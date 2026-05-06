"""
Évaluation du modèle PPO sur des données historiques.
Usage: python -m agents.ppo.evaluate_ppo
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    from .train_ppo import ContinuousBinanceEnv, PolicyNetwork
except ImportError:
    from train_ppo import ContinuousBinanceEnv, PolicyNetwork

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate(csv_path, policy_path, num_episodes=50, volume_threshold=1000):
    env = ContinuousBinanceEnv(csv_path, volume_threshold=volume_threshold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Chargement du modèle : {policy_path}")
    input_dim = env.observation_space.shape[0]
    policy_net = PolicyNetwork(input_dim).to(device)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device))
    policy_net.eval()

    episode_pnls = []
    episode_trades = []
    episode_avg_inventories = []
    all_cash_curves = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        cash_curve = [0.0]

        while not done:
            obs = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(obs).squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, done, _, _ = env.step(action)
            cash_curve.append(env.cash)

        final_price = env.closes[env.current_step]
        pnl = env.cash + env.inventory * final_price
        avg_inv = np.mean(np.abs(env.episode_inventories)) if env.episode_inventories else 0.0

        episode_pnls.append(pnl)
        episode_trades.append(env.episode_trades)
        episode_avg_inventories.append(avg_inv)
        all_cash_curves.append(cash_curve)

        print(f"Episode {ep+1}/{num_episodes} | "
              f"PnL: {pnl:.4f} | "
              f"Trades: {env.episode_trades} | "
              f"Inv moy: {avg_inv:.2f} | "
              f"Inv final: {env.inventory}")

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 60)
    print(f"  Episodes:        {num_episodes}")
    print(f"  PnL moyen:       {np.mean(episode_pnls):.4f}")
    print(f"  PnL médian:      {np.median(episode_pnls):.4f}")
    print(f"  PnL std:         {np.std(episode_pnls):.4f}")
    print(f"  PnL min:         {np.min(episode_pnls):.4f}")
    print(f"  PnL max:         {np.max(episode_pnls):.4f}")
    print(f"  Win rate:        {100 * np.mean(np.array(episode_pnls) > 0):.1f}%")
    print(f"  Trades moyen:    {np.mean(episode_trades):.1f}")
    print(f"  Inv moyen:       {np.mean(episode_avg_inventories):.2f}")
    print("=" * 60)

    # Graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Évaluation PPO — Market Maker", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(episode_pnls, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax.axvline(x=np.mean(episode_pnls), color="green", linestyle="-", linewidth=2, label=f"Moyenne: {np.mean(episode_pnls):.4f}")
    ax.set_title("Distribution des PnL")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Fréquence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episode_pnls, marker="o", markersize=3, color="steelblue", alpha=0.7)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("PnL par épisode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episode_trades, marker="o", markersize=3, color="teal", alpha=0.7)
    ax.set_title("Trades par épisode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Nombre de trades")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for curve in all_cash_curves[:10]:
        ax.plot(curve, alpha=0.4)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Courbes de cash (10 premiers épisodes)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cash")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_ppo.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nGraphiques sauvegardés : {output_path}")
    plt.show()


if __name__ == "__main__":
    csv_path = os.path.join(PROJECT_ROOT, "data", "SOLUSDT-aggTrades-2026-03.csv")
    policy_path = os.path.join(PROJECT_ROOT, "agents", "ppo", "models_checkpoints", "ppo_policy_FINAL.pth")
    evaluate(csv_path, policy_path, num_episodes=50, volume_threshold=5)
