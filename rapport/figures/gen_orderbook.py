"""Génère une illustration du carnet d'ordres (bids verts à gauche, asks rouges à droite)."""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5.5))

# Données fictives
bid_prices = [103.20, 103.21, 103.22, 103.23, 103.24, 103.25]
bid_qty    = [3200, 2700, 2400, 2145, 1450, 1234]

ask_prices = [103.27, 103.28, 103.29, 103.30, 103.31, 103.32]
ask_qty    = [1567, 1700, 2100, 2400, 2750, 3100]

y_bids = np.arange(len(bid_prices))
y_asks = np.arange(len(ask_prices)) + len(bid_prices) + 1

ax.barh(y_bids, bid_qty, color="#2ca02c", alpha=0.85, edgecolor="white")
ax.barh(y_asks, ask_qty, color="#d62728", alpha=0.85, edgecolor="white")

# Étiquettes prix
for i, (p, q) in enumerate(zip(bid_prices, bid_qty)):
    ax.text(-100, i, f"{p:.2f}", ha="right", va="center", fontsize=11, color="#2ca02c", fontweight="bold")
    ax.text(q + 50, i, f"{q}", ha="left", va="center", fontsize=10)

for i, (p, q) in enumerate(zip(ask_prices, ask_qty)):
    ax.text(-100, y_asks[i], f"{p:.2f}", ha="right", va="center", fontsize=11, color="#d62728", fontweight="bold")
    ax.text(q + 50, y_asks[i], f"{q}", ha="left", va="center", fontsize=10)

# Spread
spread_y = len(bid_prices) + 0.5
ax.axhspan(len(bid_prices) - 0.4, len(bid_prices) + 0.4, color="gold", alpha=0.3)
ax.text(2000, spread_y, "SPREAD = 0.02", ha="center", va="center", fontsize=12,
        fontweight="bold", color="#7a5901")

# Légendes côté
ax.text(-700, len(bid_prices)/2 - 0.5, "BIDS\n(achats)", fontsize=14, fontweight="bold",
        color="#2ca02c", ha="center", va="center", rotation=90)
ax.text(-700, y_asks.mean(), "ASKS\n(ventes)", fontsize=14, fontweight="bold",
        color="#d62728", ha="center", va="center", rotation=90)

ax.set_xlim(-800, 4000)
ax.set_ylim(-0.6, y_asks[-1] + 0.6)
ax.set_yticks([])
ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Carnet d'ordres : meilleur bid vs meilleur ask", fontsize=14, pad=15)

plt.tight_layout()
plt.savefig("orderbook_viz.png", dpi=160, bbox_inches="tight")
print("ok orderbook_viz.png")
