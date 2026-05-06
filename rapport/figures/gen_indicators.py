"""Illustrations pour SMA / mean reversion et RSI."""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)

# --- SMA + Mean Reversion ------------------------------------------------
n = 200
t = np.arange(n)
price = 100 + np.cumsum(np.random.randn(n) * 0.4) + 5*np.sin(t/12)
sma = np.convolve(price, np.ones(20)/20, mode="same")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, price, color="#1f77b4", label="Prix", lw=1.5)
ax.plot(t, sma,   color="#ff7f0e", label="SMA(20)", lw=2)
ax.fill_between(t, sma*0.98, sma*1.02, color="#ff7f0e", alpha=0.15, label="±2 % SMA")

# Signaux
buy = np.where(price < sma*0.98)[0]
sell = np.where(price > sma*1.02)[0]
ax.scatter(buy[::3], price[buy[::3]], marker="^", s=80, color="#2ca02c",
           edgecolor="black", zorder=5, label="Achat")
ax.scatter(sell[::3], price[sell[::3]], marker="v", s=80, color="#d62728",
           edgecolor="black", zorder=5, label="Vente")

ax.set_title("Mean reversion : achat sous la SMA, vente au-dessus", fontsize=13)
ax.set_xlabel("Pas de temps")
ax.set_ylabel("Prix")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("indic_meanrev.png", dpi=160)
plt.close()
print("ok indic_meanrev.png")

# --- RSI illustration -----------------------------------------------------
def compute_rsi(p, period=14):
    delta = np.diff(p, prepend=p[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    g = np.convolve(gain, np.ones(period)/period, mode="same")
    l = np.convolve(loss, np.ones(period)/period, mode="same")
    rs = g / (l + 1e-9)
    return 100 - 100/(1+rs)

rsi = compute_rsi(price)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                                gridspec_kw={"height_ratios":[2,1]})
ax1.plot(t, price, color="#1f77b4", lw=1.5)
ax1.set_ylabel("Prix")
ax1.set_title("RSI : zones d'achat (<30) et de vente (>70)", fontsize=13)
ax1.grid(alpha=0.3)

ax2.plot(t, rsi, color="#9467bd", lw=1.5)
ax2.axhline(70, color="#d62728", linestyle="--", lw=1)
ax2.axhline(30, color="#2ca02c", linestyle="--", lw=1)
ax2.fill_between(t, 70, 100, color="#d62728", alpha=0.15)
ax2.fill_between(t, 0,  30,  color="#2ca02c", alpha=0.15)
ax2.text(5, 85, "Surachat → VENTE", color="#d62728", fontsize=11, fontweight="bold")
ax2.text(5, 8,  "Survente → ACHAT", color="#2ca02c", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI(14)")
ax2.set_xlabel("Pas de temps")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("indic_rsi.png", dpi=160)
plt.close()
print("ok indic_rsi.png")

# --- Bollinger Bands illustration ----------------------------------------
window = 20
ma = np.array([price[max(0,i-window):i+1].mean() for i in range(n)])
sd = np.array([price[max(0,i-window):i+1].std()  for i in range(n)])
upper = ma + 2*sd
lower = ma - 2*sd

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(t, lower, upper, color="#9467bd", alpha=0.18, label="Bande ±2σ")
ax.plot(t, ma,    color="#9467bd", lw=2,   label="SMA(20)")
ax.plot(t, upper, color="#9467bd", lw=1, ls="--")
ax.plot(t, lower, color="#9467bd", lw=1, ls="--")
ax.plot(t, price, color="#1f77b4", lw=1.5, label="Prix")

ax.set_title("Bollinger Bands : volatilité = largeur de la bande", fontsize=13)
ax.set_xlabel("Pas de temps")
ax.set_ylabel("Prix")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("indic_bollinger.png", dpi=160)
plt.close()
print("ok indic_bollinger.png")
