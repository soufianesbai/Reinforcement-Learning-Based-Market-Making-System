# Reinforcement-Learning-Based Market Making System

Système de trading et de market making combinant un moteur de carnet d'ordres haute performance en C++20, des agents
Python (stratégies heuristiques et apprentissage par renforcement) connectés au testnet Binance, et un dashboard web de
supervision en temps réel.

Projet de fin d'année (PFA) 2025–2026 — ENSEIRB-MATMECA.

## Vue d'ensemble

```
 Binance API ──► Moteur C++ (LiveMarketData) ──► data/market_state.json ──► Agent Python (stratégie)
                        ▲                                                         │
                        └──────────────── data/agent_orders.json ◄────────────────┘

 Dashboard (dashboard/server.py) ── lit market_state.json et dashboard_agent.log (observateur passif)
```

- **Moteur C++** : récupère le carnet d'ordres Binance en temps réel, le reconstruit et exécute les ordres de l'agent
  avec un matching prix-temps.
- **Agent Python** : lit l'état du marché, délègue la décision à une stratégie interchangeable et envoie ses ordres.
- **Dashboard** : interface web qui observe les deux processus via des fichiers partagés, sans jamais ajouter de latence
  au trading.

## Stratégies

| Stratégie            | Type                                  | Nom (`STRATEGY`) |
|----------------------|---------------------------------------|------------------|
| SAC (Soft Actor-Critic) | RL off-policy — market making      | `sac`            |
| PPO                  | RL policy-based, on-policy            | `ppo`            |
| A2C                  | RL actor-critic, on-policy            | `a2c`            |
| DQN                  | RL value-based                        | `dqn`            |
| Dreamer v3           | RL model-based                        | — (entraînement) |
| Mean Reversion       | Heuristique (écart à la SMA)          | `mean_reversion` (défaut) |
| RSI (14)             | Heuristique momentum                  | `rsi`            |
| Bollinger Bands      | Heuristique volatilité                | `bollinger`      |
| SMA crossover        | Heuristique tendance                  | `sma`            |
| Random               | Référence aléatoire                   | `random`         |

### SAC : market making

Contrairement aux autres agents RL, qui sont directionnels, SAC place simultanément un ordre d'achat et un ordre de vente
autour du midprice. Son action est continue : deux spreads entre 0 et 0,5 %. La reward s'inspire du modèle
Avellaneda-Stoikov :

- gain du spread capturé à chaque fill ;
- pénalité quadratique sur l'inventaire, pour rester neutre ;
- pénalité de sélection adverse.

Évaluation sur données jamais vues (5 seeds × 50 épisodes) : reward moyenne de 160,68 ± 4,86, Sharpe de 8,82, drawdown
maximal de 7,09 %, taux de fill de 18,3 %. C'est environ 2× mieux que la politique aléatoire et 3× mieux qu'un spread
fixe de 0,2 %.

## Installation

### Prérequis

- CMake 3.10+ et un compilateur C++20 (GCC 10+, Clang 10+, MSVC 2019+)
- Python 3.10+
- Un compte [Binance testnet](https://testnet.binance.vision/) pour les clés API

### Moteur C++

```bash
./build.sh
```

`libcurl` et `nlohmann/json` sont téléchargés automatiquement par CMake (`FetchContent`). Les exécutables sont générés
dans `build/` : `LiveMarketData` et `OrderBookTests`.

### Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch tensorflow gymnasium numpy pandas matplotlib requests
```

### Clés Binance

```bash
cp agents/exchange/config.example.py agents/exchange/config.py
# puis renseigner BINANCE_TESTNET_API_KEY et BINANCE_TESTNET_API_SECRET
```

`config.py` contient des secrets : ne pas le committer.

## Utilisation

### Lancer le marché et un agent

```bash
# Terminal 1 — moteur C++ : ./run_market.sh [SYMBOL] [REFRESH_SECONDS] [DEPTH_LEVELS]
./run_market.sh SOLUSDT 1 50

# Terminal 2 — agent Python
./run_agent.sh sac            # ou : STRATEGY=sac SYMBOL=SOLUSDT python3 -m agents.main
```

### Dashboard

```bash
./run_dashboard.sh            # http://127.0.0.1:8090  (--restart pour relancer)
```

Le dashboard affiche :

- un graphique en chandeliers ;
- le carnet d'ordres live (8 niveaux) ;
- la courbe de PnL ;
- la dernière décision de l'agent ;
- l'inventaire ;
- les logs de l'agent et du marché.

Il permet aussi de changer d'agent ou de paire à chaud. L'accès est protégé par authentification.

### Entraîner les agents RL

Les agents sont entraînés hors ligne sur des historiques `aggTrades` de Binance (fichiers CSV placés dans `data/` ou
`training_data/`). Les poids entraînés sont enregistrés dans `models/`.

```bash
python3 -m agents.strategies.train_sac     # SAC
python3 agents/strategies/eval_sac.py      # évaluation multi-seed de SAC
python3 -m agents.train_a2c                # A2C
python3 -m agents.train_dreamer            # Dreamer v3 (ou : python3 run_dreamer.py --fast)
python3 agents/ppo/train_ppo.py            # PPO
python3 agents/dqn/train_dqn.py            # DQN
python3 compute_metrics.py                 # métriques comparatives
```

### Tests et benchmarks

```bash
./build/OrderBookTests                     # tests du moteur C++
./benchmark.sh                             # benchmark du carnet d'ordres
./scripts/test_dashboard_all.sh            # tests du dashboard (API + end-to-end Playwright)
```

### Rapport et présentation

```bash
make pdf                                   # compile rapport/rapport.pdf et rapport/presentation.pdf
```

## Moteur de carnet d'ordres C++

- **Types d'ordres** : GoodTillCancel, Market, ImmediateOrCancel, FillOrKill, GoodForDay
- **Matching** : priorité prix-temps (FIFO au sein d'un niveau de prix). Les fills partiels sont acceptés, sauf pour
  FillOrKill.
- **Structures de données** : maps triées séparées pour les bids et les asks, itérateurs stockés pour une annulation
  en O(1).

| Opération        | Complexité | Débit mesuré        |
|------------------|------------|---------------------|
| Ajout d'ordre    | O(log n)   | ~400K ops/s         |
| Annulation       | O(1)       | ~2M ops/s           |
| Modification     | O(log n)   | ~270K ops/s         |
| Matching         | O(k log n) | ~350K matches/s     |

*n = nombre de niveaux de prix, k = nombre de matches. Latence typique : 2–4 μs.*

## Structure du projet

```
├── agents/                 # Agents Python
│   ├── main.py             # Point d'entrée (choix de la stratégie via STRATEGY)
│   ├── engine.py           # Moteur d'exécution : I/O avec le C++, positions, ordres
│   ├── interfaces.py       # Interface BaseStrategy
│   ├── strategies/         # Stratégies heuristiques et RL (SAC, PPO, A2C, DQN, Dreamer…)
│   ├── ppo/ dqn/           # Entraînement et évaluation PPO / DQN
│   ├── train_*.py          # Entraînement A2C, Dreamer
│   └── exchange/           # Client Binance testnet
├── include/ src/           # Moteur C++ (OrderBook, LiveMarketData)
├── tests/                  # Tests et benchmark C++
├── dashboard/              # Serveur HTTP Python et front-end JS
├── models/                 # Poids entraînés et courbes d'apprentissage
├── data/                   # Fichiers JSON échangés entre les processus
└── rapport/                # Rapport et présentation LaTeX
```

### Ajouter une stratégie

1. Créer `agents/strategies/ma_strategie.py` en implémentant l'interface `BaseStrategy`.
2. L'enregistrer dans `agents/main.py`.
3. La lancer avec `STRATEGY=ma_strategie python3 -m agents.main`.

## Auteurs

Bella Taleb El Alem, Yani Haouchine, Ihab Jaouad, Anas Benbaki, Mohamed Aziz Migaou, Soufiane Sbai.
