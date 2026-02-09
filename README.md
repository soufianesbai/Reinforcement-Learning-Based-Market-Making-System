# Order Book Engine

Moteur de carnet d'ordres haute performance en C++20 avec intégration données de marché temps réel.

## Features

- **Types d'ordres** : GoodTillCancel, Market, ImmediateOrCancel, FillOrKill, GoodForDay
- **Algorithme de matching** : Priorité prix-temps (FIFO au sein d'un niveau de prix)
- **Structures de données** : O(1) recherche, O(log n) accès aux niveaux
- **Live market data** : Intégration Binance API

## Performance

- Insertion : ~400K ops/sec
- Annulation : ~2M ops/sec
- Modification : ~270K ops/sec
- Latence : 2-4 μs

## Installation

### Prérequis

- CMake 3.10+
- C++20 compiler (GCC 10+, Clang 10+, MSVC 2019+)
- Dépendances : libcurl 8.4.0, nlohmann/json 3.11.3 (téléchargées automatiquement)

### Compilation

**Première fois (configuration initiale)** :
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release -j$(nproc)
```

**Après modification du code** :
```bash
# Méthode simple (recommandé)
./build.sh

# Ou manuellement
cd build
cmake --build . --config Release -j$(nproc)
```

## Utilisation

### Méthode Rapide (Scripts)

```bash
# Terminal 1 - Market C++ (lance uniquement)
./run_market.sh

# Terminal 2 - Agent Python
./run_agent.sh
```

### Méthode Manuelle

```bash
# Tests et benchmarks
./build/OrderBookTests

# Affichage temps réel
./build/LiveMarketData SOLUSDT 1 20
# [SYMBOL] [REFRESH_SECONDS] [DEPTH_LEVELS]
```

## Trading Agent Python

Un agent de trading simple qui observe le marché et exécute des ordres selon une stratégie mean-reversion.

### Exécution

```bash
python3 -m agents.main
# ou
python3 agents/main.py
```

**Fichiers utilisés** :
- data/market_state.json (lecture)
- data/agent_orders.json (écriture)

### Stratégie

L'agent implémente une stratégie **Mean Reversion** :
- **ACHETER** quand le prix est 2% sous la moyenne mobile (SMA)
- **VENDRE** quand le prix est 2% au-dessus de la SMA
- **ATTENDRE** sinon

### Fonctionnalités

- 📊 Observation en temps réel de l'orderbook Binance
- 🎯 Analyse technique (SMA, déviation de prix)
- 📝 Gestion des ordres et positions
- 💰 Suivi P&L non réalisé
- 📈 Affichage détaillé du marché et des performances

### Exemple de sortie

```
================================================================================
 MARKET STATE                         SOLUSDT        13:45:23
================================================================================
Bid:    $103.25  |  Ask: $103.27  |  Mid: $103.26
Spread:   1.94 bps  |  SMA:    $103.15

Orderbook Bids:
    $103.25  x  1234.56
    $103.24  x   891.23
    $103.23  x  2145.67

Orderbook Asks:
    $103.27  x  1567.89
    $103.28  x   723.45
    $103.29  x  1234.56

================================================================================
 Decision: BUY
   Reason: Price -2.1% below SMA (-2.15%)

✅ Order #1 placed: BUY 5.0 @ $103.25

================================================================================
 POSITION
================================================================================
  Qty:      5.00 | Entry: $ 103.25 | Current: $ 103.26 | PnL: $   0.05
```

## Structure

```
orderbook-simulator-cpp/
├── agents/               # Agents Python (structure modulaire)
│   ├── __init__.py
│   ├── main.py          # Point d'entrée
│   ├── agent.py         # Orchestrateur TradingAgent
│   ├── strategy.py      # Stratégie de trading
│   ├── market_data.py   # Gestion données marché
│   ├── display.py       # Fonctions d'affichage
│   └── types.py         # Types et enums
├── data/                 # Fichiers JSON échangés
│   ├── agent_orders.json
│   └── market_state.json
├── include/              # En-têtes
│   ├── Core/
│   ├── OrderBook/
│   ├── MarketData/
│   └── LiveMarketData/
├── src/                  # Implémentations
│   ├── OrderBook/
│   └── LiveMarketData/
├── tests/                # Tests
│   └── tests.cpp
└── CMakeLists.txt
```

### Matching Logic

Orders match when:

- Buy price ≥ Best ask price, or
- Sell price ≤ Best bid price

Matching proceeds in price-time priority:

1. Best price levels matched first
2. Within a price level, earliest orders matched first (FIFO)
3. Partial fills supported for all order types except FillOrKill

#
## Performance Characteristics

| Operation      | Complexity | Measured Throughput |
|----------------|------------|---------------------|
| Add Order      | O(log n)   | ~400K ops/sec       |
| Cancel Order   | O(1)       | ~2M ops/sec         |
| Modify Order   | O(log n)   | ~270K ops/sec       |
| Match Orders   | O(k log n) | ~350K matches/sec   |
| Get Order Info | O(m)       | ~500K snapshots/sec |

*n = number of price levels, k = number of matches, m = number of orders*

Benchmarks run on typical development hardware. Actual performance depends on system configuration and workload
characteristics.

## Implementation Notes

### Design Decisions

**Why separate bids/asks maps?** Allows different sorting orders (descending for bids, ascending for asks) and
simplifies best bid/ask access.

**Why store iterators in OrderEntry?** Enables O(1) removal from the price level list when canceling orders.

**Why shared_ptr for orders?** Allows orders to be referenced in multiple places (main map, price level list) without
copying. Trade-off between memory overhead and simplicity.

**Market order conversion:** Market orders are converted to limit orders at extreme prices (max/min) to reuse the
matching logic.

### Order Type Behavior

- **GoodTillCancel**: Remains active until filled or cancelled
- **Market**: Immediately converted to aggressive limit order
- **ImmediateOrCancel**: Partial fills accepted, unfilled portion cancelled
- **FillOrKill**: All-or-nothing execution, rejected if can't fill completely
- **GoodForDay**: Cancelled at configured time (default 15:59)

