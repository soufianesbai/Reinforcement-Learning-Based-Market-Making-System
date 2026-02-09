#!/bin/bash
# Script pour lancer le market C++

PROJECT_ROOT="$(dirname "$0")"

echo "🚀 Lancement du LiveMarketData..."
echo ""
cd "$PROJECT_ROOT/build" || exit 1
./LiveMarketData SOLUSDT 1 50
