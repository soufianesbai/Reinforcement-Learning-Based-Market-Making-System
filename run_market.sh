#!/bin/bash
# Script pour lancer le market C++

PROJECT_ROOT="$(dirname "$0")"

echo "🚀 Lancement du LiveMarketData..."
echo ""
cd "$PROJECT_ROOT/build" || exit 1
./LiveMarketData "${1:-SOLUSDT}" "${2:-1}" "${3:-50}"
