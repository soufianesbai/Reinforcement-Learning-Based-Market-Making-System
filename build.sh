#!/bin/bash
# Script simple de compilation (revient à la racine)

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "Compilation du projet OrderBook..."

# Créer le dossier build si besoin
if [ ! -d "$BUILD_DIR" ]; then
    echo "Création du dossier build..."
    mkdir -p "$BUILD_DIR"
fi

# Aller dans build
cd "$BUILD_DIR" || exit 1

# Configurer si nécessaire
if [ ! -f "CMakeCache.txt" ]; then
    echo "⚙️  Configuration CMake..."
    cmake .. || { echo "Erreur de configuration CMake!"; cd "$PROJECT_ROOT"; exit 1; }
fi

# Clean
echo "🧹 make clean..."
cmake --build . --target clean 2>/dev/null || true

# Compiler
echo "🔧 Compilation (Release)..."
cmake --build . --config Release -j$(nproc)
build_status=$?

if [ $build_status -ne 0 ]; then
    echo "Erreur de compilation!"
    cd "$PROJECT_ROOT"
    exit 1
fi

echo "Compilation réussie!"
echo "Exécutables: ./build/LiveMarketData | ./build/OrderBookTests"

# Revenir à la racine
cd "$PROJECT_ROOT"
