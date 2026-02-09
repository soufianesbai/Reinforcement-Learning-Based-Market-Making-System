#!/bin/bash
# Script pour lancer l'agent Python

echo "🤖 Démarrage de l'agent Python..."
cd "$(dirname "$0")"

# Vérifier que Python3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé!"
    exit 1
fi

echo "✅ Python trouvé: $(python3 --version)"
echo "🚀 Lancement de l'agent..."
echo ""
python3 -m agents.main
