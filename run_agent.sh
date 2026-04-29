#!/bin/bash
# Script pour lancer l'agent Python avec une stratégie au choix

echo "🤖 Démarrage de l'agent Python..."
cd "$(dirname "$0")"

# Vérifier que Python3 est installé
if ! command -v python3 &> /dev/null; then
    echo "Python3 n'est pas installé!"
    exit 1
fi

echo "Python trouvé: $(python3 --version)"

# Stratégie par défaut
STRATEGY="${1:-mean_reversion}"

# Afficher l'aide si demandé
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo ""
    echo "Usage: ./run_agent.sh [STRATEGY]"
    echo ""
    echo "Stratégies disponibles:"
    echo "  mean_reversion  - Mean Reversion Strategy (par défaut)"
    echo "  rsi             - RSI Strategy"
    echo "  bollinger       - Bollinger Bands Strategy"
    echo ""
    echo "Exemples:"
    echo "  ./run_agent.sh              # Lance Mean Reversion"
    echo "  ./run_agent.sh rsi          # Lance RSI Strategy"
    echo "  ./run_agent.sh bollinger    # Lance Bollinger Bands"
    echo ""
    exit 0
fi

echo "Stratégie: $STRATEGY"
echo "Lancement de l'agent..."
echo ""

STRATEGY=$STRATEGY python3 -m agents.main
