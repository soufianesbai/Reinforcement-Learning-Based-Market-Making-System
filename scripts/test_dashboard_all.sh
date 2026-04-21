#!/bin/bash
# Script complet pour lancer le serveur dashboard, attendre qu'il soit prêt, puis lancer les tests API et E2E

set -e

# 1. Libérer le port 8000
bash "$(dirname "$0")/kill_port_8000.sh"


# 2. Lancer le serveur en arrière-plan et capturer les logs
SERVER_LOG="/tmp/dashboard_server.log"
rm -f "$SERVER_LOG"
python3 dashboard/server.py > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# 3. Attendre que le serveur soit prêt (ou crash)
for i in {1..15}; do
  if curl -s http://localhost:8000 > /dev/null; then
    echo "[OK] Dashboard server is up."
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[FAIL] Le serveur dashboard a crashé au démarrage. Voici les logs :"
    cat "$SERVER_LOG"
    exit 1
  fi
  echo "[WAIT] Waiting for dashboard server... ($i)"
  sleep 1
done

# Vérifier que le serveur est bien UP
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "[FAIL] Le serveur dashboard n'est pas lancé. Voici les logs :"
  cat "$SERVER_LOG"
  exit 1
fi

# 4. Lancer les tests API Python
.venv/bin/python -m pytest dashboard/tests/test_api_backend.py
API_STATUS=$?

# 5. Lancer les tests E2E Playwright (si installés)
if [ -f node_modules/.bin/playwright ]; then
  npx playwright test dashboard/tests/test_dashboard_e2e.spec.js
  E2E_STATUS=$?
else
  echo "[WARN] Playwright n'est pas installé. Skipping E2E tests."
  E2E_STATUS=0
fi

# 6. Arrêter le serveur
kill $SERVER_PID

# 7. Résumé
if [ $API_STATUS -eq 0 ] && [ $E2E_STATUS -eq 0 ]; then
  echo "\n✅ Tous les tests sont PASSÉS."
  exit 0
else
  echo "\n❌ Certains tests ont échoué."
  exit 1
fi
