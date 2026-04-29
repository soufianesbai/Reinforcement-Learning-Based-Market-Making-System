#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "Launching MarketMind Dashboard server..."
echo "URL: http://127.0.0.1:8090"
echo ""

if [[ "$1" == "--restart" ]]; then
	EXISTING=$(lsof -ti TCP:8090 -sTCP:LISTEN 2>/dev/null)
	if [[ -n "$EXISTING" ]]; then
		echo "Restarting: killing existing process (PID $EXISTING)..."
		kill $EXISTING 2>/dev/null
		sleep 1
	fi
fi

if lsof -iTCP:8090 -sTCP:LISTEN >/dev/null 2>&1; then
	echo "ℹDashboard is already running on port 8090"
	echo "Open: http://127.0.0.1:8090"
	echo "Use './run_dashboard.sh --restart' to kill it and relaunch."
	exit 0
fi

python3 dashboard/server.py
