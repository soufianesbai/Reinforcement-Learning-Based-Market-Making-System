#!/bin/bash
# Script pour tuer le processus qui occupe le port 8000
PID=$(lsof -ti:8000)
if [ -n "$PID" ]; then
  echo "Killing process on port 8000 (PID: $PID)"
  kill -9 $PID
else
  echo "No process found on port 8000."
fi
