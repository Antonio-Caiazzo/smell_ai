#!/bin/bash
# stop_minimal_services.sh

cd "$(dirname "$0")"

PID_FILE="minimal_services.pids"

if [ ! -f "$PID_FILE" ]; then
    echo "[Info] No PID file ($PID_FILE) found. Services might not be running."
    exit 0
fi

echo "[Info] Stopping services listed in $PID_FILE..."

while read -r PID; do
    if ps -p "$PID" > /dev/null; then
        echo "  - Killing PID $PID"
        kill "$PID"
    else
        echo "  - PID $PID not found/already stopped."
    fi
done < "$PID_FILE"

rm "$PID_FILE"
echo "[Info] Services stopped and PID file removed."
