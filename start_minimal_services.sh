#!/bin/bash
# start_minimal_services.sh
# Starts Gateway, Static Analysis, and Report Service.
# EXCLUDES: AI Service, Frontend.

# cd to script directory
cd "$(dirname "$0")"
ROOT=$(pwd)

echo "[Info] Starting Minimal Services (Gateway, StaticAnalysis, Report)..."

# Check for uvicorn
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

if ! command -v uvicorn &> /dev/null; then
    echo "[Error] uvicorn not found even after checking .venv/venv. Please install it."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH=$ROOT

# Start Services in background
# Gateway -> 8000
echo "[Start] Gateway on port 8000"
nohup uvicorn webapp.gateway.main:app --host 0.0.0.0 --port 8000 > output/gateway.log 2>&1 &
PID_GW=$!

# Static Analysis -> 8002
echo "[Start] Static Analysis on port 8002"
nohup uvicorn webapp.services.staticanalysis.app.main:app --host 0.0.0.0 --port 8002 > output/staticanalysis.log 2>&1 &
PID_SA=$!

# Report -> 8003
echo "[Start] Report Service on port 8003"
nohup uvicorn webapp.services.report.app.main:app --host 0.0.0.0 --port 8003 > output/report.log 2>&1 &
PID_REP=$!

echo "[Info] Services started. PIDs: GW=$PID_GW, SA=$PID_SA, REP=$PID_REP"
echo "[Info] AI Service (8001) is NOT started (Intentional)."
echo "[Info] Logs are being written to output/*.log (ensure output dir exists)"

# Save PIDs to file for easy stopping if needed
echo "$PID_GW" > minimal_services.pids
echo "$PID_SA" >> minimal_services.pids
echo "$PID_REP" >> minimal_services.pids

echo "[Ready] System ready for TC_17..TC_20 testing."
