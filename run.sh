#!/usr/bin/env bash
# ============================================================================
# Startup Script for Sentinel Graph AML Forensic System
# CSE4074 Forensic Analytics - Third Review
# Authors: Aakash R (23MIA1011) & Gowthamkrishna R (23MIA1087)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================="
echo "  SENTINEL GRAPH // AML NETWORK FORENSICS (NEO4J)        "
echo "========================================================="

# Ensure IBM AML dataset exists
if [ ! -f "data/HI-Small_Trans.csv" ] && [ ! -f "data/HI-Small_Trans_sample.csv" ]; then
    echo "[!] ERROR: IBM HI-Small dataset not found in data/ directory."
    echo "    Download it from: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
    echo "    Or from HuggingFace: https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans"
    exit 1
fi

echo "[+] Starting FastAPI server and forensic dashboard on http://localhost:8000..."
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
