#!/usr/bin/env bash
# CS7642 Project 4 v1.1.5b — RESTORE + PATCH-R one-shot script
# Run from your project_4 directory: bash restore_and_patch.sh
set -e

echo "=== Step 1: Check local run.py ==="
if [ ! -f run.py ]; then
    echo "ERROR: run.py not found. Are you in the project_4 directory?"
    exit 1
fi
LOCAL_SIZE=$(wc -c < run.py)
echo "Local run.py: $LOCAL_SIZE bytes"
if [ "$LOCAL_SIZE" -lt 50000 ]; then
    echo "WARNING: Local run.py looks too small ($LOCAL_SIZE bytes). Aborting."
    exit 1
fi

echo "=== Step 2: Push local run.py to fix broken remote ==="
git add run.py
git commit -m "restore: fix broken run.py (was 21 bytes from MCP truncation)" 2>/dev/null || echo "(nothing staged)"
git push origin main
echo "Remote restored."

echo "=== Step 3: Pull apply_patch_r.py ==="
git pull origin main

echo "=== Step 4: Apply PATCH-R ==="
python apply_patch_r.py

echo "=== Step 5: Push patched run.py ==="
git add run.py
git commit -m "v1.1.5b PATCH-R: BSTS-gated vperp-scaled off-track penalty"
git push origin main
echo "=== DONE ==="
