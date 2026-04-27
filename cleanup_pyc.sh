#!/bin/bash
# v1.3.0 fix: one-time cleanup of ALL stale .pyc files and git-cached bytecode
# Run this ONCE from the project root before pulling the new zip:
#
#   bash cleanup_pyc.sh

set -e
cd "$(dirname "$0")"

echo "[cleanup] Removing all __pycache__ directories..."
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo "[cleanup] Removing all .pyc / .pyo files..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

echo "[cleanup] Removing these from git index (so they stop being tracked)..."
git rm -r --cached --ignore-unmatch "__pycache__" 2>/dev/null || true
git rm -r --cached --ignore-unmatch "*.pyc" 2>/dev/null || true

echo "[cleanup] Done. Now:"
echo "  1. Drop the v1.3.0 zip files into the project root"
echo "  2. git add .gitignore && git commit -m 'fix: purge stale bytecode, add .gitignore'"
echo "  3. python run.py"
