#!/bin/bash
set -e
echo "Part II: OA + H2B (200k steps each, warm-start from Part I)"
if ! apptainer instance list 2>/dev/null | grep -q deepracer; then
    echo "Starting simulator..."; bash scripts/start_deepracer.sh; sleep 10
fi
echo "=== Object Avoidance ==="; python src/train_curriculum.py --stage oa
echo "=== Head-to-Bot ==="; python src/train_curriculum.py --stage h2b
echo "=== Demos ==="; python src/train_curriculum.py --stage demo
echo "=== Eval ==="; python src/train_curriculum.py --stage eval
echo "All done! Check saved_models/, runs/, demos/"
