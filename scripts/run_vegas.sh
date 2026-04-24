#!/bin/bash
set -e
echo "Stage 3: Vegas Track (200k steps, warm-start)"
if ! apptainer instance list 2>/dev/null | grep -q deepracer; then
    echo "Starting simulator..."; bash scripts/start_deepracer.sh; sleep 10
fi
python src/train_curriculum.py --stage 3
echo "Done! Next: bash scripts/run_part2.sh"
