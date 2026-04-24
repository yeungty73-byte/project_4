#!/bin/bash
echo "═══ P4 DeepRacer Status ═══"
echo "Checkpoints:"; find . -name "*.pt" -o -name "*.torch" 2>/dev/null | sort
echo ""; echo "TensorBoard runs:"; ls -lhrt runs/ 2>/dev/null || echo "  (none)"
echo ""; echo "Saved models:"; ls -lh saved_models/ 2>/dev/null || echo "  (none)"
echo ""; echo "Latest log:"; tail -15 $(ls -t *.log training*.log 2>/dev/null | head -1) 2>/dev/null || echo "  (none)"
echo ""; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "No GPU"
echo ""; apptainer instance list 2>/dev/null | head -5 || echo "apptainer not loaded"
