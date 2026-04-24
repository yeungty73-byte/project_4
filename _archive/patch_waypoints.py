#!/bin/bash
set -e
cd /home/hice1/tleung37/scratch/P4_deepracer

git checkout --orphan clean_push2
git rm -rf --cached .

# Source .py files (run.py + dependencies)
git add \
  project_4/run.py \
  project_4/agents.py \
  project_4/context_aware_agent.py \
  project_4/failure_analysis.py \
  project_4/stuck_tracker.py \
  project_4/corner_analysis.py \
  project_4/utils.py \
  project_4/federated_pool.py \
  project_4/race_line_engine.py \
  project_4/bsts_seasonal.py \
  project_4/live_metrics.py \
  project_4/live_dashboard.py \
  project_4/analyze_logs.py \
  project_4/td3_sac_ensemble.py \
  project_4/brake_field.py

# Support files
git add \
  project_4/references.bib \
  project_4/configs/hyper_params.yaml \
  project_4/configs/environment_params.yaml \
  project_4/README.md

# Scripts for DeepRacer sim management
git add project_4/scripts/

git commit -m "v1.0.7: waypoint tracking + DeepRacer termination conditions (clean repo)"
git branch -M main
git push --force origin main
echo "DONE - clean push with scripts/"
