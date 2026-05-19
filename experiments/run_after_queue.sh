#!/bin/bash
# Post-queue: inject _meta everywhere + aggregate.
# Run manually AFTER run_final.sh completes (sequencer finishes).
set -e
cd "$(dirname "$0")/.."
echo "=== [1] inject _meta into every metrics.json ==="
python3 experiments/lib/inject_meta.py
echo ""
echo "=== [2] aggregate all results ==="
python3 experiments/aggregate_results.py
echo ""
echo "DONE — paper-ready metrics.json files all have _meta with"
echo "git commit + torch/numpy versions + GPU name + seed."
