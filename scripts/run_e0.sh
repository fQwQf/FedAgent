#!/bin/bash
# E0: Gradient Consistency Probe
# Verifies P1: ||Δ^{Attn}||² vs ||Δ^{MLP}||²

set -euo pipefail

EXPERIMENT="e0_gradient_probe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "  FedRNK E0: Gradient Consistency Probe"
echo "  Timestamp: ${TIMESTAMP}"
echo "=========================================="

conda run --no-banner -n realm python -m src.analysis.gradient_analysis \
    --experiment e0 \
    logging.experiment_name="${EXPERIMENT}_${TIMESTAMP}" \
    training.num_rounds=3 \
    "$@"

echo ""
echo "Done. Results in outputs/${EXPERIMENT}_${TIMESTAMP}/"
