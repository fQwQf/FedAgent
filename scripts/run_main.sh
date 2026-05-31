#!/bin/bash
# Main experiment: 5 methods × 5 environments

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
METHODS=("local" "fedavg" "fedrnk" "fedrnk_inv" "centralized")

echo "=========================================="
echo "  FedRNK Main Experiment"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Methods: ${METHODS[*]}"
echo "=========================================="

for method in "${METHODS[@]}"; do
    echo ""
    echo "--- Running method: ${method} ---"

    conda run --no-banner -n realm python -m src.train \
        --method "${method}" \
        logging.experiment_name="main_${method}_${TIMESTAMP}" \
        training.num_rounds=20 \
        logging.log_gradients=false \
        "$@"
done

echo ""
echo "Done. All results in outputs/main_*_${TIMESTAMP}/"
