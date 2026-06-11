#!/bin/bash
# run_experiments.sh
# Runs all combinations of --mask_type and --wsmse_tag sequentially.
# Usage: bash run_experiments.sh
# Logs are saved to logs/<mask_type>_wsmse<tag>.out

set -e

WORKDIR="./experiments"
TYPE="other"
LOGDIR="./logs"
mkdir -p "$LOGDIR"

MASK_TYPES=("full" "erp")
WSMSE_TAGS=(0 1)

for mask_type in "${MASK_TYPES[@]}"; do
    for wsmse_tag in "${WSMSE_TAGS[@]}"; do
        LOG="$LOGDIR/${mask_type}_wsmse${wsmse_tag}.out"
        echo "========================================"
        echo "Starting: mask_type=${mask_type}  wsmse_tag=${wsmse_tag}"
        echo "Log: ${LOG}"
        echo "========================================"

        nohup python -u train.py \
            --type "$TYPE" \
            --mask_type "$mask_type" \
            --wsmse_tag "$wsmse_tag" \
            --workdir "$WORKDIR" \
            --train_steps_1 100000 \
            --train_steps_2 10000 \
            > "$LOG" 2>&1

        echo "Finished: mask_type=${mask_type}  wsmse_tag=${wsmse_tag}"
        echo ""
    done
done

echo "All experiments completed. Results in ${WORKDIR}/results.csv"
