#!/bin/bash
# run_experiments.sh
# Runs all combinations of --mask_type and --wsmse_tag sequentially.
# Usage: bash run_experiments.sh
# Logs are saved to logs/<mask_type>_wsmse<tag>.out

set -e

WORKDIR="./experiments/02"
TYPE="other"
LOGDIR="./experiments/02/logs"
mkdir -p "$LOGDIR"

MASK_TYPES=("full" "erp")
#MASK_TYPES=("erp")
WSMSE_TAGS=(0 1)
SWHDC_TAGS=(0 1)
ERP_UPSAMPLING_TAGS=(0 1)
for erp_upsampling_tag in "${ERP_UPSAMPLING_TAGS[@]}"; do
    for swhdc_tag in "${SWHDC_TAGS[@]}"; do
        for mask_type in "${MASK_TYPES[@]}"; do
            for wsmse_tag in "${WSMSE_TAGS[@]}"; do
                LOG="$LOGDIR/${mask_type}_wsmse${wsmse_tag}_swhdc${swhdc_tag}_erpUpsampling${erp_upsampling_tag}.out"
                echo "========================================"
                echo "Starting: mask_type=${mask_type}  wsmse_tag=${wsmse_tag}"
                echo "Log: ${LOG}"
                echo "========================================"

                nohup python -u train.py \
                    --type "$TYPE" \
                    --mask_type "$mask_type" \
                    --wsmse_tag "$wsmse_tag" \
                    --swhdc_tag "$swhdc_tag" \
                    --workdir "$WORKDIR" \
                    --train_steps_1 1000 \
                    --train_steps_2 1000 \
                    --erp_upsampling "$erp_upsampling_tag" \
                    > "$LOG" 2>&1

                echo "Finished: mask_type=${mask_type}  wsmse_tag=${wsmse_tag}"
                echo ""
            done
        done
    done
done
echo "All experiments completed. Results in ${WORKDIR}/results.csv"
