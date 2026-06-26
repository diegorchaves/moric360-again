#!/bin/bash
# run_experiments.sh
# Executa todas as combinações de mask_type × wsmse_tag × swhdc_tag × erp_padding.
# Total: 2 × 2 × 2 × 2 = 16 experimentos.
# Usage: bash run_experiments.sh
# Logs: experiments/busca_por_lambdas/logs/<mask>_wsmse<w>_swhdc<s>_erp<e>.out

WORKDIR="./experiments/13"
TYPE="other"
LOGDIR="${WORKDIR}/logs"
mkdir -p "$LOGDIR"

MASK_TYPES=("full" "erp")
WSMSE_TAGS=(0 1)
SWHDC_TAGS=(0 1)
ERP_PADDING_TAGS=(0 1)

TOTAL=$(( ${#MASK_TYPES[@]} * ${#WSMSE_TAGS[@]} * ${#SWHDC_TAGS[@]} * ${#ERP_PADDING_TAGS[@]} ))
COUNT=0

for mask_type in "${MASK_TYPES[@]}"; do
    for wsmse_tag in "${WSMSE_TAGS[@]}"; do
        for swhdc_tag in "${SWHDC_TAGS[@]}"; do
            for erp_padding in "${ERP_PADDING_TAGS[@]}"; do
                COUNT=$(( COUNT + 1 ))
                LOG="${LOGDIR}/${mask_type}_wsmse${wsmse_tag}_swhdc${swhdc_tag}_erp${erp_padding}.out"

                echo "========================================"
                echo "Experiment ${COUNT}/${TOTAL}"
                echo "  mask_type   = ${mask_type}"
                echo "  wsmse_tag   = ${wsmse_tag}"
                echo "  swhdc_tag   = ${swhdc_tag}"
                echo "  erp_padding = ${erp_padding}"
                echo "  Log: ${LOG}"
                echo "========================================"

                python -u train.py \
                    --type "$TYPE" \
                    --mask_type "$mask_type" \
                    --wsmse_tag "$wsmse_tag" \
                    --swhdc_tag "$swhdc_tag" \
                    --erp_padding "$erp_padding" \
                    --workdir "$WORKDIR" \
                    --train_steps_1 50000 \
                    --train_steps_2 10000 \
                    > "$LOG" 2>&1

                STATUS=$?
                if [ $STATUS -eq 0 ]; then
                    echo "✓ Finished [${COUNT}/${TOTAL}]: mask=${mask_type} wsmse=${wsmse_tag} swhdc=${swhdc_tag} erp=${erp_padding}"
                else
                    echo "✗ FAILED   [${COUNT}/${TOTAL}]: mask=${mask_type} wsmse=${wsmse_tag} swhdc=${swhdc_tag} erp=${erp_padding} (exit ${STATUS}) — continuando..."
                fi
                echo ""
            done
        done
    done
done

echo "========================================"
echo "Todos os ${TOTAL} experimentos concluídos."
echo "Resultados em: ${WORKDIR}/results.csv"
echo "Logs em:       ${LOGDIR}/"
echo "========================================"
