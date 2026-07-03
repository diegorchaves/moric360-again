#!/bin/bash
# run_experiments.sh
# Usage: bash run_experiments.sh
# Logs: ${WORKDIR}/logs/<label>.out

WORKDIR="./experiments/vcip_othim19_bpp_busca_4"
TYPE="other"
LOGDIR="${WORKDIR}/logs"
mkdir -p "$LOGDIR"

# ============================================================
# MODO ATIVO: apenas cenários com loss combined
# Itera: mask_type × swhdc_tag × erp_padding
# Total: 2 × 2 × 1 = 4 experimentos
# ============================================================

MASK_TYPES=("erp")
SWHDC_TAGS=(1)
ERP_PADDING_TAGS=(1)

# Pesos padrão do TF (deixe vazio para usar os defaults do argparse):
# LAMBDA_DIST="--lambda_dist 2.34375e-3"
# LAMBDA_PERCEP="--lambda_percep 1.0"

LAMBDA_RATE_LIST=(0.0022)

TOTAL=$(( ${#MASK_TYPES[@]} * ${#SWHDC_TAGS[@]} * ${#ERP_PADDING_TAGS[@]} ))
COUNT=0

for mask_type in "${MASK_TYPES[@]}"; do
    for swhdc_tag in "${SWHDC_TAGS[@]}"; do
        for erp_padding in "${ERP_PADDING_TAGS[@]}"; do
            COUNT=$(( COUNT + 1 ))
            LOG="${LOGDIR}/${mask_type}_wsmse1_swhdc${swhdc_tag}_erp${erp_padding}.out"

            echo "========================================"
            echo "Experiment ${COUNT}/${TOTAL}"
            echo "  mask_type   = ${mask_type}"
            echo "  swhdc_tag   = ${swhdc_tag}"
            echo "  erp_padding = ${erp_padding}"
            echo "  Log: ${LOG}"
            echo "========================================"

            python -u train.py \
                --type "$TYPE" \
                --mask_type "$mask_type" \
                --wsmse_tag 1 \
                --swhdc_tag "$swhdc_tag" \
                --erp_padding "$erp_padding" \
                --workdir "$WORKDIR" \
                --train_steps_1 5000 \
                --train_steps_2 1000 \
                --lambda_rate_list "${LAMBDA_RATE_LIST[@]}" \
                > "$LOG" 2>&1

            STATUS=$?
            if [ $STATUS -eq 0 ]; then
                echo "Finished [${COUNT}/${TOTAL}]: mask=${mask_type} loss=combined swhdc=${swhdc_tag} erp=${erp_padding}"
            else
                echo "FAILED   [${COUNT}/${TOTAL}]: mask=${mask_type} loss=combined swhdc=${swhdc_tag} erp=${erp_padding} (exit ${STATUS}) — continuando..."
            fi
            echo ""
        done
    done
done

echo "========================================"
echo "Todos os ${TOTAL} experimentos concluídos."
echo "Resultados em: ${WORKDIR}/results.csv"
echo "Logs em:       ${LOGDIR}/"
echo "========================================"


# ============================================================
# MODO COMPLETO (COMENTADO) — descomente para rodar todos os
# cenários: mask_type × loss_type × swhdc_tag × erp_padding
# Total: 2 × 3 × 2 × 2 = 24 experimentos
# ============================================================

# WORKDIR="./experiments/vcip_full"
# TYPE="other"
# LOGDIR="${WORKDIR}/logs"
# mkdir -p "$LOGDIR"
#
# MASK_TYPES=("full" "erp")
# LOSS_TYPES=("mse" "wsmse" "combined")
# SWHDC_TAGS=(0 1)
# ERP_PADDING_TAGS=(0 1)
#
# TOTAL=$(( ${#MASK_TYPES[@]} * ${#LOSS_TYPES[@]} * ${#SWHDC_TAGS[@]} * ${#ERP_PADDING_TAGS[@]} ))
# COUNT=0
#
# for mask_type in "${MASK_TYPES[@]}"; do
#     for loss_type in "${LOSS_TYPES[@]}"; do
#         for swhdc_tag in "${SWHDC_TAGS[@]}"; do
#             for erp_padding in "${ERP_PADDING_TAGS[@]}"; do
#                 COUNT=$(( COUNT + 1 ))
#                 LOG="${LOGDIR}/${mask_type}_${loss_type}_swhdc${swhdc_tag}_erp${erp_padding}.out"
#
#                 # Mapeia loss_type → wsmse_tag para backward compat no nome do arquivo de imagem
#                 case "$loss_type" in
#                     wsmse)   WSMSE_TAG=1 ;;
#                     *)       WSMSE_TAG=0 ;;
#                 esac
#
#                 echo "========================================"
#                 echo "Experiment ${COUNT}/${TOTAL}"
#                 echo "  loss_type   = ${loss_type}"
#                 echo "  mask_type   = ${mask_type}"
#                 echo "  swhdc_tag   = ${swhdc_tag}"
#                 echo "  erp_padding = ${erp_padding}"
#                 echo "  Log: ${LOG}"
#                 echo "========================================"
#
#                 python -u train.py \
#                     --type "$TYPE" \
#                     --mask_type "$mask_type" \
#                     --loss_type "$loss_type" \
#                     --wsmse_tag "$WSMSE_TAG" \
#                     --swhdc_tag "$swhdc_tag" \
#                     --erp_padding "$erp_padding" \
#                     --workdir "$WORKDIR" \
#                     --train_steps_1 100000 \
#                     --train_steps_2 10000 \
#                     > "$LOG" 2>&1
#
#                 STATUS=$?
#                 if [ $STATUS -eq 0 ]; then
#                     echo "Finished [${COUNT}/${TOTAL}]: mask=${mask_type} loss=${loss_type} swhdc=${swhdc_tag} erp=${erp_padding}"
#                 else
#                     echo "FAILED   [${COUNT}/${TOTAL}]: mask=${mask_type} loss=${loss_type} swhdc=${swhdc_tag} erp=${erp_padding} (exit ${STATUS}) — continuando..."
#                 fi
#                 echo ""
#             done
#         done
#     done
# done
#
# echo "========================================"
# echo "Todos os ${TOTAL} experimentos concluídos."
# echo "Resultados em: ${WORKDIR}/results.csv"
# echo "Logs em:       ${LOGDIR}/"
# echo "========================================"
