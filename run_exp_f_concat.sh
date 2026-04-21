#!/usr/bin/env bash
# Exp F — TCN input = concat(x_pre, visual_features). Base anchor: 4B
# (CLAS2 + TCN, CLASM/cts off). 2 variants:
#   F1 (concat_detach): visual_features.detach() → no grad back into GCN
#                       from TCN losses. Clean axis-isolation.
#   F2 (concat_joint) : no detach → TCN losses flow into GCN. Joint shaping.
#
# Expected (hypothesis):
#   F1 > 4B (22.59)                 → concat helps, detach sufficient
#   F2 > F1                         → joint gradient useful
#   F1 ≈ F2 ≈ 4B                    → features redundant with x_pre
#   F2 collapse / << F1             → CLAS2 vs TCN gradient conflict
#
# Run plan: F1 first. If F1 ≤ 22.59 → skip F2 (redundancy hypothesis).
#
# Usage:
#   bash run_exp_f_concat.sh           # F1 then F2
#   bash run_exp_f_concat.sh f1        # F1 only
#   bash run_exp_f_concat.sh f2        # F2 only
set -euo pipefail

mkdir -p logs final_model

if [ -t 2 ] && [ -w /dev/tty ]; then
    SWEEP_MODE=tty
    echo "[sweep] mode: interactive (tqdm → /dev/tty)"
else
    SWEEP_MODE=headless
    echo "[sweep] mode: headless (stderr merged into log)"
fi

run_train() {
    local logfile=$1
    shift
    if [[ "$SWEEP_MODE" == "tty" ]]; then
        python src/train.py "$@" 2>/dev/tty | tee "$logfile"
    else
        python src/train.py "$@" 2>&1 | tee "$logfile"
    fi
}

# 4B anchor (shared between F1 and F2)
BASE=(
  --max-epoch 15
  --phase1-epochs 0 --phase2-epochs 0
  --scheduler-milestones 9 13
  --lr-tcn 2e-4
  --lambda-nce 0 --lambda-cts 0
)

run_f1() {
    echo "===================================================================="
    echo "Exp F1: concat(x_pre, visual_features.detach())"
    echo "===================================================================="
    run_train logs/expf1_concat_detach.log "${BASE[@]}" \
        --tcn-input concat_detach \
        --checkpoint-path final_model/ckpt_expf1_concat_detach.pth \
        --model-path final_model/model_expf1_concat_detach.pth
}

run_f2() {
    echo "===================================================================="
    echo "Exp F2: concat(x_pre, visual_features)  [joint gradient]"
    echo "===================================================================="
    run_train logs/expf2_concat_joint.log "${BASE[@]}" \
        --tcn-input concat_joint \
        --checkpoint-path final_model/ckpt_expf2_concat_joint.pth \
        --model-path final_model/model_expf2_concat_joint.pth
}

if [[ $# -eq 0 ]]; then
    targets=(f1 f2)
else
    targets=("$@")
fi

for t in "${targets[@]}"; do
    case "$t" in
        f1) run_f1 ;;
        f2) run_f2 ;;
        *) echo "Unknown: $t (expect f1|f2)"; exit 1 ;;
    esac
done

echo "===================================================================="
echo "Exp F summary (best ckpt = max mAP_all, tiebreak mAP_abn)"
echo "===================================================================="
printf "  ref 4B         : mAP_abn=21.02 / mAP_all=19.24 / over_cov=2.03x  (α=0 TCN-only)\n"
printf "  ref 4B α=0.2   : mAP_abn=22.59 / mAP_all=21.43                   (fusion ceiling)\n"
for t in "${targets[@]}"; do
    log="logs/expf${t#f}_concat_$([ "$t" = "f1" ] && echo detach || echo joint).log"
    if [[ -f "$log" ]]; then
        final=$(grep -oE 'Final best mAP_all = [0-9.]+ \| mAP_abn = [0-9.]+' "$log" | tail -1)
        echo "  Exp ${t^^} : ${final:-<unfinished>}"
    else
        echo "  Exp ${t^^} : <log missing>"
    fi
done
