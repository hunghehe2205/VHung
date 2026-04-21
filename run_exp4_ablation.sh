#!/usr/bin/env bash
# Exp 4 — dead-weight loss ablation, all anchored on 2D config:
#   no-curriculum + ms 9/13 + lr_tcn 2e-4 + default dilations [1,2,4] + 15 ep
#
#   Run A (4A): TCN-only           — lambda_clas2=0 lambda_nce=0 lambda_cts=0
#               Q: 3 TCN losses có tự shape S đủ? Backbone collapse?
#
#   Run B (4B): CLAS2 + TCN        — lambda_nce=0 lambda_cts=0 (giữ CLAS2)
#               Q: CLASM + cts có dead weight không cho task 1-class?
#
#   Run C (4C): 2D full reproduce  — baseline trong cùng env/code path
#
# Code change: option.py thêm --lambda-clas2 (default 1.0);
#              train.py nhân args.lambda_clas2 vào loss_clas2 (line 207).
#
# Expected (hypothesis):
#   A ~ 2D → 3 legacy losses = dead weight
#   A collapse (mAP<15, bsh flat) → backbone cần CLAS2 anchor
#   B ~ 2D → CLASM+cts là dead weight, CLAS2 đủ làm prior
#   B < 2D → CLASM hoặc cts thật sự contribute (surprising)
#
# Usage:
#   bash run_exp4_ablation.sh            # all 3
#   bash run_exp4_ablation.sh 4a 4c      # cherry-pick
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

# 2D config anchor (all 3 runs share these)
BASE=(
  --max-epoch 15
  --phase1-epochs 0 --phase2-epochs 0
  --scheduler-milestones 9 13
  --lr-tcn 2e-4
)

run_4a() {
    echo "===================================================================="
    echo "Exp 4A: TCN-only (lambda_clas2=0 lambda_nce=0 lambda_cts=0)"
    echo "===================================================================="
    run_train logs/exp4a_tcn_train.log "${BASE[@]}" \
        --lambda-clas2 0 --lambda-nce 0 --lambda-cts 0 \
        --checkpoint-path final_model/ckpt_exp4a_tcn.pth \
        --model-path final_model/model_exp4a_tcn.pth
}

run_4b() {
    echo "===================================================================="
    echo "Exp 4B: CLAS2 + TCN  (lambda_nce=0 lambda_cts=0)"
    echo "===================================================================="
    run_train logs/exp4b_tcn_train.log "${BASE[@]}" \
        --lambda-nce 0 --lambda-cts 0 \
        --checkpoint-path final_model/ckpt_exp4b_tcn.pth \
        --model-path final_model/model_exp4b_tcn.pth
}

run_4c() {
    echo "===================================================================="
    echo "Exp 4C: 2D full reproduce (all losses ON)"
    echo "===================================================================="
    run_train logs/exp4c_tcn_train.log "${BASE[@]}" \
        --checkpoint-path final_model/ckpt_exp4c_tcn.pth \
        --model-path final_model/model_exp4c_tcn.pth
}

if [[ $# -eq 0 ]]; then
    targets=(4a 4b 4c)
else
    targets=("$@")
fi

for t in "${targets[@]}"; do
    case "$t" in
        4a) run_4a ;;
        4b) run_4b ;;
        4c) run_4c ;;
        *) echo "Unknown: $t (expect 4a|4b|4c)"; exit 1 ;;
    esac
done

echo "===================================================================="
echo "Exp 4 summary (best ckpt = max mAP_all, tiebreak mAP_abn)"
echo "===================================================================="
printf "  ref 2D (all ON) : mAP_abn=20.52 / mAP_all=19.16 / over_cov=2.66x\n"
for t in "${targets[@]}"; do
    log="logs/exp${t}_tcn_train.log"
    if [[ -f "$log" ]]; then
        final=$(grep -oE 'Final best mAP_all = [0-9.]+ \| mAP_abn = [0-9.]+' "$log" | tail -1)
        echo "  Exp ${t} : ${final:-<unfinished>}"
    else
        echo "  Exp ${t} : <log missing>"
    fi
done
