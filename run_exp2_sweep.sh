#!/usr/bin/env bash
# Exp 2 sweep — 6 runs across 2 strategies:
#
#   Group I  (loss ablation, keep Exp18 curriculum 3/6 + ms 12/18):
#     2A : tcn_bce only                  (ablate dice and ctr)
#     2B : tcn_bce + tcn_ctr             (ablate dice)
#     2C : tcn_bce + tcn_dice            (ablate ctr)
#
#   Group II (keep all 3 TCN losses, vary schedule / architecture):
#     2D : no-curriculum + ms 12/18 + lr_tcn 2e-4   (let TCN learn longer/louder)
#     2E : light curriculum 2/4 + ms 10/16          (2-ep backbone warmup)
#     2F : 2E + dilations [1,1,2]                   (sharpen bsh, RF 7 vs 15)
#
# Constants (fair within each group; differ between groups):
#   baseline finetune from model/model_ucf.pth (default)
#   --lr 2e-5     --batch-size 64    --max-epoch 20
#   --tcn-pos-weight 6.0    --gauss-sigma 2.0    --contrast-margin 0.3
#
# Evidence feeding these choices (prior):
#   Exp 1 (bce+dice+ctr, no curriculum)       : mAP 20.69 @ ep 3, bsh 0.0011
#   Exp 5 2d-ct (TCN + upsample)              : bsh collapse 0.0026
#   Exp 4 2d' (deeper TCN)                    : mAP 19.68, bsh 0.0160
#   dev_reframe 2c (1B' + parallel raw α=0.3) : mAP 22.17 (proven ceiling)
#
# Each run ≈ 90 min CPU. 6 runs ≈ 9 h. Expect to narrow to 1-2 winners then
# stack with hướng-A2 (parallel raw CLIP) for the real push past 22.17.
#
# Usage (from repo root, conda env vadclip activated):
#   bash run_exp2_sweep.sh                         # all 6
#   bash run_exp2_sweep.sh 2a 2e                   # cherry-pick
#   bash run_exp2_sweep.sh group1                  # 2a, 2b, 2c
#   bash run_exp2_sweep.sh group2                  # 2d, 2e, 2f
set -euo pipefail

mkdir -p logs final_model

# ---------- Group I: loss ablation (Exp18 curriculum + later LR drop) ----------
GROUP1_COMMON=(
  --phase1-epochs 3 --phase2-epochs 6
  --scheduler-milestones 12 18
)

run_2a() {
    echo "===================================================================="
    echo "Exp 2A [GroupI]: tcn_bce only  (ablate dice + ctr)"
    echo "===================================================================="
    python src/train.py "${GROUP1_COMMON[@]}" \
        --lambda-tcn-dice 0 --lambda-tcn-ctr 0 \
        --checkpoint-path final_model/ckpt_exp2a_tcn.pth \
        --model-path final_model/model_exp2a_tcn.pth \
        2>/dev/tty | tee logs/exp2a_tcn_train.log
}

run_2b() {
    echo "===================================================================="
    echo "Exp 2B [GroupI]: tcn_bce + tcn_ctr  (ablate dice)"
    echo "===================================================================="
    python src/train.py "${GROUP1_COMMON[@]}" \
        --lambda-tcn-dice 0 \
        --checkpoint-path final_model/ckpt_exp2b_tcn.pth \
        --model-path final_model/model_exp2b_tcn.pth \
        2>/dev/tty | tee logs/exp2b_tcn_train.log
}

run_2c() {
    echo "===================================================================="
    echo "Exp 2C [GroupI]: tcn_bce + tcn_dice  (ablate ctr)"
    echo "===================================================================="
    python src/train.py "${GROUP1_COMMON[@]}" \
        --lambda-tcn-ctr 0 \
        --checkpoint-path final_model/ckpt_exp2c_tcn.pth \
        --model-path final_model/model_exp2c_tcn.pth \
        2>/dev/tty | tee logs/exp2c_tcn_train.log
}

# ---------- Group II: schedule / architecture sweep (all 3 TCN losses) ----------

run_2d() {
    echo "===================================================================="
    echo "Exp 2D [GroupII]: no-curriculum + ms 12/18 + lr_tcn 2e-4"
    echo "===================================================================="
    python src/train.py \
        --phase1-epochs 0 --phase2-epochs 0 \
        --scheduler-milestones 12 18 \
        --lr-tcn 2e-4 \
        --checkpoint-path final_model/ckpt_exp2d_tcn.pth \
        --model-path final_model/model_exp2d_tcn.pth \
        2>/dev/tty | tee logs/exp2d_tcn_train.log
}

run_2e() {
    echo "===================================================================="
    echo "Exp 2E [GroupII]: light-curriculum 2/4 + ms 10/16"
    echo "===================================================================="
    python src/train.py \
        --phase1-epochs 2 --phase2-epochs 4 \
        --scheduler-milestones 10 16 \
        --checkpoint-path final_model/ckpt_exp2e_tcn.pth \
        --model-path final_model/model_exp2e_tcn.pth \
        2>/dev/tty | tee logs/exp2e_tcn_train.log
}

run_2f() {
    echo "===================================================================="
    echo "Exp 2F [GroupII]: 2E + dilations [1,1,2]  (sharpen bsh, RF=7)"
    echo "===================================================================="
    python src/train.py \
        --phase1-epochs 2 --phase2-epochs 4 \
        --scheduler-milestones 10 16 \
        --tcn-dilations 1 1 2 \
        --checkpoint-path final_model/ckpt_exp2f_tcn.pth \
        --model-path final_model/model_exp2f_tcn.pth \
        2>/dev/tty | tee logs/exp2f_tcn_train.log
}

# ---------- dispatcher ----------
expand_alias() {
    case "$1" in
        group1|g1) echo "2a 2b 2c" ;;
        group2|g2) echo "2d 2e 2f" ;;
        all)       echo "2a 2b 2c 2d 2e 2f" ;;
        *)         echo "$1" ;;
    esac
}

if [[ $# -eq 0 ]]; then
    targets=(2a 2b 2c 2d 2e 2f)
else
    targets=()
    for a in "$@"; do
        for t in $(expand_alias "$a"); do
            targets+=("$t")
        done
    done
fi

for t in "${targets[@]}"; do
    case "$t" in
        2a) run_2a ;;
        2b) run_2b ;;
        2c) run_2c ;;
        2d) run_2d ;;
        2e) run_2e ;;
        2f) run_2f ;;
        *) echo "Unknown target: $t (expect 2a|2b|2c|2d|2e|2f|group1|group2|all)"; exit 1 ;;
    esac
done

echo "===================================================================="
echo "Sweep done. Summary (best mAP_abn + bsh_med at that epoch):"
echo "===================================================================="
printf "  ref Exp 1 (bce+dice+ctr, no-curr, ms 6/11) : 20.69 / bsh 0.0011\n"
for t in "${targets[@]}"; do
    log="logs/exp${t}_tcn_train.log"
    if [[ -f "$log" ]]; then
        final=$(grep -oE 'Final best avg_mAP_abn = [0-9.]+' "$log" | tail -1)
        best_val=$(echo "$final" | grep -oE '[0-9]+\.[0-9]+')
        if [[ -n "$best_val" ]]; then
            bsh=$(grep "mAP_abn=${best_val} " "$log" | tail -1 \
                  | grep -oE 'bsh_med=[0-9.]+' | head -1)
        else
            bsh=""
        fi
        echo "  Exp ${t} : ${final:-<unfinished>}  ${bsh:-}"
    else
        echo "  Exp ${t} : <log missing>"
    fi
done
