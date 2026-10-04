#!/bin/bash
# Run the full model comparison on the datasets written by phy/wifiPosGenerateData.m.
#
# Usage: bash scripts/run_all.sh [data_dir] [epochs] [seeds]
#   data_dir: directory with the data_*.mat files (default: current directory)
#   epochs:   maximum number of epochs per run (default: 60)
#   seeds:    quoted list of seeds (default: "0 1 2")
#
# Runs that already have a result file are skipped, so the script can be restarted.

DATA=${1:-.}
EPOCHS=${2:-60}
SEEDS=${3:-"0 1 2"}
OUT=results/camera_ready
COMMON="--data_dir $DATA --output_dir $OUT --epochs $EPOCHS --warmup_epochs 5 --patience 15"

mkdir -p $OUT/logs

run() {
    name=$1; seed=$2; shift 2
    if [ -f "$OUT/${name}_seed${seed}.json" ]; then
        echo "Skipping $name seed $seed (already done)"
        return
    fi
    echo "Running $name seed $seed"
    python3 scripts/run_experiments.py --name $name --seed $seed $COMMON "$@" \
        > $OUT/logs/${name}_seed${seed}.log 2>&1
}

for seed in $SEEDS; do
    # Main comparison: proposed model and CNN baseline, complex and magnitude input
    run transformer_appos_complex   $seed --model transformer --use_anchor_position
    run cnn_complex                 $seed --model cnn
    run transformer_appos_magnitude $seed --model transformer --use_anchor_position --input magnitude
    run cnn_magnitude               $seed --model cnn --input magnitude
    run knn_complex                 $seed --model knn
    run knn_magnitude               $seed --model knn --input magnitude

    # Ablations of the proposed model
    run transformer_complex         $seed --model transformer
    run deepsets_appos_complex      $seed --model transformer --use_anchor_position --num_layers 0
    run attnpool_appos_complex      $seed --model transformer --use_anchor_position --pooling attention
    run apdropout_appos_complex     $seed --model transformer --use_anchor_position --anchor_dropout 0.25

    # Synchronized control: trained and tested without clock offset and phase noise
    run transformer_appos_complex_sync $seed --model transformer --use_anchor_position --train_condition synchronized
done

python3 scripts/summarize_results.py --results_dir $OUT
