export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export CALIB_DATASETS="c4"
# Mask method kwargs
export INTER_LAYER_METHOD="loss_coverage"
export INTRA_LAYER_METHOD="attr_coverage"
export INTRA_EXPERT_METRIC="activation"

# Adjust masks kwargs
export ALIGN_HIDDEN="0"
export ALIGN_KV_GROUP="0"
export ALIGN_INTER="0"
export MIN_PER_EXPERT="0"

# Quick test config
CUDA_VISIBLE_DEVICES=0 python src/calibration/channel_scoring/main.py \
    --model-name-or-path "Qwen/Qwen1.5-MoE-A2.7B" \
    --calib-datasets ${CALIB_DATASETS} \
    --calib-batches 10 \
    --per-device-train-batch-size 2 \
    --max-seq-length 256 \
    --device "cuda" \
    --dtype "bfloat16" \
    --output-dir "./results/" \
    --collect-h-data \
    --verbose
