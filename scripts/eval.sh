export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0
python src/train/merge_slim_eval.py --config configs/eval/qwen1_5_moe_a2_7b.yaml