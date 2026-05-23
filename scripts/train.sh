export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=8 --master_port=29502 src/train/train.py \
           --config configs/train/qwen1_5_moe_a2_7b_e2e_alpaca.yaml
