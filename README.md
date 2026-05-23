# Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression

> **[Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression](https://openreview.net/pdf?id=oreET6Wz52)**  
> Yifu Ding, Jiacheng Wang, Ge Yang, Yongcheng Jing, Jinyang Guo, Xianglong Liu, Dacheng Tao  
> *Forty-Third International Conference on Machine Learning (ICML 2026) — **Spotlight***

---

## Overview

This repository contains the official implementation of our ICML 2026 Spotlight paper. We propose a structured pruning framework for **Mixture-of-Experts (MoE) Large Language Models** that achieves significant parameter reduction with minimal performance degradation.

Our method operates in four stages:

1. **Channel Scoring** — Computes attribution-based importance scores for each expert's intermediate dimensions using first-order Taylor approximations over calibration data.
2. **Mask Generation** — Plans pruning budgets across layers (*inter-layer*) and across experts within each layer (*intra-layer*) via coverage-maximized allocation.
3. **Structural Pruning** — Physically removes pruned channels from expert feed-forward networks, producing a smaller dense model checkpoint.
4. **Fine-tuning** — Recovers accuracy through LoRA-based fine-tuning with optional gradual mask annealing.

### Supported Models

| Model | Parameters (Total / Active) |
|---|---|
| Qwen1.5-MoE-A2.7B | 14.3B / 2.7B |
| DeepSeek-MoE-16B | 16.4B / 2.8B |
| DeepSeek-V2-Lite | 15.7B / 2.4B |
| Qwen3-30B-A3B | 30.5B / 3.3B |

---

## Configuration

Configuration files are located in `configs/`. Select the appropriate file for your target model:

```
configs/
├── train/
│   ├── qwen1_5_moe_a2_7b_e2e_alpaca.yaml
│   ├── deepseek-ai_deepseek-moe-16b_alpaca.yaml
│   ├── deepseek-v2-lite.yaml
│   └── qwen3_30b_a3b_e2e_alpaca.yaml
└── eval/
    └── ...
```

Edit the relevant YAML to set model paths, output directories, pruning ratios, and training hyperparameters before running each stage.

---

## Project Structure

```
├── src/
│   ├── base/              # Shared utilities: models, datasets, argument parsing
│   ├── calibration/       # Channel scoring and expert importance analysis
│   ├── prune/             # Mask generation and structural pruning
│   └── train/             # LoRA fine-tuning and evaluation
├── configs/               # YAML configuration files
├── scripts/               # Shell scripts for each pipeline stage
├── eval/                  # lm-eval integration utilities
└── requirements.txt
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ding2026attribution,
  title     = {Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression},
  author    = {Ding, Yifu and Wang, Jiacheng and Yang, Ge and Jing, Yongcheng and Guo, Jinyang and Liu, Xianglong and Tao, Dacheng},
  booktitle = {Proceedings of the Forty-Third International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  note      = {Proceedings URL will be updated once available on PMLR}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
