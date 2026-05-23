from __future__ import annotations

import os
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from src.base.shared_utils import _print

class SavePruningArtifactsCallback(TrainerCallback):
    """Save pruning-related objects (masks, drop_idx_plan, drop_kv_idx_plan, etc.) whenever the model is saved."""
    
    def __init__(
        self, 
        masks=None, 
        verbose: bool = True
    ):
        """
        Args:
            masks: Masks object to save
            drop_idx_plan: drop_idx_plan object to save
            drop_kv_idx_plan: drop_kv_idx_plan object to save
            verbose: Whether to print save information
        """
        self.artifacts = {}
        if masks is not None:
            self.artifacts['masks'] = masks
        self.verbose = verbose
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Triggered when saving model, also save pruning artifacts."""
        if not self.artifacts:
            if self.verbose:
                _print("[SavePruningArtifactsCallback] No artifacts to save, skipping")
            return
        
        # Determine save path
        # If intermediate checkpoint, save to corresponding checkpoint directory
        # If final save, save to output_dir
        if state.global_step > 0:
            checkpoint_folder = f"checkpoint-{state.global_step}"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)
        else:
            output_dir = args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, obj in self.artifacts.items():
            artifact_path = os.path.join(output_dir, f"{name}.pth")
            torch.save(obj, artifact_path)
            if self.verbose:
                _print(f"[SavePruningArtifactsCallback] {name} saved to {artifact_path}")
