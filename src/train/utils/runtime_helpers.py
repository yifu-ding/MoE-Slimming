import os
from typing import Optional
from src.base.shared_utils import _print

def resolve_resume_checkpoint(args) -> Optional[str]:
    if not args.resume_path or not args.resume_training:
        return None
    trainer_state = os.path.join(args.resume_path, "trainer_state.json")
    if os.path.isfile(trainer_state):
        _print(f"[peft] resume training from: {trainer_state}")
        return args.resume_path
    return None

