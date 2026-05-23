from transformers import Trainer
from src.train.utils.optim import _create_optimizer

class SlimTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Keep your callback wiring.
        from src.train.callbacks import ProgressCallback
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, ProgressCallback):
                callback._trainer = self
                
    def create_optimizer(self):
        _create_optimizer(self)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **inp_kwargs):
        outputs = model(**inputs)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        loss_task = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        self.loss_task = float(loss_task.detach().cpu()) if loss_task is not None else 0.0
        self.loss_kd = 0.0  # SlimTrainer has no KD loss
        self.loss_logits = 0.0  # SlimTrainer has no logits loss
        
        if return_outputs:
            return loss_task, outputs
        return loss_task