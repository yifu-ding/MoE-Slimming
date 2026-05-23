import torch
try:
    from peft import PeftModel  # type: ignore
except ImportError:  # compatible with no peft environment
    PeftModel = None

__all__ = [
    "remove_hooks",
    "add_mlp_hook_for_model",
    "add_down_proj_hook",
    "forward_inp_out_hook",
    "add_gate_hook",
    "add_up_proj_hook",
    "add_gate_proj_hook",
    "add_block_hook_for_model",
    "add_mlp_hook_for_model",
    "add_down_proj_hook",
    "add_up_proj_hook",
    "add_gate_proj_hook",
]


def forward_inp_hook(module, input, output):
    if hasattr(module, "save_tensors") and module.save_tensors == False:
        return 
    module.saved_input = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
    # module.saved_output = output[0].detach().clone()

def forward_out_hook(detach: bool = True):
    def _forward_out_hook(module, input, output):
        if hasattr(module, "save_tensors") and module.save_tensors == False:
            return 
        out = output[0] if isinstance(output, (tuple, list)) else output
        if detach:
            out = out.detach()
        # clone to ensure subsequent modifications do not affect the original tensor; when detach=False, still keep gradient
        module.saved_output = out.clone()

    return _forward_out_hook

def forward_inp_out_hook(stop_forward: bool = False):
    def _forward_inp_out_hook(module, input, output):
        if hasattr(module, "save_tensors") and module.save_tensors == False:
            return 
        module.saved_input = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
        module.saved_output = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        if stop_forward:
            raise ValueError("forward_inp_out_hook")
    return _forward_inp_out_hook

def backward_grad_inp_hook(module, grad_input, grad_output):
    if hasattr(module, "save_tensors") and module.save_tensors == False:
        return 
    if grad_input[0] is not None:
        module.saved_grad_in = grad_input[0].detach().clone() if isinstance(grad_input, tuple) else grad_input.detach().clone()
    else:
        module.saved_grad_in = None

def backward_grad_out_hook(module, grad_input, grad_output):
    if hasattr(module, "save_tensors") and module.save_tensors == False:
        return 
    if grad_output[0] is not None:
        module.saved_grad_out = grad_output[0].detach().clone() if isinstance(grad_output, tuple) else grad_output.detach().clone()
    else:
        module.saved_grad_out = None

def backward_grad_inp_out_hook(module, grad_input, grad_output):
    if hasattr(module, "save_tensors") and module.save_tensors == False:
        return 
    if grad_input[0] is not None:
        module.saved_grad_in = grad_input[0].detach().clone() if isinstance(grad_input, tuple) else grad_input.detach().clone()
    else:
        module.saved_grad_in = None
    if grad_output[0] is not None:
        module.saved_grad_out = grad_output[0].detach().clone() if isinstance(grad_output, tuple) else grad_output.detach().clone()
    else:
        module.saved_grad_out = None


def block_forward_inp_out_hook():
    def _apply(obj):
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone()
        if isinstance(obj, (list, tuple)):
            return type(obj)(_apply(v) for v in obj)
        if isinstance(obj, dict):
            return {k: _apply(v) for k, v in obj.items()}
        return obj

    def _forward_inp_out_hook(module, args, kwargs, output):
        if hasattr(module, "save_tensors") and module.save_tensors == False:
            return 
        # args is positional arguments tuple, kwargs is keyword arguments dict
        module.saved_input_args = _apply(args)
        module.saved_input_kwargs = _apply(kwargs)
        module.saved_output = _apply(output)

        if module.stop_forward:
            raise ValueError("block_forward_inp_out_hook")

    return _forward_inp_out_hook


def add_block_hook_for_model(model, layer_idx, stop_forward=True):
    block = model.model.layers[layer_idx]
    block.stop_forward = stop_forward
    hook = block.register_forward_hook(
        block_forward_inp_out_hook(),
        with_kwargs=True,      
        always_call=True,      
    )
    block.saved_input_args = None
    block.saved_input_kwargs = None
    block.saved_output = None
    return [hook]

def _get_block_for_layer(model, layer_idx):
    """
    get corresponding layer block according to model type:
    - PeftModel: structure is usually model.model.model.layers
    - normal HF model: model.model.layers
    """
    if PeftModel is not None and isinstance(model, PeftModel):
        # try three layer model first
        if hasattr(model, "model") and hasattr(model.model, "model"):
            return model.model.model.layers[layer_idx]
        # compatible with some implementations using base_model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            return model.base_model.model.layers[layer_idx]
    # default path: two layer model
    return model.model.layers[layer_idx]


def add_block_out_hook_for_model(model, layer_idx, detach: bool = True):
    block = _get_block_for_layer(model, layer_idx)
    hook = block.register_forward_hook(
        forward_out_hook(detach=detach),
    )
    block.saved_output = None
    return [hook]


def add_mlp_hook_for_model(model, layer_idx):
    mlp = model.model.layers[layer_idx].mlp
    hook = mlp.register_forward_hook(forward_inp_out_hook(stop_forward=True))
    mlp.saved_input = None
    mlp.saved_output = None
    return [hook]

def add_gate_hook(mlp):
    # import ipdb; ipdb.set_trace()
    # mlp.gate.requires_grad = True
    fwd_hook = mlp.gate.register_forward_hook(forward_inp_out_hook(stop_forward=False))
    bwd_hook = mlp.gate.register_full_backward_hook(backward_grad_inp_out_hook)
    mlp.gate.saved_input = None
    mlp.gate.saved_output = None
    mlp.gate.saved_grad_in = None
    mlp.gate.saved_grad_out = None
    return [fwd_hook, bwd_hook]

def add_down_proj_hook(mlp):
    hooks = []
    for expert in mlp.experts:
        hook = expert.down_proj.register_forward_hook(forward_inp_out_hook(stop_forward=False))
        expert.down_proj.saved_input = None
        expert.down_proj.saved_output = None
        hooks.append(hook)
        hook = expert.down_proj.register_full_backward_hook(backward_grad_inp_out_hook)
        expert.down_proj.saved_grad_in = None
        expert.down_proj.saved_grad_out = None
        hooks.append(hook)
    return hooks

def add_up_proj_hook(mlp):
    hooks = []
    for expert in mlp.experts:
        hook = expert.up_proj.register_forward_hook(forward_inp_out_hook(stop_forward=False))
        expert.up_proj.saved_input = None
        expert.up_proj.saved_output = None
        hooks.append(hook)
        hook = expert.up_proj.register_full_backward_hook(backward_grad_inp_out_hook)
        expert.up_proj.saved_grad_in = None
        expert.up_proj.saved_grad_out = None
        hooks.append(hook)
    return hooks

def add_gate_proj_hook(mlp):
    hooks = []
    for expert in mlp.experts:  
        hook = expert.gate_proj.register_forward_hook(forward_inp_out_hook(stop_forward=False))
        expert.gate_proj.saved_input = None
        expert.gate_proj.saved_output = None
        hooks.append(hook)
        hook = expert.gate_proj.register_full_backward_hook(backward_grad_inp_out_hook)
        expert.gate_proj.saved_grad_in = None
        expert.gate_proj.saved_grad_out = None
        hooks.append(hook)
    return hooks



def remove_hooks(hooks):
    if hooks is None:
        return
    if isinstance(hooks, (list, tuple)):
        for hook in hooks:
            hook.remove()
    else:
        hooks.remove()


def clear_hooks_for_mlp(mlp):
    mlp.saved_input = None
    mlp.saved_output = None
    mlp.saved_grad_in = None
    mlp.saved_grad_out = None
    mlp.gate.saved_input = None
    mlp.gate.saved_output = None
    mlp.gate.saved_grad_in = None
    mlp.gate.saved_grad_out = None
    for expert in mlp.experts:
        expert.down_proj.saved_input = None
        expert.down_proj.saved_output = None
        expert.down_proj.saved_grad_in = None
        expert.down_proj.saved_grad_out = None
        expert.up_proj.saved_input = None
        expert.up_proj.saved_output = None
        expert.up_proj.saved_grad_in = None
        expert.up_proj.saved_grad_out = None
        expert.gate_proj.saved_input = None
        expert.gate_proj.saved_output = None
        expert.gate_proj.saved_grad_in = None
        expert.gate_proj.saved_grad_out = None
    return mlp