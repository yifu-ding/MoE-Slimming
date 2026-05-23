import torch
from peft import PeftModel


def demote_uint8_params_to_buffers(model):
    first_param = next(model.parameters(), None)
    model_device = first_param.device if first_param is not None else torch.device('cuda:0')
    
    for module_name, module in model.named_modules():
        if module is model:
            continue
        to_convert = []
        for name, param in list(module.named_parameters(recurse=False)):
            if param.dtype == torch.uint8:
                to_convert.append((name, param))
        for name, param in to_convert:
            delattr(module, name)
            buffer_data = param.data.to(model_device)
            module.register_buffer(name, buffer_data)
    return model


def prepare_model_for_training(pruned_model, args):
    pruned_model = demote_uint8_params_to_buffers(pruned_model) 
    pruned_model.set_adapter("default")
    
    for name, param in pruned_model.named_parameters():
        if ".lora_" in name and "default" in name:
            param.requires_grad_(True)
        elif hasattr(param, "requires_grad") and param.requires_grad:
            param.requires_grad_(False)
            
    pruned_model.train()
    pruned_model.config.use_cache = False
    if args.gradient_checkpointing:
        pruned_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
    
    if isinstance(pruned_model, PeftModel) and hasattr(pruned_model, "enable_input_require_grads"):
        pruned_model.enable_input_require_grads()
        
    if getattr(args, "train_router_aux_loss", False) and hasattr(pruned_model.config, "output_router_logits"):
        pruned_model.config.output_router_logits = True
        pruned_model.config.router_aux_loss_coef = getattr(args, "router_aux_loss_coef", 3e-2)
        
    return pruned_model
