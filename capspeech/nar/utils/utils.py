import torch
import numpy as np
import yaml
import os


def load_yaml_with_includes(yaml_file):
    def loader_with_include(loader, node):
        # Load the included file
        include_path = os.path.join(os.path.dirname(yaml_file), loader.construct_scalar(node))
        with open(include_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    yaml.add_constructor('!include', loader_with_include, Loader=yaml.FullLoader)

    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def initialize_controlnet(controlnet, model):
    model_state_dict = model.state_dict()
    controlnet_state_dict = controlnet.state_dict()

    # Create a new state_dict for controlnet
    new_state_dict = {}
    for k, v in controlnet_state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            new_state_dict[k] = model_state_dict[k]
        else:
            print(f'new layer in controlnet: {k}')
            new_state_dict[k] = v  # Keep the original if unmatched

    # Load the new state_dict into controlnet
    controlnet.load_state_dict(new_state_dict)
    return controlnet


def load_checkpoint(model, ckpt_path, device, use_ema = True):
    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location=device)

    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('ema_model.transformer'):
            new_key = key.replace('ema_model.transformer.', '')
            new_state_dict[new_key] = value

    load_info = model.load_state_dict(new_state_dict, strict=False)
    # The returned object provides two lists: 'missing_keys' and 'unexpected_keys'
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)
    return model


def customized_lr_scheduler(optimizer, warmup_steps=10000, decay_steps=1e6, end_factor=1e-4):
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=min(1 / warmup_steps, 1),
                                end_factor=1.0, total_iters=warmup_steps)

    decay_scheduler = LinearLR(optimizer,
                               start_factor=1.0,
                               end_factor=end_factor,
                               total_iters=decay_steps)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], 
                             milestones=[warmup_steps])
    return scheduler


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths <= lengths.unsqueeze(-1)