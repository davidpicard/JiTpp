import torch.nn as nn


def add_weight_decay(model: nn.Module, weight_decay: float = 0.0, skip_list: tuple = ()):
    """Split model parameters into two groups: with and without weight decay.

    1-D parameters (norms, biases) are always placed in the no-decay group.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay,    "weight_decay": weight_decay},
    ]
