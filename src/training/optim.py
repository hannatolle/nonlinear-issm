"""
Optimizer construction.
"""

import torch
from typing import List, Sequence


def get_optimizer(models: Sequence[torch.nn.Module],
                  learning_rate: float,
                  weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Build an AdamW optimizer over all trainable params of models.

    Weight decay is applied to params with dim >= 2 only; biases, 1-d params and
    any names listed in a submodule's no_decay_params attribute get none.
    """
    if not isinstance(models, (list, tuple)):
        raise TypeError(f"`models` must be a list/tuple of nn.Module, got {type(models)}")

    no_decay_param_ids = set()

    for model in models:
        for submodule in model.modules():
            no_decay_names = getattr(submodule, "no_decay_params", None)
            if no_decay_names is None:
                continue
            for name, param in submodule.named_parameters(recurse=False):
                if name in no_decay_names:
                    no_decay_param_ids.add(id(param))

    # Deduplicated across models (shared submodules)
    decay_group: List[torch.nn.Parameter] = []
    no_decay_group: List[torch.nn.Parameter] = []

    seen = set()

    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)

            if (pid in no_decay_param_ids
                    or "bias" in name
                    or param.dim() < 2):
                no_decay_group.append(param)
            else:
                decay_group.append(param)

    param_groups = []
    if len(decay_group) > 0:
        param_groups.append({"params": decay_group, "weight_decay": float(weight_decay)})
    if len(no_decay_group) > 0:
        param_groups.append({"params": no_decay_group, "weight_decay": 0.0})

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found (all requires_grad=False?).")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(learning_rate))

    return optimizer