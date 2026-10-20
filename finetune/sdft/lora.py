"""Low-rank adapters (LoRA) in plain PyTorch.

A LoRA wraps a frozen Linear or Conv2d layer and adds a trainable low-rank
update: ``y = W x + (alpha / rank) * B (A x)``. ``A`` is initialised randomly
and ``B`` with zeros, so training starts from the original model exactly.
Merging adds ``(alpha / rank) * B A`` into ``W`` and removes the wrapper, which
is how the result becomes an ordinary checkpoint that the C++ binary loads.

Implemented here rather than with ``peft`` so that what is trained, saved and
merged is visible in one short file and does not depend on another library's
version.
"""

import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=torch.float32)
        self.lora_B.to(device=base.weight.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype))))
        return self.base(x) + update.to(x.dtype) * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base.weight += delta.to(self.base.weight.dtype)
        return self.base


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Conv2d(base.in_channels, rank, base.kernel_size, base.stride, base.padding, bias=False)
        self.lora_B = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=torch.float32)
        self.lora_B.to(device=base.weight.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype))))
        return self.base(x) + update.to(x.dtype) * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Conv2d:
        out_c, rank = self.lora_B.weight.shape[:2]
        delta = (self.lora_B.weight.view(out_c, rank) @ self.lora_A.weight.view(rank, -1)).view_as(self.base.weight)
        self.base.weight += (delta * self.scaling).to(self.base.weight.dtype)
        return self.base


LoRAModule = (LoRALinear, LoRAConv2d)


def _name_matches(name: str, suffixes: Sequence[str]) -> bool:
    return any(name == s or name.endswith("." + s) for s in suffixes)


def _passes_filters(name: str, include: Optional[Sequence[str]], exclude: Sequence[str]) -> bool:
    if include is not None and not any(re.search(p, name) for p in include):
        return False
    return not any(re.search(p, name) for p in exclude)


def inject_lora(model: nn.Module, targets: Sequence[str], rank: int, alpha: float, dropout: float = 0.0,
                include: Optional[Sequence[str]] = None, exclude: Sequence[str] = ()) -> List[str]:
    """Wraps every matching Linear/Conv2d in `model` and returns their names."""
    injected = []
    for name, module in list(model.named_modules()):
        if not _name_matches(name, targets) or not _passes_filters(name, include, exclude):
            continue
        if isinstance(module, LoRAModule):
            continue
        if isinstance(module, nn.Linear):
            wrapper = LoRALinear(module, rank, alpha, dropout)
        elif isinstance(module, nn.Conv2d):
            wrapper = LoRAConv2d(module, rank, alpha, dropout)
        else:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, wrapper)
        injected.append(name)
    return injected


def lora_modules(model: nn.Module) -> Dict[str, nn.Module]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRAModule)}


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for m in lora_modules(model).values():
        yield from m.lora_A.parameters()
        yield from m.lora_B.parameters()


def lora_state_dict(model: nn.Module, prefix: str) -> Dict[str, torch.Tensor]:
    """Adapter weights keyed like diffusers/peft: `<prefix>.<module>.lora_A.weight`."""
    out = {}
    for name, m in lora_modules(model).items():
        out[f"{prefix}.{name}.lora_A.weight"] = m.lora_A.weight.detach().cpu().contiguous()
        out[f"{prefix}.{name}.lora_B.weight"] = m.lora_B.weight.detach().cpu().contiguous()
    return out


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], prefix: str) -> Tuple[int, List[str]]:
    """Loads matching adapter weights; returns (loaded count, missing module names)."""
    loaded, missing = 0, []
    for name, m in lora_modules(model).items():
        a, b = f"{prefix}.{name}.lora_A.weight", f"{prefix}.{name}.lora_B.weight"
        if a in state and b in state:
            m.lora_A.weight.data.copy_(state[a].to(m.lora_A.weight.dtype))
            m.lora_B.weight.data.copy_(state[b].to(m.lora_B.weight.dtype))
            loaded += 1
        else:
            missing.append(name)
    return loaded, missing


def merge_lora(model: nn.Module) -> int:
    """Folds every adapter into its base layer and removes the wrappers."""
    count = 0
    for name, m in list(lora_modules(model).items()):
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, m.merged())
        count += 1
    return count


def save_lora_file(path: str, tensors: Dict[str, torch.Tensor], rank: int, alpha: float, extra: Optional[dict] = None):
    from safetensors.torch import save_file
    meta = {"format": "sdft-lora-v1", "rank": str(rank), "alpha": str(alpha)}
    if extra:
        meta["extra"] = json.dumps(extra)
    save_file(tensors, path, metadata=meta)


def load_lora_file(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    from safetensors import safe_open
    tensors, meta = {}, {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta
