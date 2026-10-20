"""Loading the SD 1.x components and selecting what to train."""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn

from .config import ComponentWeights, Config
from .lora import inject_lora, lora_modules, lora_parameters, load_lora_file, load_lora_state_dict
from .utils import count_params, fmt_count

# Files that `diffusers` needs beside the weights. scripts/download_sd15.sh only
# fetches the weights, so these are pulled from `model.config_repo` on demand.
CONFIG_FILES = [
    "unet/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "scheduler/scheduler_config.json",
]


def ensure_configs(base: str, config_repo: str) -> None:
    if not os.path.isdir(base):
        return  # a hub id: diffusers downloads everything itself
    missing = [f for f in CONFIG_FILES if not os.path.exists(os.path.join(base, f))]
    if not missing:
        return
    from huggingface_hub import hf_hub_download
    print(f"fetching {len(missing)} config file(s) from {config_repo} into {base}")
    for f in missing:
        hf_hub_download(repo_id=config_repo, filename=f, local_dir=base)


def detect_variant(base: str, requested: Optional[str]) -> Optional[str]:
    if requested is not None or not os.path.isdir(base):
        return requested
    if os.path.exists(os.path.join(base, "unet", "diffusion_pytorch_model.safetensors")):
        return None
    if os.path.exists(os.path.join(base, "unet", "diffusion_pytorch_model.fp16.safetensors")):
        return "fp16"
    return None


@dataclass
class Components:
    tokenizer: object
    text_encoder: nn.Module
    vae: nn.Module
    unet: nn.Module
    noise_scheduler: object
    variant: Optional[str] = None


def load_components(cfg: Config) -> Components:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    base = cfg.model.base
    ensure_configs(base, cfg.model.config_repo)
    variant = detect_variant(base, cfg.model.variant)
    kw = {"variant": variant} if variant else {}
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", **kw).to(torch.float32)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", **kw).to(torch.float32)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", **kw).to(torch.float32)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
    return Components(tokenizer, text_encoder, vae, unet, noise_scheduler, variant)


# ----------------------------------------------------------------------------
# Trainable parameter selection
# ----------------------------------------------------------------------------
@dataclass
class ParamGroup:
    component: str
    lr: float
    names: List[str]
    params: List[nn.Parameter]


@dataclass
class TrainSetup:
    method: str
    groups: List[ParamGroup] = field(default_factory=list)
    lora_injected: Dict[str, List[str]] = field(default_factory=dict)  # component -> module names

    @property
    def parameters(self) -> List[nn.Parameter]:
        return [p for g in self.groups for p in g.params]

    @property
    def trainable_count(self) -> int:
        return count_params(self.parameters)

    def optimizer_groups(self) -> List[dict]:
        return [{"params": g.params, "lr": g.lr, "name": g.component} for g in self.groups]

    def summary(self, comps: Components) -> str:
        lines = [f"method: {self.method}"]
        for comp_name, module in (("unet", comps.unet), ("text_encoder", comps.text_encoder)):
            total = count_params(module.parameters())
            group = next((g for g in self.groups if g.component == comp_name), None)
            if group is None:
                lines.append(f"  {comp_name}: {fmt_count(total)} params, frozen")
                continue
            n = count_params(group.params)
            extra = f", {len(self.lora_injected.get(comp_name, []))} LoRA modules" if self.method == "lora" else ""
            lines.append(f"  {comp_name}: {fmt_count(total)} params, training {fmt_count(n)} ({100 * n / max(1, total):.2f}%)"
                         f" in {len(group.names)} tensors, lr {group.lr:g}{extra}")
        lines.append(f"  total trainable: {fmt_count(self.trainable_count)}")
        return "\n".join(lines)


def _passes(name: str, cw: ComponentWeights) -> bool:
    if cw.include is not None and not any(re.search(p, name) for p in cw.include):
        return False
    return not any(re.search(p, name) for p in cw.exclude)


def apply_weights_config(cfg: Config, comps: Components) -> TrainSetup:
    """Freezes everything, then unfreezes (full) or injects adapters into (lora)
    the parameters selected by cfg.weights. Returns the optimizer groups."""
    w = cfg.weights
    setup = TrainSetup(method=w.method)
    components = (("unet", comps.unet, w.unet, w.lora.unet_targets),
                  ("text_encoder", comps.text_encoder, w.text_encoder, w.lora.text_encoder_targets))
    for comp_name, module, cw, targets in components:
        if not cw.train:
            continue
        lr = cw.lr if cw.lr is not None else cfg.train.lr
        if w.method == "full":
            names, params = [], []
            for name, p in module.named_parameters():
                if _passes(name, cw):
                    p.requires_grad_(True)
                    names.append(name)
                    params.append(p)
            if not params:
                raise ValueError(f"weights.{comp_name}: include/exclude patterns selected no parameters")
            setup.groups.append(ParamGroup(comp_name, lr, names, params))
        else:
            injected = inject_lora(module, targets, w.lora.rank, w.lora.alpha, w.lora.dropout, cw.include, cw.exclude)
            if not injected:
                raise ValueError(f"weights.{comp_name}: no modules matched lora targets {targets} with the include/exclude patterns")
            setup.lora_injected[comp_name] = injected
            params = list(lora_parameters(module))
            names = [f"{n}.lora_A/B" for n in injected]
            setup.groups.append(ParamGroup(comp_name, lr, names, params))
    if w.method == "lora" and w.init_from:
        tensors, _ = load_lora_file(w.init_from)
        for comp_name, module in (("unet", comps.unet), ("text_encoder", comps.text_encoder)):
            if comp_name in setup.lora_injected:
                loaded, missing = load_lora_state_dict(module, tensors, comp_name)
                print(f"init_from {w.init_from}: loaded {loaded} {comp_name} adapters" + (f", {len(missing)} not found in file" if missing else ""))
    return setup


def trainable_state_dict(comps: Components, setup: TrainSetup) -> Dict[str, torch.Tensor]:
    """Only what is being trained, with component prefixes (checkpoints)."""
    from .lora import lora_state_dict
    out = {}
    for comp_name, module in (("unet", comps.unet), ("text_encoder", comps.text_encoder)):
        group = next((g for g in setup.groups if g.component == comp_name), None)
        if group is None:
            continue
        if setup.method == "lora":
            out.update(lora_state_dict(module, comp_name))
        else:
            for name, p in zip(group.names, group.params):
                out[f"{comp_name}.{name}"] = p.detach().cpu().contiguous()
    return out


def load_trainable_state_dict(comps: Components, setup: TrainSetup, state: Dict[str, torch.Tensor]) -> None:
    for comp_name, module in (("unet", comps.unet), ("text_encoder", comps.text_encoder)):
        group = next((g for g in setup.groups if g.component == comp_name), None)
        if group is None:
            continue
        if setup.method == "lora":
            load_lora_state_dict(module, state, comp_name)
        else:
            for name, p in zip(group.names, group.params):
                key = f"{comp_name}.{name}"
                if key in state:
                    p.data.copy_(state[key].to(p.dtype))
