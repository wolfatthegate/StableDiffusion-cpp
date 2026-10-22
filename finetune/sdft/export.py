"""Exporting trained weights to a diffusers directory that `./sd` can load."""

import json
import os
import shutil
from typing import Optional

import torch

from .config import Config, dump_config
from .lora import lora_state_dict, merge_lora, save_lora_file
from .model import Components, TrainSetup


def export_model(cfg: Config, comps: Components, setup: TrainSetup) -> str:
    out = cfg.export.output_dir
    os.makedirs(out, exist_ok=True)
    e = cfg.export

    if setup.method == "lora":
        if e.save_lora or e.save_lora_only:
            tensors = {}
            for comp_name, module in (("unet", comps.unet), ("text_encoder", comps.text_encoder)):
                if comp_name in setup.lora_injected:
                    tensors.update(lora_state_dict(module, comp_name))
            path = os.path.join(out, "lora.safetensors")
            save_lora_file(path, tensors, cfg.weights.lora.rank, cfg.weights.lora.alpha,
                           extra={"unet_targets": cfg.weights.lora.unet_targets,
                                  "text_encoder_targets": cfg.weights.lora.text_encoder_targets,
                                  "base": cfg.model.base})
            print(f"wrote {path} ({len(tensors)} tensors)")
        if e.save_lora_only:
            dump_config(cfg, os.path.join(out, "training_config.yaml"))
            return out
        if e.fuse_lora:
            n = merge_lora(comps.unet) + merge_lora(comps.text_encoder)
            print(f"merged {n} LoRA modules into the base weights")
        else:
            print("warning: export.fuse_lora is false, so the exported model equals the base model; "
                  "the C++ binary cannot apply lora.safetensors on its own")

    if comps.family == "z-image":
        save_zimage(cfg, comps, out)
    else:
        save_pipeline(comps, out, torch.float16 if e.dtype == "fp16" else torch.float32)
        ensure_tokenizer_files(cfg.model.base, comps.tokenizer, out)
    dump_config(cfg, os.path.join(out, "training_config.yaml"))
    verify_export(out, comps.family)
    return out


def _link_or_copy(src: str, dst: str) -> None:
    """Shares large unchanged files with the base directory via a relative
    symlink, copying when links are unavailable."""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(os.path.relpath(os.path.abspath(src), os.path.dirname(os.path.abspath(dst))), dst)
    except OSError:
        shutil.copyfile(src, dst)


def save_zimage(cfg: Config, comps: Components, out: str) -> None:
    """Writes the trained transformer in bf16 and shares the untouched text
    encoder, tokenizer, VAE and scheduler with the base directory."""
    base = cfg.model.base
    comps.unet.to(torch.bfloat16).save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)
    for sub in ("text_encoder", "tokenizer", "vae", "scheduler"):
        src_dir = os.path.join(base, sub)
        if not os.path.isdir(src_dir):
            continue
        for name in os.listdir(src_dir):
            _link_or_copy(os.path.join(src_dir, name), os.path.join(out, sub, name))
    index = os.path.join(base, "model_index.json")
    if os.path.exists(index):
        shutil.copyfile(index, os.path.join(out, "model_index.json"))
    print(f"wrote transformer to {out}/transformer; other components link to {base}")


def ensure_tokenizer_files(base: str, tokenizer, out: str) -> None:
    """The C++ loader reads vocab.json and merges.txt. Newer `transformers`
    versions save only tokenizer.json, so write the classic files too."""
    tok_dir = os.path.join(out, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    need = ["vocab.json", "merges.txt"]
    if all(os.path.exists(os.path.join(tok_dir, n)) for n in need):
        return
    try:
        tokenizer.save_vocabulary(tok_dir)
    except Exception:
        pass
    if all(os.path.exists(os.path.join(tok_dir, n)) for n in need):
        return
    for n in need:
        dst = os.path.join(tok_dir, n)
        if os.path.exists(dst):
            continue
        src = os.path.join(base, "tokenizer", n)
        if os.path.isdir(base) and os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            from huggingface_hub import hf_hub_download
            shutil.copyfile(hf_hub_download(repo_id=base, filename=f"tokenizer/{n}"), dst)


def save_pipeline(comps: Components, out: str, dtype: torch.dtype) -> None:
    """Writes a complete diffusers pipeline directory (no safety checker)."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from diffusers.utils import logging as dlog
    verbosity = dlog.get_verbosity()
    dlog.set_verbosity_error()  # silences "cannot run fp16 on cpu" advice; we only save
    try:
        for m in (comps.vae, comps.text_encoder, comps.unet):
            m.to(torch.device("cpu"), dtype)
        pipe = StableDiffusionPipeline(vae=comps.vae, text_encoder=comps.text_encoder, tokenizer=comps.tokenizer, unet=comps.unet,
                                       scheduler=DDIMScheduler.from_config(comps.noise_scheduler.config),
                                       safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipe.save_pretrained(out, safe_serialization=True)
        # Restore fp32 so the caller's models stay usable (e.g. sampling after export).
        for m in (comps.vae, comps.text_encoder, comps.unet):
            m.to(torch.float32)
    finally:
        dlog.set_verbosity(verbosity)


def verify_export(out: str, family: str = "sd1") -> None:
    """Checks the files `./sd` needs and prints their sizes."""
    if family == "z-image":
        needed = ["model_index.json", "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
                  "text_encoder/model.safetensors.index.json", "transformer/config.json", "vae/diffusion_pytorch_model.safetensors",
                  "scheduler/scheduler_config.json"]
        tr = os.path.join(out, "transformer")
        if os.path.isdir(tr) and not any(f.startswith("diffusion_pytorch_model") and f.endswith(".safetensors") for f in os.listdir(tr)):
            needed.append("transformer/diffusion_pytorch_model.safetensors")
    else:
        needed = ["tokenizer/vocab.json", "tokenizer/merges.txt", "text_encoder/model.safetensors",
                  "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"]
    ok = True
    for rel in needed:
        p = os.path.join(out, rel)
        if os.path.exists(p):
            print(f"  {rel:45s} {os.path.getsize(p) / 1e6:9.1f} MB")
        else:
            print(f"  {rel:45s} MISSING")
            ok = False
    if ok:
        print(f"ready: ./sd --model-dir {out} --prompt \"...\"")
    else:
        print("export incomplete; the C++ binary will not load this directory")
