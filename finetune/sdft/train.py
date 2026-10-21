"""The training loop."""

import glob
import math
import os
import random
import shutil
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, dump_config
from .data import ImageCaptionDataset, collate, load_image
from .export import export_model
from .model import Components, TrainSetup, apply_weights_config, load_components, load_trainable_state_dict, trainable_state_dict
from .utils import RunLogger, pick_device, seed_everything


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_lr_lambda(kind: str, total: int, warmup: int):
    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        if kind == "constant":
            return 1.0
        if kind == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine
    return f


class EMA:
    """Exponential moving average of the trainable parameters."""

    def __init__(self, params: List[torch.nn.Parameter], decay: float):
        self.decay = decay
        self.params = params
        self.shadow = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self) -> None:
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to_params(self) -> None:
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s)

    def state_dict(self) -> List[torch.Tensor]:
        return [s.cpu() for s in self.shadow]

    def load_state_dict(self, state: List[torch.Tensor]) -> None:
        for s, t in zip(self.shadow, state):
            s.copy_(t.to(s.device))


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def snr_weights(noise_scheduler, timesteps: torch.Tensor, gamma: float) -> torch.Tensor:
    ac = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    snr = ac / (1.0 - ac)
    w = torch.clamp(snr, max=gamma) / snr
    if noise_scheduler.config.prediction_type == "v_prediction":
        w = torch.clamp(snr, max=gamma) / (snr + 1.0)
    return w


def save_checkpoint(path: str, comps: Components, setup: TrainSetup, optimizer, lr_sched, step: int, ema: Optional[EMA]) -> None:
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    save_file(trainable_state_dict(comps, setup), os.path.join(path, "trainable.safetensors"))
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "lr_scheduler": lr_sched.state_dict(),
                "ema": ema.state_dict() if ema else None}, os.path.join(path, "state.pt"))


def load_checkpoint(path: str, comps: Components, setup: TrainSetup, optimizer, lr_sched, ema: Optional[EMA]) -> int:
    from safetensors.torch import load_file
    load_trainable_state_dict(comps, setup, load_file(os.path.join(path, "trainable.safetensors")))
    state_path = os.path.join(path, "state.pt")
    if not os.path.exists(state_path):
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if lr_sched is not None and state.get("lr_scheduler"):
        lr_sched.load_state_dict(state["lr_scheduler"])
    if ema is not None and state.get("ema"):
        ema.load_state_dict(state["ema"])
    return int(state.get("step", 0))


def prune_checkpoints(run_dir: str, keep: int) -> None:
    dirs = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")), key=lambda d: int(d.rsplit("-", 1)[1]))
    for d in dirs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


@torch.no_grad()
def encode_latents(vae, pixel_values: torch.Tensor) -> torch.Tensor:
    z = vae.encode(pixel_values).latent_dist.sample()
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    return (z - shift) * vae.config.scaling_factor


def encode_text(comps: Components, input_ids: torch.Tensor, grad: bool):
    """Returns the conditioning the generator expects.
    sd1: CLIP last hidden state [B, 77, 768].
    z-image: a list of per-sample Qwen3 penultimate hidden states [L_i, 2560] without padding."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if comps.family == "z-image":
            pad_id = comps.tokenizer.pad_token_id
            mask = input_ids != pad_id
            out = comps.text_encoder(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
            hidden = out.hidden_states[-2]
            return [hidden[i][mask[i]] for i in range(hidden.shape[0])]
        return comps.text_encoder(input_ids)[0]


def flow_sigmas(batch: int, shift: float, device) -> torch.Tensor:
    """Logit-normal timestep sampling (SD3 / FLUX recipe) followed by the
    scheduler's static shift, so training covers the same sigmas inference uses."""
    u = torch.sigmoid(torch.randn(batch, device=device))
    return shift * u / (1.0 + (shift - 1.0) * u)


@torch.no_grad()
def generate_samples(cfg: Config, comps: Components, device: torch.device, out_dir: str, step: int) -> None:
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    if comps.family == "z-image":
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline(scheduler=comps.noise_scheduler, vae=comps.vae, text_encoder=comps.text_encoder,
                              tokenizer=comps.tokenizer, transformer=comps.unet)
    else:
        pipe = StableDiffusionPipeline(vae=comps.vae, text_encoder=comps.text_encoder, tokenizer=comps.tokenizer, unet=comps.unet,
                                       scheduler=DDIMScheduler.from_config(comps.noise_scheduler.config),
                                       safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    os.makedirs(out_dir, exist_ok=True)
    was_training = comps.unet.training
    comps.unet.eval()
    gen = torch.Generator(device="cpu").manual_seed(cfg.train.seed)
    for i, prompt in enumerate(cfg.logging.sample_prompts):
        img = pipe(prompt, num_inference_steps=cfg.logging.sample_steps, guidance_scale=cfg.logging.sample_cfg,
                   height=cfg.logging.sample_size, width=cfg.logging.sample_size, generator=gen).images[0]
        img.save(os.path.join(out_dir, f"step{step:06d}-{i}.png"))
    if was_training:
        comps.unet.train()


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def train(cfg: Config) -> str:
    """Runs training as configured and returns the export directory."""
    device = pick_device(cfg.train.device)
    seed_everything(cfg.train.seed)
    run_dir = cfg.checkpoint.dir
    log = RunLogger(run_dir)
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    log.info(f"device: {device}")

    comps = load_components(cfg)
    setup = apply_weights_config(cfg, comps)
    log.info(f"family: {comps.family}, base weights {str(comps.weights_dtype).replace('torch.', '')}")
    log.info(setup.summary(comps))
    is_flow = comps.family == "z-image"
    flow_shift = float(getattr(comps.noise_scheduler.config, "shift", 3.0)) if is_flow else 0.0

    comps.vae.to(device).eval()
    comps.text_encoder.to(device)
    comps.unet.to(device)
    te_trainable = any(g.component == "text_encoder" for g in setup.groups)
    unet_trainable = any(g.component == "unet" for g in setup.groups)
    comps.unet.train(unet_trainable)
    comps.text_encoder.train(te_trainable)
    if cfg.train.gradient_checkpointing:
        if unet_trainable:
            comps.unet.enable_gradient_checkpointing()
        if te_trainable:
            comps.text_encoder.gradient_checkpointing_enable()

    dataset = ImageCaptionDataset(cfg.data, comps.tokenizer, seed=cfg.train.seed)
    log.info(f"dataset: {len(dataset.files)} images x {cfg.data.repeats} repeats, resolution {cfg.data.resolution}")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=len(dataset) >= cfg.train.batch_size,
                        num_workers=cfg.data.num_workers, collate_fn=collate)

    latent_cache: Dict[str, torch.Tensor] = {}
    if cfg.data.cache_latents:
        # Fixed center crops, no flip (flips are applied to the latents instead).
        log.info("caching latents...")
        for path in dataset.files:
            px = load_image(path, cfg.data.resolution, True, False, random.Random(0))
            latent_cache[path] = encode_latents(comps.vae, px[None].to(device))[0].cpu()
        log.info(f"cached {len(latent_cache)} latents")

    optimizer = torch.optim.AdamW(setup.optimizer_groups(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.999))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(cfg.train.lr_scheduler, cfg.train.steps, cfg.train.warmup_steps))
    ema = EMA(setup.parameters, cfg.train.ema_decay) if cfg.train.ema_decay else None
    use_scaler = cfg.train.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_scaler)

    step = 0
    if cfg.checkpoint.resume:
        step = load_checkpoint(cfg.checkpoint.resume, comps, setup, optimizer, lr_sched, ema)
        log.info(f"resumed from {cfg.checkpoint.resume} at step {step}")

    T = int(getattr(comps.noise_scheduler.config, "num_train_timesteps", 1000))
    loss_ema = None
    t_last = time.time()
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    while step < cfg.train.steps:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.train.grad_accum):
            batch = next_batch()
            input_ids = batch["input_ids"].to(device)
            if latent_cache:
                latents = torch.stack([latent_cache[p] for p in batch["path"]]).to(device)
                if cfg.data.random_flip:
                    flip = torch.rand(latents.shape[0], device=device) < 0.5
                    latents = torch.where(flip[:, None, None, None], latents.flip(-1), latents)
            else:
                latents = encode_latents(comps.vae, batch["pixel_values"].to(device))
            noise = torch.randn_like(latents)
            if cfg.train.noise_offset > 0:
                noise = noise + cfg.train.noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=device)

            with autocast_context(device, cfg.train.precision):
                hidden = encode_text(comps, input_ids, te_trainable)
                if is_flow:
                    # x_t = (1 - sigma) x0 + sigma noise; the model predicts x0 - noise
                    # (the pipeline negates its output before the Euler step).
                    sigmas = flow_sigmas(latents.shape[0], flow_shift, device)
                    s = sigmas[:, None, None, None]
                    noisy = (1.0 - s) * latents + s * noise
                    t_model = 1.0 - sigmas
                    outs = comps.unet(x=[n.unsqueeze(1).to(comps.weights_dtype) for n in noisy], t=t_model,
                                      cap_feats=[h.to(comps.weights_dtype) for h in hidden], return_dict=False)[0]
                    pred = torch.stack([o.squeeze(1) for o in outs])
                    target = latents - noise
                    timesteps = (sigmas * T).long()
                else:
                    timesteps = torch.randint(0, T, (latents.shape[0],), device=device, dtype=torch.long)
                    noisy = comps.noise_scheduler.add_noise(latents, noise, timesteps)
                    pred = comps.unet(noisy.to(comps.weights_dtype), timesteps, encoder_hidden_states=hidden.to(comps.weights_dtype)).sample
                    if comps.noise_scheduler.config.prediction_type == "v_prediction":
                        target = comps.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise
            loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
            if cfg.train.snr_gamma and not is_flow:
                loss = loss * snr_weights(comps.noise_scheduler, timesteps, cfg.train.snr_gamma)
            loss = loss.mean() / cfg.train.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(setup.parameters, cfg.train.max_grad_norm).item()
        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()
        if ema:
            ema.update()
        step += 1

        loss_ema = loss_accum if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_accum
        if step % cfg.logging.every == 0 or step == cfg.train.steps:
            now = time.time()
            log.log(step=step, loss=loss_accum, loss_ema=loss_ema, lr=lr_sched.get_last_lr()[0], grad_norm=grad_norm,
                    sec_per_step=(now - t_last) / cfg.logging.every)
            t_last = now
        if cfg.checkpoint.every and step % cfg.checkpoint.every == 0 and step < cfg.train.steps:
            path = os.path.join(run_dir, f"checkpoint-{step}")
            save_checkpoint(path, comps, setup, optimizer, lr_sched, step, ema)
            prune_checkpoints(run_dir, cfg.checkpoint.keep)
            log.info(f"saved {path}")
        if cfg.logging.sample_every and step % cfg.logging.sample_every == 0 and cfg.logging.sample_prompts:
            generate_samples(cfg, comps, device, os.path.join(run_dir, "samples"), step)

    final = os.path.join(run_dir, f"checkpoint-{step}")
    save_checkpoint(final, comps, setup, optimizer, lr_sched, step, ema)
    log.info(f"saved {final}")
    if ema:
        ema.copy_to_params()
        log.info("using EMA weights for export")
    out = export_model(cfg, comps, setup)
    log.info(f"exported to {out}")
    return out
