# sdft: configurable fine-tuning for Z-Image and Stable Diffusion 1.x

A small PyTorch training framework that lives beside the C++ inference code.
One YAML file says which weights to train, how, and on what; the result is
exported as a diffusers model directory that `./sd` and `./sd-server` load
with no further conversion.

```
YAML config ──> sdft train ──> runs/<name>/checkpoint-N ──> models/<name>/ ──> ./sd --model-dir models/<name>
```

## Install

```sh
python3 -m venv finetune/.venv && source finetune/.venv/bin/activate
pip install -e finetune            # installs the `sdft` command and dependencies
```

Any Python 3.10 or newer works. Training needs PyTorch with a GPU backend
(CUDA, or Apple's MPS) for anything beyond a smoke test; a CPU works but is
very slow.

Without installing, run from the repository root with
`PYTHONPATH=finetune python -m sdft ...`.

## Five-minute check

```sh
python finetune/scripts/make_sample_dataset.py data/sample-shapes   # 16 synthetic images with captions
sdft dataset finetune/configs/smoke.yaml                            # shows the images and captions it found
sdft inspect finetune/configs/smoke.yaml                            # shows exactly what will be trained
sdft train   finetune/configs/smoke.yaml                            # 6 LoRA steps at 128px, then export
./sd --model-dir models/smoke-export --prompt "a red circle on a white background" --steps 20
```

The first run fetches the small `config.json` files that `diffusers` needs
beside the weights, since `scripts/download_sd15.sh` only fetches the
weights.

## Z-Image

Z-Image is the 6B transformer model the C++ binary runs by default. The
framework detects it from `model_index.json` (or `model.family: z-image`)
and switches to the flow-matching objective, the Qwen3 text encoder with
the chat template, and the FLUX VAE. Points to know:

- Train on the non-distilled base, `Tongyi-MAI/Z-Image`, downloaded with
  `ZIMAGE_REPO=Tongyi-MAI/Z-Image scripts/download_zimage.sh models/z-image`.
  The Turbo checkpoint is distilled for 8 steps and adapts poorly.
- `model.weights_dtype` defaults to `bf16` for Z-Image: the frozen 6B base
  takes 12 GB and the adapters are float32. Full fine-tuning needs
  `weights_dtype: fp32`, which is 25 GB of weights before gradients, so it
  is a multi-GPU job; LoRA is the practical choice.
- `weights.unet` means the generator, here the transformer. The default
  LoRA targets `to_q, to_k, to_v, to_out.0` match its attention; add
  `w1, w2, w3` for the feed-forward layers. `include: ["^layers\\."]`
  restricts training to the 30 main blocks.
- Timesteps are drawn logit-normally and shifted like the sampler
  (`shift: 3`), and the loss is the mean squared error against `x0 - noise`.
  `snr_gamma` does not apply.
- The export writes the merged transformer in bf16 and links the unchanged
  text encoder, tokenizer, VAE and scheduler to the base directory, so a
  fine-tune costs 12 GB rather than 33.

`configs/zimage_lora.yaml` is a starting point; `configs/zimage_smoke.yaml`
runs a two-block LoRA for four steps at 256 px as a check, which takes about
two minutes on an M1 Max.

## Preparing a dataset

Put images in a folder. Captions come from, in order of preference:

1. `metadata.jsonl` with one `{"file_name": "a.jpg", "text": "caption"}` per line
2. a sidecar text file with the same name, `a.txt` next to `a.jpg`
3. `data.default_caption`

Images are resized so the short side equals `data.resolution`, then
center-cropped (or randomly cropped with `center_crop: false`). Run
`sdft dataset config.yaml` to see what will be used.

For a subject, 10 to 30 images and captions that use a rare token for it
("a photo of sks dog") work well. For a style, 50 to several hundred images
with descriptive captions.

## The configuration file

Every key has a default; a config only needs what differs. Unknown keys are
rejected, so typos fail immediately. Values can be overridden on the command
line: `sdft train cfg.yaml --set train.steps=500 --set weights.lora.rank=32`.

### Choosing the weights

This is the `weights` section, and it is the point of the framework.

```yaml
weights:
  method: lora            # lora: small adapters on chosen layers; full: update the weights themselves
  unet:
    train: true
    lr: 1e-4              # per-component learning rate (defaults to train.lr)
    include: ["attn2"]    # regexes on names; only matching parameters/modules train
    exclude: ["mid_block"]
  text_encoder:
    train: false          # true adapts the prompt side too; use a small lr such as 1e-5
    lr: 1e-5
  lora:
    rank: 16              # adapter size; 4 to 8 for a subject, 16 to 64 for a style
    alpha: 16             # update scale (alpha / rank); alpha = rank is the common choice
    dropout: 0.0
    unet_targets: [to_q, to_k, to_v, to_out.0]          # which layers get adapters
    text_encoder_targets: [q_proj, k_proj, v_proj, out_proj]
  init_from: null         # start from an earlier lora.safetensors
```

| goal | settings |
|---|---|
| teach a subject (DreamBooth-style) | `method: lora`, rank 4 to 8, `text_encoder.train: true` with lr 1e-5, 500 to 1500 steps |
| teach a style | `method: lora`, rank 16 to 32, 1000 to 3000 steps, descriptive captions |
| bind prompts to concepts only | `unet.include: ["attn2"]` trains just the cross-attention layers |
| adapt convolutions too | add `conv1`, `conv2` or `proj_in`, `proj_out` to `unet_targets` |
| full fine-tune on a large dataset | `method: full`, lr 1e-6 to 5e-6, `ema_decay: 0.999`, see `configs/full.yaml` |
| partial full fine-tune | `method: full` with `unet.include` patterns such as `["attn", "transformer_blocks"]` |

`sdft inspect` prints the parameter count per component, the percentage being
trained, and the first tensor names, so a pattern can be checked before a run.

Learning rates matter more than anything else here. Adam moves every trained
parameter by roughly the learning rate each step whatever the gradient size.
`1e-4` is the usual starting point for LoRA and `1e-6` to `5e-6` for full
fine-tuning; an order of magnitude higher tends to overfit or degrade the
model within a few hundred steps. Watch the sample images.

Names to match against: UNet parameters look like
`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight`; text
encoder parameters like `text_model.encoder.layers.3.self_attn.q_proj.weight`.
With `method: lora`, patterns match module names (the same without `.weight`).

### Training

```yaml
train:
  steps: 1500
  batch_size: 2
  grad_accum: 2            # effective batch = batch_size * grad_accum
  lr: 1e-4
  lr_scheduler: cosine     # constant | linear | cosine
  warmup_steps: 50
  weight_decay: 0.01
  max_grad_norm: 1.0
  precision: bf16          # fp32 | fp16 | bf16 (autocast; fp16 uses a loss scaler on CUDA)
  gradient_checkpointing: true
  snr_gamma: 5.0           # min-SNR loss weighting; null to disable
  noise_offset: 0.0        # 0.05 to 0.1 helps very dark or bright images
  ema_decay: null          # 0.999 for full fine-tunes
  seed: 0
  device: auto             # auto | cuda | mps | cpu
```

`data.cache_latents: true` encodes every image once and trains on the stored
latents, which removes the VAE from the training step. Flips are applied to
the latents; crops are fixed to center crops.

### Checkpoints, logging, export

```yaml
checkpoint:
  dir: runs/my-run         # config.yaml, log.jsonl, checkpoint-N/, samples/
  every: 250
  keep: 3
  resume: null             # runs/my-run/checkpoint-500 to continue

logging:
  every: 10
  sample_every: 250        # generate sample_prompts into runs/my-run/samples/
  sample_prompts: ["a photo of sks dog on a beach"]

export:
  output_dir: models/my-model
  dtype: fp16              # fp16 halves the size; fp32 keeps full precision
  fuse_lora: true          # merge adapters into the weights so ./sd can load them
  save_lora: true          # also write lora.safetensors for diffusers or further training
  save_lora_only: false
```

A checkpoint holds only the trained tensors plus optimizer state. Export a
checkpoint without retraining with
`sdft export cfg.yaml --checkpoint runs/my-run/checkpoint-500`.

## Using the result

```sh
./sd --model-dir models/my-model --prompt "a photo of sks dog on a beach"
./sd-server --model-dir models/my-model
```

In Python, the same directory loads with
`StableDiffusionPipeline.from_pretrained("models/my-model")`. The exported
`lora.safetensors` uses `unet.<module>.lora_A.weight` keys and can be loaded
into a base pipeline with `pipe.load_lora_weights("models/my-model/lora.safetensors")`
or passed back to `weights.init_from` for further training.

## How it works

| file | contents |
|---|---|
| `sdft/config.py` | the dataclass schema, strict YAML loading, `--set` overrides |
| `sdft/lora.py` | LoRA for Linear and Conv2d in plain PyTorch: inject, save, load, merge |
| `sdft/model.py` | loading the components with `diffusers`; selecting trainable weights from the config |
| `sdft/data.py` | the image + caption dataset |
| `sdft/train.py` | the loop: noise, timestep, UNet prediction, MSE (optionally SNR-weighted), AdamW, EMA, checkpoints, samples |
| `sdft/export.py` | merging adapters and writing the diffusers directory |
| `sdft/cli.py` | `train`, `inspect`, `export`, `dataset` |
| `tests/` | config and LoRA unit tests: `python finetune/tests/test_config.py`, `python finetune/tests/test_lora.py` |

The loss is the standard denoising objective: sample a timestep, noise the
latent, predict the noise, mean squared error. Models with
`prediction_type: v_prediction` in their scheduler config are handled, though
the C++ binary expects epsilon prediction.

## Memory guide

| setting | 512 resolution, batch 1 |
|---|---|
| LoRA, UNet only, bf16, gradient checkpointing | about 6 GB |
| LoRA, UNet + text encoder | about 7 GB |
| full UNet fine-tune, bf16, checkpointing | about 20 GB, more with EMA |

Apple Silicon shares memory with the system; a 32 GB machine handles LoRA
comfortably at 512. Figures are approximate.
