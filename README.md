# StableDiffusion-cpp

Stable Diffusion 1.x text-to-image inference written from scratch in C++17.
No PyTorch, no BLAS, no image library: every operator (GEMM, convolution,
attention, normalisation), the safetensors loader, the CLIP BPE tokenizer,
the samplers and the PNG writer live in `src/`. It runs on the CPU in float32
and reads the standard diffusers weight layout.

## Build

```sh
make -j            # builds ./sd
make test          # builds and runs the kernel tests against naive references
```

Only a C++17 compiler and `make` are needed. On macOS you can swap the
built-in GEMM for Apple's BLAS with `make ACCELERATE=1`; everything else stays
the same.

## Weights

```sh
scripts/download_sd15.sh            # ~2.1 GB into models/sd15 (fp16 safetensors)
```

Any SD 1.x checkpoint in diffusers format works. The directory must contain:

```
models/sd15/
  tokenizer/vocab.json, merges.txt
  text_encoder/model[.fp16].safetensors
  unet/diffusion_pytorch_model[.fp16].safetensors
  vae/diffusion_pytorch_model[.fp16].safetensors
```

## Run

```sh
./sd --prompt "a photo of an astronaut riding a horse on mars" \
     --negative "blurry, low quality" \
     --steps 20 --cfg 7.5 --seed 42 --sampler euler --output astronaut.png
```

| flag | default | meaning |
|---|---|---|
| `--steps` | 20 | sampling steps |
| `--cfg` | 7.5 | classifier-free guidance scale (1 disables the unconditional pass) |
| `--width`, `--height` | 512 | image size, multiples of 8 (SD 1.x is trained at 512) |
| `--seed` | 42 | latent noise seed |
| `--sampler` | euler | `euler` (k-diffusion Euler) or `ddim` |
| `--threads` | all cores | worker threads |

On a 10-core Apple Silicon machine a 512x512 image at 20 steps takes about
95 seconds (4.3 s per step, 7 s for the VAE decode); each step is two UNet
evaluations (conditional and unconditional) of roughly 340 GFLOP each.

| `--prompt "a photo of an astronaut riding a horse on mars, highly detailed"` <br> `--negative "blurry, low quality, deformed" --steps 20 --seed 42` | `--prompt "a watercolor painting of a lighthouse at sunset"` <br> `--sampler ddim --steps 15 --seed 7` |
|---|---|
| ![astronaut](examples/astronaut.png) | ![lighthouse](examples/lighthouse.png) |

## How it works

```
prompt ──> CLIPTokenizer ──> CLIPTextEncoder ──┐ context [2,77,768]
                                               v
noise ──> scheduler loop: x_in = scale(x) ──> UNet(x_in, t, context) ──> eps ──> step ──> x
                                               ^
                                               └─ CFG: eps_u + s (eps_c - eps_u)
x / 0.18215 ──> VAEDecoder ──> [-1,1] RGB ──> write_png
```

| file | contents |
|---|---|
| `src/tensor.*` | `Tensor`, thread pool, blocked `gemm_nt` (4x4 register tile, SIMD via vector extensions), im2col conv2d, group/layer norm, attention, GEGLU, reshapes |
| `src/safetensors.*`, `src/json.*` | mmap-based safetensors reader with F16/BF16 to F32 conversion; a tiny JSON parser |
| `src/tokenizer.*` | CLIP byte-level BPE: byte-to-unicode table, regex-equivalent pre-tokeniser, merge ranks |
| `src/modules.*` | `ResnetBlock`, `CrossAttention`, GEGLU `FeedForward`, `Transformer2D`, VAE attention, bound to diffusers tensor names |
| `src/clip.*` | 12-layer pre-LN causal transformer with quick-GELU |
| `src/unet.*` | `UNet2DConditionModel`: sinusoidal time embedding, down/mid/up blocks with skip concatenation |
| `src/vae.*` | `AutoencoderKL` decoder path |
| `src/scheduler.*` | scaled-linear beta schedule, Euler and DDIM steps |
| `src/image.*` | PNG writer using stored deflate blocks |
| `src/main.cpp` | CLI and the sampling loop |

Model structure (block counts, channel widths, presence of attention or
shortcut convolutions) is discovered from the weight names and shapes rather
than hard-coded, so other SD 1.x fine-tunes load without changes.

## Limitations

- CPU float32 only; no fp16 compute, no GPU.
- The pre-tokeniser treats every non-ASCII code point as a letter, which
  differs from CLIP's Unicode-aware regex for non-Latin punctuation.
- SD 2.x (different head count, linear projections, v-prediction) and SDXL
  are not supported.
- Random numbers come from `std::mt19937_64`, so a seed does not reproduce
  the exact image a PyTorch pipeline makes with the same seed.
