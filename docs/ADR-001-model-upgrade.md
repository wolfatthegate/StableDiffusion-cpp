# ADR-001: Replace Stable Diffusion 1.5 with Z-Image

Status: accepted and implemented (phases 0 to 7). Date: 2026-10-21.

Measured on the M1 Max after implementation, Accelerate build, Z-Image-Turbo:

| | 512 x 512 | 1024 x 1024 |
|---|---|---|
| per transformer step | 8.2 s | 38.4 s |
| full image (9 steps + decode) | 85 s | 372 s |
| peak memory | 22 GB | 25 GB |

The C++ transformer matches the PyTorch reference to a relative error of
6e-6, the Qwen3 encoder to 1.4e-6 and the VAE to 1.1e-6. The estimates
below were written before implementation and are kept for the record; the
measured 1024 time beat them because Accelerate reaches about 2 TFLOP/s on
the bf16 weight path, twice the assumed rate.

## Context

Stable Diffusion 1.5 (2022) is no longer competitive: weak prompt following,
poor hands and text, 512-pixel native resolution. The project's constraints
for a replacement are fixed by its purpose and by the machine it runs on:

| constraint | value |
|---|---|
| inference | CPU only, float32 arithmetic, from-scratch C++ with no dependencies |
| memory | 32 GB unified memory (Apple M1 Max), so weights plus activations must stay well under that |
| licence | must allow commercial use and redistribution of fine-tuned weights |
| training | the Python framework in `finetune/` must be able to fine-tune it on a single GPU |
| verification | a PyTorch reference must run on this machine so the C++ port can be checked tensor by tensor |

## Candidates (open weights, October 2026)

| model | params (generator + text encoder) | weights at bf16 | steps per image | licence | arena standing | verdict |
|---|---|---|---|---|---|---|
| Qwen-Image-2.1 | 20B + 7B | ~54 GB | 50 with CFG | research only | #1 open weights (1,035 Elo) | licence and memory rule it out |
| FLUX.2 [dev] | 32B + 24B | ~112 GB | 50 with CFG | non-commercial | #3 open weights | licence and memory rule it out |
| Qwen-Image-2512 | 20B + 7B | ~54 GB | 50 with CFG | Apache 2.0 | top tier | needs 8-bit weights to fit; 4x the compute of Z-Image |
| **Z-Image-Turbo** | **6.15B + 4B (Qwen3-4B)** | **12.3 + 8.0 GB** | **8, no CFG** | **Apache 2.0** | **#1 open weights at release (Dec 2025), still top tier** | **chosen** |
| FLUX.2 [klein] 4B | ~4B + 4B (Qwen3-4B) | 7.4 + 8.0 GB | 4, with CFG | Apache 2.0 | below Z-Image-Turbo | fallback if Z-Image proves too slow |
| SD 3.5 Medium | 2.5B + 5.8B (T5 + 2 CLIPs) | ~17 GB | 28 with CFG | community licence, gated | older | superseded |
| SDXL | 2.6B + 1.5B | ~8 GB | 25 with CFG | CreativeML OpenRAIL++ | far below | easiest port, not good enough |

Sources: Artificial Analysis leaderboard and announcement
([leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image?open-weights=true),
[Z-Image #1 post](https://x.com/ArtificialAnlys/status/2002839525609865575)),
the [Z-Image-Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
the [Z-Image paper](https://arxiv.org/pdf/2511.22699),
[Z-Image-Base release coverage](https://comfyui-wiki.com/en/news/2026-01-28-alibaba-z-image-base-release),
[FLUX.2 licensing and sizes](https://www.runpod.io/articles/guides/deploying-flux-2),
[FLUX.2 klein details](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF),
[SD 3.5 Medium licence](https://replicate.com/stability-ai/stable-diffusion-3.5-medium/readme),
and a [2026 open-model survey](https://pinggy.io/blog/best_free_open_source_ai_image_generators_to_self_host/).

## Decision

Adopt **Z-Image-Turbo** (`Tongyi-MAI/Z-Image-Turbo`) for inference and
**Z-Image** (`Tongyi-MAI/Z-Image`, the non-distilled base) for fine-tuning.
Both are Apache 2.0 and share one architecture, so one C++ implementation
serves both. Z-Image-Turbo is the strongest model that fits the memory
budget without quantisation, and its 8 guidance-free steps make it the
cheapest top-tier model per image, which matters more on a CPU than
anywhere else.

Stable Diffusion 1.5 support stays in the code base. The two model families
sit behind one `Pipeline` interface selected by the model directory's
`model_index.json`.

## What changes

The architecture is entirely different from SD 1.5. From the diffusers
reference implementation (`transformer_z_image.py`, `pipeline_z_image.py`):

| component | SD 1.5 (today) | Z-Image |
|---|---|---|
| text encoder | CLIP ViT-L/14, 123M, 77 tokens, causal transformer | Qwen3-4B language model, 36 layers, 2560 wide, grouped-query attention (32 query / 8 key-value heads), RoPE, SwiGLU; prompt wrapped in the chat template, up to 512 tokens, penultimate hidden state used |
| generator | UNet, 860M, convolutions + cross-attention | single-stream diffusion transformer, 6.15B: 30 blocks 3840 wide, 30 heads of 128, SwiGLU 10240, RMSNorm with QK-norm, 3-axis RoPE (theta 256), adaLN modulation from a 256-dim timestep embedding, plus 2 "noise refiner" and 2 "context refiner" blocks before the joint stack |
| latent | 4 channels, 1/8 resolution | 16 channels, 1/8 resolution, patched 2x2 into 64-feature tokens (4096 tokens at 1024x1024) |
| VAE | SD AutoencoderKL | FLUX AutoencoderKL: same block structure, 16 channels, `scaling_factor` and `shift_factor` from its config. The existing decoder code reads channel counts from the weights and should load it unchanged |
| sampler | Euler / DDIM on a beta schedule, CFG 7.5 | flow-matching Euler: sigmas from 1 to 1/N with a resolution-dependent time shift, no CFG for Turbo, model predicts velocity |
| weights on disk | fp16, 2.1 GB | fp32 transformer (24.6 GB) + bf16 text encoder (8 GB); the C++ loader will keep bf16 in memory |

### Memory plan (32 GB machine)

Loading everything as float32 would need 41 GB, so the C++ side gains a
bf16 weight path: tensors stay bf16 in memory and are expanded to float32
tile by tile inside the GEMM. The text encoder (8 GB) runs first and is
freed before the transformer (12.3 GB) loads. Activations at 1024x1024 are
under 1 GB (the largest attention score matrix is 4,400 x 4,400 floats per
head, 77 MB). Expected peak: about 14 GB, against 6 GB today.

### Speed estimate (M1 Max, 10 cores)

One transformer forward is roughly 11 GFLOP per token per pass through the
stack, plus attention. These are estimates to be measured in phase 0 and
revised:

| resolution | tokens | per forward | 8 forwards + VAE | built-in GEMM (0.55 TFLOP/s) | Accelerate (about 1 TFLOP/s) |
|---|---|---|---|---|---|
| 512 x 512 | 1,024 + text | ~13 TFLOP | ~110 TFLOP | about 3.5 min | about 2 min |
| 1024 x 1024 | 4,096 + text | ~57 TFLOP | ~460 TFLOP | about 14 min | about 8 min |

SD 1.5 takes 95 s for a 512x512 image today. The upgrade trades time for
quality; the web interface's live preview makes the wait tolerable, and the
Accelerate build becomes the default on macOS. Z-Image's native resolution
is 1024; 512 and 768 are usable for drafts.

## Implementation plan

Each phase ends with a numeric check against the PyTorch reference on the
same inputs, using tensors dumped from diffusers.

0. **Reference and baseline.** Run `ZImagePipeline` on this machine with
   diffusers on Apple MPS; record quality, time and memory; dump per-component
   reference tensors (text embeddings, one transformer forward, VAE decode).
   Gate: the reference runs here.
1. **bf16 weights in C++.** `Weights` keeps a bf16 storage option; `gemm_nt`
   gains a bf16 B-operand path (weights are always the B operand); sharded
   safetensors (`*.safetensors.index.json`) are loaded. Also new primitives:
   RMSNorm, RoPE, SwiGLU, grouped-query attention with key/value head
   sharing, and a padding mask.
2. **Qwen3-4B text encoder.** Tokenizer (byte-level BPE with the Qwen
   vocabulary and the chat template), the 36-layer decoder with causal
   attention, output of the penultimate layer. Check against the dump.
3. **VAE.** Load the FLUX VAE with the existing decoder; apply scaling and
   shift factors. Check against the dump.
4. **S3-DiT transformer.** Patch embedding, refiners, 30 joint blocks with
   adaLN, final layer, unpatchify, 3-axis RoPE with the pipeline's position
   assignment. Check one forward against the dump.
5. **Flow-matching sampler and pipeline.** Sigma schedule with shift, Euler
   step on the negated prediction, `Pipeline` dispatch by model type, server
   and web page updates (resolution presets, no CFG for Turbo, previews from
   16-channel latents), download script for the weights.
6. **Fine-tuning.** `sdft` gains a `z-image` model type: Qwen3 encoder,
   16-channel latents, flow-matching loss with the logit-normal timestep
   sampling, LoRA targets for the DiT (`to_q`, `to_k`, `to_v`, `to_out.0`,
   `w1`, `w2`, `w3`), export in the Z-Image layout.
7. **Performance.** Accelerate default on macOS, tiled attention, parallel
   heads, bf16 GEMM tuning. Target: under 2 minutes for 512x512 and under 8
   for 1024x1024 on this machine.

## Risks

- **Speed on CPU.** If phase 0 and 4 measurements come out much worse than
  the estimates, FLUX.2 [klein] 4B (same text encoder, smaller generator,
  4 steps) is the fallback and reuses most of the new code.
- **Reference fidelity.** The port is validated against diffusers, not
  against the authors' code; diffusers is the de facto reference and is what
  the fine-tuning framework uses too.
- **Tokenizer.** The Qwen tokenizer has a 151k vocabulary, pre-tokenisation
  rules and special chat-template tokens; the existing CLIP tokenizer is a
  template, but this is new code and must be checked against `tokenizers`
  output for many strings.
- **Disk.** Z-Image-Turbo is 33 GB on disk as published (fp32 transformer);
  the download script will fetch it as is, and a one-time conversion to bf16
  halves it.

## Consequences

The code base grows a second model family and a bf16 weight path, and the
word "Stable Diffusion" in the project name becomes historical. The
dependency-free, from-scratch character is kept: no new libraries on the
C++ side.
