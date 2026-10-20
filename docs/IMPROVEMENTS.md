# Performance and improvement guide

This document records where the time goes in StableDiffusion-cpp, how results
change from one processor to another, and which parts of the code to change
to get better or faster output.

Line references point to commit `c77ed85` on `main`. They will drift as the
code changes, so search for the function name if a link lands on the wrong
line.

## Contents

1. [Measured performance](#measured-performance)
2. [How performance varies between processors](#how-performance-varies-between-processors)
3. [Two ways to improve results](#two-ways-to-improve-results)
4. [Using your own fine-tuned weights](#using-your-own-fine-tuned-weights)
5. [Making it faster: code map](#making-it-faster-code-map)
6. [Making output better: code map](#making-output-better-code-map)
7. [Recommended order of work](#recommended-order-of-work)

## Measured performance

### Test machine

| item | value |
|---|---|
| processor | Apple M1 Max: 8 performance cores and 2 efficiency cores |
| operating system | macOS 26.6 |
| compiler | Apple Clang 21, `-O3` |
| model | Stable Diffusion 1.5, fp16 weights expanded to float32 |
| image | 512x512, classifier-free guidance on, so the UNet runs on a batch of 2 |
| threads | 10 |

### Built-in GEMM versus Apple Accelerate

GEMM is general matrix multiplication, the operation behind every layer.
The project has its own GEMM and can optionally use Apple's Accelerate
library instead.

| stage | built-in GEMM | Accelerate (`make ACCELERATE=1`) |
|---|---|---|
| one denoising step | 4.3 s | 2.5 s |
| VAE decode and PNG write | 7.1 s | 4.3 s |
| full 20-step image | about 94 s | about 55 s |
| largest pixel difference between the two outputs | | 1 level out of 255 |

The 55-second figure is 20 measured steps of 2.5 seconds plus the measured
decode time. A full 20-step run with Accelerate has not been timed end to end.

| other measurements | value |
|---|---|
| built-in GEMM throughput, 4096x320x2880 | 550 to 600 GFLOP/s |
| peak memory footprint, all three models resident | about 6.0 GB |

### Where the time goes in a denoising step

The profile below comes from the macOS `sample` tool. It sampled the built-in
GEMM build for 8 seconds during the UNet loop.

| work | share of samples | code |
|---|---|---|
| matrix multiplication | 70% | `gemm_nt` |
| threads waiting while serial code runs | 13% | thread pool wait |
| softmax in attention, mostly `expf` | 9% | `softmax_rows_` |
| im2col patch copying before convolutions | 4% | `conv2d` |
| `erff` in the GEGLU activation | 1% | `geglu` |
| everything else: norms, layout changes, adds | 3% | |

Matrix multiplication is the obvious target. The waiting time is the second
biggest cost. It comes from code that runs on one thread while the others
are idle, mainly the per-head loop and the transpose of V in `attention`.

### Reproducing these numbers

```sh
make clean && make -j && ./test_ops                  # GEMM benchmark is the last line
./sd --prompt "a photo of a cat" --steps 3           # per-step times on stderr
make clean && make -j ACCELERATE=1 && ./sd --prompt "a photo of a cat" --steps 3
/usr/bin/time -l ./sd --prompt "a photo of a cat" --steps 1   # memory
```

To profile on macOS, start `./sd` and run `sample <pid> 8` while it is in the
step loop. On Linux, use `perf record -g ./sd ...` and then `perf report`.

## How performance varies between processors

Yes, these numbers are specific to the M1 Max. Expect large differences on
other machines, for these reasons:

- **Core count.** The UNet's work splits across all threads, so time per
  step falls roughly in proportion to the number of fast cores. A 4-core
  laptop will be about twice as slow as this machine, and a 32-core
  workstation can be much faster.
- **Performance and efficiency cores.** Apple and recent Intel chips mix fast
  and slow cores. The thread pool hands out work dynamically, so slow cores
  take less of it, but they still add less than a fast core would.
- **Vector width.** The built-in GEMM uses 4-float vectors, which is the full
  width of ARM NEON. On x86 processors with AVX2 (8 floats) or AVX-512
  (16 floats), the same code leaves much of the hardware unused. Widening
  the vector type is the biggest single gain available on x86.
- **Accelerate is Apple-only.** It uses the matrix coprocessor in Apple
  Silicon, which is why it is so much faster here. On Linux, the equivalent
  is a tuned BLAS library such as OpenBLAS or Intel MKL. The code does not
  support those yet, but the change is small; see the code map below.
- **Memory bandwidth and cache size.** The UNet reads 3.4 GB of float32
  weights each step. Machines with less memory bandwidth or smaller caches
  will spend more time waiting on memory. The M1 Max has unusually high
  bandwidth.
- **Thermal limits.** A fanless laptop can slow down after a minute or two
  of full load, so later steps may be slower than the first ones.
- **Compiler.** GCC and Clang produce different code for the vector loops.
  On x86 the Makefile adds `-march=native`, so a binary built on one
  machine may not run on an older one.

To compare machines fairly, use the `test_ops` GEMM benchmark and the
per-step time from `./sd`, both at the same thread count.

## Two ways to improve results

There are two separate levers, and they live in different places.

| lever | what it changes | where the work happens |
|---|---|---|
| model weights | what the model draws: style, subjects, prompt following | in PyTorch, outside this repo. This repo only runs inference. |
| inference code | speed, memory, sampler quality, prompt handling, features | in this repo |

Fine-tuning changes the weights. The C++ code cannot train, because it has
no backward pass, no optimiser and no training data pipeline. The Python
framework in [finetune/](../finetune/README.md) does the training with
PyTorch and exports a directory that `./sd` loads. As long as the
architecture stays SD 1.x, no C++ changes are needed.

## Using your own fine-tuned weights

### With the bundled framework

```sh
pip install -e finetune
sdft inspect finetune/configs/lora.yaml      # which weights will train, and how many
sdft train   finetune/configs/lora.yaml      # trains, then exports to models/sd15-lora
./sd --model-dir models/sd15-lora --prompt "..."
```

The `weights` section of the YAML file selects the method (LoRA or full),
the components, per-component learning rates, include/exclude patterns on
layer names, and the LoRA rank and targets. The export merges adapters into
the weights and writes `vocab.json` and `merges.txt`, which the C++ loader
needs. See the framework README for the full option list.

### Export from your own diffusers code

If you train elsewhere, save the pipeline in diffusers format with
safetensors:

```python
pipe.save_pretrained("models/my-model", safe_serialization=True)
# optional: halve the file sizes
# pipe.to(torch.float16).save_pretrained("models/my-model", safe_serialization=True, variant="fp16")
```

Then run:

```sh
./sd --model-dir models/my-model --prompt "..."
```

The loader is `find_weights` in [src/pipeline.cpp](../src/pipeline.cpp).
It accepts `*.safetensors` and `*.fp16.safetensors` in the `text_encoder/`,
`unet/` and `vae/` folders, plus `tokenizer/vocab.json` and
`tokenizer/merges.txt`. Recent `transformers` versions save the tokenizer as
a single `tokenizer.json`; copy the two classic files from the base model if
they are missing.

### What kinds of fine-tune work without code changes

| fine-tuning method | works today | notes |
|---|---|---|
| full UNet fine-tune | yes | |
| full fine-tune including the text encoder | yes | |
| DreamBooth | yes | if saved as a full pipeline |
| LoRA, merged into the weights before export | yes | the bundled framework does this with `export.fuse_lora: true`; in your own code call `pipe.fuse_lora()` before `save_pretrained` |
| LoRA as a separate file, loaded at run time | no | needs code; see below |
| Textual inversion embeddings | no | needs code; see below |
| a replacement VAE, such as `stabilityai/sd-vae-ft-mse` | yes | copy its files into `vae/` |
| v-prediction models | no | needs a sampler change |
| SD 2.x or SDXL | no | different architecture |

### Architecture assumptions to keep in mind

The code discovers block counts and channel widths from the weight names, so
changing the depth or width of the UNet works. A few values are fixed:

| assumption | where it is set |
|---|---|
| 8 attention heads in every UNet transformer | `UNet(const Weights&, int heads = 8)` in [src/unet.h:40](../src/unet.h#L40) |
| 12 attention heads in CLIP | [src/clip.h:27](../src/clip.h#L27) |
| 32 GroupNorm groups | `GroupNorm` in [src/modules.h](../src/modules.h) |
| latent scale factor 0.18215 | `kScalingFactor` in [src/vae.h:36](../src/vae.h#L36) |
| epsilon (noise) prediction | `step()` in [src/scheduler.cpp:58](../src/scheduler.cpp#L58) and [src/scheduler.cpp:76](../src/scheduler.cpp#L76) |
| scaled-linear beta schedule, 0.00085 to 0.012 | `Scheduler::Scheduler` in [src/scheduler.cpp:14](../src/scheduler.cpp#L14) |
| final CLIP layer output, no "clip skip" | [src/clip.cpp:61](../src/clip.cpp#L61) |

If your training changes any of these, change the matching line.

## Making it faster: code map

Ordered by expected payoff on Apple Silicon.

### 1. Use Accelerate by default on macOS

- **Where:** [Makefile:19-23](../Makefile#L19-L23).
- **What:** turn `ACCELERATE=1` on automatically when `uname -s` is `Darwin`.
- **Gain:** measured at 1.7x per step on the M1 Max.
- **On Linux:** add a similar `BLAS=1` option that links OpenBLAS or MKL. The
  call site already exists: the `cblas_sgemm` branch in `gemm_nt` at
  [src/tensor.cpp:254](../src/tensor.cpp#L254) uses the standard CBLAS
  interface. Only the `#include` and the linker flags differ.

### 2. Fewer steps with a better sampler

- **Where:** add a class next to `EulerScheduler` in
  [src/scheduler.h:64](../src/scheduler.h#L64) and register it in
  `make_scheduler` at [src/scheduler.cpp:95](../src/scheduler.cpp#L95).
- **What:** DPM++ 2M with Karras sigmas. It reaches good quality in about 15
  steps, against 20 to 30 for Euler.
- **Gain:** step time is unchanged, but each image needs fewer steps. This
  also improves output, so it appears in both lists.

### 3. Parallelise and tile attention

- **Where:** `attention` at [src/tensor.cpp:465](../src/tensor.cpp#L465).
- **Now:** heads run one after another. Each head builds a full score matrix,
  64 MB at the 64x64 latent resolution, in the buffer at
  [line 473](../src/tensor.cpp#L473). The V transpose at
  [line 486](../src/tensor.cpp#L486) runs on one thread.
- **What to change:** run heads, or blocks of query rows, in parallel with
  one scratch buffer per thread. Better still, process keys in tiles with a
  running maximum and sum, as flash attention does, so the full score matrix
  never exists.
- **Gain:** most of the 13% waiting time and part of the 9% softmax time.

### 4. Faster `exp` and `erf`

- **Where:** `softmax_rows_` at [src/tensor.cpp:441](../src/tensor.cpp#L441),
  `geglu` at [src/tensor.cpp:494](../src/tensor.cpp#L494), and `silu_` at
  [src/tensor.cpp:519](../src/tensor.cpp#L519).
- **What:** replace `std::exp` and `std::erf` with vectorised polynomial
  approximations, or use the tanh form of GELU.
- **Gain:** up to about 5%. Check image quality afterwards, because GELU
  approximations change the output slightly.

### 5. A faster built-in GEMM, for non-Apple machines

- **Where:** the vector type at [src/tensor.cpp:184](../src/tensor.cpp#L184),
  the 4x4 micro-kernel at [src/tensor.cpp:209](../src/tensor.cpp#L209), and
  the tile sizes at [src/tensor.cpp:271](../src/tensor.cpp#L271).
- **What:**
  - On x86, widen the vector type to 8 floats (AVX2) or 16 floats (AVX-512).
  - Copy ("pack") tiles of A and B into contiguous buffers before the inner
    loop, so loads are sequential.
  - Use a larger register tile, such as 8x8 or 6x16, so each load feeds more
    multiply-adds.
  - Tune the tile sizes for the target cache.
- **Gain:** large on x86; modest on Apple Silicon, where Accelerate is
  faster anyway.

### 6. Skip im2col for 1x1 convolutions

- **Where:** `conv2d` at [src/tensor.cpp:330](../src/tensor.cpp#L330).
- **What:** a 1x1, stride-1 convolution is a plain matrix multiply, so the
  patch-copying step is wasted work. The input is laid out `[Cin, HW]`,
  though, and `gemm_nt` needs `[HW, Cin]`, so this needs either a transpose
  or a second GEMM variant that takes B untransposed.
- **Gain:** part of the 4% im2col time.

### 7. Fuse small operations

- **Where:** `ResnetBlock::forward` at
  [src/modules.cpp:33](../src/modules.cpp#L33) and
  `BasicTransformerBlock::forward` at
  [src/modules.cpp:80](../src/modules.cpp#L80).
- **What:** do GroupNorm and SiLU in one pass, and add the bias, the time
  embedding and the residual inside the GEMM output loop. Each fusion saves
  one full read and write of a large activation tensor.
- **Gain:** a few percent, mostly from lower memory traffic.

### 8. Cache the text keys and values

- **Where:** `CrossAttention::forward` at
  [src/modules.cpp:68](../src/modules.cpp#L68).
- **What:** the text context never changes between steps, but `to_k` and
  `to_v` are applied to it again in all 16 cross-attention layers at every
  step. Compute them once before the loop.
- **Gain:** small, because the context has only 77 tokens.

### 9. Keep weights in fp16

- **Where:** the conversion loop in `Weights::load_safetensors` at
  [src/safetensors.cpp:143](../src/safetensors.cpp#L143).
- **What:** store weights as 16-bit and convert inside the GEMM.
- **Gain:** memory drops by about 1.7 GB. Speed improves only on chips with
  native fp16 arithmetic, and only if the GEMM uses it.

### Settings that trade quality for speed, no code needed

| setting | effect |
|---|---|
| `--cfg 1` | skips the unconditional pass, halving UNet time, but weakens prompt following |
| fewer `--steps` | linear speed-up; DDIM is usable at 15 |
| smaller `--width` and `--height` | time falls faster than the pixel count; keep multiples of 64 |

## Making output better: code map

Ordered by expected improvement per unit of effort.

### 1. Better weights

- **Where:** no code; use `--model-dir`.
- **What:** your own fine-tune, a community SD 1.5 fine-tune, or an improved
  VAE such as `sd-vae-ft-mse`, which gives sharper faces and fine detail.
  Copy a replacement VAE's `diffusion_pytorch_model.safetensors` into the
  model's `vae/` folder.

### 2. Better samplers

- **Where:** [src/scheduler.h](../src/scheduler.h) and
  [src/scheduler.cpp](../src/scheduler.cpp).
- **What:**
  - Add DPM++ 2M Karras, as in the speed list above.
  - Add Karras sigma spacing as an option. It changes only the sigma values
    chosen in `EulerScheduler::set_timesteps`, at
    [src/scheduler.cpp:46](../src/scheduler.cpp#L46).
  - Add Euler Ancestral, which injects fresh noise at each step. It needs
    the random generator to be passed into `step()`.

### 3. Long prompts and prompt weighting

- **Where:** `CLIPTokenizer::encode` at
  [src/tokenizer.h:34](../src/tokenizer.h#L34), which truncates to 77 tokens,
  and the text-encoding line in `Pipeline::generate` at
  [src/pipeline.cpp:73](../src/pipeline.cpp#L73).
- **What:**
  - For long prompts, split the tokens into chunks of 75, encode each chunk
    with CLIP, and concatenate the results to `[1, 77*k, 768]`. The UNet's
    cross-attention accepts any context length already.
  - For weighting, parse `(word:1.2)` before tokenising, remember which
    tokens each weight applies to, and scale those rows of the CLIP output.

### 4. Clip skip

- **Where:** the layer loop in `CLIPTextEncoder::forward` at
  [src/clip.cpp:52](../src/clip.cpp#L52).
- **What:** many fine-tunes, especially anime-style models, were trained on
  CLIP's second-to-last layer. Add an option to stop one layer early and
  still apply `final_ln_`.

### 5. Guidance refinements

- **Where:** the guidance line at [src/pipeline.cpp:96](../src/pipeline.cpp#L96).
- **What:** guidance rescale, which reduces over-saturation at high `--cfg`,
  or a guidance scale that changes over the steps.

### 6. Exact tokenizer for non-Latin text

- **Where:** `is_letter` at [src/tokenizer.cpp:59](../src/tokenizer.cpp#L59)
  and `pretokenize` at [src/tokenizer.cpp:113](../src/tokenizer.cpp#L113).
- **What:** use real Unicode letter and number tables, and apply Unicode
  lower-casing and the text clean-up CLIP performs.

### 7. New features that change what you can do

| feature | where to start |
|---|---|
| image-to-image | add the VAE encoder to [src/vae.cpp](../src/vae.cpp), built from the same `ResnetBlock` and `VAEAttention` modules; then start the loop in `main` at a middle timestep |
| run-time LoRA | after `load_safetensors` in `main`, read the LoRA file and add `scale * up * down` to each matching weight in `Weights::tensors` |
| textual inversion | add the new embedding rows to the token table used in `CLIPTextEncoder::forward`, and the new token to the tokenizer vocabulary |
| v-prediction models | add a prediction-type option to the samplers' `step()` and convert v to eps there |

### 8. Verify against PyTorch

The output has never been compared value by value with the reference
implementation. To do it once, dump the CLIP output, one UNet output and the
VAE output from both diffusers and this code for the same input, and compare.
That catches small errors that still produce plausible-looking images. Do it
again after any kernel change, because approximations in items 4 and 6 of the
speed list change the numbers slightly.

## Recommended order of work

1. Turn on Accelerate by default on macOS. Small change, measured 1.7x.
2. Add DPM++ 2M Karras. Fewer steps and better images.
3. Add the PyTorch comparison, so later changes can be checked numerically.
4. Parallelise and tile attention.
5. Add long prompts, prompt weighting and clip skip, which matter most when
   using fine-tuned models.
6. On Linux or x86, add OpenBLAS or MKL support and widen the built-in GEMM.
