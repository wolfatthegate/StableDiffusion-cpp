# StableDiffusion-cpp

Stable Diffusion 1.x text-to-image inference written from scratch in C++17.

There are no third-party dependencies. PyTorch, BLAS, ONNX, image libraries and
JSON libraries are all absent. Every piece of the pipeline is implemented in
`src/` in about 2,500 lines:

- the numeric kernels: GEMM, convolution, attention, normalisation
- the safetensors weight loader and a JSON parser
- the CLIP byte-level BPE tokenizer
- the CLIP text encoder, the UNet and the VAE decoder
- the Euler and DDIM samplers
- a PNG writer

It runs on the CPU in float32 and reads the standard Hugging Face diffusers
weight layout, so any SD 1.x checkpoint in that format works.

| `--prompt "a photo of an astronaut riding a horse on mars, highly detailed"` <br> `--negative "blurry, low quality, deformed" --steps 20 --seed 42` | `--prompt "a watercolor painting of a lighthouse at sunset"` <br> `--sampler ddim --steps 15 --seed 7` |
|---|---|
| ![astronaut](examples/astronaut.png) | ![lighthouse](examples/lighthouse.png) |

## Contents

1. [Quick start](#quick-start)
2. [Building](#building)
3. [Model weights](#model-weights)
4. [Command-line reference](#command-line-reference)
5. [How Stable Diffusion works](#how-stable-diffusion-works)
6. [Code tour](#code-tour)
7. [Component reference](#component-reference)
8. [Design decisions](#design-decisions)
9. [Performance](#performance)
10. [Testing](#testing)
11. [Extending the code](#extending-the-code)
12. [Limitations](#limitations)
13. [Troubleshooting](#troubleshooting)

## Quick start

```sh
make -j                          # build ./sd
scripts/download_sd15.sh         # fetch SD 1.5 weights (~2.1 GB) into models/sd15
./sd --prompt "a photo of a cat" --output cat.png
```

A 512x512 image with the default 20 steps takes about 95 seconds on a 10-core
Apple Silicon machine.

## Building

Requirements:

- A C++17 compiler (Clang or GCC). Apple Clang on macOS works out of the box.
- `make`.
- A POSIX system, because the weight loader uses `mmap`. macOS and Linux are
  supported. Windows would need a replacement for the memory map in
  `src/safetensors.cpp`.

Targets:

```sh
make                 # ./sd, the command-line tool
make test            # build and run ./test_ops (kernel tests, no weights needed)
make test_tokenizer  # build ./test_tokenizer (needs the tokenizer files)
make clean           # remove build/ and all binaries
```

Build options:

| option | effect |
|---|---|
| `make ACCELERATE=1` | macOS only. Routes `gemm_nt` to Apple's Accelerate BLAS instead of the built-in kernel. Nothing else changes. |
| `CXX=g++ make` | use a different compiler |
| `CXXFLAGS="..." make` | replace the default `-std=c++17 -O3 -Wall -Wextra -pthread` |

On x86-64 the Makefile adds `-march=native` so the vector code uses AVX and
FMA. On arm64, NEON is used automatically.

## Model weights

`scripts/download_sd15.sh [dir]` downloads Stable Diffusion 1.5 from the
`stable-diffusion-v1-5/stable-diffusion-v1-5` repository on Hugging Face into
`models/sd15` by default. It fetches the fp16 safetensors files, which are
half the size of the fp32 ones, and converts nothing on disk. Downloads resume
if interrupted. Set `HF_TOKEN` if the repository ever requires a login.

The model directory must look like this:

```
models/sd15/
  tokenizer/vocab.json                  CLIP vocabulary, 49408 entries
  tokenizer/merges.txt                  BPE merge list
  text_encoder/model.safetensors        CLIP ViT-L/14 text model, 123M params
  unet/diffusion_pytorch_model.safetensors   UNet, 860M params
  vae/diffusion_pytorch_model.safetensors    autoencoder, 84M params
```

Each `.safetensors` file may instead be named `*.fp16.safetensors`. The
full-precision name is tried first. F32, F16 and BF16 tensors are accepted
and converted to float32 at load time.

Other SD 1.x checkpoints, such as fine-tunes and DreamBooth models, work as
long as they are in diffusers format. A single-file `.ckpt` or `.safetensors`
checkpoint in the original CompVis layout must first be converted with the
diffusers conversion script, because the tensor names differ.

## Command-line reference

```
./sd --prompt TEXT [options]
```

| flag | short | default | meaning |
|---|---|---|---|
| `--prompt` | `-p` | required | what to draw |
| `--negative` | `-n` | empty | what to steer away from; also `--negative-prompt` |
| `--model-dir` | `-m` | `models/sd15` | diffusers-layout model directory |
| `--steps` | | 20 | number of sampler steps |
| `--cfg` | | 7.5 | classifier-free guidance scale; `1` disables guidance and halves the work |
| `--width` | `-W` | 512 | image width; multiple of 8, at least 64 |
| `--height` | `-H` | 512 | image height; multiple of 8, at least 64 |
| `--seed` | `-s` | 42 | seed for the initial noise |
| `--sampler` | | `euler` | `euler` or `ddim` |
| `--threads` | `-t` | all cores | worker threads |
| `--output` | `-o` | `output.png` | output PNG path |
| `--help` | `-h` | | print usage |

Progress goes to stderr:

```
threads: 10
text encoder: 12 layers, 0.2s
unet: 859.52M params loaded in 0.4s
step  1/20  t= 999.00  4.4s
...
vae decode + save: 7.2s -> output.png
```

Practical notes:

- SD 1.x was trained at 512x512. Other sizes work, but sizes far from 512
  tend to produce duplicated subjects or poor composition.
- A width or height that is a multiple of 8 but not of 64 can fail with a
  channel-concatenation error, because the UNet halves the latent three times
  and the skip connections must line up. Use multiples of 64.
- Prompts longer than 75 tokens are truncated.
- 20 to 30 Euler steps, or 15 to 50 DDIM steps, give good results. Guidance
  between 5 and 9 is typical.

## How Stable Diffusion works

Stable Diffusion is a *latent* diffusion model. Rather than denoising pixels,
it denoises a small 4-channel latent image 8 times smaller than the output,
and a separate decoder turns the final latent into pixels.

```
prompt ──> CLIPTokenizer ──> CLIPTextEncoder ──┐ context [2, 77, 768]
                                               v
noise ──> for each step:  x_in = scale(x) ──> UNet(x_in, t, context) ──> eps ──> sampler step ──> x
                                               ^
                                               └─ guidance: eps = eps_u + cfg * (eps_c - eps_u)
x / 0.18215 ──> VAEDecoder ──> RGB in [-1, 1] ──> PNG
```

1. **Text encoding.** The prompt is split into at most 77 CLIP tokens
   (start token, up to 75 text tokens, end token, then end-token padding).
   The CLIP text transformer turns them into 77 vectors of 768 features. The
   negative prompt, or an empty string, is encoded the same way.
2. **Initial noise.** A latent of shape `[1, 4, H/8, W/8]` is filled with
   Gaussian noise, scaled by the sampler's initial sigma.
3. **Denoising loop.** At each step the UNet predicts the noise in the
   current latent. It runs once with the negative-prompt embedding and once
   with the prompt embedding, as a single batch of two. Classifier-free
   guidance extrapolates from the unconditional prediction towards the
   conditional one by the factor `cfg`. The sampler then uses this noise
   estimate to move the latent to the next, less noisy timestep.
4. **Decoding.** The final latent is divided by the training scale factor
   0.18215 and passed through the VAE decoder, which upsamples it 8x to RGB.

The diffusion process itself is defined in `src/scheduler.h`. Training used
1000 noise levels with a "scaled linear" schedule. A noisy latent at level
`t` is `x_t = sqrt(a_t) * x_0 + sqrt(1 - a_t) * eps`, where `a_t` is the
cumulative product of `1 - beta`. The samplers visit a short subsequence of
these levels.

## Code tour

```
src/
  tensor.h / .cpp        Tensor type, thread pool, GEMM, and every operator
  json.h / .cpp          minimal JSON parser
  safetensors.h / .cpp   weight file loader with F16/BF16 conversion
  tokenizer.h / .cpp     CLIP byte-level BPE tokenizer
  modules.h / .cpp       reusable layers bound to weights by name
  clip.h / .cpp          CLIP text encoder
  unet.h / .cpp          UNet2DConditionModel
  vae.h / .cpp           AutoencoderKL decoder
  scheduler.h / .cpp     beta schedule, Euler and DDIM samplers
  image.h / .cpp         PNG writer
  main.cpp               CLI and the generation pipeline
tests/
  test_ops.cpp           kernel tests against naive references, plus a GEMM benchmark
  test_tokenizer.cpp     tokenizer tests against known token ids
scripts/
  download_sd15.sh       weight downloader
examples/                sample outputs shown above
```

Dependencies flow strictly upwards:

```
main.cpp
  ├── clip, unet, vae ──> modules ──> safetensors ──> json
  │                          └──────> tensor
  ├── tokenizer ──> json
  ├── scheduler ──> tensor
  └── image
```

Suggested reading order for a review:

1. `src/tensor.h`, for the data model and operator contracts.
2. `src/main.cpp`, for the end-to-end pipeline.
3. `src/modules.h` and `src/modules.cpp`, for the layers.
4. `src/unet.h`, `src/clip.h` and `src/vae.h`, which describe each model's
   layout in their header comments.
5. `src/scheduler.h`, for the sampling maths.
6. `src/tensor.cpp`, for how the kernels are implemented.

Every header documents its public API. Every source file starts with a
summary and comments its non-obvious steps.

## Component reference

### Tensor and operators (`src/tensor.h`)

`Tensor` is a shape plus a contiguous row-major `std::vector<float>`. There are
no strides, views, dtypes or devices. Operators return new tensors, except
those whose names end in `_`, which work in place.

Two activation layouts are used:

| layout | shape | used by |
|---|---|---|
| NCHW | `[batch, channels, height, width]` | convolution, group norm, upsampling, skip concatenation |
| NLC | `[batch, tokens, features]` | linear layers, layer norm, attention |

`nchw_to_nlc` and `nlc_to_nchw` convert between them inside each
transformer block.

| operator | semantics |
|---|---|
| `gemm_nt(M, N, K, A, lda, B, ldb, C, ldc, bias, accumulate)` | `C = A * B^T (+ bias)`; the core of everything below |
| `linear(x, W, b)` | `nn.Linear`: `x[..., in] -> [..., out]` |
| `conv2d(x, W, b, stride, pad)` | `nn.Conv2d` via im2col + GEMM |
| `group_norm(x, groups, gamma, beta, eps)` | `nn.GroupNorm` |
| `layer_norm(x, gamma, beta, eps)` | `nn.LayerNorm` over the last dimension |
| `attention(q, k, v, heads, causal)` | multi-head `softmax(Q K^T / sqrt(d)) V` |
| `geglu(x)` | `a * gelu(g)` where `[a, g]` split the last dimension |
| `silu_`, `gelu_`, `quick_gelu_` | in-place activations |
| `upsample_nearest2x`, `concat_channels`, `concat_batch`, `slice_batch`, `repeat_batch` | shape helpers |

Threading uses one global pool. `parallel_for(n, fn)` hands out item indices
dynamically from an atomic counter and the calling thread joins in.
`parallel_chunks(n, min_chunk, fn)` hands out contiguous ranges for cheap
elementwise loops. Neither is re-entrant, so operators parallelise at one
level only.

### GEMM kernel (`src/tensor.cpp`)

All of the arithmetic intensity in Stable Diffusion is in matrix multiplies:
linear layers, convolutions after im2col, and attention. One kernel handles
all of them, and it always takes the second operand transposed. That is the
natural layout here, because PyTorch stores linear weights as `[out, in]`,
im2col rows are `[pixels, K]`, and attention computes `Q K^T`.

The kernel splits C into 64x128 tiles, one task per tile, so threads never
share output. Inside a tile it walks K in slices of 128 for cache reuse. The
micro-kernel computes a 4x4 block of C with 16 four-lane vector accumulators,
written with Clang/GCC vector extensions so that the same source becomes
NEON or SSE/AVX. It reaches about 550 to 600 GFLOP/s on a 10-core M-series
CPU. There is no operand packing, which keeps the code short at some cost in
peak speed.

### Convolution

`conv2d` uses im2col. Each output pixel's input patch is copied into one row of
a scratch matrix, ordered `(input channel, ky, kx)`. That ordering matches
PyTorch's `[Cout, Cin, kh, kw]` weight layout, so the weight is used directly
as a `[Cout, K]` matrix. A single GEMM then writes the NCHW output. Pixels
are processed in chunks so the scratch matrix stays at or below 64 MB; a
full-image im2col in the VAE's last stage would need more than 2 GB.

### Attention

`attention` loops over heads. For each head it computes the full score matrix
`Q_h K_h^T`, applies a numerically stable softmax in place, transposes `V_h`,
and multiplies. The heads are strided slices of the projected tensors, so no
reshaping copies are needed. At the UNet's highest resolution the score
matrix is 4096 x 4096 floats, or 64 MB. There is no flash-attention style
tiling.

### Weight loading (`src/safetensors.h`)

A `.safetensors` file is an 8-byte header length, a JSON header that
describes each tensor's dtype, shape and byte range, and then raw data.
`Weights::load_safetensors` memory-maps the file, parses the header,
allocates every tensor, and converts them to float32 in parallel. Half
precision conversion is bit-exact, including subnormals, infinities and NaN.
Tensors of other dtypes are skipped with a message. The only such tensor in
SD 1.5 is CLIP's integer `position_ids` buffer, which is not needed.

`Weights` maps names to tensors. Model classes keep raw pointers into it, so
it must outlive them. `main.cpp` loads each model's weights in its own scope
and frees them before loading the next model.

### Tokenizer (`src/tokenizer.h`)

`CLIPTokenizer` reproduces the tokenizer of `openai/clip-vit-large-patch14`:

1. Lower-case the text.
2. Split it the way CLIP's regex does, into special tokens, contractions such
   as `'s`, runs of letters, single digits, and runs of other symbols.
   Whitespace is dropped.
3. Map every byte of a piece to a printable Unicode stand-in using GPT-2's
   `bytes_to_unicode` table, and append `</w>` to the last symbol.
4. Merge adjacent symbol pairs greedily, lowest `merges.txt` rank first.
5. Look up each final symbol in `vocab.json`.

`encode()` wraps the ids as `[49406] ids [49407] [49407]...` to exactly 77
tokens. SD 1.x pads with the end token rather than a separate pad token.

### Modules (`src/modules.h`)

Each module mirrors a diffusers class and binds weights by name prefix. For
example, `Linear(w, "mid_block.attentions.0.proj_in")` reads
`mid_block.attentions.0.proj_in.weight` and the optional `.bias`. Modules hold
pointers into the weights, own no memory, and have `const` forward passes.

| module | diffusers class | computation |
|---|---|---|
| `ResnetBlock` | `ResnetBlock2D` | `conv2(silu(norm2(conv1(silu(norm1(x))) + temb))) + shortcut(x)` |
| `CrossAttention` | `Attention` | `to_out(attention(to_q(x), to_k(ctx), to_v(ctx)))` |
| `FeedForward` | `FeedForward` (GEGLU) | `out(geglu(proj(x)))`, widening 4x inside |
| `BasicTransformerBlock` | `BasicTransformerBlock` | self-attention, cross-attention, feed-forward, each pre-norm with a residual |
| `Transformer2D` | `Transformer2DModel` | group norm, `proj_in`, blocks, `proj_out`, plus residual |
| `VAEAttention` | VAE mid-block `Attention` | single-head self-attention over pixels |

### CLIP text encoder (`src/clip.h`)

A 12-layer, 768-wide, 12-head pre-norm transformer with causal self-attention
and quick-GELU MLPs. It takes the 77 token ids and returns the final hidden
state `[1, 77, 768]` after the last layer norm. SD 1.x conditions on all 77
positions, not on the pooled output.

### UNet (`src/unet.h`)

The UNet predicts noise from a latent, a timestep and the text context.
For a 512x512 image, with a 64x64 latent:

| stage | contents | channels | resolution |
|---|---|---|---|
| `conv_in` | 3x3 conv | 4 -> 320 | 64 |
| down 0 | 2 x (resnet + transformer), downsample | 320 | 64 -> 32 |
| down 1 | 2 x (resnet + transformer), downsample | 640 | 32 -> 16 |
| down 2 | 2 x (resnet + transformer), downsample | 1280 | 16 -> 8 |
| down 3 | 2 x resnet | 1280 | 8 |
| mid | resnet, transformer, resnet | 1280 | 8 |
| up 0 | 3 x resnet, upsample | 1280 | 8 -> 16 |
| up 1 | 3 x (resnet + transformer), upsample | 1280 | 16 -> 32 |
| up 2 | 3 x (resnet + transformer), upsample | 640 | 32 -> 64 |
| up 3 | 3 x (resnet + transformer) | 320 | 64 |
| out | group norm, SiLU, 3x3 conv | 320 -> 4 | 64 |

The timestep is encoded as 320 sinusoidal features (cosines then sines) and
passed through a two-layer MLP to 1280 features. Each resnet adds a
projection of this embedding to every pixel. Down-path outputs are pushed
onto a stack of 12 skip tensors, and each up-path resnet pops one and
concatenates it to its input along the channel axis.

The constructor discovers block counts, attention placement and channel
widths from the weight names and shapes. Nothing is hard-coded except the
head count of 8 and the GroupNorm settings.

### VAE decoder (`src/vae.h`)

Only the decoder is implemented, because text-to-image never encodes an
image. It applies `post_quant_conv` and `conv_in` to reach 512 channels, a
mid block with single-head self-attention, then four up blocks of three
resnets each, the first three of which upsample 2x. A final group norm, SiLU
and conv produce RGB in roughly `[-1, 1]`. `main.cpp` maps this to
`[0, 255]` with clamping.

### Samplers (`src/scheduler.h`)

Both samplers share the training schedule: 1000 betas spaced linearly in
square-root space between 0.00085 and 0.012, and their cumulative product
`alpha_bar`.

**Euler** (`--sampler euler`) matches diffusers' `EulerDiscreteScheduler`
with linspace timesteps. It works in terms of
`sigma = sqrt((1 - alpha_bar) / alpha_bar)` and the unscaled latent
`x = x_0 + sigma * eps`. One step is `x += eps * (sigma_next - sigma)`.
Before each UNet call, the latent is divided by `sqrt(sigma^2 + 1)` to
recover the input distribution the UNet was trained on. The initial noise is
scaled by the largest sigma, about 14.6.

**DDIM** (`--sampler ddim`) matches diffusers' `DDIMScheduler` with eta 0,
leading spacing and `steps_offset = 1`. It predicts the clean latent from the
noise estimate and re-noises it to the next level deterministically:

```
x0     = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
x_prev = sqrt(a_prev) * x0 + sqrt(1 - a_prev) * eps
```

### PNG writer (`src/image.h`)

`write_png` writes an 8-bit RGB PNG with three chunks: IHDR, IDAT and IEND.
The pixel data uses deflate "stored" blocks, which are uncompressed, so no
zlib is needed. A 512x512 image is about 790 KB.

## Design decisions

- **Readability over peak speed.** Every operator is a plain loop or a call
  to the one GEMM. There is no graph compiler, operator fusion, or
  hand-written assembly.
- **float32 everywhere.** Weights are stored as fp16 on disk but expanded to
  float32 in memory. This doubles memory use to about 3.4 GB for the UNet
  but keeps the arithmetic simple and accurate.
- **Structure from weights.** Models discover their layout from tensor names
  instead of a config file, so there is no `config.json` parser and
  fine-tunes load unchanged.
- **One model in memory at a time.** The text encoder, UNet and VAE are
  loaded and freed in sequence, which keeps peak memory near the UNet's size.
- **Batched guidance.** The conditional and unconditional UNet passes run as
  one batch of two, so weights are read once per step.

## Performance

Measured on a 10-core Apple Silicon Mac with the built-in GEMM:

| stage | time |
|---|---|
| load and run the text encoder | 0.2 s |
| load the UNet weights, after the first run warms the page cache | 0.4 s |
| one UNet step with guidance at 512x512 | 4.3 s |
| VAE decode and PNG write at 512x512 | 7.2 s |
| full 20-step image | about 94 s |

Peak memory footprint at 512x512 is about 4.1 GB, and the maximum resident
set size reported by the OS is about 5.4 GB because the memory-mapped weight
file is counted too.

Ways to make it faster, roughly in order of payoff:

- Build with `make ACCELERATE=1` on macOS to use Apple's tuned BLAS. This
  has not been benchmarked here.
- Use `--cfg 1` to skip the unconditional pass, at a cost in prompt adherence.
- Use fewer steps; DDIM is usable at 15.
- Generate at a smaller size such as 384x384.

## Testing

```sh
make test                 # kernel tests
make test_tokenizer && ./test_tokenizer [model-dir]
```

`test_ops` compares `gemm_nt`, `conv2d`, `attention`, `group_norm` and
`layer_norm` against naive double-precision references on random inputs. It
uses odd sizes and padded leading dimensions to exercise the edge-tile and
stride paths. It then benchmarks one GEMM shaped like a UNet convolution. It
needs no weights and exits non-zero on any failure.

`test_tokenizer` checks case folding, contractions, digits, punctuation runs,
empty input and the 77-token padding against ids from the real vocabulary.

End-to-end correctness was checked by generating the example images above.
The pipeline has not been compared tensor by tensor against PyTorch.

## Extending the code

- **New operator.** Declare it in `src/tensor.h` with a comment describing
  shapes, implement it in `src/tensor.cpp` using `parallel_for` or
  `parallel_chunks`, and add a reference check to `tests/test_ops.cpp`.
- **New sampler.** Subclass `Scheduler` in `src/scheduler.h`, implement
  `set_timesteps` and `step`, and override `init_noise_sigma` and
  `scale_model_input` if the sampler works in sigma space. Register it in
  `make_scheduler`.
- **Image-to-image.** Implement the VAE encoder in `src/vae.cpp` with the
  same modules as the decoder, encode the input image, noise it to an
  intermediate timestep, and start the sampling loop from there.
- **SD 2.x.** Change the UNet head count to use a fixed head dimension of 64,
  swap the text encoder for OpenCLIP ViT-H with penultimate-layer output, and
  add v-prediction to the samplers.
- **Faster kernels.** Everything heavy goes through `gemm_nt`, so packing,
  larger register tiles or a BLAS backend there speed up the whole model.

## Limitations

- CPU and float32 only; there is no GPU backend and no fp16 or quantised
  compute.
- The pre-tokeniser treats every non-ASCII code point as a letter, and
  lower-casing only affects ASCII. This differs from CLIP's Unicode-aware
  regex for non-Latin text and punctuation.
- Prompt weighting syntax such as `(word:1.2)` is not parsed; it is tokenised
  as plain text.
- SD 2.x and SDXL are not supported.
- Random numbers come from `std::mt19937_64`, so a seed does not reproduce
  the image a PyTorch pipeline makes with the same seed.
- Only text-to-image is implemented. There is no image-to-image, inpainting,
  LoRA or ControlNet support.

## Troubleshooting

| message | cause and fix |
|---|---|
| `cannot find models/sd15/unet/diffusion_pytorch_model[.fp16].safetensors` | Weights are missing or `--model-dir` is wrong. Run `scripts/download_sd15.sh`. |
| `missing weight: <name>` | The checkpoint is not in diffusers SD 1.x layout. Convert it with the diffusers conversion script. |
| `concat_channels: [...] vs [...]` | Width or height is not a multiple of 64. |
| `cannot open .../vocab.json` | The tokenizer files are missing from the model directory. |
| Very slow generation | Check that `threads:` on the first line matches your core count, and that no other heavy process is running. |
