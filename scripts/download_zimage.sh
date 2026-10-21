#!/bin/sh
# Downloads Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo, Apache 2.0) into a
# diffusers-layout directory: Qwen3-4B text encoder (bf16, 8 GB), the 6B
# transformer (fp32 as published, 24.6 GB), the FLUX VAE and configs.
#   usage: scripts/download_zimage.sh [target-dir]   (default: models/z-image-turbo)
# Downloads resume (curl -C -) if interrupted. Set HF_TOKEN if required.
set -eu
DIR="${1:-models/z-image-turbo}"
REPO="${ZIMAGE_REPO:-Tongyi-MAI/Z-Image-Turbo}"
BASE="https://huggingface.co/$REPO/resolve/main"
AUTH=""
if [ -n "${HF_TOKEN:-}" ]; then AUTH="-H Authorization: Bearer $HF_TOKEN"; fi

fetch() {
    mkdir -p "$DIR/$(dirname "$1")"
    echo "-> $1"
    # shellcheck disable=SC2086
    curl -L --fail --retry 3 -C - $AUTH -o "$DIR/$1" "$BASE/$1"
}

fetch model_index.json
fetch scheduler/scheduler_config.json
fetch vae/config.json
fetch vae/diffusion_pytorch_model.safetensors
fetch tokenizer/tokenizer_config.json
fetch tokenizer/tokenizer.json
fetch tokenizer/vocab.json
fetch tokenizer/merges.txt
fetch text_encoder/config.json
fetch text_encoder/generation_config.json
fetch text_encoder/model.safetensors.index.json
fetch text_encoder/model-00001-of-00003.safetensors
fetch text_encoder/model-00002-of-00003.safetensors
fetch text_encoder/model-00003-of-00003.safetensors
fetch transformer/config.json
fetch transformer/diffusion_pytorch_model.safetensors.index.json
fetch transformer/diffusion_pytorch_model-00001-of-00003.safetensors
fetch transformer/diffusion_pytorch_model-00002-of-00003.safetensors
fetch transformer/diffusion_pytorch_model-00003-of-00003.safetensors
echo "done: $DIR"
