#!/bin/sh
# Downloads the Stable Diffusion 1.5 weights (fp16 safetensors, ~2.1 GB) and the
# CLIP tokenizer files into a diffusers-style directory.
#   usage: scripts/download_sd15.sh [target-dir]   (default: models/sd15)
# Set HF_TOKEN if the repository ever requires authentication.
set -eu
DIR="${1:-models/sd15}"
BASE="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main"
AUTH=""
if [ -n "${HF_TOKEN:-}" ]; then AUTH="-H Authorization: Bearer $HF_TOKEN"; fi

fetch() {  # fetch <relative path>
    mkdir -p "$DIR/$(dirname "$1")"
    echo "-> $1"
    # shellcheck disable=SC2086
    curl -L --fail --retry 3 -C - $AUTH -o "$DIR/$1" "$BASE/$1"
}

fetch tokenizer/vocab.json
fetch tokenizer/merges.txt
fetch text_encoder/model.fp16.safetensors
fetch vae/diffusion_pytorch_model.fp16.safetensors
fetch unet/diffusion_pytorch_model.fp16.safetensors
echo "done: $DIR"
