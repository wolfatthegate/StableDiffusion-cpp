#!/usr/bin/env python3
"""Dumps reference tensors from the PyTorch Z-Image models so that the C++
port can be checked numerically. Each dump is <prefix>.bin (raw float32) plus
<prefix>.json (shape and inputs).

    dump_reference.py text        --prompt "a photo of a cat"      # penultimate Qwen3 hidden state
    dump_reference.py vae         --size 256                       # decode of a fixed random latent
    dump_reference.py transformer --size 256                       # one DiT forward on fixed inputs
    dump_reference.py generate    --prompt "..." --size 1024       # a full image + timing

All run in float32 on the CPU for exactness unless --device is given.
"""

import argparse
import json
import os
import time

import numpy as np
import torch


def save(prefix, tensor, **meta):
    arr = tensor.detach().float().cpu().contiguous().numpy()
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    arr.astype(np.float32).tofile(prefix + ".bin")
    meta = {"shape": list(arr.shape), **meta}
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f)
    print(f"wrote {prefix}.bin {list(arr.shape)}")


def cmd_text(a):
    from transformers import AutoTokenizer, Qwen3Model
    tok = AutoTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    enc = Qwen3Model.from_pretrained(a.model, subfolder="text_encoder", dtype=torch.float32).to(a.device).eval()
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok(text, max_length=512, truncation=True, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = enc(input_ids=ids.input_ids.to(a.device), attention_mask=ids.attention_mask.to(a.device), output_hidden_states=True)
    hs = out.hidden_states
    print(f"{len(hs)} hidden states, {ids.input_ids.shape[1]} tokens, {time.time() - t0:.1f}s")
    # hidden_states[-2]: output of decoder layer n_layers-2, i.e. n_layers-1 layers run, no final norm.
    save(a.out, hs[-2][0], prompt=a.prompt, ids=ids.input_ids[0].tolist(), layer=len(hs) - 2)


def cmd_vae(a):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32).to(a.device).eval()
    g = torch.Generator().manual_seed(a.seed)
    z = torch.randn(1, vae.config.latent_channels, a.size // 8, a.size // 8, generator=g)
    with torch.no_grad():
        img = vae.decode(z.to(a.device) / vae.config.scaling_factor + vae.config.shift_factor).sample
    save(a.out + "_in", z, seed=a.seed, scaling_factor=vae.config.scaling_factor, shift_factor=vae.config.shift_factor)
    save(a.out, img[0])


def cmd_transformer(a):
    from diffusers import ZImageTransformer2DModel
    tr = ZImageTransformer2DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=torch.float32).to(a.device).eval()
    if a.round_bf16:  # mimic the C++ numerics: bf16-stored weights, float32 arithmetic
        with torch.no_grad():
            for p in tr.parameters():
                p.data = p.data.to(torch.bfloat16).to(torch.float32)
    g = torch.Generator().manual_seed(a.seed)
    C = tr.config.in_channels
    x = torch.randn(1, C, a.size // 8, a.size // 8, generator=g)
    cap = torch.randn(a.text_tokens, tr.config.cap_feat_dim, generator=g)
    t = torch.tensor([a.t])
    # Trace of intermediate outputs, for locating divergences in the C++ port.
    traces = {}
    def hook(name):
        def f(mod, inp, out):
            traces[name] = out[0] if isinstance(out, (tuple, list)) else out
        return f
    tr.t_embedder.register_forward_hook(hook("adaln"))
    tr.all_x_embedder["2-1"].register_forward_hook(hook("x_embed"))
    tr.noise_refiner[0].register_forward_hook(hook("noise_refiner0"))
    tr.noise_refiner[-1].register_forward_hook(hook("noise_refined"))
    tr.cap_embedder.register_forward_hook(hook("cap_embed"))
    tr.context_refiner[-1].register_forward_hook(hook("cap_refined"))
    tr.layers[0].register_forward_hook(hook("layer0"))
    tr.layers[-1].register_forward_hook(hook("layers_out"))
    tr.all_final_layer["2-1"].register_forward_hook(hook("final"))
    t0 = time.time()
    with torch.no_grad():
        out = tr(x=[x[0].unsqueeze(1).to(a.device)], t=t.to(a.device), cap_feats=[cap.to(a.device)], return_dict=False)[0]
    for name, v in traces.items():
        save(a.out + "_trace_" + name, v.reshape(-1, v.shape[-1]))
    print(f"transformer forward: {time.time() - t0:.1f}s")
    save(a.out + "_x", x[0], seed=a.seed, t=a.t)
    save(a.out + "_cap", cap)
    save(a.out, out[0].squeeze(1) if out[0].dim() == 4 else out[0], t=a.t)


def cmd_generate(a):
    from diffusers import ZImagePipeline
    dtype = torch.bfloat16 if a.device != "cpu" else torch.float32
    pipe = ZImagePipeline.from_pretrained(a.model, torch_dtype=dtype).to(a.device)
    pipe.set_progress_bar_config(disable=False)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    img = pipe(prompt=a.prompt, height=a.size, width=a.size, num_inference_steps=a.steps, guidance_scale=0.0, generator=g).images[0]
    print(f"generated {a.size}x{a.size} in {time.time() - t0:.1f}s on {a.device}")
    img.save(a.out + ".png")
    print(f"wrote {a.out}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["text", "vae", "transformer", "generate"])
    ap.add_argument("--model", default="models/z-image-turbo")
    ap.add_argument("--prompt", default="a photo of a cat")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--text-tokens", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--round-bf16", action="store_true", help="round weights to bf16 before the forward (transformer)")
    a = ap.parse_args()
    if a.out is None:
        a.out = f"tests/fixtures/ref_{a.command}"
    torch.manual_seed(a.seed)
    {"text": cmd_text, "vae": cmd_vae, "transformer": cmd_transformer, "generate": cmd_generate}[a.command](a)


if __name__ == "__main__":
    main()
