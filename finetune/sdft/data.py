"""Image + caption dataset."""

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DataConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(root: str) -> List[str]:
    files = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                files.append(os.path.join(dirpath, n))
    return files


def read_captions(root: str, files: List[str], default: str) -> Dict[str, str]:
    captions = {}
    meta_path = os.path.join(root, "metadata.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                captions[os.path.join(root, row["file_name"])] = str(row.get("text", row.get("caption", "")))
    for path in files:
        if path in captions:
            continue
        txt = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                captions[path] = f.read().strip()
        else:
            captions[path] = default
    return captions


def load_image(path: str, resolution: int, center_crop: bool, random_flip: bool, rng: random.Random) -> torch.Tensor:
    """Returns a float tensor [3, res, res] in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = resolution / min(w, h)
    nw, nh = max(resolution, round(w * scale)), max(resolution, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    if center_crop:
        left, top = (nw - resolution) // 2, (nh - resolution) // 2
    else:
        left, top = rng.randint(0, nw - resolution), rng.randint(0, nh - resolution)
    img = img.crop((left, top, left + resolution, top + resolution))
    if random_flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ImageCaptionDataset(Dataset):
    def __init__(self, cfg: DataConfig, tokenizer, seed: int = 0, max_tokens: int = 512):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        if not os.path.isdir(cfg.path):
            raise FileNotFoundError(f"data.path '{cfg.path}' is not a directory")
        self.files = find_images(cfg.path)
        if not self.files:
            raise FileNotFoundError(f"no images found under '{cfg.path}'")
        self.captions = read_captions(cfg.path, self.files, cfg.default_caption)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files) * max(1, self.cfg.repeats)

    def tokenize(self, caption: str) -> torch.Tensor:
        # CLIP: fixed 77 ids. Qwen (Z-Image): chat-template text, padded to
        # max_tokens with the pad id; the trainer drops the padding again.
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self, "chat_template", True) and \
                "Qwen" in type(self.tokenizer).__name__:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": caption}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=True)
            return self.tokenizer(text, padding="max_length", max_length=self.max_tokens, truncation=True,
                                  return_tensors="pt").input_ids[0]
        return self.tokenizer(caption, padding="max_length", max_length=self.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt").input_ids[0]

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx % len(self.files)]
        pixels = load_image(path, self.cfg.resolution, self.cfg.center_crop, self.cfg.random_flip, self.rng)
        caption = self.captions[path]
        if self.cfg.caption_dropout > 0 and self.rng.random() < self.cfg.caption_dropout:
            caption = ""
        return {"pixel_values": pixels, "input_ids": self.tokenize(caption), "caption": caption, "path": path}


def collate(batch: List[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "path": [b["path"] for b in batch],
    }
