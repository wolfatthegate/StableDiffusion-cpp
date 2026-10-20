#!/usr/bin/env python3
"""Generates a small synthetic dataset of coloured shapes with captions, for
smoke-testing the training pipeline without real images.

    python finetune/scripts/make_sample_dataset.py data/sample-shapes --count 16 --size 256
"""

import argparse
import os
import random

from PIL import Image, ImageDraw

COLORS = {"red": (220, 50, 50), "green": (50, 180, 80), "blue": (50, 90, 220), "yellow": (240, 200, 40),
          "purple": (150, 60, 200), "orange": (240, 140, 30)}
BACKGROUNDS = {"white": (245, 245, 245), "black": (15, 15, 15), "gray": (128, 128, 128)}
SHAPES = ["circle", "square", "triangle"]


def draw(shape: str, color, bg, size: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    r = rng.randint(size // 5, size // 3)
    cx, cy = rng.randint(r, size - r), rng.randint(r, size - r)
    if shape == "circle":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif shape == "square":
        d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.count):
        shape = rng.choice(SHAPES)
        cname, color = rng.choice(list(COLORS.items()))
        bname, bg = rng.choice(list(BACKGROUNDS.items()))
        draw(shape, color, bg, a.size, rng).save(os.path.join(a.out, f"{i:03d}.png"))
        with open(os.path.join(a.out, f"{i:03d}.txt"), "w") as f:
            f.write(f"a {cname} {shape} on a {bname} background")
    print(f"wrote {a.count} images to {a.out}")


if __name__ == "__main__":
    main()
