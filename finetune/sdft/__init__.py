"""sdft: configurable fine-tuning for Stable Diffusion 1.x.

Train LoRA adapters or full weights for the UNet and/or the CLIP text encoder
from a single YAML file, then export a diffusers-layout model directory that
both `diffusers` and the C++ `sd` binary in this repository can load.

Entry points:
    python -m sdft train   config.yaml
    python -m sdft inspect config.yaml
    python -m sdft export  config.yaml --checkpoint runs/x/checkpoint-500
"""

from .config import Config, load_config  # noqa: F401

__version__ = "0.1.0"
