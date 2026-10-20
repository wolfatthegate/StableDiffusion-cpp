"""Small helpers: device selection, seeding, logging, parameter counting."""

import json
import os
import random
import time
from typing import Iterable, Optional

import numpy as np
import torch


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def fmt_count(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


class RunLogger:
    """Prints to stdout and appends JSON lines to <dir>/log.jsonl."""

    def __init__(self, run_dir: Optional[str]):
        self.path = os.path.join(run_dir, "log.jsonl") if run_dir else None
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
        self.t0 = time.time()

    def log(self, **fields) -> None:
        fields = {"time": round(time.time() - self.t0, 1), **fields}
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items()), flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fields) + "\n")

    def info(self, msg: str) -> None:
        print(msg, flush=True)
