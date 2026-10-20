"""Configuration schema and YAML loading.

Every option lives in one of the dataclasses below. The YAML file mirrors the
dataclass nesting, so ``train.steps`` in the file becomes ``cfg.train.steps``.
Unknown keys are errors, which catches typos before a long run starts.
"""

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    pass


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Local diffusers-layout directory (for example the models/sd15 used by the
    # C++ binary) or a Hugging Face repository id.
    base: str = "models/sd15"
    # Weight file variant: None for *.safetensors, "fp16" for *.fp16.safetensors.
    # Auto-detected for local directories when None.
    variant: Optional[str] = None
    # Where to fetch the small config.json files from when `base` is a local
    # directory that only contains weights (as scripts/download_sd15.sh makes).
    config_repo: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"


@dataclass
class DataConfig:
    # Folder of images. Captions come from, in order of preference:
    #   metadata.jsonl lines {"file_name": "x.jpg", "text": "..."}
    #   a sidecar x.txt next to x.jpg
    #   default_caption
    path: str = "data/train"
    resolution: int = 512          # training image size (square), multiple of 8
    center_crop: bool = True       # False = random crop after resizing the short side
    random_flip: bool = True
    caption_dropout: float = 0.1   # probability of training on an empty caption (enables CFG)
    default_caption: str = ""
    repeats: int = 1               # virtual dataset size multiplier
    cache_latents: bool = False    # encode every image once with the VAE and reuse the latents
    num_workers: int = 0


@dataclass
class ComponentWeights:
    """Which parameters of one component (unet or text_encoder) to train.

    `include` and `exclude` are regular expressions matched against parameter
    names such as ``down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight``
    (full fine-tuning) or module names such as ``...attn1.to_q`` (LoRA).
    """
    train: bool = False
    lr: Optional[float] = None               # overrides train.lr for this component
    include: Optional[List[str]] = None      # None = everything
    exclude: List[str] = field(default_factory=list)


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    # Module-name suffixes that receive adapters. Any nn.Linear or nn.Conv2d
    # whose qualified name ends with one of these is wrapped.
    unet_targets: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    text_encoder_targets: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])


@dataclass
class WeightsConfig:
    method: str = "lora"                     # "lora" or "full"
    unet: ComponentWeights = field(default_factory=lambda: ComponentWeights(train=True))
    text_encoder: ComponentWeights = field(default_factory=ComponentWeights)
    lora: LoraConfig = field(default_factory=LoraConfig)
    # A previously exported lora.safetensors to start from (LoRA method only).
    init_from: Optional[str] = None


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 1
    grad_accum: int = 1
    lr: float = 1e-4
    lr_scheduler: str = "cosine"             # constant | linear | cosine
    warmup_steps: int = 0
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    precision: str = "fp32"                  # fp32 | fp16 | bf16 (autocast)
    gradient_checkpointing: bool = False
    snr_gamma: Optional[float] = None        # min-SNR loss weighting, 5.0 is common
    noise_offset: float = 0.0
    seed: int = 0
    device: str = "auto"                     # auto | cuda | mps | cpu
    ema_decay: Optional[float] = None        # for full fine-tunes, e.g. 0.999


@dataclass
class ExportConfig:
    output_dir: str = "models/finetuned"
    dtype: str = "fp16"                      # fp16 | fp32
    fuse_lora: bool = True                   # merge adapters into the weights (needed by ./sd)
    save_lora: bool = True                   # also write lora.safetensors
    save_lora_only: bool = False             # skip the full model export


@dataclass
class CheckpointConfig:
    dir: str = "runs/run"
    every: int = 250                         # 0 disables periodic checkpoints
    keep: int = 3
    resume: Optional[str] = None             # path to a checkpoint-N directory


@dataclass
class LoggingConfig:
    every: int = 10
    sample_every: int = 0                    # generate sample_prompts every N steps (0 = never)
    sample_prompts: List[str] = field(default_factory=list)
    sample_steps: int = 20
    sample_cfg: float = 7.5
    sample_size: int = 512


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        w, t, d, e = self.weights, self.train, self.data, self.export
        _check(w.method in ("lora", "full"), "weights.method must be 'lora' or 'full'")
        _check(w.unet.train or w.text_encoder.train, "nothing to train: set weights.unet.train or weights.text_encoder.train")
        _check(w.lora.rank >= 1, "weights.lora.rank must be >= 1")
        _check(t.steps >= 1, "train.steps must be >= 1")
        _check(t.batch_size >= 1 and t.grad_accum >= 1, "train.batch_size and train.grad_accum must be >= 1")
        _check(t.lr > 0, "train.lr must be > 0")
        _check(t.lr_scheduler in ("constant", "linear", "cosine"), "train.lr_scheduler must be constant, linear or cosine")
        _check(t.precision in ("fp32", "fp16", "bf16"), "train.precision must be fp32, fp16 or bf16")
        _check(t.device in ("auto", "cuda", "mps", "cpu"), "train.device must be auto, cuda, mps or cpu")
        _check(t.ema_decay is None or 0 < t.ema_decay < 1, "train.ema_decay must be in (0, 1)")
        _check(d.resolution % 8 == 0 and d.resolution >= 64, "data.resolution must be a multiple of 8 and >= 64")
        _check(0 <= d.caption_dropout <= 1, "data.caption_dropout must be in [0, 1]")
        _check(e.dtype in ("fp16", "fp32"), "export.dtype must be fp16 or fp32")
        _check(self.checkpoint.keep >= 1, "checkpoint.keep must be >= 1")
        if w.method == "full":
            _check(w.init_from is None, "weights.init_from only applies to the lora method")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ConfigError(msg)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _build(cls, data: Any, path: str, base=None):
    """Overlays a dict onto a dataclass instance, rejecting unknown keys.

    `base` is the instance to start from (the parent's default for this
    section), so a section like ``weights.unet: {include: [...]}`` keeps the
    parent's ``train: true`` default instead of the class default.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; valid keys: {sorted(fields)}")
    obj = base if base is not None else cls()
    for name, f in fields.items():
        if name not in data:
            continue
        value = data[name]
        if dataclasses.is_dataclass(f.type):
            setattr(obj, name, _build(f.type, value, f"{path}.{name}", base=getattr(obj, name)))
        else:
            setattr(obj, name, _coerce(f.type, value, f"{path}.{name}"))
    return obj


def _coerce(tp, value, path):
    """Light type checking for leaf values."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:  # Optional[X]
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value, path)
    if origin is list or tp is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        return [str(v) if args and args[0] is str else v for v in value]
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false")
        return value
    # PyYAML reads "1e-4" (no decimal point) as a string, so numeric strings
    # are accepted for number fields.
    if tp is int:
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise ConfigError(f"{path}: expected an integer")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer")
        return value
    if tp is float:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise ConfigError(f"{path}: expected a number")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string")
        return value
    return value


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Applies `a.b.c=value` strings (values parsed as YAML) to a nested dict."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override '{item}': '{p}' is not a section")
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def config_from_dict(data: Dict[str, Any]) -> Config:
    cfg = _build(Config, data, "config")
    cfg.validate()
    return cfg


def load_config(path: str, overrides: Optional[List[str]] = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    if overrides:
        apply_overrides(data, overrides)
    return config_from_dict(data)


def dump_config(cfg: Config, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
