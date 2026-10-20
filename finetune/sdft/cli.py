"""Command-line entry point: python -m sdft <command> config.yaml [--set a.b=c ...]"""

import argparse
import os
import sys

from .config import ConfigError, load_config


def cmd_inspect(cfg) -> None:
    from .model import apply_weights_config, load_components
    comps = load_components(cfg)
    setup = apply_weights_config(cfg, comps)
    print(setup.summary(comps))
    for g in setup.groups:
        print(f"\n{g.component}: first tensors of {len(g.names)}")
        for n in g.names[:12]:
            print(f"  {n}")
        if len(g.names) > 12:
            print(f"  ... {len(g.names) - 12} more")


def cmd_train(cfg) -> None:
    from .train import train
    train(cfg)


def cmd_export(cfg, checkpoint: str) -> None:
    from safetensors.torch import load_file
    from .export import export_model
    from .model import apply_weights_config, load_components, load_trainable_state_dict
    comps = load_components(cfg)
    setup = apply_weights_config(cfg, comps)
    load_trainable_state_dict(comps, setup, load_file(os.path.join(checkpoint, "trainable.safetensors")))
    print(f"loaded {checkpoint}")
    export_model(cfg, comps, setup)


def cmd_dataset(cfg) -> None:
    from .data import ImageCaptionDataset
    from transformers import CLIPTokenizer
    from .model import ensure_configs
    ensure_configs(cfg.model.base, cfg.model.config_repo)
    tok = CLIPTokenizer.from_pretrained(cfg.model.base, subfolder="tokenizer")
    ds = ImageCaptionDataset(cfg.data, tok)
    print(f"{len(ds.files)} images under {cfg.data.path} ({len(ds)} samples with repeats)")
    for p in ds.files[:10]:
        print(f"  {os.path.relpath(p, cfg.data.path)}: {ds.captions[p]!r}")
    if len(ds.files) > 10:
        print(f"  ... {len(ds.files) - 10} more")
    sample = ds[0]
    print(f"sample tensor {tuple(sample['pixel_values'].shape)}, {int((sample['input_ids'] != tok.eos_token_id).sum())} tokens before padding")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sdft", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_text in (("train", "train and export"), ("inspect", "show what would be trained"),
                            ("export", "export a checkpoint without training"), ("dataset", "check the dataset and captions")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", help="YAML config file")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.steps=50")
        if name == "export":
            p.add_argument("--checkpoint", required=True, help="checkpoint-N directory")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)
    if args.command == "train":
        cmd_train(cfg)
    elif args.command == "inspect":
        cmd_inspect(cfg)
    elif args.command == "export":
        cmd_export(cfg, args.checkpoint)
    elif args.command == "dataset":
        cmd_dataset(cfg)
