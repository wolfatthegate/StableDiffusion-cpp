"""Run: python -m finetune.tests.test_config  (from the repo root, with finetune on sys.path)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sdft.config import ConfigError, apply_overrides, config_from_dict, load_config  # noqa: E402


def expect_error(data, fragment):
    try:
        config_from_dict(data)
    except ConfigError as e:
        assert fragment in str(e), f"expected '{fragment}' in '{e}'"
        return
    raise AssertionError(f"expected ConfigError containing '{fragment}'")


def main():
    cfg = config_from_dict({"weights": {"unet": {"train": True}}})
    assert cfg.weights.method == "lora" and cfg.train.steps == 1000
    expect_error({"weights": {"unet": {"train": False}, "text_encoder": {"train": False}}}, "nothing to train")
    expect_error({"trian": {}}, "unknown key")
    expect_error({"train": {"steps": "many"}}, "expected an integer")
    expect_error({"weights": {"method": "qlora"}}, "weights.method")
    expect_error({"data": {"resolution": 100}}, "resolution")
    data = apply_overrides({}, ["train.steps=5", "weights.lora.rank=2", "weights.text_encoder.train=true"])
    cfg = config_from_dict(data)
    assert cfg.train.steps == 5 and cfg.weights.lora.rank == 2 and cfg.weights.text_encoder.train is True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("train:\n  steps: 3\n  lr: 2e-5\nweights:\n  unet:\n    include: ['attn2']\n")
        path = f.name
    cfg = load_config(path, ["export.dtype=fp32"])
    assert cfg.train.lr == 2e-5 and cfg.weights.unet.include == ["attn2"] and cfg.export.dtype == "fp32"
    os.unlink(path)
    print("test_config: ok")


if __name__ == "__main__":
    main()
