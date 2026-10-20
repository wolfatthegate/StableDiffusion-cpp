"""Pure-PyTorch LoRA checks: zero-init equivalence, merge correctness, save/load, filtering."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from sdft.lora import (inject_lora, load_lora_file, load_lora_state_dict, lora_modules, lora_state_dict,  # noqa: E402
                       merge_lora, save_lora_file)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(16, 16)
        self.to_k = nn.Linear(16, 16)
        self.proj = nn.Conv2d(4, 4, 1)

    def forward(self, x, img):
        return self.to_k(self.to_q(x)), self.proj(img)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Block()
        self.b = Block()


def main():
    torch.manual_seed(0)
    net = Net()
    x, img = torch.randn(3, 16), torch.randn(2, 4, 5, 5)
    ref = net.a(x, img)
    names = inject_lora(net, ["to_q", "to_k", "proj"], rank=2, alpha=4.0, exclude=["^b\\."])
    assert names == ["a.to_q", "a.to_k", "a.proj"], names
    assert len(lora_modules(net)) == 3
    out = net.a(x, img)
    assert torch.allclose(out[0], ref[0]) and torch.allclose(out[1], ref[1]), "zero-initialised LoRA must not change outputs"

    # Perturb adapters, then check that merging reproduces the adapted outputs exactly.
    for m in lora_modules(net).values():
        nn.init.normal_(m.lora_B.weight, std=0.1)
    adapted = net.a(x, img)
    state = lora_state_dict(net, "unet")
    assert set(state) == {f"unet.a.{n}.lora_{ab}.weight" for n in ("to_q", "to_k", "proj") for ab in "AB"}
    merged = merge_lora(net)
    assert merged == 3 and not lora_modules(net)
    after = net.a(x, img)
    assert torch.allclose(after[0], adapted[0], atol=1e-5) and torch.allclose(after[1], adapted[1], atol=1e-5)

    # Save, reload into a fresh network, compare.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lora.safetensors")
        save_lora_file(path, state, 2, 4.0)
        tensors, meta = load_lora_file(path)
        assert meta["rank"] == "2" and len(tensors) == 6
        net2 = Net()
        net2.load_state_dict(Net().state_dict())
        inject_lora(net2, ["to_q", "to_k", "proj"], 2, 4.0, exclude=["^b\\."])
        loaded, missing = load_lora_state_dict(net2, tensors, "unet")
        assert loaded == 3 and not missing
    print("test_lora: ok")


if __name__ == "__main__":
    main()
