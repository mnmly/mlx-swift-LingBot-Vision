# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors", "pyyaml", "numpy"]
# ///
"""Convert a LingBot-Vision backbone `.pt` checkpoint to the MLX-swift layout.

Produces two files under --out:
  model.safetensors  — fp32 weights with keys matching the Swift module tree.
  config.json        — architecture parameters read by the Swift loader.

Transformations applied (all numerically neutral):
  * unwrap the common state-dict wrappers and strip a `_orig_mod.` prefix,
  * bake the fused-QKV `bias_mask` into `attn.qkv.bias` and drop the mask
    buffer, so the Swift side is a plain fused Linear (mask_k_bias zeroes the
    K third of the bias — see LinearKMaskedBias in the Python reference),
  * drop `mask_token` (MIM-only; the inference forward uses `cls_token + 0 *
    mask_token`, i.e. the token is a no-op at eval time).

Conv2d weight transposition (PyTorch NCHW -> MLX NHWC) is done on the Swift
side at load time, matching the sibling MLXDINOv3 port.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

_WRAPPER_KEYS = ("teacher", "model_state", "state_dict", "model", "backbone")


def unwrap(sd):
    if isinstance(sd, dict):
        for k in _WRAPPER_KEYS:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def strip_backbone_prefix(sd):
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model.pt")
    ap.add_argument("--config", required=True, help="path to the variant yaml")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    student = cfg["student"]
    hcfg = student.get("lbot_vision", {}) or {}

    arch_dims = {
        "vit_small": (384, 12, 6, 4.0),
        "vit_base": (768, 12, 12, 4.0),
        "vit_large": (1024, 24, 16, 4.0),
        "vit_so400m": (1152, 27, 18, 3.777777778),
        "vit_huge2": (1280, 32, 20, 4.0),
        "vit_giant2": (1536, 40, 24, 4.0),
        "vit_7b": (4096, 40, 32, 3.0),
    }
    embed_dim, depth, num_heads, ffn_ratio = arch_dims[student["arch"]]

    norm_eps = {"layernorm": 1e-6, "layernormbf16": 1e-5, "rmsnorm": 1e-5}[
        hcfg.get("norm_layer", "layernorm")
    ]

    config = {
        "arch": student["arch"],
        "patchSize": int(student["patch_size"]),
        "inChannels": 3,
        "embedDim": embed_dim,
        "depth": depth,
        "numHeads": num_heads,
        "ffnRatio": ffn_ratio,
        "imgSize": int(cfg["crops"]["global_crops_size"]),
        "qkvBias": bool(student["qkv_bias"]),
        "projBias": bool(student["proj_bias"]),
        "ffnBias": bool(student["ffn_bias"]),
        "layerscaleInit": float(student["layerscale"]),
        "nStorageTokens": int(student["num_register_tokens"]),
        "normLayer": hcfg.get("norm_layer", "layernorm"),
        "normEps": norm_eps,
        "ffnLayer": hcfg.get("ffn_layer", "mlp"),
        "ropeBase": float(hcfg.get("pos_embed_rope_base", 100.0)),
        "maskKBias": bool(hcfg.get("mask_k_bias", False)),
    }

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = strip_backbone_prefix(unwrap(sd))

    out_sd = {}
    baked = 0
    for k, v in sd.items():
        if k == "mask_token":
            continue
        if k.endswith(".bias_mask"):
            continue  # consumed together with the matching bias below
        if k.endswith("attn.qkv.bias"):
            mask = sd.get(k + "_mask")
            if mask is None:
                mask = sd.get(k[: -len("bias")] + "bias_mask")
            if mask is not None:
                v = v * mask
                baked += 1
        out_sd[k] = v.contiguous().to(torch.float32)

    save_file(out_sd, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[convert] arch={config['arch']} keys={len(out_sd)} baked_qkv_bias={baked}")
    print(f"[convert] wrote {out/'model.safetensors'} and {out/'config.json'}")


if __name__ == "__main__":
    main()
