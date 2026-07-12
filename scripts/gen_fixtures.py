# /// script
# requires-python = ">=3.10"
# dependencies = ["safetensors", "numpy"]
# ///
"""Generate a numerical-parity fixture from the Python LingBot-Vision reference.

Runs the reference backbone in fp32 on CPU over the example image and writes a
safetensors bundle the Swift test loads directly:

  input_nhwc     [1, H, W, 3]   ImageNet-normalized input (NHWC, MLX layout)
  patch_tokens   [1, N, D]      x_norm_patchtokens
  cls_token      [1, D]         x_norm_clstoken

The same normalized input is fed to both sides so the fixture isolates model
parity from any preprocessing (PIL vs CoreGraphics) differences.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the python lingbot-vision repo")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from lingbot_vision.loader import load_backbone, load_backbone_state, load_config
    from lingbot_vision.preprocess import load_image

    cfg = load_config(args.config)
    ckpt = load_backbone_state(args.ckpt)
    backbone, embed_dim = load_backbone(cfg, ckpt, device="cpu", dtype=torch.float32, verbose=True)

    img_norm, _, (H, W) = load_image(args.image, size=args.size, patch_size=backbone.patch_size, mode="square")
    img_norm = img_norm.to(torch.float32)  # [1, 3, H, W]

    with torch.no_grad():
        out = backbone(img_norm, is_training=True)

    patch = out["x_norm_patchtokens"].float().cpu().numpy()  # [1, N, D]
    cls = out["x_norm_clstoken"].float().cpu().numpy()  # [1, D]
    input_nhwc = img_norm.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [1, H, W, 3]

    tensors = {
        "input_nhwc": input_nhwc.astype(np.float32),
        "patch_tokens": patch.astype(np.float32),
        "cls_token": cls.astype(np.float32),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out)
    print(f"[fixture] size={H}x{W} embed_dim={embed_dim} patch={patch.shape} cls={cls.shape}")
    print(f"[fixture] wrote {args.out}")


if __name__ == "__main__":
    main()
