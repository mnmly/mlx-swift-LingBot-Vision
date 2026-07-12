# ``MLXLingBotVision``

An MLX-Swift port of the LingBot-Vision self-supervised ViT backbones for
dense spatial perception on Apple Silicon.

## Overview

LingBot-Vision is a DINOv2/DINOv3-lineage Vision Transformer: axial 2D RoPE,
register/storage tokens, LayerScale, and a fused-QKV attention with a masked
K-bias. This module reproduces the eval-time forward pass — normalized
CLS / storage / patch token outputs — and adds a PCA feature-visualization
pipeline.

The consumer-facing entry point is ``LingBotVisionSession``: it loads a
converted checkpoint once and runs many images through the backbone. The same
session backs both the `lbv-tool` CLI and the SwiftUI demo app, so their
behavior can never drift.

```swift
import MLXLingBotVision

let session = try LingBotVisionSession.load(
    SessionConfig(modelDirectory: modelDir, dtype: .float16))

// Full token output (cls / storage / patch tokens + patch grid dims)
let out = try session.features(imageURL: imageURL, size: 512)

// PCA RGB visualization of the patch tokens
let cg = try session.pcaCGImage(imageURL: imageURL, size: 512)
```

Weights ship as PyTorch `.pt` files; convert them to the MLX layout
(`config.json` + `model.safetensors`) with `scripts/convert.py` before loading.

## Topics

### Running inference

- ``LingBotVisionSession``
- ``SessionConfig``
- ``PCAResult``
- ``defaultImageSize``

### The model

- ``LingBotVisionTransformer``
- ``LingBotVisionOutput``
- ``LingBotVisionConfiguration``

### Preprocessing and visualization

- ``ImageProcessor``
- ``PatchPCA``
- ``makeCGImage(rgb:h:w:upscaleTo:)``
- ``writePNG(rgb:h:w:to:upscaleTo:)``

### Errors

- ``LingBotVisionError``
