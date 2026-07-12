# MLXLingBotVision

An [MLX Swift](https://github.com/ml-explore/mlx-swift) port of the
[LingBot-Vision](https://github.com/Robbyant/lingbot-vision) self-supervised ViT
backbones for dense spatial perception, running natively on Apple Silicon.

📖 **[API documentation](https://mnmly.github.io/mlx-swift-LingBot-Vision/)** · 🤗 **[MLX weights (ViT-L/16)](https://huggingface.co/mnmly/lingbot-vision-vit-large-mlx)**

LingBot-Vision is a DINOv2/DINOv3-lineage Vision Transformer with axial 2D RoPE,
register/storage tokens, LayerScale, and a fused-QKV attention with a masked
K-bias. This package reproduces the eval-time forward pass (CLS / storage /
patch token outputs) and ships a PCA feature-visualization pipeline.

## Status

| variant | embed / depth / heads | FFN | ported | parity |
|---------|-----------------------|-----|--------|--------|
| ViT-S/16 | 384 / 12 / 6   | mlp    | ✅ (arch) | — |
| ViT-B/16 | 768 / 12 / 12  | mlp    | ✅ (arch) | — |
| **ViT-L/16** | **1024 / 24 / 16** | **mlp** | ✅ | **cosine 0.99999, maxAbs ~1e-2** |
| ViT-g/16 | 1536 / 40 / 24 | swiglu | ✅ (arch) | — |

ViT-L is verified end-to-end against the Python reference. The other variants
share the same code paths (the SwiGLU FFN and RMSNorm branches are implemented)
but have not been numerically checked here.

## Convert a checkpoint

The released weights are PyTorch `.pt` files. Convert one to the MLX layout
(`config.json` + `model.safetensors`) with the bundled script. It bakes the
fused-QKV `bias_mask` into the bias and drops the inference-unused `mask_token`.

```bash
hf download robbyant/lingbot-vision-vit-large            # -> ~/.cache/huggingface

uv run --with safetensors scripts/convert.py \
  --ckpt   ~/.cache/huggingface/hub/models--robbyant--lingbot-vision-vit-large/snapshots/*/model.pt \
  --config /path/to/lingbot-vision/lingbot_vision/configs/lbot_vision_vitl.yaml \
  --out    ~/lingbot-vision-mlx/vit-large
```

## Use the library

`LingBotVisionSession` is the shared inference driver — load a converted
checkpoint once, run many images. It backs both the CLI and the SwiftUI demo
app, so their behavior can't drift.

```swift
import MLXLingBotVision

let session = try LingBotVisionSession.load(
    SessionConfig(modelDirectory: URL(fileURLWithPath: "~/lingbot-vision-mlx/vit-large"),
                  dtype: .float16))

// Full token output (cls / storage / patch tokens + patch grid dims)
let out = try session.features(imageURL: imageURL, size: 512)
print(out.patchTokens.shape)   // [1, 1024, 1024]  (32x32 patches, dim 1024)

// PCA RGB visualization of the patch tokens
let result = try session.pca(imageURL: imageURL, size: 512)
try writePNG(rgb: result.rgb, h: result.gridH, w: result.gridW, to: outURL, upscaleTo: 512)

// …or straight to a CGImage (used by the SwiftUI app)
let cg = try session.pcaCGImage(imageURL: imageURL, size: 512)
```

## Command-line tools

```bash
# PCA feature visualization
xcodebuild -scheme lbv-tool -configuration Release -destination 'platform=macOS' \
  -derivedDataPath .xcdd build
.xcdd/Build/Products/Release/lbv-tool \
  --model ~/lingbot-vision-mlx/vit-large --image example.png --out pca.png --size 512

# In-process benchmark + memory-leak check
.xcdd/Build/Products/Release/lbv-bench \
  --model ~/lingbot-vision-mlx/vit-large --image example.png --size 512 --iterations 20
```

`lbv-bench` loops the session and reports the active-memory drift across
iterations (≈0 ⇒ no leak) alongside the peak footprint (MLX's reusable buffer
cache, not leaked memory).

## SwiftUI demo app

`Examples/LingBotVisionDemo` is a small macOS app that drives the same
`LingBotVisionSession`. It pulls in this package as a local Swift Package
dependency; open the project and run:

```bash
open Examples/LingBotVisionDemo/LingBotVisionDemo.xcodeproj
```

Pick a converted model folder and an image, choose a size, and hit **Run** —
the backbone runs off the main thread and the PCA map renders next to the
input. The model I/O and compute live entirely in the library Session; the app
(`DemoModel.swift`) owns only the `Task.detached`, `autoreleasepool`,
`MainActor` hops, and `CGImage` display.

## Documentation

The published site lives at
**<https://mnmly.github.io/mlx-swift-LingBot-Vision/>** — auto-deployed from
`main` by `.github/workflows/docs.yml`.

Reference docs are generated with DocC. Because the package depends on mlx-swift
(Metal), the docs build uses `xcodebuild docbuild` + `docc process-archive`
rather than the SwiftPM plugin:

```bash
Scripts/build_docs.sh            # -> ./docs (GitHub Pages-ready)
Scripts/build_docs.sh preview    # build then open index.html
```

## Build & test

Use `xcodebuild` — `swift build`/`swift test` skip the Metal-capable toolchain
MLX needs at runtime.

```bash
# Build everything
xcodebuild -scheme MLXLingBotVision-Package -destination 'platform=macOS' \
  -derivedDataPath .xcdd build

# Numerical parity test (needs a converted model at ~/lingbot-vision-mlx/vit-large,
# or set LBV_MODEL_DIR)
xcodebuild -scheme MLXLingBotVision-Package -destination 'platform=macOS' test
```

Regenerate the parity fixture from the Python reference:

```bash
uv run --with safetensors scripts/gen_fixtures.py \
  --repo /path/to/lingbot-vision --ckpt .../model.pt \
  --config .../lbot_vision_vitl.yaml --image examples/example.png --size 224 \
  --out Tests/MLXLingBotVisionTests/Fixtures/vitl_parity.safetensors
```

## Verified results (ViT-L/16, Apple Silicon)

- **Parity vs Python fp32:** patch tokens cosine `0.9999987`, maxAbs `0.010`,
  meanAbs `0.0004`; CLS token cosine `0.99999624`.
- **Throughput:** ~63 ms / image median at 512×512 (32×32 patch grid), fp16.
- **Memory:** flat active memory across 20 iterations (no leak); ~1.3 GB peak
  reusable buffer cache.

## Notes on the port

- Weight keys follow the original DINOv2 layout (`blocks.N.attn.qkv`,
  `blocks.N.ls1.gamma`, `cls_token`, `storage_tokens`, `rope_embed.periods`),
  loaded with `verify: [.noUnusedKeys]`.
- `mask_k_bias` (zeroing the K third of the QKV bias) is baked into `qkv.bias`
  at conversion, so the Swift attention is a plain fused `Linear`.
- The RoPE `periods` buffer is loaded from the checkpoint and kept in fp32
  regardless of the compute dtype, matching the reference's fp32 rope tables.
- Conv2d patch-embed weights are transposed NCHW→NHWC at load time.

## License

`MLXLingBotVision` is licensed under the **Apache License 2.0** — see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

It is an **unofficial** port and derivative work of
[LingBot-Vision](https://github.com/robbyant/lingbot-vision) (weights on
[Hugging Face](https://huggingface.co/robbyant/lingbot-vision-vit-large)), which
is itself Apache 2.0. This project is **not affiliated with or endorsed by** the
original authors (Ant Group). The self-supervised pretraining method and the
released checkpoints are their work; this repository only re-implements the
eval-time forward pass for MLX Swift.

Converted MLX weights are published separately on Hugging Face at
[`mnmly/lingbot-vision-vit-large-mlx`](https://huggingface.co/mnmly/lingbot-vision-vit-large-mlx).
