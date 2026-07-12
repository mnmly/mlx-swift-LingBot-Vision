import Foundation
import MLX
import MLXNN

/// Patchify a NHWC image with a strided Conv2d (`patch_embed.proj`).
///
/// Input `[B, H, W, C]` -> `[B, H/patch, W/patch, embedDim]` (kept in NHWC,
/// grid layout; the transformer flattens the grid to a token sequence).
final class PatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(patchSize: Int, inChannels: Int, embedDim: Int) {
        _proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize),
            bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(x)
    }
}
