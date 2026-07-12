import Foundation
import MLX
import MLXNN

extension LingBotVisionTransformer {
    /// Build and load a converted LingBot-Vision backbone from a directory
    /// produced by `scripts/convert.py` (containing `config.json` and
    /// `model.safetensors`).
    ///
    /// - Parameters:
    ///   - directory: folder holding `config.json` + `model.safetensors`.
    ///   - dtype: compute dtype for the weights. Defaults to `.float32` for
    ///     exact parity; pass `.float16` for a lighter footprint. The RoPE
    ///     `periods` buffer is always kept in `.float32` (matching the
    ///     reference's fp32 rope tables).
    public static func loadPretrained(
        directory: URL,
        dtype: DType = .float32
    ) throws -> LingBotVisionTransformer {
        let config = try LingBotVisionConfiguration.load(from: directory.appendingPathComponent("config.json"))
        let model = LingBotVisionTransformer(config)

        var weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        for (key, value) in weights {
            var v = value
            // PyTorch Conv2d weights are NCHW `(out, in, kH, kW)`; MLX Conv2d
            // expects NHWC `(out, kH, kW, in)`.
            if key.hasSuffix(".weight") && v.ndim == 4 {
                v = v.transposed(0, 2, 3, 1)
            }
            // Keep the rope frequency table in fp32 regardless of compute dtype.
            if key == "rope_embed.periods" {
                v = v.asType(.float32)
            } else {
                v = v.asType(dtype)
            }
            weights[key] = v
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        eval(model)
        return model
    }
}
