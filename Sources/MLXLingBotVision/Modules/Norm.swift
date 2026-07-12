import Foundation
import MLX
import MLXNN

/// Build the normalization layer named by the config. Both variants expose a
/// `weight` (and LayerNorm additionally `bias`) parameter, matching the
/// checkpoint keys directly. LingBot's `layernorm`/`layernormbf16` differ only
/// in epsilon (baked into `config.normEps`).
func makeNorm(_ config: LingBotVisionConfiguration, dim: Int) -> UnaryLayer {
    switch config.normLayer {
    case "rmsnorm":
        return RMSNorm(dimensions: dim, eps: config.normEps)
    default:
        return LayerNorm(dimensions: dim, eps: config.normEps)
    }
}
