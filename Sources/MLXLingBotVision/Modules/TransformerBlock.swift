import Foundation
import MLX
import MLXNN

/// Pre-norm transformer block: `x + ls1(attn(norm1(x)))` then
/// `x + ls2(mlp(norm2(x)))`. Keys: `norm1`, `attn`, `ls1`, `norm2`, `mlp`,
/// `ls2`.
final class TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: UnaryLayer
    @ModuleInfo(key: "attn") var attn: Attention
    @ModuleInfo(key: "ls1") var ls1: LayerScale
    @ModuleInfo(key: "norm2") var norm2: UnaryLayer
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "ls2") var ls2: LayerScale

    init(_ config: LingBotVisionConfiguration) {
        let dim = config.embedDim
        _norm1.wrappedValue = makeNorm(config, dim: dim)
        _attn.wrappedValue = Attention(
            dim: dim, numHeads: config.numHeads,
            qkvBias: config.qkvBias, projBias: config.projBias)
        _ls1.wrappedValue = LayerScale(dim: dim, initValues: config.layerscaleInit)
        _norm2.wrappedValue = makeNorm(config, dim: dim)
        _mlp.wrappedValue =
            config.ffnLayer.hasPrefix("swiglu")
            ? SwiGLUFFN(inFeatures: dim, hiddenFeatures: config.intermediateSize, bias: config.ffnBias)
            : Mlp(inFeatures: dim, hiddenFeatures: config.intermediateSize, bias: config.ffnBias)
        _ls2.wrappedValue = LayerScale(dim: dim, initValues: config.layerscaleInit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: (sin: MLXArray, cos: MLXArray)?) -> MLXArray {
        let xAttn = x + ls1(attn(norm1(x), rope: rope))
        return xAttn + ls2(mlp(norm2(xAttn)))
    }
}
