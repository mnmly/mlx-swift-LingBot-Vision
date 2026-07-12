import Foundation
import MLX
import MLXNN

/// Normalized token outputs of a forward pass (the eval-time
/// `x_norm_*` dictionary from the Python reference).
public struct LingBotVisionOutput {
    /// Normalized CLS token, `[B, D]`.
    public let clsToken: MLXArray
    /// Normalized storage/register tokens, `[B, nStorage, D]`.
    public let storageTokens: MLXArray
    /// Normalized patch tokens, `[B, H*W, D]`.
    public let patchTokens: MLXArray
    /// Patch grid dimensions.
    public let gridH: Int
    public let gridW: Int
}

/// LingBot-Vision ViT backbone.
///
/// Token layout mirrors the reference: `[cls, storage×n, patch×HW]`. RoPE is
/// applied to the patch tokens only. A single final norm is applied to all
/// tokens (the released checkpoints do not untie the cls/patch norms).
public final class LingBotVisionTransformer: Module {
    public let config: LingBotVisionConfiguration
    let patchSize: Int
    let nStorageTokens: Int
    let embedDim: Int

    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "storage_tokens") var storageTokens: MLXArray
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ModuleInfo(key: "rope_embed") var ropeEmbed: RoPEPositionEmbedding
    @ModuleInfo(key: "blocks") var blocks: [TransformerBlock]
    @ModuleInfo(key: "norm") var norm: UnaryLayer

    public init(_ config: LingBotVisionConfiguration) {
        self.config = config
        self.patchSize = config.patchSize
        self.nStorageTokens = config.nStorageTokens
        self.embedDim = config.embedDim

        _clsToken.wrappedValue = MLXArray.zeros([1, 1, config.embedDim])
        _storageTokens.wrappedValue = MLXArray.zeros([1, max(config.nStorageTokens, 0), config.embedDim])
        _patchEmbed.wrappedValue = PatchEmbed(
            patchSize: config.patchSize, inChannels: config.inChannels, embedDim: config.embedDim)
        _ropeEmbed.wrappedValue = RoPEPositionEmbedding(embedDim: config.embedDim, numHeads: config.numHeads)
        _blocks.wrappedValue = (0 ..< config.depth).map { _ in TransformerBlock(config) }
        _norm.wrappedValue = makeNorm(config, dim: config.embedDim)
        super.init()
    }

    /// Run the backbone on a NHWC image batch `[B, H, W, C]`.
    public func callAsFunction(_ x: MLXArray) -> LingBotVisionOutput {
        let B = x.dim(0)

        // Patchify -> [B, H, W, D] -> flatten grid -> [B, HW, D]
        let patched = patchEmbed(x)
        let H = patched.dim(1)
        let W = patched.dim(2)
        var tokens = patched.reshaped(B, H * W, embedDim)

        // Prepend cls + storage tokens.
        let cls = broadcast(clsToken, to: [B, 1, embedDim])
        var prefix = [cls]
        if nStorageTokens > 0 {
            prefix.append(broadcast(storageTokens, to: [B, nStorageTokens, embedDim]))
        }
        tokens = concatenated(prefix + [tokens], axis: 1)

        let rope = ropeEmbed(H: H, W: W, dtype: tokens.dtype)
        for block in blocks {
            tokens = block(tokens, rope: rope)
        }

        let normed = norm(tokens)
        let prefixEnd = nStorageTokens + 1
        return LingBotVisionOutput(
            clsToken: normed[0..., 0, 0...],
            storageTokens: normed[0..., 1 ..< prefixEnd, 0...],
            patchTokens: normed[0..., prefixEnd..., 0...],
            gridH: H,
            gridW: W)
    }
}
