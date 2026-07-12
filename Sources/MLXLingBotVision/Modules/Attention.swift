import Foundation
import MLX
import MLXFast
import MLXNN

private func ropeRotateHalf(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    let half = d / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

private func ropeApply(_ x: MLXArray, sin: MLXArray, cos: MLXArray) -> MLXArray {
    (x * cos) + (ropeRotateHalf(x) * sin)
}

/// Fused-QKV multi-head self-attention, matching the DINOv2-style layout of the
/// LingBot-Vision checkpoint (`attn.qkv` + `attn.proj`).
///
/// The `mask_k_bias` behavior (zeroing the K third of the QKV bias) is baked
/// into `qkv.bias` at conversion time, so this is a plain fused `Linear`.
final class Attention: Module {
    let numHeads: Int
    let headDim: Int
    let dim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(dim: Int, numHeads: Int, qkvBias: Bool, projBias: Bool) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.dim = dim
        self.scale = 1.0 / Float(dim / numHeads).squareRoot()

        _qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        _proj.wrappedValue = Linear(dim, dim, bias: projBias)
        super.init()
    }

    /// Rotate only the patch tokens; the leading prefix (cls + storage tokens)
    /// carries no spatial position and passes through unchanged.
    private func applyRope(_ q: MLXArray, _ k: MLXArray, rope: (sin: MLXArray, cos: MLXArray)) -> (MLXArray, MLXArray) {
        let (sin, cos) = rope
        let n = q.dim(-2)
        let prefix = n - sin.dim(-2)
        precondition(prefix >= 0)
        if prefix > 0 {
            let qPre = q[.ellipsis, ..<prefix, 0...]
            let kPre = k[.ellipsis, ..<prefix, 0...]
            let qTail = ropeApply(q[.ellipsis, prefix..., 0...], sin: sin, cos: cos)
            let kTail = ropeApply(k[.ellipsis, prefix..., 0...], sin: sin, cos: cos)
            return (concatenated([qPre, qTail], axis: -2), concatenated([kPre, kTail], axis: -2))
        }
        return (ropeApply(q, sin: sin, cos: cos), ropeApply(k, sin: sin, cos: cos))
    }

    func callAsFunction(_ x: MLXArray, rope: (sin: MLXArray, cos: MLXArray)? = nil) -> MLXArray {
        let B = x.dim(0)
        let N = x.dim(1)

        // [B, N, 3D] -> [B, N, 3, heads, head_dim] -> per-tensor [B, heads, N, head_dim]
        let qkvR = qkv(x).reshaped(B, N, 3, numHeads, headDim)
        var q = qkvR[0..., 0..., 0].swappedAxes(1, 2)
        var k = qkvR[0..., 0..., 1].swappedAxes(1, 2)
        let v = qkvR[0..., 0..., 2].swappedAxes(1, 2)

        if let rope {
            (q, k) = applyRope(q, k, rope: rope)
        }

        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        out = out.swappedAxes(1, 2).reshaped(B, N, dim)
        return proj(out)
    }
}
