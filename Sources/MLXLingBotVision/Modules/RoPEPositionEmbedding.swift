import Foundation
import MLX
import MLXNN

/// Axial 2D rotary position embedding over the patch grid.
///
/// Height and width use independent rotation frequencies (no coordinate
/// mixing) and there are no learnable weights — only a persistent `periods`
/// buffer that ships in the checkpoint (`rope_embed.periods`). Patch
/// coordinates are normalized per-axis to `[-1, 1]`. Training-time coordinate
/// augmentation (shift/jitter/rescale) is inference-irrelevant and omitted.
/// `callAsFunction(H:W:)` returns `(sin, cos)` tables of shape `[H*W, D_head]`.
final class RoPEPositionEmbedding: Module {
    @ParameterInfo(key: "periods") var periods: MLXArray

    init(embedDim: Int, numHeads: Int) {
        precondition(embedDim % (4 * numHeads) == 0)
        let dHead = embedDim / numHeads
        _periods.wrappedValue = MLXArray.zeros([dHead / 4], dtype: .float32)
        super.init()
    }

    func callAsFunction(H: Int, W: Int, dtype: DType = .float32) -> (sin: MLXArray, cos: MLXArray) {
        // Per-axis normalized patch-center coordinates in [0, 1).
        let coordsH = MLXArray(stride(from: Float(0.5), to: Float(H), by: 1.0)) / Float(H)
        let coordsW = MLXArray(stride(from: Float(0.5), to: Float(W), by: 1.0)) / Float(W)

        let hGrid = broadcast(coordsH[0..., .newAxis], to: [H, W])
        let wGrid = broadcast(coordsW[.newAxis, 0...], to: [H, W])
        var coords = stacked([hGrid, wGrid], axis: -1).reshaped(H * W, 2)  // [HW, 2]
        coords = 2.0 * coords - 1.0

        let dq = periods.dim(0)
        let hw = coords.dim(0)

        // angles[hw, axis, dq] = 2*pi * coords / periods
        let coordsExp = coords[0..., 0..., .newAxis]              // [HW, 2, 1]
        let periodsExp = periods[.newAxis, .newAxis, 0...]         // [1, 1, dq]
        var angles = (2.0 * Float.pi) * coordsExp / periodsExp     // [HW, 2, dq]
        angles = angles.reshaped(hw, 2 * dq)                       // [HW, D/2]
        angles = tiled(angles, repetitions: [1, 2])               // [HW, D]

        return (sin: MLX.sin(angles).asType(dtype), cos: MLX.cos(angles).asType(dtype))
    }
}
