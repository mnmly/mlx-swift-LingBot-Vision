import Foundation
import MLX
import MLXNN

/// Per-channel learned scale applied to a residual branch (`ls1.gamma` /
/// `ls2.gamma`).
final class LayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int, initValues: Float) {
        _gamma.wrappedValue = MLXArray.full([dim], values: MLXArray(initValues))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}
