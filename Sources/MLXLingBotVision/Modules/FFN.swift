import Foundation
import MLX
import MLXNN

/// Standard two-layer MLP FFN (`mlp.fc1` / `mlp.fc2`) with exact-GELU.
final class Mlp: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let act = GELU()

    init(inFeatures: Int, hiddenFeatures: Int, bias: Bool) {
        _fc1.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _fc2.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(act(fc1(x)))
    }
}

/// SwiGLU FFN (`mlp.w1` / `mlp.w2` / `mlp.w3`). The hidden width follows the
/// Python reference: `d = int(hidden * 2/3)` rounded up to `alignTo`.
final class SwiGLUFFN: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(inFeatures: Int, hiddenFeatures: Int, bias: Bool, alignTo: Int = 8) {
        let d = (hiddenFeatures * 2) / 3
        let rem = ((-d) % alignTo + alignTo) % alignTo
        let hidden = d + rem
        _w1.wrappedValue = Linear(inFeatures, hidden, bias: bias)
        _w2.wrappedValue = Linear(inFeatures, hidden, bias: bias)
        _w3.wrappedValue = Linear(hidden, inFeatures, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w3(silu(w1(x)) * w2(x))
    }
}
