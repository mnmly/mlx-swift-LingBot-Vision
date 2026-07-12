import Foundation
import MLX
import MLXLinalg

/// PCA projection of patch tokens to an RGB visualization, matching the
/// reference `pca_demo._pca_rgb`: center, take the top-3 right singular
/// vectors, project, then per-channel 1–99 percentile normalize.
public enum PatchPCA {
    /// - Parameters:
    ///   - patchTokens: `[1, N, D]` (or `[N, D]`) normalized patch tokens.
    ///   - h, w: patch grid dimensions with `h * w == N`.
    /// - Returns: row-major RGB in `[0, 1]`, shape `[h, w, 3]`.
    public static func rgb(patchTokens: MLXArray, h: Int, w: Int) -> [Float] {
        var x = patchTokens
        if x.ndim == 3 { x = x[0] }  // [N, D]
        x = x.asType(.float32)
        x = x - x.mean(axis: 0, keepDims: true)

        // SVD is CPU-only in mlx-swift; run it on the CPU stream.
        let (_, _, vt) = MLXLinalg.svd(x, stream: .cpu)
        let top3 = vt[..<3]                      // [3, D]
        let proj = matmul(x, top3.transposed())  // [N, 3]
        eval(proj)

        var flat = proj.asArray(Float.self)      // row-major [N*3]
        percentileNormalizeInPlace(&flat, channels: 3, lowPct: 1, highPct: 99)
        return flat
    }

    /// Per-channel percentile stretch to [0, 1] with clipping. Operates in
    /// place on a row-major `[N, channels]` buffer.
    static func percentileNormalizeInPlace(_ data: inout [Float], channels: Int, lowPct: Double, highPct: Double) {
        let n = data.count / channels
        guard n > 0 else { return }
        for c in 0 ..< channels {
            var channel = [Float](repeating: 0, count: n)
            for i in 0 ..< n { channel[i] = data[i * channels + c] }
            channel.sort()
            let lo = percentile(sorted: channel, pct: lowPct)
            let hi = percentile(sorted: channel, pct: highPct)
            let denom = max(hi - lo, 1e-6)
            for i in 0 ..< n {
                let v = (data[i * channels + c] - lo) / denom
                data[i * channels + c] = min(max(v, 0), 1)
            }
        }
    }

    /// Linear-interpolated percentile of a pre-sorted array (numpy default).
    private static func percentile(sorted: [Float], pct: Double) -> Float {
        let n = sorted.count
        if n == 1 { return sorted[0] }
        let rank = pct / 100.0 * Double(n - 1)
        let lo = Int(rank.rounded(.down))
        let hi = min(lo + 1, n - 1)
        let frac = Float(rank - Double(lo))
        return sorted[lo] * (1 - frac) + sorted[hi] * frac
    }
}
