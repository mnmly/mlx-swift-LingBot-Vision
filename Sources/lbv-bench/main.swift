import ArgumentParser
import Foundation
import MLX
import MLXLingBotVision

/// In-process benchmark that loops the pipeline over one image and watches
/// `MLX.GPU`/`MLX.Memory` counters. A flat active-memory curve across
/// iterations means no leak; the large "peak" figure is MLX's reusable buffer
/// cache, not leaked memory. Run in **Release** (`-c release`) — Debug is ~5×
/// slower.
@main
struct LBVBench: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lbv-bench",
        abstract: "Benchmark LingBot-Vision inference and check for memory leaks.")

    @Option(name: .long) var model: String
    @Option(name: .long) var image: String
    @Option(name: .long) var size: Int = 512
    @Option(name: .long) var iterations: Int = 20
    @Option(name: .long) var warmup: Int = 3
    @Option(name: .long, help: "Compute dtype: float32 or float16.") var dtype: String = "float16"
    @Option(name: .long, help: "Bound the MLX buffer cache (bytes). 0 = unbounded.") var cacheLimit: Int = 0

    func run() throws {
        let dt: DType = dtype == "float16" ? .float16 : .float32
        if cacheLimit > 0 { MLX.GPU.set(cacheLimit: cacheLimit) }

        let config = SessionConfig(modelDirectory: URL(fileURLWithPath: model), dtype: dt)
        let session = try LingBotVisionSession.load(config)
        let url = URL(fileURLWithPath: image)

        func mb(_ bytes: Int) -> String { String(format: "%.1f MB", Double(bytes) / 1_048_576) }

        for _ in 0 ..< warmup { _ = try session.features(imageURL: url, size: size) }

        var times: [Double] = []
        let startActive = MLX.GPU.activeMemory
        for i in 0 ..< iterations {
            let t0 = Date()
            let out = try session.features(imageURL: url, size: size)
            eval(out.patchTokens)
            let dt = Date().timeIntervalSince(t0) * 1000
            times.append(dt)
            if i == 0 || i == iterations - 1 {
                print(String(format: "  iter %2d  %.1f ms  active=%@  peak=%@",
                             i, dt, mb(MLX.GPU.activeMemory), mb(MLX.GPU.peakMemory)))
            }
        }

        let sorted = times.sorted()
        let median = sorted[sorted.count / 2]
        let mean = times.reduce(0, +) / Double(times.count)
        let endActive = MLX.GPU.activeMemory
        let drift = endActive - startActive

        print("")
        print(String(format: "median %.1f ms  mean %.1f ms  (%d iters, size %d, %@)",
                     median, mean, iterations, size, dtype))
        print("active memory drift over run: \(mb(drift)) (≈0 ⇒ no leak)")
        print("peak footprint: \(mb(MLX.GPU.peakMemory)) (reusable buffer cache, not a leak)")
    }
}
