import ArgumentParser
import Foundation
import MLX
import MLXLingBotVision

@main
struct LBVTool: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lbv-tool",
        abstract: "Run a LingBot-Vision backbone and save a patch-token PCA visualization.")

    @Option(name: .long, help: "Converted model directory (config.json + model.safetensors).")
    var model: String

    @Option(name: .long, help: "Input image path.")
    var image: String

    @Option(name: .long, help: "Output PCA PNG path.")
    var out: String = "pca.png"

    @Option(name: .long, help: "Square input size (snapped to a multiple of patch size).")
    var size: Int = 512

    @Option(name: .long, help: "Compute dtype: float32 or float16.")
    var dtype: String = "float32"

    func run() throws {
        let dt: DType = dtype == "float16" ? .float16 : .float32
        let config = SessionConfig(modelDirectory: URL(fileURLWithPath: model), dtype: dt)
        let session = try LingBotVisionSession.load(config)

        let result = try session.pca(imageURL: URL(fileURLWithPath: image), size: size)
        let snapped = ImageProcessor.snap(size, patchSize: session.patchSize)
        try writePNG(rgb: result.rgb, h: result.gridH, w: result.gridW, to: URL(fileURLWithPath: out), upscaleTo: snapped)
        print("[lbv-tool] grid=\(result.gridH)x\(result.gridW) -> \(out) (upscaled to \(snapped)x\(snapped))")
    }
}
