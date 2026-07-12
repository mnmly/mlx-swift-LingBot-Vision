import Foundation
import MLX
import XCTest

@testable import MLXLingBotVision

final class MLXLingBotVisionTests: XCTestCase {
    /// Directory holding the converted vit-large model (config.json +
    /// model.safetensors). Override with LBV_MODEL_DIR; defaults to
    /// ~/lingbot-vision-mlx/vit-large produced by scripts/convert.py.
    private func modelDirectory() throws -> URL {
        let env = ProcessInfo.processInfo.environment["LBV_MODEL_DIR"]
        let path = env ?? NSString(string: "~/lingbot-vision-mlx/vit-large").expandingTildeInPath
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else {
            throw XCTSkip("converted model not found at \(url.path); set LBV_MODEL_DIR")
        }
        return url
    }

    private func fixture() throws -> [String: MLXArray] {
        guard let url = Bundle.module.url(forResource: "vitl_parity", withExtension: "safetensors") else {
            throw XCTSkip("parity fixture not bundled")
        }
        return try loadArrays(url: url)
    }

    func testNumericalParity() throws {
        let model = try LingBotVisionTransformer.loadPretrained(directory: try modelDirectory(), dtype: .float32)
        let fx = try fixture()

        let input = fx["input_nhwc"]!  // [1, H, W, 3]
        let refPatch = fx["patch_tokens"]!.asType(.float32)  // [1, N, D]
        let refCls = fx["cls_token"]!.asType(.float32)  // [1, D]

        let out = model(input.asType(.float32))
        eval(out.patchTokens, out.clsToken)

        assertClose(out.patchTokens, refPatch, name: "patch_tokens", maxAbsTol: 2e-2, meanAbsTol: 1e-3)
        assertClose(out.clsToken, refCls, name: "cls_token", maxAbsTol: 2e-2, meanAbsTol: 1e-3)
    }

    /// The Session contract: load a session and run one image end-to-end
    /// (preprocess -> forward -> PCA). Both frontends inherit correctness from
    /// this — it exercises the exact same driver.
    func testSessionRun() throws {
        guard let imageURL = Bundle.module.url(forResource: "example", withExtension: "png") else {
            throw XCTSkip("example image not bundled")
        }
        let config = SessionConfig(modelDirectory: try modelDirectory(), dtype: .float32)
        let session = try LingBotVisionSession.load(config)

        let size = 256
        let result = try session.pca(imageURL: imageURL, size: size)
        let grid = size / session.patchSize

        XCTAssertEqual(result.gridH, grid)
        XCTAssertEqual(result.gridW, grid)
        XCTAssertEqual(result.rgb.count, grid * grid * 3)
        XCTAssertEqual(result.output.patchTokens.dim(1), grid * grid)
        // PCA output is a normalized [0, 1] RGB buffer.
        XCTAssertGreaterThanOrEqual(result.rgb.min() ?? -1, 0)
        XCTAssertLessThanOrEqual(result.rgb.max() ?? 2, 1)
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, name: String,
        maxAbsTol: Float, meanAbsTol: Float
    ) {
        let diff = MLX.abs(a - b)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)

        // Cosine similarity of the flattened tensors.
        let af = a.reshaped(-1).asType(.float32)
        let bf = b.reshaped(-1).asType(.float32)
        let cos = (MLX.sum(af * bf) / (MLX.sqrt(MLX.sum(af * af)) * MLX.sqrt(MLX.sum(bf * bf)))).item(Float.self)

        print("[parity] \(name): maxAbs=\(maxAbs) meanAbs=\(meanAbs) cosine=\(cos)")
        XCTAssertLessThan(maxAbs, maxAbsTol, "\(name) maxAbs \(maxAbs) exceeds \(maxAbsTol)")
        XCTAssertLessThan(meanAbs, meanAbsTol, "\(name) meanAbs \(meanAbs) exceeds \(meanAbsTol)")
        XCTAssertGreaterThan(cos, 0.9999, "\(name) cosine \(cos) too low")
    }
}
