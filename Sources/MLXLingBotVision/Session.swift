import CoreGraphics
import Foundation
import MLX

/// Session-construction knobs (what to load). The per-image input size is a
/// request parameter on ``LingBotVisionSession/features(imageURL:size:)`` — a
/// session loads the weights once and can run images of varying resolution —
/// so it is deliberately not here. Defaults work for both the CLI and the app.
public struct SessionConfig: Sendable {
    /// Converted model directory (`config.json` + `model.safetensors`).
    public var modelDirectory: URL
    /// Compute dtype for the weights.
    public var dtype: DType

    public init(modelDirectory: URL, dtype: DType = .float16) {
        self.modelDirectory = modelDirectory
        self.dtype = dtype
    }
}

/// Default square input size (snapped to a multiple of the patch size).
public let defaultImageSize = 512

/// Result of running one image through the backbone: the raw token output plus
/// its PCA visualization.
public struct PCAResult {
    /// Row-major RGB in `[0, 1]`, shape `[gridH, gridW, 3]`.
    public let rgb: [Float]
    public let gridH: Int
    public let gridW: Int
    public let output: LingBotVisionOutput
}

/// The shared, reusable inference driver consumed identically by the CLI
/// (`lbv-tool` / `lbv-bench`) and the SwiftUI demo app.
///
/// It owns everything that is *not* presentation: model I/O, preprocessing,
/// the forward pass, PCA, and the per-step lazy-graph evaluation. The
/// frontends own only their loop, cadence, and presentation surface (stdout /
/// PNG for the CLI; `Task.detached` + `MainActor` + `CGImage` for the GUI).
///
/// - Important: The contract is *single-writer-at-a-time*. It is
///   `@unchecked Sendable` on that basis — the CLI is single-threaded and the
///   app's detached task is the sole caller of ``features(imageURL:size:)`` /
///   ``pca(imageURL:size:)`` while the UI only reads snapshots it hands back.
///   Do not step one session from two concurrent callers.
///
///   The escape hatch exists only because the wrapped ``LingBotVisionTransformer``
///   (an MLXNN `Module`) is not `Sendable`. If mlx-swift ever makes `Module`
///   `Sendable`, drop `@unchecked` and let conformance be inferred. Until then,
///   a second concurrent stepping consumer must build its own session or a
///   thread-safe wrapper.
public final class LingBotVisionSession: @unchecked Sendable {
    public let config: SessionConfig
    public let model: LingBotVisionTransformer

    public var patchSize: Int { model.patchSize }

    private init(config: SessionConfig, model: LingBotVisionTransformer) {
        self.config = config
        self.model = model
    }

    /// Build a ready-to-run session from a converted model directory.
    public static func load(_ config: SessionConfig) throws -> LingBotVisionSession {
        let model = try LingBotVisionTransformer.loadPretrained(directory: config.modelDirectory, dtype: config.dtype)
        return LingBotVisionSession(config: config, model: model)
    }

    /// Run the backbone on one image, returning the full token output. The
    /// per-step `eval` (materializing the lazy graph) lives here so neither
    /// frontend can forget it and drift.
    public func features(imageURL: URL, size: Int = defaultImageSize) throws -> LingBotVisionOutput {
        let loaded = try ImageProcessor.load(url: imageURL, size: size, patchSize: patchSize)
        let out = model(loaded.input.asType(model.clsTokenDType))
        eval(out.patchTokens, out.clsToken, out.storageTokens)
        return out
    }

    /// Run the backbone and project its patch tokens to an RGB PCA map.
    public func pca(imageURL: URL, size: Int = defaultImageSize) throws -> PCAResult {
        let out = try features(imageURL: imageURL, size: size)
        let rgb = PatchPCA.rgb(patchTokens: out.patchTokens, h: out.gridH, w: out.gridW)
        return PCAResult(rgb: rgb, gridH: out.gridH, gridW: out.gridW, output: out)
    }

    /// Run the backbone and return the PCA map as a `CGImage`, optionally
    /// nearest-neighbor upscaled back to the (snapped) input resolution.
    public func pcaCGImage(imageURL: URL, size: Int = defaultImageSize, upscale: Bool = true) throws -> CGImage {
        let result = try pca(imageURL: imageURL, size: size)
        let target = upscale ? ImageProcessor.snap(size, patchSize: patchSize) : nil
        return try makeCGImage(rgb: result.rgb, h: result.gridH, w: result.gridW, upscaleTo: target)
    }
}

extension LingBotVisionTransformer {
    /// Compute dtype of the loaded weights (inferred from the cls token).
    var clsTokenDType: DType { clsToken.dtype }
}
