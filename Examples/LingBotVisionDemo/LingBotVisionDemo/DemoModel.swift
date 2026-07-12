import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXLingBotVision

/// SwiftUI-side driver for the demo. It owns only presentation concerns:
/// file selection, the detached run `Task`, `autoreleasepool`, MainActor hops,
/// and `CGImage` conversion. All model I/O and compute live in the shared
/// ``LingBotVisionSession`` — the exact same driver the `lbv-tool` CLI uses.
@MainActor
@Observable
final class DemoModel {
    var modelDirectory: URL?
    var imageURL: URL?
    var inputImage: CGImage?
    var pcaImage: CGImage?
    var status = "Choose a converted model folder and an image, then Run."
    var isRunning = false
    var useFloat16 = true
    var imageSize = defaultImageSize

    /// Cached session, reused across runs while the model folder + dtype are
    /// unchanged (avoids reloading ~1 GB of weights on every Run).
    private var session: LingBotVisionSession?
    private var sessionKey: String?

    var canRun: Bool { modelDirectory != nil && imageURL != nil && !isRunning }

    func setModelDirectory(_ url: URL) {
        modelDirectory = url
        session = nil
        sessionKey = nil
        status = "Model: \(url.lastPathComponent)"
    }

    func setImage(_ url: URL) {
        imageURL = url
        inputImage = Self.loadCGImage(url)
        status = inputImage == nil
            ? "Could not read \(url.lastPathComponent)"
            : "Image: \(url.lastPathComponent)"
    }

    func run() {
        // Enforce the session's single-writer contract at the source, not just
        // via the disabled Run button: a fast double-trigger (Return + click)
        // must not launch a second run against the same cached session.
        guard !isRunning, let modelDirectory, let imageURL else { return }
        isRunning = true

        let dtype: DType = useFloat16 ? .float16 : .float32
        let size = imageSize
        let key = "\(modelDirectory.path)|fp16=\(useFloat16)"
        let cached = (sessionKey == key) ? session : nil
        status = cached == nil ? "Loading model…" : "Running…"

        // Structured: this outer Task inherits @MainActor, so applying the
        // result to `self` below is race-free (no `weak self` juggling). The
        // heavy MLX work runs inside an awaited `Task.detached`, which captures
        // only Sendable locals — never `self` — and executes OFF the main actor
        // so the UI stays responsive.
        Task {
            do {
                let result = try await Task.detached {
                    let dirScope = modelDirectory.startAccessingSecurityScopedResource()
                    let imgScope = imageURL.startAccessingSecurityScopedResource()
                    defer {
                        if dirScope { modelDirectory.stopAccessingSecurityScopedResource() }
                        if imgScope { imageURL.stopAccessingSecurityScopedResource() }
                    }
                    let session = try cached
                        ?? LingBotVisionSession.load(
                            SessionConfig(modelDirectory: modelDirectory, dtype: dtype))
                    let start = Date()
                    let image = try autoreleasepool {
                        try session.pcaCGImage(imageURL: imageURL, size: size, upscale: true)
                    }
                    return InferResult(
                        session: session, image: image,
                        milliseconds: Date().timeIntervalSince(start) * 1000)
                }.value

                session = result.session
                sessionKey = key
                pcaImage = result.image
                status = String(
                    format: "Done — %d×%d px in %.0f ms",
                    result.image.width, result.image.height, result.milliseconds)
            } catch {
                status = "Error: \(error)"
            }
            isRunning = false
        }
    }

    /// Decode a user-selected image into an eagerly-materialized `CGImage`.
    ///
    /// Reads the whole file into memory *while the security scope is held* and
    /// decodes from that `Data`. `CGImageSourceCreateWithURL` would defer the
    /// file read until SwiftUI renders the image — by then the sandbox scope is
    /// gone and the lazy `open` fails with EPERM ("Operation not permitted").
    private static func loadCGImage(_ url: URL) -> CGImage? {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let data = try? Data(contentsOf: url),
            let src = CGImageSourceCreateWithData(data as CFData, nil)
        else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceShouldCache: true,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        return CGImageSourceCreateImageAtIndex(src, 0, options as CFDictionary)
    }
}

/// Result handed back from the off-main inference task. `@unchecked Sendable`:
/// its `CGImage` (not formally `Sendable`) is freshly created inside the task
/// and never mutated or shared afterward, and `LingBotVisionSession` carries
/// its own single-writer contract.
private struct InferResult: @unchecked Sendable {
    let session: LingBotVisionSession
    let image: CGImage
    let milliseconds: Double
}
