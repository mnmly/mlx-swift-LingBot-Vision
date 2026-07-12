import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXLingBotVision
import os

/// Diagnostic log for the Hugging Face download flow — visible in Xcode's
/// console and Console.app (filter subsystem "LingBotVisionDemo").
private let downloadLog = Logger(subsystem: "LingBotVisionDemo", category: "download")

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
    var status = "Download a model or choose a folder, pick an image, then Run."
    var isRunning = false
    var useFloat16 = true
    var imageSize = defaultImageSize

    /// Fraction (0…1) downloaded while pulling weights from Hugging Face.
    var downloadProgress: Double = 0
    var isDownloading = false
    /// True until the large file reports a total size — the bar shows an
    /// indeterminate "working" state so activity is visible from the first byte.
    var downloadIndeterminate = false
    var downloadedBytes: Int64 = 0

    /// Category of the current status line, so the UI can pick an icon + color
    /// (status / completion / error feedback).
    enum StatusKind { case info, success, error }
    var statusKind: StatusKind = .info

    /// Bumped on every new PCA result so the view can replay its "materialize"
    /// transition even when one result replaces another.
    var resultToken = 0

    /// Cached session, reused across runs while the model folder + dtype are
    /// unchanged (avoids reloading ~1 GB of weights on every Run).
    private var session: LingBotVisionSession?
    private var sessionKey: String?

    var canRun: Bool { modelDirectory != nil && imageURL != nil && !isRunning && !isDownloading }

    /// Hugging Face repo holding the converted MLX weights, and the two files
    /// the Swift loader needs from it.
    static let hubRepo = "mnmly/lingbot-vision-vit-large-mlx"
    private static let hubFiles = ["config.json", "model.safetensors"]

    func setModelDirectory(_ url: URL) {
        modelDirectory = url
        session = nil
        sessionKey = nil
        status = "Model: \(url.lastPathComponent)"
        statusKind = .info
    }

    func setImage(_ url: URL) {
        imageURL = url
        inputImage = Self.loadCGImage(url)
        if inputImage == nil {
            status = "Could not read \(url.lastPathComponent)"
            statusKind = .error
        } else {
            status = "Image: \(url.lastPathComponent)"
            statusKind = .info
        }
    }

    /// Download the published MLX weights from Hugging Face into a subfolder of
    /// `parentDir`, then select that folder as the active model.
    ///
    /// `parentDir` must come from a file dialog (e.g. defaulting to
    /// `~/.cache/huggingface`): the App Sandbox only grants write access to a
    /// user-picked location, so the picker is what makes the download possible.
    func downloadFromHub(into parentDir: URL) {
        guard !isDownloading, !isRunning else { return }
        isDownloading = true
        downloadProgress = 0
        downloadedBytes = 0
        downloadIndeterminate = true
        status = "Downloading model from Hugging Face…"
        statusKind = .info

        let repo = Self.hubRepo
        let files = Self.hubFiles
        downloadLog.info("start → \(parentDir.path, privacy: .public)")

        // Outer Task inherits @MainActor, so the writes to `self` after the
        // await are race-free. The network + disk I/O runs in a detached task
        // that captures only Sendable values (the URL and this Sendable actor).
        Task {
            do {
                let modelDir = try await Task.detached { () -> URL in
                    let scoped = parentDir.startAccessingSecurityScopedResource()
                    defer { if scoped { parentDir.stopAccessingSecurityScopedResource() } }

                    let dir = parentDir.appendingPathComponent(
                        "lingbot-vision-vit-large-mlx", isDirectory: true)
                    try FileManager.default.createDirectory(
                        at: dir, withIntermediateDirectories: true)
                    downloadLog.info("dir ready → \(dir.path, privacy: .public)")

                    let base = "https://huggingface.co/\(repo)/resolve/main/"
                    for name in files {
                        guard let url = URL(string: base + name) else { throw URLError(.badURL) }
                        downloadLog.info("GET \(name, privacy: .public)")
                        // config.json is tiny; drive the progress bar off the
                        // large model.safetensors only.
                        let reportsProgress = (name == "model.safetensors")
                        try await downloadFile(from: url, to: dir.appendingPathComponent(name)) { written, total in
                            guard reportsProgress else { return }
                            Task { @MainActor in
                                self.downloadedBytes = written
                                if total > 0 {
                                    self.downloadIndeterminate = false
                                    self.downloadProgress = Double(written) / Double(total)
                                }
                            }
                        }
                        downloadLog.info("done \(name, privacy: .public)")
                    }
                    return dir
                }.value

                setModelDirectory(modelDir)
                status = "Downloaded — ready. Choose an image and Run."
                statusKind = .success
                downloadLog.info("complete")
            } catch {
                status = "Download failed: \(error.localizedDescription)"
                statusKind = .error
                downloadLog.error("failed: \(error.localizedDescription, privacy: .public)")
            }
            isDownloading = false
        }
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
        statusKind = .info

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
                resultToken += 1
                status = String(
                    format: "Done — %d×%d px in %.0f ms",
                    result.image.width, result.image.height, result.milliseconds)
                statusKind = .success
            } catch {
                status = "Error: \(error)"
                statusKind = .error
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

/// Download `url` to `dest`, reporting fractional progress. Uses a download
/// task (streamed to a temp file, no in-memory buffering) so the ~1 GB
/// `model.safetensors` never has to fit in RAM.
private func downloadFile(
    from url: URL, to dest: URL,
    onProgress: @escaping @Sendable (_ written: Int64, _ total: Int64) -> Void
) async throws {
    let delegate = DownloadProgress(onProgress)
    let (tempURL, response) = try await URLSession.shared.download(
        for: URLRequest(url: url), delegate: delegate)
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
        throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode)"])
    }
    try? FileManager.default.removeItem(at: dest)
    try FileManager.default.moveItem(at: tempURL, to: dest)
}

/// Task-scoped delegate that forwards download progress as (bytesWritten,
/// totalExpected); `total` is `-1` when the server doesn't send a length.
/// `@unchecked Sendable`: its only stored property is an immutable `@Sendable`
/// closure, and URLSession delivers the callback on its own serial queue.
private final class DownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Int64, Int64) -> Void

    init(_ onProgress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    // Required by the protocol; the async `download(for:)` returns the file URL,
    // so nothing to do here.
    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {}
}
