import CoreGraphics
import Foundation
import ImageIO
import MLX

#if canImport(UniformTypeIdentifiers)
import UniformTypeIdentifiers
#endif

/// ImageNet preprocessing for LingBot-Vision.
///
/// Loads an image with ImageIO/CoreGraphics (Apple-native, no third-party
/// decoders), resizes to a square multiple of the patch size, and normalizes
/// with ImageNet statistics into an MLX NHWC batch `[1, size, size, 3]`.
public enum ImageProcessor {
    public static let mean: [Float] = [0.485, 0.456, 0.406]
    public static let std: [Float] = [0.229, 0.224, 0.225]

    public struct Loaded {
        /// Normalized input, `[1, size, size, 3]` (NHWC).
        public let input: MLXArray
        /// Resized RGB pixels in `[0, 1]`, `[size, size, 3]` — for visualization.
        public let rgb: MLXArray
        public let size: Int
    }

    /// Snap `size` down to a multiple of `patchSize` (never below one patch).
    public static func snap(_ size: Int, patchSize: Int) -> Int {
        max(patchSize, (size / patchSize) * patchSize)
    }

    public static func load(url: URL, size requested: Int, patchSize: Int) throws -> Loaded {
        let size = snap(requested, patchSize: patchSize)
        let pixels = try rgbPixels(url: url, size: size)  // [size*size*3] Float in [0,1], row-major RGB

        let rgb = MLXArray(pixels, [size, size, 3])
        let meanA = MLXArray(mean, [1, 1, 3])
        let stdA = MLXArray(std, [1, 1, 3])
        let normalized = ((rgb - meanA) / stdA).expandedDimensions(axis: 0)  // [1, size, size, 3]
        return Loaded(input: normalized, rgb: rgb, size: size)
    }

    /// Decode + square-resize to `size×size`, returning row-major RGB floats in [0, 1].
    private static func rgbPixels(url: URL, size: Int) throws -> [Float] {
        // Read the file eagerly and decode from in-memory Data rather than
        // letting CGImageSource lazily map the file. Under App Sandbox the
        // caller holds the security scope only for the duration of this call;
        // a lazy read could fire later (outside the scope) and fail with EPERM.
        guard let data = try? Data(contentsOf: url),
            let src = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            throw LingBotVisionError.imageDecodeFailed(url)
        }

        let bytesPerRow = size * 4
        var buffer = [UInt8](repeating: 0, count: size * size * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard
            let ctx = CGContext(
                data: &buffer,
                width: size, height: size,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw LingBotVisionError.imageDecodeFailed(url)
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        var out = [Float](repeating: 0, count: size * size * 3)
        for i in 0 ..< (size * size) {
            out[i * 3 + 0] = Float(buffer[i * 4 + 0]) / 255.0
            out[i * 3 + 1] = Float(buffer[i * 4 + 1]) / 255.0
            out[i * 3 + 2] = Float(buffer[i * 4 + 2]) / 255.0
        }
        return out
    }
}

public enum LingBotVisionError: Error, CustomStringConvertible {
    case imageDecodeFailed(URL)
    case renderFailed

    public var description: String {
        switch self {
        case .imageDecodeFailed(let url): return "failed to decode image at \(url.path)"
        case .renderFailed: return "failed to render RGB buffer to a CGImage"
        }
    }
}
