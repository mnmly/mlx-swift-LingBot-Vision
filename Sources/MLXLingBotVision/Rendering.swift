import CoreGraphics
import Foundation
import ImageIO

#if canImport(UniformTypeIdentifiers)
import UniformTypeIdentifiers
#endif

/// Build a `CGImage` from a row-major `[h, w, 3]` RGB buffer in `[0, 1]`,
/// optionally nearest-neighbor upscaled to `targetSize × targetSize`. Shared by
/// both the file-writing CLI path and the in-memory GUI display path.
public func makeCGImage(rgb: [Float], h: Int, w: Int, upscaleTo targetSize: Int? = nil) throws -> CGImage {
    var bytes = [UInt8](repeating: 255, count: h * w * 4)
    for i in 0 ..< (h * w) {
        bytes[i * 4 + 0] = UInt8(min(max(rgb[i * 3 + 0], 0), 1) * 255)
        bytes[i * 4 + 1] = UInt8(min(max(rgb[i * 3 + 1], 0), 1) * 255)
        bytes[i * 4 + 2] = UInt8(min(max(rgb[i * 3 + 2], 0), 1) * 255)
    }
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
        let ctx = CGContext(
            data: &bytes, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
        let base = ctx.makeImage()
    else {
        throw LingBotVisionError.renderFailed
    }

    guard let target = targetSize, target != w || target != h else { return base }

    var up = [UInt8](repeating: 255, count: target * target * 4)
    guard
        let upCtx = CGContext(
            data: &up, width: target, height: target,
            bitsPerComponent: 8, bytesPerRow: target * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { throw LingBotVisionError.renderFailed }
    upCtx.interpolationQuality = .none
    upCtx.draw(base, in: CGRect(x: 0, y: 0, width: target, height: target))
    return upCtx.makeImage() ?? base
}

/// Write a row-major `[h, w, 3]` RGB buffer in `[0, 1]` to a PNG file.
public func writePNG(rgb: [Float], h: Int, w: Int, to url: URL, upscaleTo targetSize: Int? = nil) throws {
    let image = try makeCGImage(rgb: rgb, h: h, w: w, upscaleTo: targetSize)

    let type: CFString
    #if canImport(UniformTypeIdentifiers)
    type = UTType.png.identifier as CFString
    #else
    type = "public.png" as CFString
    #endif
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type, 1, nil) else {
        throw LingBotVisionError.renderFailed
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw LingBotVisionError.renderFailed
    }
}
