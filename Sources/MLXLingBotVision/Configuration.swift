import Foundation

/// Architecture parameters for a LingBot-Vision backbone.
///
/// Mirrors the fields written by `scripts/convert.py` into `config.json`
/// (already camelCase, so no `CodingKeys` remapping is needed). One config
/// fully determines the module tree and weight shapes.
public struct LingBotVisionConfiguration: Codable, Sendable {
    public var arch: String
    public var patchSize: Int
    public var inChannels: Int
    public var embedDim: Int
    public var depth: Int
    public var numHeads: Int
    public var ffnRatio: Float
    public var imgSize: Int
    public var qkvBias: Bool
    public var projBias: Bool
    public var ffnBias: Bool
    public var layerscaleInit: Float
    public var nStorageTokens: Int
    public var normLayer: String
    public var normEps: Float
    public var ffnLayer: String
    public var ropeBase: Float
    public var maskKBias: Bool

    /// FFN hidden width. For the plain MLP this is `int(dim * ffn_ratio)`; the
    /// SwiGLU branch derives its own `2/3`-aligned width from this value.
    public var intermediateSize: Int { Int(Float(embedDim) * ffnRatio) }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        arch = try c.decode(String.self, forKey: .arch)
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        embedDim = try c.decode(Int.self, forKey: .embedDim)
        depth = try c.decode(Int.self, forKey: .depth)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        ffnRatio = try c.decode(Float.self, forKey: .ffnRatio)
        imgSize = try c.decodeIfPresent(Int.self, forKey: .imgSize) ?? 512
        qkvBias = try c.decode(Bool.self, forKey: .qkvBias)
        projBias = try c.decode(Bool.self, forKey: .projBias)
        ffnBias = try c.decode(Bool.self, forKey: .ffnBias)
        layerscaleInit = try c.decodeIfPresent(Float.self, forKey: .layerscaleInit) ?? 1.0e-5
        nStorageTokens = try c.decode(Int.self, forKey: .nStorageTokens)
        normLayer = try c.decodeIfPresent(String.self, forKey: .normLayer) ?? "layernorm"
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1.0e-6
        ffnLayer = try c.decodeIfPresent(String.self, forKey: .ffnLayer) ?? "mlp"
        ropeBase = try c.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 100.0
        maskKBias = try c.decodeIfPresent(Bool.self, forKey: .maskKBias) ?? false
    }

    public static func load(from url: URL) throws -> LingBotVisionConfiguration {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(LingBotVisionConfiguration.self, from: data)
    }
}
