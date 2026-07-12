// swift-tools-version: 6.0
import PackageDescription

let mlxDeps: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
    .product(name: "MLXLinalg", package: "mlx-swift"),
]

let package = Package(
    name: "MLXLingBotVision",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "MLXLingBotVision", targets: ["MLXLingBotVision"]),
        .executable(name: "lbv-tool", targets: ["lbv-tool"]),
        .executable(name: "lbv-bench", targets: ["lbv-bench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "MLXLingBotVision",
            dependencies: mlxDeps,
            path: "Sources/MLXLingBotVision"
        ),
        .executableTarget(
            name: "lbv-tool",
            dependencies: ["MLXLingBotVision", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/lbv-tool"
        ),
        .executableTarget(
            name: "lbv-bench",
            dependencies: ["MLXLingBotVision", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/lbv-bench"
        ),
        .testTarget(
            name: "MLXLingBotVisionTests",
            dependencies: ["MLXLingBotVision"],
            path: "Tests/MLXLingBotVisionTests",
            resources: [.process("Fixtures")]
        ),
    ]
)

// Pull in swift-docc-plugin only when generating documentation, so normal
// builds and downstream consumers don't have to resolve an extra dependency.
if Context.environment["SPI_GENERATE_DOCS"] == "1"  // Swift Package Index
    || Context.environment["BUILD_DOC"] == "1"       // local / CI
{
    package.dependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3")
    )
}
