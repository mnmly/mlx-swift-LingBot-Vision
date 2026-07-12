# CLAUDE.md — mlx-swift-LingBot-Vision

## Shared driver

Both frontends — the `lbv-tool` / `lbv-bench` CLIs and the SwiftUI app in
`Examples/LingBotVisionDemo` — drive the **same** library-side type
``LingBotVisionSession`` (`Sources/MLXLingBotVision/Session.swift`). All model
I/O, preprocessing, the forward pass, PCA, and per-step `MLX.eval` live in the
Session. The frontends own only their loop, cadence, and presentation surface
(stdout / PNG for the CLI; `Task.detached` + `autoreleasepool` + `MainActor` +
`CGImage` for the app).

When adding shared compute, put it in the Session — never copy-paste bootstrap
between `Sources/lbv-tool/main.swift` and `Examples/.../DemoModel.swift`. Keep
the Session actor-agnostic (no `@MainActor`) so the CLI can use it synchronously.

## Build / test

Use `xcodebuild` (not `swift build`/`swift test` — they skip the Metal
toolchain MLX needs):

```bash
xcodebuild -scheme MLXLingBotVision-Package -destination 'platform=macOS' -derivedDataPath .xcdd build
xcodebuild -scheme MLXLingBotVision-Package -destination 'platform=macOS' test
```

## Documentation

`MLXLingBotVision` ships DocC-generated reference docs (see
`Sources/MLXLingBotVision/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public symbols are published** to the static site.
Because the package depends on mlx-swift (Metal), the docs build uses
`xcodebuild docbuild` + `docc process-archive`, not the SwiftPM plugin.

When you add or modify a `public` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if the
  *why* is non-obvious. Don't restate the signature.
- Document each parameter with `- Parameter name:` using the **internal** name
  when there's an external label (DocC warns otherwise).
- Cross-reference related symbols with signature-sensitive double-backtick
  links, e.g. `` ``LingBotVisionSession/pca(imageURL:size:)`` ``.
- File new top-level symbols under a `## Topics` group (by *user task*) in
  `Sources/MLXLingBotVision/Documentation.docc/MLXLingBotVision.md`.

Verify before declaring documentation work done:

```bash
Scripts/build_docs.sh
```

Expect exit 0 and no new "doesn't exist at" / "external name used to document
parameter" warnings attributable to your changes (preexisting mlx-swift catalog
warnings are out of scope).
