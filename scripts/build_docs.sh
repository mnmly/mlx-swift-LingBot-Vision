#!/usr/bin/env bash
# Build a static DocC site for MLXLingBotVision into ./docs (GitHub Pages-ready).
#
# This package depends on mlx-swift, whose Metal kernels do NOT compile under
# `swift package generate-documentation` (SwiftPM's Metal path fails on
# steel_attention.metal). `xcodebuild docbuild` compiles Metal correctly, so we
# generate a .doccarchive with xcodebuild and then transform it for static
# hosting with `docc process-archive`. The swift-docc-plugin is still listed
# (gated) in Package.swift for Swift Package Index.
#
# Usage:
#   Scripts/build_docs.sh            # build ./docs
#   Scripts/build_docs.sh preview    # build, then open the generated index.html
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET="${TARGET:-MLXLingBotVision}"
SCHEME="${SCHEME:-MLXLingBotVision}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-mlx-swift-LingBot-Vision}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"
DERIVED="${DERIVED:-.xcdd-docs}"

MODE="build"
[[ "${1:-}" == "preview" ]] && MODE="preview"

echo ">> xcodebuild docbuild ($SCHEME)"
xcodebuild docbuild \
    -scheme "$SCHEME" \
    -destination 'generic/platform=macOS' \
    -derivedDataPath "$DERIVED" \
    OTHER_DOCC_FLAGS="--emit-digest" \
    | grep -E "Compiling|BUILD SUCCEEDED|BUILD FAILED|error:" || true

ARCHIVE="$(find "$DERIVED/Build/Products" -name "$TARGET.doccarchive" -type d | head -1)"
if [[ -z "$ARCHIVE" ]]; then
    echo "error: no $TARGET.doccarchive produced" >&2
    exit 1
fi
echo ">> archive: $ARCHIVE"

DOCC_BIN="$(xcrun --find docc)"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/$TARGET"

"$DOCC_BIN" process-archive transform-for-static-hosting "$ARCHIVE" \
    --output-path "$OUTPUT_DIR/$TARGET" \
    --hosting-base-path "$HOSTING_BASE_PATH/$TARGET"

# Top-level redirect so the Pages root URL lands on the module docs page
# instead of 404ing (the site itself lives under $OUTPUT_DIR/$TARGET/).
slug="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"
redirect_url="/${HOSTING_BASE_PATH}/${TARGET}/documentation/${slug}/"
cat > "$OUTPUT_DIR/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>${HOSTING_BASE_PATH}</title>
<meta http-equiv="refresh" content="0; url=${redirect_url}">
<link rel="canonical" href="${redirect_url}">
<p>Redirecting to <a href="${redirect_url}">${redirect_url}</a>.</p>
HTML

echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/$TARGET/index.html"

if [[ "$MODE" == "preview" ]]; then
    open "$OUTPUT_DIR/$TARGET/index.html"
fi
