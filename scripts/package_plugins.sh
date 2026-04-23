#!/usr/bin/env bash
# Package all plugins into a tar.gz archive for distribution.
# Usage:
#   ./scripts/package_plugins.sh [VERSION] [--with-models] [OUTPUT_DIR]
# Examples:
#   ./scripts/package_plugins.sh                     # plugins only, v0.1.0
#   ./scripts/package_plugins.sh 1.0.0               # plugins only, v1.0.0
#   ./scripts/package_plugins.sh 0.1.0 --with-models # include models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGINS_DIR="$PROJECT_DIR/plugins"

VERSION="0.1.0"
WITH_MODELS=false
OUTPUT_DIR="$PROJECT_DIR/dist"

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --with-models) WITH_MODELS=true; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)
            if [ "$VERSION" = "0.1.0" ] && [ "$WITH_MODELS" = false ]; then
                VERSION="$1"
            else
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

SUFFIX=""
if [ "$WITH_MODELS" = true ]; then
    SUFFIX="-with-models"
fi
ARCHIVE_NAME="video-scene-plugins-${VERSION}${SUFFIX}.tar.gz"

PLUGIN_DIRS=(
    insightface_plugin
    object_yolo_plugin
    scene_detect_plugin
    scene_understanding_plugin
    video_understanding_plugin
    embedding_plugin
    image_clip_plugin
)

SHARED_FILES=(
    plugin_sdk.py
    pyproject.toml
)

echo "Packaging video-scene-plugins v${VERSION}..."

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

for f in "${SHARED_FILES[@]}"; do
    cp "$PLUGINS_DIR/$f" "$STAGING/"
done

for dir in "${PLUGIN_DIRS[@]}"; do
    if [ -d "$PLUGINS_DIR/$dir" ]; then
        cp -r "$PLUGINS_DIR/$dir" "$STAGING/"
        find "$STAGING/$dir" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        echo "  Added: $dir"
    else
        echo "  Warning: $dir not found, skipping"
    fi
done

if [ "$WITH_MODELS" = true ]; then
    if [ -d "$PLUGINS_DIR/models" ]; then
        cp -r "$PLUGINS_DIR/models" "$STAGING/"
        find "$STAGING/models" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        echo "  Added: models/"
    else
        echo "  Warning: models/ not found, skipping"
    fi
fi

mkdir -p "$OUTPUT_DIR"
cd "$STAGING"
tar -czf "$OUTPUT_DIR/$ARCHIVE_NAME" .

SIZE=$(du -h "$OUTPUT_DIR/$ARCHIVE_NAME" | cut -f1)
echo ""
echo "Package created: $OUTPUT_DIR/$ARCHIVE_NAME ($SIZE)"
