#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="Luma Swarm Studio"
APP_BUNDLE="$PROJECT_DIR/dist/$APP_NAME.app"
APP_ARCHIVE="$PROJECT_DIR/dist/$APP_NAME.zip"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/luma-swarm-package.XXXXXX")"
STAGING_APP="$STAGING_DIR/$APP_NAME.app"
STAGING_ARCHIVE="$STAGING_DIR/$APP_NAME.zip"

cleanup() {
    rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

cd "$PROJECT_DIR"
cargo build --release --locked --offline

mkdir -p "$STAGING_APP/Contents/MacOS" "$STAGING_APP/Contents/Resources"
cp "$PROJECT_DIR/target/release/luma-swarm-studio" "$STAGING_APP/Contents/MacOS/luma-swarm-studio"
cp "$PROJECT_DIR/packaging/Info.plist" "$STAGING_APP/Contents/Info.plist"
cp "$PROJECT_DIR/assets/AppIcon.icns" "$STAGING_APP/Contents/Resources/AppIcon.icns"

chmod +x "$STAGING_APP/Contents/MacOS/luma-swarm-studio"
find "$STAGING_APP" -exec xattr -c {} \;
codesign --force --deep --sign - "$STAGING_APP" >/dev/null
codesign --verify --deep --strict "$STAGING_APP"
ditto -c -k --norsrc --noextattr --noqtn --keepParent "$STAGING_APP" "$STAGING_ARCHIVE"
ZIP_VERIFY_DIR="$STAGING_DIR/verify"
mkdir -p "$ZIP_VERIFY_DIR"
ditto -x -k "$STAGING_ARCHIVE" "$ZIP_VERIFY_DIR"
codesign --verify --deep --strict "$ZIP_VERIFY_DIR/$APP_NAME.app"

mkdir -p "$PROJECT_DIR/dist"
if [[ -e "$APP_BUNDLE" ]]; then
    rm -rf "$APP_BUNDLE"
fi
if [[ -e "$APP_ARCHIVE" ]]; then
    rm -f "$APP_ARCHIVE"
fi
mv "$STAGING_APP" "$APP_BUNDLE"
mv "$STAGING_ARCHIVE" "$APP_ARCHIVE"

# Documents may be backed by a file provider that attaches FinderInfo while
# the bundle is moved out of staging. Clear what is available, refresh the
# ad-hoc signature, and verify its complete code graph. The strict resource
# check is performed above on the clean ZIP extraction because file-provider
# metadata is outside the bundle's signed contents. Moving does not alter the
# signature created in staging, so do not re-sign after the provider attaches
# metadata.
find "$APP_BUNDLE" -exec xattr -c {} \;
codesign --verify --deep "$APP_BUNDLE"

echo "$APP_BUNDLE"
echo "$APP_ARCHIVE"
