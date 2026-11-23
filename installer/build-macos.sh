#!/usr/bin/env bash
# Build dist/Star Effects Lab <ver>.pkg (arm64, unsigned) from a clean checkout.
# Needs: uv, cmake, Xcode CLT, Photoshop SDK in photoshop/sdk/mac.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
stage="$here/build"
app="$stage/Star Effects Lab.app"
res="$app/Contents/Resources"
ver="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$repo/pyproject.toml" | head -1)"
bundle_id="com.starlab.StarEffectsLab"   # branding.py APPLICATION_ID is not reverse-DNS
[ "$(uname -m)" = arm64 ] || { echo "arm64 only for now" >&2; exit 1; }

cd "$repo"
rm -rf "$stage"; mkdir -p "$stage" "$app/Contents/MacOS" "$res"

echo "== wheel"
rm -f dist/star_effects_lab-*.whl
uv build --wheel
wheel="$(ls dist/star_effects_lab-*.whl)"

echo "== bundled Python"
uv python install 3.14
pyhome="$("$(uv python find --no-project --managed-python 3.14)" -c 'import sys; print(sys.base_prefix)')"
[ -f "$pyhome/lib/libpython3.14.dylib" ] || { echo "libpython missing in $pyhome" >&2; exit 1; }
ditto --noextattr --noqtn "$pyhome" "$res/python"
install_name_tool -id @rpath/libpython3.14.dylib "$res/python/lib/libpython3.14.dylib"
rm -f "$res/python/EXTERNALLY-MANAGED" "$res/python/lib/python3.14/EXTERNALLY-MANAGED"
# The app only uses PySide6 essentials: skip addons/webengine/pdf (~330 MB). The export is the full
# locked closure, so --no-deps installs exactly the locked versions.
uv export --no-dev --no-hashes --no-emit-project 2>/dev/null \
    | grep -vi '^pyside6[-_]\(addons\|webengine\|pdf\)' > "$stage/requirements.txt"
uv pip install --python "$res/python/bin/python3" --no-deps --break-system-packages \
    -r "$stage/requirements.txt" "$wheel"
# Qt GUI tools shipped inside PySide6 (nested .app bundles); not needed.
rm -rf "$res"/python/lib/python3.14/site-packages/PySide6/{Assistant,Designer,Linguist}.app
find "$res/python" -name __pycache__ -type d -prune -exec rm -rf {} +
# /Applications is read-only for users: ship bytecode so imports stay fast and the signature stays valid.
"$res/python/bin/python3" -I -m compileall -q -j 4 "$res/python/lib" >/dev/null || true  # stdlib ships deliberately broken test files

echo "== launcher"
cmake -S "$here/launcher" -B "$stage/launcher" -DPYTHON_HOME="$res/python" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
cmake --build "$stage/launcher"
cp "$stage/launcher/Star Effects Lab" "$app/Contents/MacOS/"

echo "== bundle metadata"
sed -e "s/@VERSION@/$ver/g" -e "s/@BUNDLE_ID@/$bundle_id/g" "$here/Info.plist.in" > "$app/Contents/Info.plist"
iconset="$stage/application.iconset"; mkdir "$iconset"
png="$repo/src/starlab/presentation/qt/assets/application.png"
for s in 16 32 128 256 512; do
    sips -z $s $s "$png" --out "$iconset/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) "$png" --out "$iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$res/application.icns"
codesign --force --deep -s - "$app"
codesign --verify --deep --strict "$app"

echo "== plug-in"
"$repo/photoshop/build.sh" --app "/Applications/Star Effects Lab.app/Contents/MacOS/Star Effects Lab"

echo "== pkg"
pay="$stage/payload"; mkdir -p "$pay/app" "$pay/plugin/Star Effects Lab"
ditto --noextattr --noqtn "$app" "$pay/app/Star Effects Lab.app"
ditto --noextattr --noqtn "$repo/photoshop/build/Star Effects Lab" "$pay/plugin/Star Effects Lab"
pkgbuild --analyze --root "$pay/app" "$stage/app.plist" >/dev/null
/usr/libexec/PlistBuddy -c "Add :0:BundleIsRelocatable bool false" "$stage/app.plist"
pkgbuild --root "$pay/app" --component-plist "$stage/app.plist" --identifier "$bundle_id.app" \
    --version "$ver" --install-location /Applications "$stage/app.pkg"
pkgbuild --root "$pay/plugin" --identifier "$bundle_id.plugin" --version "$ver" \
    --install-location "/Library/Application Support/Adobe/Plug-Ins/CC" "$stage/plugin.pkg"
sed "s/@VERSION@/$ver/g" "$here/distribution.xml.in" > "$stage/distribution.xml"
out="$repo/dist/Star Effects Lab $ver.pkg"
productbuild --distribution "$stage/distribution.xml" --package-path "$stage" "$out"
echo; echo "Built: $out ($(du -h "$out" | cut -f1))"
