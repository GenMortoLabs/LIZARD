#!/usr/bin/env bash
# ai: The Android decoder's build steps; core/, gen/, out/ and .tools/ are liblizard's:
#   ./build.sh tools     slang 2026.22 into ../liblizard/.tools/slang, the shader-slang release build (Linux x86-64 or
#                        macOS arm64), its sha256 checked, unless it is there already (curl, tar)
#   ./build.sh gen       the GPU tree: the host's tool listing its recipes (build/linux/lizard_gpu_tree, built first), then ../liblizard/out/{setup,blobs,slang,spv},
#                        every kernel and the engine's own shaders by slangc, and ../liblizard/gpu/send_wgsl.mjs (the
#                        web encoder's WGSL, slangc's). Needs Node, ../liblizard/build/ob.mjs and
#                        ../liblizard/bindings/wasm/dist/lizard.wasm (liblizard's ./build.sh) and slang 2026.22 (tools)
#   ./build.sh ffmpeg    FFmpeg's decoding libraries for the phone (../liblizard/ffmpeg/build.sh android, the pinned release
#                        fetched and checked): ../liblizard/build/ffmpeg/android, the file camera's (core/cam/media.cpp);
#                        android, apk and install make it first where it is not there
#   ./build.sh linux     ../liblizard/core for this desktop: build/linux/lizard_gpu_check (the 4090, the iGPU, lavapipe)
#   ./build.sh android   ../liblizard/core for the phone with the NDK: build/android/lizard_gpu_check (arm64, API 29)
#   ./build.sh phone [cmd...]   the phone build and out/ pushed to /data/local/tmp/lizard over adb, then cmd run there
#                        (default: info); a selftest also pushes its frames and references from the lab's
#                        ../research/build, which is not published
#   ./build.sh apk       the receiver app (app/, Gradle): app/build/outputs/apk/debug/app-debug.apk, out/ as its assets
#   ./build.sh install   the app installed on the phone over adb, the camera granted (extra args go to Gradle)
set -euo pipefail
cd "$(dirname "$0")"
LIB=../liblizard; OUT=$LIB/out; RESEARCH=../research
NDK=${ANDROID_NDK_HOME:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.1.12297006}
PHONE=/data/local/tmp/lizard
step=${1:-}; shift || true
# ai: the phone's FFmpeg (the file camera's images and videos), made once
ffmpeg_libs() { [ -f $LIB/build/ffmpeg/android/jniLibs/arm64-v8a/libavformat.so ] || $LIB/ffmpeg/build.sh android; }
# ai: Slang for every module gen compiles (../liblizard/gen/gen.mjs SLANG; the kernels' source since 2026-11-24): a
# ai: module is named by its source, so the compiler is pinned; the release's own builds, each checked by the sha256 the
# ai: release states for it
SLANG=2026.22
slang() {
  local dir=$LIB/.tools/slang asset sum tmp got
  [ "$("$dir/bin/slangc" -version 2>&1 | head -1)" = "$SLANG" ] && return 0
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) asset=linux-x86_64-glibc-2.27.tar.gz; sum=e03168bb9214795d1b455755b76b569e82d9d3f6a416d02fd00602e268a57bb2 ;;
    Darwin-arm64) asset=macos-aarch64.tar.gz; sum=885b5bd9a79a1ae81fdb92212f5f7c785001f870d45c39f9bb5df01339db3fea ;;
    *) echo "slang $SLANG: no release build for $(uname -sm) here; put slang $SLANG at $dir (bin/slangc)" >&2; return 1 ;;
  esac
  tmp=$(mktemp -d)
  curl -fsSL --retry 3 -o "$tmp/slang.tar.gz" "https://github.com/shader-slang/slang/releases/download/v$SLANG/slang-$SLANG-$asset"
  got=$( (sha256sum "$tmp/slang.tar.gz" 2>/dev/null || shasum -a 256 "$tmp/slang.tar.gz") | cut -d' ' -f1)
  if [ "$got" != "$sum" ]; then echo "slang-$SLANG-$asset: sha256 $got, the release's is $sum" >&2; rm -rf "$tmp"; return 1; fi
  rm -rf "$dir"; mkdir -p "$dir"
  tar xzf "$tmp/slang.tar.gz" -C "$dir"
  rm -rf "$tmp"
  echo "slang $("$dir/bin/slangc" -version 2>&1 | head -1) in $dir"
}
case $step in
  tools)
    slang
    ;;
  gen)
    # ai: the host's tool that lists the recipes gen compiles (core/tools/gpu_tree.cpp), built first
    cmake -S $LIB/core -B build/linux -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
    cmake --build build/linux --target lizard_gpu_tree -j"$(nproc)"
    node $LIB/gen/gen.mjs "$@"
    ;;
  ffmpeg)
    $LIB/ffmpeg/build.sh android
    ;;
  linux)
    cmake -S $LIB/core -B build/linux -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
    cmake --build build/linux -j"$(nproc)"
    ;;
  android)
    ffmpeg_libs
    cmake -S $LIB/core -B build/android -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build build/android -j"$(nproc)"
    ;;
  phone)
    adb get-state >/dev/null || { echo "no phone on adb (plug it in, USB debugging on)" >&2; exit 1; }
    adb shell mkdir -p $PHONE/out/spv $PHONE/research/build
    adb push build/android/lizard_gpu_check $PHONE/ >/dev/null
    adb push $OUT/setup $OUT/blobs $PHONE/out/ >/dev/null
    adb push $OUT/spv/. $PHONE/out/spv/ >/dev/null
    if [ "${1:-}" = selftest ]; then
      adb push $RESEARCH/build/gpu_selftest_frames $RESEARCH/build/gpu_selftest_ref.json $RESEARCH/build/gpu_selftest_ref_int8.json $PHONE/research/build/ >/dev/null
    fi
    adb shell "cd $PHONE && LIZ_OUT=out LIZ_BARCODE=research LIZ_CACHE=$PHONE/pipeline.cache ./lizard_gpu_check ${*:-info}"
    ;;
  apk)
    ffmpeg_libs
    ./gradlew assembleDebug "$@"
    ;;
  install)
    adb get-state >/dev/null || { echo "no phone on adb (plug it in, USB debugging on)" >&2; exit 1; }
    ffmpeg_libs
    ./gradlew installDebug "$@"
    adb shell pm grant dev.fosslabs.lizard android.permission.CAMERA
    ;;
  *)
    sed -n 2,20p ./build.sh; exit 2   # ai: the script's own folder, where it has changed to
    ;;
esac
