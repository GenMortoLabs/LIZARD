#!/usr/bin/env bash
# ai: build.sh <linux|windows|android|macos|wasm> [out]: FFmpeg's decoding libraries for the file camera (../core/cam/media_source.cpp,
# ai: 2026-11-07): the pinned release (FFMPEG, FFMPEG_SHA256) fetched once into ../.tools/ffmpeg and checked against its
# ai: SHA-256, configured LGPL (no GPL or non-free part) with everything off but reading files and decoding the formats
# ai: listed (DEMUXERS, DECODERS, PARSERS), built as shared libraries into out (../build/ffmpeg/<platform> unless given):
# ai: include/, the headers media.cpp compiles against, and lib/ (bin/ on Windows): libavutil, libavcodec, libavformat,
# ai: which the apps carry beside their native library and the core loads at run time.
# ai:   linux    the host's gcc; the system's zlib
# ai:   windows  MinGW-w64 (x86_64-w64-mingw32-), cross; zlib built in from its pinned release (ZLIB, ZLIB_SHA256)
# ai:   android  the NDK's clang (ANDROID_NDK_HOME, else $ANDROID_HOME/ndk/27.1.12297006), arm64, API 29; the NDK's zlib;
# ai:            aligned for 16 KB pages, as the app's own library; also out/jniLibs/arm64-v8a/, the APK's layout
# ai:   macos    the host's clang (on a Mac), arm64; the system's zlib; install names @loader_path, so each finds the others
# ai:   wasm     Emscripten's emcc (source <emsdk>/emsdk_env.sh): static libraries for the web's module (../CMakeLists.txt
# ai:            lizard_media), no threads (no SharedArrayBuffer), no assembly, WebAssembly SIMD as the codec's module; zlib
# ai:            built in from its pinned release, libz.a beside them
# ai: x86 assembly where nasm is on the PATH (the decode about twice as fast), else none. About 1 to 3 minutes a platform.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PLATFORM=${1:?usage: build.sh <linux|windows|android|macos|wasm> [out]}
OUT=${2:-$HERE/../build/ffmpeg/$PLATFORM}
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)   # ai: absolute (macOS has no realpath -m)
CACHE=$HERE/../.tools/ffmpeg
FFMPEG=ffmpeg-9.0.2
FFMPEG_SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
ZLIB=zlib-1.3.1
ZLIB_SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# ai: a pinned tarball into the cache, once; refused where its SHA-256 is not the pinned one
fetch() {
  local name=$1 url=$2 sum=$3 file=$CACHE/$(basename "$2")
  mkdir -p "$CACHE"
  # ai: a connection the server drops is tried again (ffmpeg.org reset one on CI's macOS runner, 2026-11-08)
  [ -f "$file" ] || curl -sSfL --retry 5 --retry-all-errors --retry-delay 5 -o "$file.part" "$url" && { [ -f "$file" ] || mv "$file.part" "$file"; }
  local got
  got=$( (sha256sum "$file" 2>/dev/null || shasum -a 256 "$file") | cut -d' ' -f1)
  [ "$got" = "$sum" ] || { echo "$name: $file is $got, not the pinned $sum" >&2; rm -f "$file"; exit 1; }
  echo "$file"
}

DEMUXERS=mov,matroska,avi,mpegts,flv,ivf,h264,hevc,m4v,mjpeg,image_png_pipe,image_jpeg_pipe,image_webp_pipe,image_bmp_pipe,image_tiff_pipe,gif
DECODERS=h264,hevc,vp8,vp9,mpeg4,mpeg1video,mpeg2video,mjpeg,prores,png,apng,bmp,tiff,webp,gif
PARSERS=h264,hevc,vp8,vp9,mpeg4video,mpegvideo,mjpeg,png,bmp,webp,gif
# ai: a neutral prefix, the libraries staged and moved to out: FFmpeg keeps its configure line inside them
# ai: (av_configuration), so no path of the build machine is written into what the apps carry
FLAGS=(--prefix=/ffmpeg --enable-shared --disable-static --enable-pic --disable-programs --disable-doc --disable-network
  --disable-autodetect --disable-everything --disable-avdevice --disable-avfilter --disable-swresample --disable-swscale
  --disable-iconv --enable-zlib --enable-protocol=file --enable-demuxer=$DEMUXERS --enable-decoder=$DECODERS
  --enable-parser=$PARSERS)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
tar -xf "$(fetch FFmpeg https://ffmpeg.org/releases/$FFMPEG.tar.xz $FFMPEG_SHA256)" -C "$WORK"
SRC=$WORK/$FFMPEG

case $PLATFORM in
  linux)
    command -v nasm >/dev/null || FLAGS+=(--disable-x86asm)
    ;;
  windows)
    tar -xf "$(fetch zlib https://github.com/madler/zlib/releases/download/v${ZLIB#zlib-}/$ZLIB.tar.gz $ZLIB_SHA256)" -C "$WORK"
    make -s -C "$WORK/$ZLIB" -f win32/Makefile.gcc PREFIX=x86_64-w64-mingw32- libz.a >/dev/null
    command -v nasm >/dev/null || FLAGS+=(--disable-x86asm)
    # ai: zlib's folder through the environment, which configure takes and does not write into the libraries
    export CFLAGS="-I$WORK/$ZLIB" LDFLAGS="-L$WORK/$ZLIB -static-libgcc"
    FLAGS+=(--target-os=mingw32 --arch=x86_64 --enable-cross-compile --cross-prefix=x86_64-w64-mingw32-)
    ;;
  android)
    NDK=${ANDROID_NDK_HOME:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.1.12297006}
    TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64
    # ai: the toolchain by name on the PATH (its own sysroot found by the clang wrapper), so its folder is not written in
    export PATH="$TC/bin:$PATH"
    FLAGS+=(--target-os=android --arch=aarch64 --enable-cross-compile --cc=aarch64-linux-android29-clang
      --cxx=aarch64-linux-android29-clang++ --ar=llvm-ar --nm=llvm-nm --ranlib=llvm-ranlib --strip=llvm-strip
      --extra-ldflags="-Wl,-z,max-page-size=16384")
    ;;
  macos)
    FLAGS+=(--arch=arm64 --cc=clang --install-name-dir=@loader_path)
    ;;
  wasm)
    command -v emcc >/dev/null || { echo "wasm: no emcc on the PATH (source <emsdk>/emsdk_env.sh)" >&2; exit 1; }
    tar -xf "$(fetch zlib https://github.com/madler/zlib/releases/download/v${ZLIB#zlib-}/$ZLIB.tar.gz $ZLIB_SHA256)" -C "$WORK"
    ( cd "$WORK/$ZLIB" && CFLAGS="-O3 -msimd128" emconfigure ./configure --static > /dev/null && emmake make -s libz.a > /dev/null )
    export CFLAGS="-I$WORK/$ZLIB" LDFLAGS="-L$WORK/$ZLIB"
    # ai: static libraries in the shared ones' place, nothing position independent
    STATIC=()
    for f in "${FLAGS[@]}"; do
      case $f in --enable-shared) STATIC+=(--disable-shared) ;; --disable-static) STATIC+=(--enable-static) ;; --enable-pic) ;; *) STATIC+=("$f") ;; esac
    done
    FLAGS=("${STATIC[@]}" --target-os=none --arch=x86_32 --enable-cross-compile --disable-asm --disable-pthreads
      --disable-runtime-cpudetect --cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=emnm --dep-cc=emcc
      --extra-cflags=-msimd128)
    ;;
  *) echo "platform: linux, windows, android, macos or wasm" >&2; exit 2 ;;
esac

rm -rf "$OUT"
( cd "$SRC" && ./configure "${FLAGS[@]}" > "$WORK/configure.log" 2>&1 ) || { tail -30 "$WORK/configure.log" >&2; exit 1; }
make -s -C "$SRC" -j"$JOBS" >/dev/null
make -s -C "$SRC" install DESTDIR="$WORK/stage" >/dev/null
mkdir -p "$OUT"
cp -a "$WORK/stage/ffmpeg/." "$OUT/"
rm -rf "$OUT/share" "$OUT/lib/pkgconfig"
if [ "$PLATFORM" = android ]; then mkdir -p "$OUT/jniLibs/arm64-v8a" && cp "$OUT"/lib/libav*.so "$OUT/jniLibs/arm64-v8a/"; fi
if [ "$PLATFORM" = wasm ]; then cp "$WORK/$ZLIB/libz.a" "$OUT/lib/"; fi
# ai: the configuration the libraries were built with, for the credits' source offer (README.md)
# ai: (the build machine's paths named by role: the output, the scratch folder, the NDK)
grep -m1 'FFMPEG_CONFIGURATION' "$SRC/config.h" | sed 's/^#define FFMPEG_CONFIGURATION //' |
  sed -e "s|$OUT|<out>|g" -e "s|$WORK|<scratch>|g" -e "s|${TC:-/nonexistent}|<ndk toolchain>|g" > "$OUT/configuration.txt"
# ai: the license texts the apps carry beside the libraries (out/licenses): FFmpeg's LGPL, headed with what it covers and
# ai: where its source is; Windows' and the web's zlib, built into libavcodec there
mkdir -p "$OUT/licenses"
{ echo "FFmpeg ${FFMPEG#ffmpeg-} (libavutil, libavcodec, libavformat), under the GNU Lesser General Public License 2.1 or later."
  echo "Built unmodified from https://ffmpeg.org/releases/$FFMPEG.tar.xz (SHA-256 $FFMPEG_SHA256) by"
  echo "liblizard/ffmpeg/build.sh in LIZARD's source; configured: $(cat "$OUT/configuration.txt")"
  echo; cat "$SRC/COPYING.LGPLv2.1"; } > "$OUT/licenses/ffmpeg.txt"
if [ "$PLATFORM" = windows ] || [ "$PLATFORM" = wasm ]; then
  { echo "zlib ${ZLIB#zlib-} (built into libavcodec on Windows and in the web's module), under the zlib license."; echo; cat "$WORK/$ZLIB/LICENSE"; } > "$OUT/licenses/zlib.txt"
fi
echo "$PLATFORM: $(du -sh "$OUT" | cut -f1) in $OUT"
