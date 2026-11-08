# FFmpeg for the file camera

The receivers can play an image or a video file in place of a camera (`../core/cam/media.cpp`, the camera id
`file:<path>`; the web receiver's File). They read it through one reader (`../core/cam/media_source.cpp`) over FFmpeg's
decoding libraries, which `build.sh` builds from the pinned release:

| | |
|---|---|
| Release | FFmpeg 9.0.2 |
| Source | https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz |
| SHA-256 | `8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e` |
| Licence | GNU Lesser General Public License 2.1 or later (configured with no GPL or non-free part) |
| Libraries | libavutil, libavcodec, libavformat, shared (static for WebAssembly) |
| Windows and WebAssembly | zlib 1.3.1 (https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz, SHA-256 `9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23`), built in |

The tarballs are fetched once into `../.tools/ffmpeg/` and refused where their SHA-256 is not the pinned one. The source
is built unmodified.

## Building

```
liblizard/ffmpeg/build.sh linux      # ../build/ffmpeg/linux     the host's gcc, the system's zlib
liblizard/ffmpeg/build.sh windows    # ../build/ffmpeg/windows   MinGW-w64, cross from Linux
liblizard/ffmpeg/build.sh android    # ../build/ffmpeg/android   the NDK's clang, arm64, API 29, 16 KB pages
liblizard/ffmpeg/build.sh macos      # ../build/ffmpeg/macos     on a Mac, arm64
liblizard/ffmpeg/build.sh wasm       # ../build/ffmpeg/wasm      Emscripten's emcc: static, no threads, no assembly, WebAssembly SIMD
```

Each output holds `include/` (the headers the reader compiles against), the libraries (`lib/`, `bin/` on Windows;
also `jniLibs/arm64-v8a/` for Android; `libz.a` beside them for WebAssembly), `configuration.txt` (the configure line, the build machine's paths named by
role) and `licenses/` (the licence texts the apps carry). x86 assembly is built where nasm is on the PATH. About one to
three minutes a platform.

The configuration turns everything off but reading local files and decoding these formats:

- Containers: MP4 and QuickTime, Matroska and WebM, AVI, MPEG-TS, FLV, IVF, raw H.264, HEVC and MPEG-4, MJPEG.
- Video: H.264, HEVC, VP8, VP9, MPEG-4 Part 2, MPEG-1 and MPEG-2, Motion JPEG, ProRes.
- Images: PNG and APNG, JPEG, BMP, TIFF, WebP, GIF.

## Use

`../core/CMakeLists.txt` compiles the reader against `../build/ffmpeg/<platform>/include` where it exists
(`LIZ_FFMPEG_DIR` names another), defining `LIZ_HAVE_FFMPEG`; without it the file camera says the build reads no
images or videos. The libraries are loaded at run time: from `LIZ_FFMPEG`'s folder alone where it is set (set and
empty: none), else from the folder of the library holding the core, then by the system's search. The apps carry them beside their native
library: the desktop packages (`../../lizard-desktop/build.gradle.kts`) and the APK (`../../lizard-android/`).

For the web the same reader is linked with the static WebAssembly build into a module of its own,
`../bindings/wasm/dist/lizard-media.mjs` and its `.wasm` (`../CMakeLists.txt` `lizard_media`, made by `../build.sh`
beside the codec's module, which builds FFmpeg for it first where it is missing). The web receiver loads it in a worker
when a file is played, mounting the picked file read-only so that it is read as FFmpeg asks and never copied whole.
No SharedArrayBuffer and no threads: FFmpeg decodes on the worker's one thread.

Frames are decoded to their luma plane at 8 bits (10- and 12-bit video scaled down; RGB and palette images
converted as (77 R + 150 G + 29 B) / 256). An image is one frame; a video's frames are handed over in order, none
dropped: the receiver waits for room for each where a camera's frame would be skipped.

## Source offer

The libraries are FFmpeg's, unmodified, built by `build.sh` from the release named above with the configuration in
`configuration.txt`. Being shared libraries loaded at run time, they can be replaced by any compatible build of the
same major versions (libavutil 61, libavcodec 63, libavformat 63). The web's module links them statically; its own
source and build (`../bindings/wasm/media.cpp`, `../core/cam/media_source.cpp`, `../CMakeLists.txt` `lizard_media`) are
in this repository, so it can be built again against a modified FFmpeg.
