// ai: An image's or a video's frames as luma, read by FFmpeg (2026-11-07): the file camera's reader (media.cpp, the
// ai: apps') and the web's (../../bindings/wasm/media.cpp, the same reader in its own WebAssembly module). An image
// ai: (PNG, JPEG, WebP, BMP, TIFF, GIF) is one frame, a video (MP4, MOV, MKV, WebM, AVI, MPEG-TS; H.264, HEVC, VP8, VP9,
// ai: MPEG-4, MPEG-2, ProRes) its frames in order. Each frame's luma at 8 bits: a YUV or grey frame's Y (a 10-bit video's
// ai: taken down to 8, its HDR curve as it is), an RGB or paletted one's BT.601 luma. FFmpeg's libraries are loaded at
// ai: run time (LIZ_FFMPEG's folder, else this library's own, else the system's search), or linked in where
// ai: LIZ_FFMPEG_STATIC is defined (the wasm module); without LIZ_HAVE_FFMPEG nothing is read.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lizard::cam {

// ai: A file's frames in order, each its luma at 8 bits, w x h rows of w bytes (w and h once the first is read)
struct Source {
  virtual ~Source() = default;
  // ai: the next frame into out: true; false at the file's end (why left empty) or on a failure (why said)
  virtual bool next(std::vector<uint8_t>& out, std::string& why) = 0;
  // ai: from another thread: a next() under way returns soon (the camera's stop)
  virtual void cancel() {}
  uint32_t w = 0, h = 0;
  double fps = 0;       // ai: the video's rate where it names one
  long frames = -1;     // ai: its frame count where it names one
  std::string kind;     // ai: "image" or "video"
};

// ai: the file's frames from its start and the first one read (it gives the size); why names the file, then what is wrong
bool openSource(const std::string& path, std::unique_ptr<Source>& s, std::vector<uint8_t>& first, std::string& why);

}  // namespace lizard::cam
