// ai: The web's file camera (2026-11-07): the apps' reader (../../core/cam/media_source.h) with FFmpeg linked in
// ai: (../../ffmpeg/build.sh wasm), its own module (lizard-media.mjs, ../../CMakeLists.txt lizard_media) beside the
// ai: codec's, loaded by the page's media worker (lizard-web/media-worker.mjs) only where a file is played. One file at
// ai: a time, read through the module's file system (the worker mounts the picked File there read-only, WORKERFS, so
// ai: nothing of it is copied in whole); each frame's crop copied out on request, so one frame is held at a time.
#include <emscripten/emscripten.h>

#include <algorithm>
#include <cstring>

#include "../../core/cam/media_source.h"

namespace {
std::unique_ptr<lizard::cam::Source> src;
std::vector<uint8_t> frame, cut;
std::string why;
bool firstHeld = false;   // ai: the open's first frame, not yet handed out
}  // namespace

extern "C" {

// ai: the file at path (its first frame read, for the size): 1, else 0 and lizm_why says why ("<name>: <what>")
EMSCRIPTEN_KEEPALIVE int lizm_open(const char* path) {
  src.reset();
  why.clear();
  frame.clear();
  if (!lizard::cam::openSource(path ? path : "", src, frame, why)) return 0;
  firstHeld = true;
  return 1;
}

// ai: the open file's frame size, its frame count (-1 unnamed), its rate (0 unnamed), 1 for an image (0 a video)
EMSCRIPTEN_KEEPALIVE int lizm_w() { return src ? int(src->w) : 0; }
EMSCRIPTEN_KEEPALIVE int lizm_h() { return src ? int(src->h) : 0; }
EMSCRIPTEN_KEEPALIVE int lizm_frames() { return src ? int(src->frames) : -1; }
EMSCRIPTEN_KEEPALIVE double lizm_fps() { return src ? src->fps : 0; }
EMSCRIPTEN_KEEPALIVE int lizm_image() { return src && src->kind == "image" ? 1 : 0; }

// ai: The next frame's luma within (x, y, w, h), clamped to the frame, w x h rows of w bytes: its address in the
// ai: module's memory (valid until the next call), else 0: the file's end (lizm_why empty) or a failure (lizm_why says)
EMSCRIPTEN_KEEPALIVE const uint8_t* lizm_next(int x, int y, int w, int h) {
  if (!src) return nullptr;
  if (!firstHeld) {
    why.clear();
    if (!src->next(frame, why)) return nullptr;
  }
  firstHeld = false;
  const int fw = int(src->w), fh = int(src->h);
  x = std::clamp(x, 0, fw);
  y = std::clamp(y, 0, fh);
  w = std::clamp(w, 0, fw - x);
  h = std::clamp(h, 0, fh - y);
  if (!w || !h) {
    why = "an empty crop";
    return nullptr;
  }
  cut.resize(size_t(w) * size_t(h));
  for (int r = 0; r < h; r++) std::memcpy(cut.data() + size_t(r) * size_t(w), frame.data() + size_t(y + r) * size_t(fw) + size_t(x), size_t(w));
  return cut.data();
}

EMSCRIPTEN_KEEPALIVE const char* lizm_why() { return why.c_str(); }

// ai: the file let go (its decoder, its frames)
EMSCRIPTEN_KEEPALIVE void lizm_close() {
  src.reset();
  frame = {};
  cut = {};
  firstHeld = false;
}
}
