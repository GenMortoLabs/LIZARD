// ai: The file camera's reader (media_source.h): FFmpeg's, the same in the apps and the web's module
#include "media_source.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#if defined(LIZ_HAVE_FFMPEG)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}
#if defined(LIZ_FFMPEG_STATIC)
#elif defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace lizard::cam {
namespace {

#if defined(LIZ_HAVE_FFMPEG)

// ai: FFmpeg's calls the reader makes, each from its library (X(library, call)), typed by the headers it was built with
#define LIZ_FF_CALLS(X)                                                                                                     \
  X(0, av_frame_alloc) X(0, av_frame_free) X(0, av_frame_unref) X(0, av_pix_fmt_desc_get) X(0, av_read_image_line2)        \
  X(1, avcodec_alloc_context3) X(1, avcodec_free_context) X(1, avcodec_parameters_to_context) X(1, avcodec_open2)          \
  X(1, avcodec_send_packet) X(1, avcodec_receive_frame) X(1, av_packet_alloc) X(1, av_packet_free) X(1, av_packet_unref)  \
  X(2, avformat_open_input) X(2, avformat_find_stream_info) X(2, av_find_best_stream) X(2, avformat_close_input)           \
  X(2, av_read_frame)

struct Ff {
#define LIZ_FF_PTR(lib, name) decltype(&::name) name = nullptr;
  LIZ_FF_CALLS(LIZ_FF_PTR)
#undef LIZ_FF_PTR
  bool ok = false;
  std::string why;   // ai: why not, where not ok
};

#if !defined(LIZ_FFMPEG_STATIC)
// ai: each library's file as the build names it (the major versions its headers carry): avutil, avcodec, avformat
std::string libFile(int k) {
  static const char* const base[3] = {"avutil", "avcodec", "avformat"};
  const int major[3] = {LIBAVUTIL_VERSION_MAJOR, LIBAVCODEC_VERSION_MAJOR, LIBAVFORMAT_VERSION_MAJOR};
#if defined(_WIN32)
  return std::string(base[k]) + "-" + std::to_string(major[k]) + ".dll";
#elif defined(__APPLE__)
  return "lib" + std::string(base[k]) + "." + std::to_string(major[k]) + ".dylib";
#elif defined(__ANDROID__)
  (void)major;
  return "lib" + std::string(base[k]) + ".so";
#else
  return "lib" + std::string(base[k]) + ".so." + std::to_string(major[k]);
#endif
}

// ai: the folder of the module holding this code (the app's native library, lizard_receive), with its separator
std::string ownFolder() {
#if defined(_WIN32)
  HMODULE m = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&ownFolder), &m)) return "";
  wchar_t buf[32768];
  const DWORD n = GetModuleFileNameW(m, buf, 32768);
  if (!n || n >= 32768) return "";
  const std::string s = std::filesystem::path(std::wstring(buf, n)).u8string();
  const size_t k = s.find_last_of("\\/");
#else
  Dl_info d{};
  if (!dladdr(reinterpret_cast<void*>(&ownFolder), &d) || !d.dli_fname) return "";
  const std::string s = d.dli_fname;
  const size_t k = s.rfind('/');
#endif
  return k == std::string::npos ? "" : s.substr(0, k + 1);
}

#endif

// ai: The three libraries loaded once, in their order (each finds the one before it loaded: by its soname on Linux and
// ai: Android, in its own folder on Windows, by @loader_path on macOS), from the first folder that holds all three:
// ai: LIZ_FFMPEG's (set and empty: off), this library's own, then the system's search
const Ff& ff() {
  static const Ff loaded = [] {
    Ff x;
#if defined(LIZ_FFMPEG_STATIC)
    // ai: linked in (the wasm module): every call its own address
#define LIZ_FF_SET(lib, name) x.name = &::name;
    LIZ_FF_CALLS(LIZ_FF_SET)
#undef LIZ_FF_SET
    x.ok = true;
    return x;
#else
    std::vector<std::string> dirs;
    if (const char* env = std::getenv("LIZ_FFMPEG")) {
      if (!*env) { x.why = "images and videos are off (LIZ_FFMPEG is empty)"; return x; }
      std::string d = env;
      if (d.back() != '/' && d.back() != '\\') d += '/';
      dirs.push_back(d);
    } else {
      if (const std::string own = ownFolder(); !own.empty()) dirs.push_back(own);
      dirs.push_back("");
    }
    std::string tried;
    for (const std::string& dir : dirs) {
#if defined(_WIN32)
      HMODULE h[3] = {};
      bool all = true;
      for (int k = 0; k < 3 && all; k++) {
        const std::wstring f = std::filesystem::u8path(dir + libFile(k)).wstring();
        h[k] = dir.empty() ? LoadLibraryW(f.c_str()) : LoadLibraryExW(f.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!h[k]) { all = false; tried += (tried.empty() ? "" : "; ") + dir + libFile(k) + " not loaded"; }
      }
      auto sym = [&](int k, const char* name) { return reinterpret_cast<void*>(GetProcAddress(h[k], name)); };
#else
      void* h[3] = {};
      bool all = true;
      for (int k = 0; k < 3 && all; k++) {
        h[k] = dlopen((dir + libFile(k)).c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h[k]) {
          all = false;
          const char* e = dlerror();
          tried += (tried.empty() ? "" : "; ") + (e ? std::string(e) : dir + libFile(k) + " not loaded");
        }
      }
      auto sym = [&](int k, const char* name) { return dlsym(h[k], name); };
#endif
      if (!all) continue;
      int lacks = 0;
      std::string missing;
#define LIZ_FF_GET(lib, name)                                                                   \
  x.name = reinterpret_cast<decltype(x.name)>(sym(lib, #name));                                 \
  if (!x.name && lacks++ == 0) missing = #name;
      LIZ_FF_CALLS(LIZ_FF_GET)
#undef LIZ_FF_GET
      if (lacks) { tried += (tried.empty() ? "" : "; ") + dir + libFile(0) + " lacks " + missing; continue; }
      x.ok = true;
      return x;
    }
    x.why = "images and videos need the app's FFmpeg, not found (" + tried + ")";
    return x;
#endif
  }();
  return loaded;
}

// ai: a frame's sample (0 to 2^depth - 1) at 8 bits
inline uint8_t eight(uint32_t v, int depth) {
  if (depth == 8) return uint8_t(v);
  if (depth > 8) return uint8_t(v >> (depth - 8));
  return uint8_t(v * 255u / ((1u << depth) - 1u));
}

// ai: The file's first video stream (an image's one stream), decoded by FFmpeg on as many threads as it takes
class FfSource : public Source {
 public:
  ~FfSource() override {
    const Ff& f = ff();
    if (fr_) f.av_frame_free(&fr_);
    if (pkt_) f.av_packet_free(&pkt_);
    if (cc_) f.avcodec_free_context(&cc_);
    if (fc_) f.avformat_close_input(&fc_);
  }
  static std::unique_ptr<Source> open(const std::string& path, std::string& why) {
    const Ff& f = ff();
    if (!f.ok) { why = f.why; return nullptr; }
    auto s = std::make_unique<FfSource>();
    const std::string neither = "neither an image nor a video this app reads";
    if (f.avformat_open_input(&s->fc_, ("file:" + path).c_str(), nullptr, nullptr) < 0) { why = neither; return nullptr; }
    if (f.avformat_find_stream_info(s->fc_, nullptr) < 0) { why = neither; return nullptr; }
    const AVCodec* dec = nullptr;
    s->si_ = f.av_find_best_stream(s->fc_, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
    if (s->si_ == AVERROR_DECODER_NOT_FOUND) { why = "its picture's format is not one this app decodes"; return nullptr; }
    if (s->si_ < 0 || !dec) { why = neither; return nullptr; }
    const AVStream* st = s->fc_->streams[s->si_];
    s->cc_ = f.avcodec_alloc_context3(dec);
    if (!s->cc_ || f.avcodec_parameters_to_context(s->cc_, st->codecpar) < 0) { why = "its decoder did not start"; return nullptr; }
    s->cc_->thread_count = 0;   // ai: as many as FFmpeg takes (one where it was built without threads: the wasm module)
    if (f.avcodec_open2(s->cc_, dec, nullptr) < 0) { why = "its decoder did not start"; return nullptr; }
    s->pkt_ = f.av_packet_alloc();
    s->fr_ = f.av_frame_alloc();
    if (!s->pkt_ || !s->fr_) { why = "out of memory"; return nullptr; }
    if (st->avg_frame_rate.den > 0 && st->avg_frame_rate.num > 0) s->fps = double(st->avg_frame_rate.num) / st->avg_frame_rate.den;
    s->frames = st->nb_frames > 0 ? long(st->nb_frames) : -1;
    const std::string demuxer = s->fc_->iformat && s->fc_->iformat->name ? s->fc_->iformat->name : "";
    const bool image = (demuxer.size() > 5 && demuxer.compare(demuxer.size() - 5, 5, "_pipe") == 0) || s->frames == 1;
    s->kind = image ? "image" : "video";
    return s;
  }
  void cancel() override { cancelled_ = true; }
  bool next(std::vector<uint8_t>& out, std::string& why) override {
    const Ff& f = ff();
    while (!cancelled_) {
      const int r = f.avcodec_receive_frame(cc_, fr_);
      if (r == 0) {
        const bool ok = luma(out);
        f.av_frame_unref(fr_);
        if (!ok) why = "a frame of another size, or of a format this app cannot take the luma of";
        return ok;
      }
      if (r == AVERROR_EOF) return false;
      if (r != AVERROR(EAGAIN)) { why = "its decoder failed"; return false; }
      // ai: the decoder wants a packet: the stream's next, else the end (the decoder flushed for its last frames)
      if (f.av_read_frame(fc_, pkt_) < 0) {
        if (flushed_) return false;
        flushed_ = true;
        f.avcodec_send_packet(cc_, nullptr);
        continue;
      }
      if (pkt_->stream_index == si_) f.avcodec_send_packet(cc_, pkt_);   // ai: a packet it refuses (corrupt) is passed over
      f.av_packet_unref(pkt_);
    }
    return false;
  }

 private:
  AVFormatContext* fc_ = nullptr;
  AVCodecContext* cc_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* fr_ = nullptr;
  int si_ = -1;
  bool flushed_ = false;
  std::atomic<bool> cancelled_{false};
  std::vector<uint16_t> line_[3];

  // ai: The frame's luma at 8 bits: a YUV or grey frame's first component (its rows copied where it is 8-bit and
  // ai: alone in its plane, else read a row at a time and taken down to 8 bits), an RGB or paletted frame's
  // ai: (77 R + 150 G + 29 B) / 256 from its components at 8 bits each
  bool luma(std::vector<uint8_t>& out) {
    const Ff& f = ff();
    const AVPixFmtDescriptor* d = f.av_pix_fmt_desc_get(AVPixelFormat(fr_->format));
    if (!d || fr_->width <= 0 || fr_->height <= 0) return false;
    const uint32_t fw = uint32_t(fr_->width), fh = uint32_t(fr_->height);
    if (w && (fw != w || fh != h)) return false;
    w = fw;
    h = fh;
    out.resize(size_t(w) * h);
    const bool rgb = d->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL);
    const bool pal = d->flags & AV_PIX_FMT_FLAG_PAL;
    const AVComponentDescriptor& y0 = d->comp[0];
    if (!rgb && y0.depth == 8 && y0.step == 1 && y0.shift == 0 && y0.offset == 0 && !(d->flags & AV_PIX_FMT_FLAG_BITSTREAM)) {
      for (uint32_t y = 0; y < h; y++) std::memcpy(out.data() + size_t(y) * w, fr_->data[y0.plane] + size_t(y) * size_t(fr_->linesize[y0.plane]), w);
      return true;
    }
    const uint8_t* data[4] = {fr_->data[0], fr_->data[1], fr_->data[2], fr_->data[3]};
    const int lines[4] = {fr_->linesize[0], fr_->linesize[1], fr_->linesize[2], fr_->linesize[3]};
    for (auto& l : line_) l.resize(w);
    for (uint32_t y = 0; y < h; y++) {
      uint8_t* dst = out.data() + size_t(y) * w;
      if (!rgb) {
        f.av_read_image_line2(line_[0].data(), data, lines, d, 0, int(y), 0, int(w), 0, 2);
        for (uint32_t x = 0; x < w; x++) dst[x] = eight(line_[0][x], y0.depth);
        continue;
      }
      for (int c = 0; c < 3; c++) f.av_read_image_line2(line_[c].data(), data, lines, d, 0, int(y), c, int(w), pal ? 1 : 0, 2);
      const int dr = pal ? 8 : d->comp[0].depth, dg = pal ? 8 : d->comp[1].depth, db = pal ? 8 : d->comp[2].depth;
      for (uint32_t x = 0; x < w; x++)
        dst[x] = uint8_t((77u * eight(line_[0][x], dr) + 150u * eight(line_[1][x], dg) + 29u * eight(line_[2][x], db)) >> 8);
    }
    return true;
  }
};

#endif

}  // namespace

// ai: the file's frames from its start and the first one read (it gives the size)
bool openSource(const std::string& path, std::unique_ptr<Source>& s, std::vector<uint8_t>& first, std::string& why) {
  const std::string name = std::filesystem::u8path(path).filename().u8string();
#if defined(LIZ_HAVE_FFMPEG)
  std::string w;
  s = FfSource::open(path, w);
  if (!s) { why = name + ": " + w; return false; }
  std::string fwhy;
  if (!s->next(first, fwhy) || !s->w || !s->h) { why = name + ": " + (fwhy.empty() ? "no frame in it" : fwhy); s.reset(); return false; }
  return true;
#else
  (void)s;
  (void)first;
  why = name + ": this build reads no images or videos (it was built without FFmpeg)";
  return false;
#endif
}

}  // namespace lizard::cam
