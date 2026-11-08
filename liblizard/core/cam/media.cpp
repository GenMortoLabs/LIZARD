// ai: Images and videos played as a camera (2026-11-07, backends.h mediaOpen: a file:<path> that holds no .lizard,
// ai: upload.cpp fileOpen), read by the FFmpeg the app carries (../../ffmpeg/build.sh: LGPL, decoders only; libavutil,
// ai: libavcodec and libavformat loaded at run time from LIZ_FFMPEG's folder, else this library's own, else the
// ai: system's search), the same on every system and with no program or service of the system's: an image (PNG, JPEG,
// ai: WebP, BMP, TIFF, GIF) is one frame, a video (MP4, MOV, MKV, WebM, AVI, MPEG-TS; H.264, HEVC, VP8, VP9, MPEG-4,
// ai: MPEG-2, ProRes) its frames in order. Each frame's luma at 8 bits: a YUV or grey frame's Y (a 10-bit video's taken
// ai: down to 8, its HDR curve as it is), an RGB or paletted one's BT.601 luma. Every frame waits for the receiver
// ai: (Camera::waits) and goes as soon as it is taken, so all of a file's frames are decoded, as fast as the decoders
// ai: go; the file's end ends the stream ("the file ended"). One mode, the frames' size; no controls; no layout of its
// ai: own: the receiver's crop takes the frame's centre as a camera's.
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>

#include "backends.h"
#include "camera.h"
#include "clock.h"

#if defined(LIZ_HAVE_FFMPEG)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif
#endif

namespace lizard::cam {
namespace {

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

// ai: The three libraries loaded once, in their order (each finds the one before it loaded: by its soname on Linux and
// ai: Android, in its own folder on Windows, by @loader_path on macOS), from the first folder that holds all three:
// ai: LIZ_FFMPEG's (set and empty: off), this library's own, then the system's search
const Ff& ff() {
  static const Ff loaded = [] {
    Ff x;
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
    s->cc_->thread_count = 0;   // ai: as many as FFmpeg takes
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

class MediaCamera : public Camera {
 public:
  MediaCamera(std::string id, std::string path, std::unique_ptr<Source> s, std::vector<uint8_t> first)
      : path_(std::move(path)), src_(std::move(s)), first_(std::move(first)) {
    info_.id = std::move(id);
    const std::string name = std::filesystem::u8path(path_).filename().u8string();
    info_.name = name.empty() ? "File" : name;
    info_.backend = "file";
    // ai: no layout first in the detail (session.cpp ownLayout reads one for an upload or a phone alone): the
    // ai: receiver's crop is the setting's
    info_.detail = src_->kind + ", " + std::to_string(src_->w) + " x " + std::to_string(src_->h) +
                   (src_->kind == "video" && src_->frames > 0 ? ", " + std::to_string(src_->frames) + " frames" : "");
  }
  ~MediaCamera() override { stop(); }

  const Info& info() const override { return info_; }
  bool waits() const override { return true; }
  std::vector<Mode> modes() override {
    Mode m;
    m.width = src_->w;
    m.height = src_->h;
    m.fps = m.fpsMin = 0;   // ai: as fast as the receiver takes them: no rate of its own
    m.format = "GREY";
    m.pix = Pix::Grey8;
    return {m};
  }
  std::vector<Control> controls() override { return {}; }
  bool set(const std::string& id, double, bool, std::string& why) override {
    why = "a file has no control " + id;
    return false;
  }
  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override {
    stop();
    if (mode.width != src_->w || mode.height != src_->h) { why = "a file has one mode, " + std::to_string(src_->w) + " x " + std::to_string(src_->h); return false; }
    // ai: a start after the file was played (a Stop and a Start) plays it again from its start
    if (played_) {
      std::unique_ptr<Source> s;
      std::vector<uint8_t> first;
      if (!openSource(path_, s, first, why)) return false;
      if (s->w != src_->w || s->h != src_->h) { why = "the file changed since it was opened"; return false; }
      src_ = std::move(s);
      first_ = std::move(first);
    }
    played_ = true;
    sink_ = std::move(sink);
    ended_ = std::move(ended);
    {
      std::lock_guard<std::mutex> l(mu_);
      stats_ = Stats{};
      convertSum_ = 0;
    }
    quit_ = false;
    th_ = std::thread([this] { loop(); });
    return true;
  }
  void stop() override {
    quit_ = true;
    if (th_.joinable()) {
      src_->cancel();
      th_.join();
    }
  }
  Stats stats() override {
    std::lock_guard<std::mutex> l(mu_);
    Stats s = stats_;
    s.convertMs = s.frames ? convertSum_ / double(s.frames) : 0;
    return s;
  }

 private:
  Info info_;
  std::string path_;
  std::unique_ptr<Source> src_;
  std::vector<uint8_t> first_;
  bool played_ = false;
  Sink sink_;
  Ended ended_;
  std::thread th_;
  std::atomic<bool> quit_{false};
  std::mutex mu_;
  Stats stats_;
  double convertSum_ = 0;

  // ai: the reading thread: the first frame (read at the open, for the size), then each next one, handed over at once
  // ai: (the sink holds it until a decoder takes it: Camera::waits); convertMs the decode's time a frame
  void loop() {
    std::vector<uint8_t> frame;
    frame.swap(first_);
    std::string why;
    for (uint64_t k = 0; !frame.empty() && !quit_; k++) {
      Frame f;
      f.luma = frame.data();
      f.width = src_->w;
      f.height = src_->h;
      f.stride = src_->w;
      f.ns = steadyNs();
      f.seq = k;
      sink_(f);
      {
        std::lock_guard<std::mutex> l(mu_);
        stats_.frames++;
      }
      if (quit_) break;
      const int64_t t0 = steadyNs();
      if (!src_->next(frame, why)) frame.clear();
      std::lock_guard<std::mutex> l(mu_);
      convertSum_ += double(steadyNs() - t0) * 1e-6;
    }
    if (!quit_) ended_(why.empty() ? "the file ended" : why);
  }
};

}  // namespace

std::unique_ptr<Camera> mediaOpen(const std::string& id, const std::string& path, std::string& why) {
  std::unique_ptr<Source> s;
  std::vector<uint8_t> first;
  if (!openSource(path, s, first, why)) return nullptr;
  return std::make_unique<MediaCamera>(id, path, std::move(s), std::move(first));
}

}  // namespace lizard::cam
