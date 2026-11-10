// ai: Images and videos played as a camera (2026-11-07, backends.h mediaOpen: a file:<path> that holds no .lizard,
// ai: upload.cpp fileOpen), their frames read by FFmpeg (media_source.h; ../../ffmpeg/build.sh: LGPL, decoders only),
// ai: the same on every system and with no program or service of the system's. Every frame waits for the receiver
// ai: (Camera::waits) and goes as soon as it is taken, so all of a file's frames are decoded, as fast as the decoders
// ai: go; the file's end ends the stream ("the file ended"). One mode, the frames' size; no controls; no layout of its
// ai: own: the receiver's crop takes the frame's centre as a camera's.
#include <atomic>
#include <filesystem>
#include <mutex>
#include <thread>

#include "backends.h"
#include "camera.h"
#include "clock.h"
#include "media_source.h"

namespace lizard::cam {
namespace {

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
      f.whole = true;
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
