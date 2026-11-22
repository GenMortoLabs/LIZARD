// ai: The phase lock's host (2026-10-31): what a receive session does around the lock (phase_lock.h) for a camera that
// ai: can move its captures (the capture callback's order, the mode read every 30 captures, the feed, the log line every
// ai: 300). The camera backend says what its lever is (PhaseLock::Lever: a delay of any length, or pulses whose moves
// ai: the lock learns; whether it marks the capture a delay moved) and does what the lock asks (act):
// ai:   a delay of us   one capture made us microseconds longer, so every capture after it starts that much later: on
// ai:                   Camera2 one request with auto-exposure off, the newest undelayed result's exposure and
// ai:                   sensitivity, and SENSOR_FRAME_DURATION that result's frame plus us, the repeating request going
// ai:                   on after it; a GenICam camera's trigger schedule moved
// ai:   a pulse         the backend's own action of that number, whatever it moves (the lock measures it)
// ai: false where it could not be asked (no undelayed result yet, the request refused). It reports, per capture result
// ai: in order: its sensor timestamp (the clock the receiver's series keeps, since the frame pushed carries the same
// ai: timestamp) and whether it was that delayed capture where it marks one; and the delayed capture's fate where it is
// ai: not a completed result (its request failed: taken where its image was captured all the same; its sequence
// ai: aborted: not taken). A camera with no lever cannot move a capture where it was asked (opened's bar), and runs
// ai: free with the lock Off.
// ai: Threads: opened, want, captured and fate on the camera's callback thread (one thread for a capture session's
// ai: callbacks), delayOnce never calling back into the host before it returns; state() from any thread.
#pragma once
#include "phase_lock.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lizard {

class PhaseLockHost {
 public:
  // ai: series: the receiver's frames after sinceMs (Receiver::series's vector; {0} with no receiver)
  using Series = std::function<std::vector<double>(double sinceMs)>;
  using Act = std::function<bool(const PhaseLock::Ask&)>;

  PhaseLockHost(PhaseLock::Say say, Series series, Act act, bool debug = false)
      : say_(say), lock_(std::move(say), debug), series_(std::move(series)), act_(std::move(act)) {}

  // ai: A capture session begun (each open of the camera, each restart of its stream): the lock's camera and hold
  // ai: afresh (reopened), the feed from the series' start. bar: why this camera cannot move a capture where it was
  // ai: asked ("" where it can), which holds the lock Off for the session; lever: what it can do.
  void opened(const std::string& bar, const PhaseLock::Lever& lever = {}) {
    lock_.reopened();
    lock_.lever(lever);
    fedMs_ = -1.0;
    n_ = 0;
    bar_ = bar;
    if (!bar_.empty()) {
      say_("phase: no lock on this lens: " + bar_);
      lock_.set(PhaseLock::Mode::Off);
    }
  }
  // ai: The mode a tool asks (Track unless told; Android's debug.lizard.phase), taken at the session's next 30th capture
  void want(PhaseLock::Mode m) { want_ = m; }
  // ai: A capture's result: the mode (every 30 captures), the receiver's new frames to the lock, then the capture, and
  // ai: what it asks to the camera (a refusal told to the lock at once); the lock's line every 300 captures
  void captured(int64_t tsNs, bool delayed) {
    if (n_ % 30 == 15) lock_.set(bar_.empty() ? want_ : PhaseLock::Mode::Off);
    feed();
    const PhaseLock::Ask a = lock_.onCapture(tsNs, delayed);
    if (a.any() && !act_(a)) lock_.delayFate(false);
    if (n_ % 300 == 0)
      if (const auto l = lock_.line(tsNs)) say_(*l);
    n_++;
  }
  void fate(bool taken) { lock_.delayFate(taken); }
  // ai: for the stats and a replay's rows (PhaseLock::State), and the bar ("" none) where the lock is off for the camera
  std::optional<PhaseLock::State> state() const { return lock_.state(); }
  const std::string& bar() const { return bar_; }

 private:
  // ai: the frames after the newest fed, Engine.feed's way: none asked while the lock is Off
  void feed() {
    if (lock_.mode() != PhaseLock::Mode::Track) return;
    const std::vector<double> a = series_(fedMs_);
    int version = 0;
    const std::vector<PhaseLock::Frame> fs = PhaseLock::framesOf(a, version);
    if (fs.empty()) return;
    fedMs_ = a[1 + 8 * (fs.size() - 1)];
    lock_.onWindow(fs, version);
  }

  PhaseLock::Say say_;
  PhaseLock lock_;
  Series series_;
  Act act_;
  PhaseLock::Mode want_ = PhaseLock::Mode::Track;
  std::string bar_;
  double fedMs_ = -1.0;
  uint64_t n_ = 0;
};

}  // namespace lizard
