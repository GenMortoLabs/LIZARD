// ai: The desktop presenters' common parts (presenter.h; out of presenter.cpp 2026-11-25, for the Metal presenter beside
// ai: the Vulkan one): the steady clock, JSON's text and numbers, a present's record and a second's tally of them, the
// ai: sender lock's bounded take, the Lanczos-3 stretch's table and VSync's choice of a refresh. presenter.cpp (Vulkan:
// ai: Linux, Windows) and presenter_metal.mm (Metal: macOS) run the same.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lizard {
namespace pc {

inline double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ai: a string's text fit for JSON (a driver's message may hold quotes or a backslash)
inline std::string esc(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    if (static_cast<unsigned char>(ch) >= 0x20) o += ch;
  }
  return o;
}

// ai: a number as the JSON and the CSV want it whatever the process's locale: the JVM sets LC_NUMERIC to the user's,
// ai: and a decimal-comma locale (de_DE, fr_FR, en_DK) made the stats JSON {"shownFps":54,0}, which the app could not
// ai: read, so the rate line stayed empty (2026-10-04). A %.Nf conversion differs from C's only in that mark.
inline std::string fmt(const char* f, double v) {
  char b[64];
  std::snprintf(b, sizeof b, f, v);
  for (char* q = b; *q; q++) if (*q == ',') *q = '.';
  return b;
}

// ai: one present: its id, the picture it carried (the sender's count, whose mod 4 the pilots paint), whether that
// ai: picture was new with it, when (ms from the start) its acquire returned, it was submitted and it was on screen (-1:
// ai: not known), and its epoch (a new one after a pause or a new swapchain: no gap is counted across it)
struct Shown {
  uint64_t id = 0, picture = 0;
  bool fresh = false;
  double acquiredMs = 0, submittedMs = 0, shownMs = -1;
  uint32_t epoch = 0;
  int queued = -1;   // ai: presents from the last one on screen to this one at its submit (the chain's slack); -1 unknown
  // ai: on present timing (2026-10-31): the refresh this present was for (ms, -1 none) and its shown time the display's
  // ai: first pixel; a present then stands for its picture's whole hold, so no refresh between two is a miss
  double targetMs = -1;
  bool timed = false;
};

// ai: The presents counted: the refreshes between presents in a row (one, unless the loop missed a refresh and the
// ai: image before stayed up), and the refreshes each picture held (from its first present to the next picture's). A
// ai: hold of 0 is the waits' doing, not the screen's: after a miss the queued presents' waits return together.
struct Tally {
  std::map<long, uint64_t> held;
  uint64_t presents = 0, pictures = 0, missed = 0, unknown = 0, behind = 0;
  // ai: present timing's (2026-10-31): pictures on screen a refresh or more past their target, the schedule moved to
  // ai: the display's times, the largest first-pixel time off its target (ms)
  uint64_t late = 0, slips = 0;
  double errMaxMs = 0;
  double lastShown = -1, lastFresh = -1;
  uint32_t epoch = 0;
  std::vector<double> gaps;
  double queuedSum = 0;   // ai: the chain's slack at each submit, summed, and the least (-1: none known)
  uint64_t queuedN = 0;
  int queuedMin = -1;
  void add(const Shown& s, double refreshMs) {
    presents++;
    if (s.queued >= 0) {
      queuedSum += s.queued;
      queuedN++;
      if (queuedMin < 0 || s.queued < queuedMin) queuedMin = s.queued;
    }
    if (s.epoch != epoch) { epoch = s.epoch; lastShown = lastFresh = -1; }
    if (s.shownMs < 0) { unknown++; lastShown = lastFresh = -1; return; }
    if (s.timed && s.targetMs >= 0) {
      const double err = s.shownMs - s.targetMs;
      errMaxMs = std::max(errMaxMs, std::fabs(err));
      if (s.fresh && err >= 0.5 * refreshMs) late++;
    }
    if (lastShown >= 0) {
      const double gap = s.shownMs - lastShown;
      gaps.push_back(gap);
      const long k = std::lround(gap / refreshMs);
      if (k >= 2 && !s.timed) missed += static_cast<uint64_t>(k - 1);
    }
    lastShown = s.shownMs;
    if (s.fresh) {
      pictures++;
      if (lastFresh >= 0) held[std::lround((s.shownMs - lastFresh) / refreshMs)]++;
      lastFresh = s.shownMs;
    }
  }
  std::string heldJson() const {
    std::string o = "{";
    for (const auto& [k, n] : held) o += (o.size() > 1 ? ",\"" : "\"") + std::to_string(k) + "\":" + std::to_string(n);
    return o + "}";
  }
  // ai: the refresh the presents show, Hz: the mean of the gaps of about one refresh (a wait's wake-up moves each time
  // ai: by a few tenths of a ms either way, which a median of gaps leans on and a mean cancels). One refresh is the
  // ai: mode's where it is known, else the median gap (FIFO presents one a refresh while the loop keeps up), so a
  // ai: 144 Hz screen with no mode to read is not judged against 60.
  // ai: Where fewer than half the gaps are near the mode's refresh (a windowed swapchain paced by another monitor's
  // ai: CRTC than the one under the window's centre: a mixed 60/120 Hz pair), the median gap is the refresh
  // ai: instead (2026-10-04: the mode alone left the measurement at 0 and paced 120 presents against 60).
  double measuredHz(double modeHz) const {
    if (gaps.size() < 8) return 0;
    auto aroundRef = [&](double ref, double& hz) {
      double sum = 0;
      size_t n = 0;
      for (double g : gaps)
        if (g > 0.5 * ref && g < 1.5 * ref) { sum += g; n++; }
      hz = n ? 1000.0 * static_cast<double>(n) / sum : 0;
      return n;
    };
    double hz = 0;
    if (modeHz > 0 && 2 * aroundRef(1000.0 / modeHz, hz) >= gaps.size()) return hz;
    std::vector<double> g = gaps;
    std::nth_element(g.begin(), g.begin() + static_cast<long>(g.size() / 2), g.end());
    if (g[g.size() / 2] <= 0) return 0;
    aroundRef(g[g.size() / 2], hz);
    return hz;
  }
  // ai: a new window: its counts zero, the last times kept so the first gap of the next one counts
  void reset() {
    Tally t;
    t.lastShown = lastShown;
    t.lastFresh = lastFresh;
    t.epoch = epoch;
    *this = std::move(t);
  }
};

// ai: the mutex taken for at most `ms`: the present thread never waits out a configure (the sender's painters joined
// ai: and restarted, the GPU checked), only a take or a stats call's few microseconds
inline bool lockFor(std::mutex& m, double ms) {
  const double end = nowMs() + ms;
  while (!m.try_lock()) {
    if (nowMs() >= end) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return true;
}

// ai: The stretch's taps an axis (s source pixels to d): Lanczos-3's support of 3, widened by s / d where it shrinks, in
// ai: whole pixels with one to spare
inline int stretchTaps(int s, int d) { return static_cast<int>(std::ceil(6.0 * std::max(1.0, static_cast<double>(s) / d))) + 1; }

// ai: One axis of the stretch's table (scale_h.slang, scale_v.slang): for each of d destination pixels its first source
// ai: pixel, then `taps` weights: Lanczos-3 at the source pixels about the destination pixel's centre (in source pixels,
// ai: (o + 0.5) s / d - 0.5), the argument divided by the widening, normalised to sum 1, zero past the support. A whole
// ai: scale's centres fall on source pixels and every weight but the centre's is 0: an exact copy.
inline void stretchAxis(float* out, int s, int d, int taps) {
  constexpr double PI = 3.14159265358979323846;
  const double k = std::max(1.0, static_cast<double>(s) / d);
  std::vector<double> w(static_cast<size_t>(taps));
  for (int o = 0; o < d; o++) {
    const double c = (o + 0.5) * s / d - 0.5;
    const int lo = static_cast<int>(std::floor(c - 3 * k)) + 1;
    double sum = 0;
    for (int j = 0; j < taps; j++) {
      const double x = std::abs((lo + j - c) / k);
      w[static_cast<size_t>(j)] = x < 1e-9 ? 1.0 : x >= 3 ? 0.0 : 3 * std::sin(PI * x) * std::sin(PI * x / 3) / (PI * PI * x * x);
      sum += w[static_cast<size_t>(j)];
    }
    float* row = out + static_cast<size_t>(o) * static_cast<size_t>(taps + 1);
    row[0] = static_cast<float>(lo);
    for (int j = 0; j < taps; j++) row[1 + j] = static_cast<float>(w[static_cast<size_t>(j)] / sum);
  }
}

// ai: whether a refresh shows fps pictures a second evenly, each held a whole number of refreshes (period()'s 1%)
inline bool evenAt(double hz, int fps) {
  const double p = hz / std::max(1, fps), r = std::round(p);
  return r >= 1 && std::fabs(p - r) < 0.01 * r;
}
// ai: VSync's choice among a monitor's refreshes at its size: one fps divides, the nearest 60 Hz in whole Hz (the rate
// ai: every monitor has, and the camera's), a tie to the higher (60.00 over 59.94, 120.00 over 119.88); -1 for none
inline int pickRate(const std::vector<double>& hz, int fps) {
  int best = -1;
  for (int k = 0; k < static_cast<int>(hz.size()); k++) {
    if (!evenAt(hz[k], fps)) continue;
    const long a = std::lround(std::fabs(hz[k] - 60)), b = best < 0 ? 0 : std::lround(std::fabs(hz[best] - 60));
    if (best < 0 || a < b || (a == b && hz[k] > hz[best])) best = k;
  }
  return best;
}

}  // namespace pc
}  // namespace lizard
