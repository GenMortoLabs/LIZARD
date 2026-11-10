// ai: The crops of a camera frame, the one rule for the receiver (receiver.cpp, ReceiverConfig.layout) and the C API
// ai: (liz_layout_rects): 1:1 the centre square; 2:1 the centre region twice as long as it is high along the frame's
// ai: long side, its two halves (each a square, each read as a frame). The File camera's frames alone (its images and
// ai: videos, cam::Frame::whole; a live camera's and a .lizard's keep the crop) are never cropped, only sliced: 1:1 the
// ai: whole frame; 2:1 cut in half along its long side, each half the frame's whole height by half its width (upright,
// ai: its whole width by half its height), whatever the frame's shape.
#pragma once
#include <algorithm>
#include <cstdint>

namespace lizard {

// ai: n crops, each w x h, at (x[k], y[k])
struct Crops { int n; uint32_t w, h, x[2], y[2]; };

inline Crops cropsOf(uint32_t w, uint32_t h, bool two, bool whole = false) {
  Crops c{};
  const bool wide = w >= h;
  if (whole) {
    c.n = two ? 2 : 1;
    c.w = two && wide ? w / 2 : w;
    c.h = two && !wide ? h / 2 : h;
    c.x[1] = two && wide ? c.w : 0;
    c.y[1] = two && !wide ? c.h : 0;
    return c;
  }
  const uint32_t lo = std::min(w, h), hi = std::max(w, h);
  const uint32_t side = two ? std::min(lo, hi / 2) : lo;
  c.w = c.h = side;
  c.n = two ? 2 : 1;
  for (int k = 0; k < c.n; k++) {
    const uint32_t along = (hi - c.n * side) / 2 + k * side, across = (lo - side) / 2;
    c.x[k] = wide ? along : across;
    c.y[k] = wide ? across : along;
  }
  return c;
}

}  // namespace lizard
