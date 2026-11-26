// ai: Refresh's hold rule (2026-11-17; Rate refresh of 2026-11-12, 1 to 4 a code rate, before), every sender's: the
// ai: native Sender::produce (sender.h setRefresh, setRefreshBlocks) and the web's (lizard.h liz_refresh_places and
// ai: liz_refresh_held through the wasm, sim/ob.mjs Refresh, lizard-web/send-worker.mjs). Each block place of a symbol
// ai: takes a new block every `every` pictures, its delay, a whole number 1 to MAX (1, the default, every picture): picture
// ai: k (the count) shows a new block there where k is a multiple of it, and repeats the picture before's block otherwise.
// ai: The pilots stay the picture's count (painted with the picture, not kept with the block), so a receiver and its
// ai: phase lock read every picture as ever: no format change, a repeated block one a receiver has. Header-only.
#pragma once
#include <cstdint>

extern "C" {
#include "fmt.h"
#include "lizard_phy.h"
}

namespace lizard::refresh {

// ai: the largest delay the border's word states (src/fmt.h)
constexpr int MAX = OB_FMT_DELAY_MAX;
inline int clampDelay(int e) { return e < 1 ? 1 : e > MAX ? MAX : e; }
// ai: a tier's rate code (src/lizard_phy.h: 6 6/7, 4 3/4, 3 2/3, 2 1/2) to its index among the four delays
inline int rateIndex(int rate) { return rate == 6 ? 0 : rate == 4 ? 1 : rate == 3 ? 2 : 3; }
// ai: whether picture k shows a new block at a place that takes one every `every` pictures
inline bool due(int every, uint64_t k) { return every <= 1 || k % static_cast<uint64_t>(every) == 0; }

// ai: each of a layout's `blocks` places' delays from the four rates' (6/7, 3/4, 2/3, 1/2, each held to 1 .. MAX), its
// ai: tiers laid from the inside out, a place past them the 1/2's (the tiers lay every block: a guard, not a case)
inline void places(const lizard_tier_t* tier, int tiers, int blocks, const int every[4], int* place) {
  int p = 0;
  for (int t = 0; t < tiers; t++) for (int k = 0; k < tier[t].blocks && p < blocks; k++) place[p++] = clampDelay(every[rateIndex(tier[t].rate)]);
  while (p < blocks) place[p++] = clampDelay(every[3]);
}

// ai: The frame painted seq-th, `total` blocks (codes x the symbol's `blocks` places, place[p] place p's delay): held[t] 1
// ai: where block t repeats the frame before's at its place (there is one, `before`: a configure's first frame, or the
// ai: first after a stop, is new throughout), 0 where it takes a new block. Returns the new blocks' count.
inline int held(const int* place, int blocks, int total, uint64_t seq, bool before, uint8_t* out) {
  int fresh = 0;
  for (int t = 0; t < total; t++) {
    out[t] = before && !due(place[t % blocks], seq) ? 1 : 0;
    fresh += !out[t];
  }
  return fresh;
}

}  // namespace lizard::refresh
