// ai: Layer 1, the format's arithmetic (lizard.h): a format's geometry as lizard_init paints it, and the room a symbol
// ai: needs and the largest that fits one, ported from the web sender's (sim/lizard_pick.mjs ROOM_FOR, pickVersion;
// ai: two codes' gap as lizard-web/send.mjs gapFrac and desktop Pick.kt). The library is where this arithmetic lives now;
// ai: tests/pick_ref.mjs holds it to the JS.
#include <algorithm>
#include <cmath>

#include "api.hpp"

extern "C" {
#include "lizard_phy.h"
}

namespace liz {

static constexpr int OB_MARGIN = 15;   // ai: the border's depth in modules (the 12-module corner mark and LIZARD_CP's 3)
static constexpr double T_DISPLAY_CYCLE = 2.7, PI = 3.14159265358979323846;

int ringIndex(int ring) {
  if (ring == LIZ_RING_DEFAULT) return LIZARD_RING_DEFAULT;
  need(ring >= 0 && ring < LIZARD_RINGS, "ring is 0 to " + std::to_string(LIZARD_RINGS - 1) + " or LIZ_RING_DEFAULT, not " + std::to_string(ring));
  return ring;
}
static int spanOf(int r) { return 2 * LIZARD_RING[r]; }
static int modulesOf(int r) { return spanOf(r) + 2 * OB_MARGIN; }
static void needBlocks(int blocks) { need(blocks >= LIZ_MIN_BLOCKS && blocks <= LIZ_MAX_BLOCKS, "blocks is " + std::to_string(LIZ_MIN_BLOCKS) + " to " + std::to_string(LIZ_MAX_BLOCKS) + " (LIZARD-32 to LIZARD-1024), not " + std::to_string(blocks)); }

liz_geometry geometryOf(const liz_format& f) {
  needBlocks(f.blocks);
  need(f.fps >= 1 && f.fps <= 240, "fps is 1 to 240, not " + std::to_string(f.fps));   // ai: src/fmt.h OB_FMT_FPS_MAX
  need(f.codes == 1 || f.codes == 2, "codes is 1 or 2, not " + std::to_string(f.codes));
  const int r = ringIndex(f.ring);
  liz_geometry g{};
  g.n = lizard_n_for(LIZARD_STEP * f.blocks);
  g.span = spanOf(r);
  g.pxm = lizard_pxm_for(g.n, g.span);
  g.side = (modulesOf(r) + 2 * LIZARD_QUIET) * g.pxm;
  g.gap = f.codes > 1 ? LIZ_GAP_MODULES * g.pxm : 0;
  g.width = f.codes * g.side + (f.codes - 1) * g.gap;
  g.height = g.side;
  g.frame_blocks = f.codes * lizard_blocks_for(LIZARD_STEP * f.blocks);
  return g;
}

// ai: ROOM_FOR: the symbol's width over the picture's, times 2.7 display pixels a cycle of the top ring's radius
static double roomFor(int blocks, int r) {
  const double radius = std::sqrt(2.0 * LIZARD_SUB * LIZARD_STEP * blocks / PI);
  return double(modulesOf(r) + 2 * LIZARD_QUIET) / spanOf(r) * T_DISPLAY_CYCLE * radius;
}

}  // namespace liz

using namespace liz;

extern "C" {

LIZ_API int liz_ring_cells(int ring) { return guard([&] { return LIZARD_RING[ringIndex(ring)]; }); }

LIZ_API int liz_geometry_of(const liz_format* format, liz_geometry* out) {
  return guard([&] {
    need(format && out, "a format and a place for its geometry");
    *out = geometryOf(*format);
    return LIZ_OK;
  });
}

LIZ_API double liz_room_for(int blocks, int ring) {
  double v = 0;
  const int rc = guard([&] { needBlocks(blocks); v = roomFor(blocks, ringIndex(ring)); return LIZ_OK; });
  return rc < 0 ? rc : v;
}

LIZ_API int liz_blocks_for(int blocks) { return blocks >= LIZ_MIN_BLOCKS && blocks <= LIZ_MAX_BLOCKS ? lizard_blocks_for(LIZARD_STEP * blocks) : 0; }

// ai: The format's tiers in order (lizard_tiers_for, inner rates first), their blocks packed from slot 0, each its rate's
// ai: slots (lizard_block_slots); a sub-channel 2 x LIZARD_SUB slots
LIZ_API int liz_block_place_of(int blocks, int b, liz_block_place* out) {
  return guard([&] {
    needBlocks(blocks);
    need(out, "a place for the block's place");
    lizard_tier_t tier[LIZARD_TIERS];
    const int tiers = lizard_tiers_for(LIZARD_STEP * blocks, tier);
    need(b >= 0 && b < lizard_blocks_for(LIZARD_STEP * blocks), "a block of the format's symbol");
    for (int t = 0, at = 0, first = 0; t < tiers; t++) {
      const int slots = lizard_block_slots(tier[t].rate);
      if (b < first + tier[t].blocks) {
        const int start = at + (b - first) * slots;
        out->rate = tier[t].rate == 6 ? 0 : tier[t].rate == 4 ? 1 : tier[t].rate == 3 ? 2 : 3;
        out->first_slot = start;
        out->slots = slots;
        out->first_subch = start / (2 * LIZARD_SUB);
        out->last_subch = (start + slots - 1) / (2 * LIZARD_SUB);
        return LIZ_OK;
      }
      at += tier[t].blocks * slots;
      first += tier[t].blocks;
    }
    throw Error(LIZ_E_INTERNAL, "the tiers lay fewer blocks than the format carries");
  });
}

LIZ_API int liz_pick(double w, double h, int codes, int ring, int top) {
  return guard([&] {
    need(codes == 1 || codes == 2, "codes is 1 or 2");
    needBlocks(top);
    const int r = ringIndex(ring);
    const double gapFrac = codes > 1 ? (codes - 1) * double(LIZ_GAP_MODULES) / (modulesOf(r) + 2 * LIZARD_QUIET) : 0;
    const double room = std::max(64.0, std::min(w / (codes + gapFrac), h));
    // ai: no room for even the smallest: LIZARD-64, as the web sender's pickVersion
    int best = LIZ_MIN_BLOCKS;
    for (int b = LIZ_MIN_BLOCKS; b <= top; b++) if (roomFor(b, r) <= room) best = b;
    return best;
  });
}

}  // extern "C"
