// TAB's readers and a frame's layout, for the rule stage (TAB the buffer named TAB; its layout
// ../../plan/back_tiers.mjs tabLayout). The four codes' constants: nOf the sent bits, slotsOf a block's slots (its sent
// bits: the pilots apart from the blocks), barOf the decline bar (a float's bits). A frame's layout from its
// configuration g (the gate's list, lib/cfg.slang): blocksOf its blocks, ttOf a code's (first block, first data slot,
// blocks, 0), blockAt a block's (code, index in it), each from the four counts by fixed comparisons (no array indexed
// at run time). fcoefIn: a sub-channel's data coefficient r (0 to DCOEF - 1), its coefficient there, past the count
// pilot (0) and the reference (PB_REF) (src/lizard_phy.c pb_fcoef); whiteOf: coefficient c of sub-channel j's two
// whitening bits (x, then y), the frame's slots 640 j + 2 c and the next. Defines (back_tiers.mjs): TAB_WHITE_AT,
// TAB_PB_REF, TAB_SUB_SLOTS, TAB_NCODES, CODE0_N to CODE3_N, CODE0_BAR to CODE3_BAR (uint); lib/cfg.slang's (CODE0_SLOTS
// to CODE3_SLOTS among them).
static const uint WHITE_AT = TAB_WHITE_AT;
static const uint PB_REF = TAB_PB_REF;
static const uint SUB_SLOTS = TAB_SUB_SLOTS;
uint nOf(uint c) { return (c == 0u ? CODE0_N : (c == 1u ? CODE1_N : (c == 2u ? CODE2_N : CODE3_N))); }
uint slotsOf(uint c) { return (c == 0u ? CODE0_SLOTS : (c == 1u ? CODE1_SLOTS : (c == 2u ? CODE2_SLOTS : CODE3_SLOTS))); }
float barOf(uint c) { return asfloat((c == 0u ? CODE0_BAR : (c == 1u ? CODE1_BAR : (c == 2u ? CODE2_BAR : CODE3_BAR)))); }
static const uint NCODES = TAB_NCODES;
#include "lib/cfg.slang"
uint blocksOf(uint2 g) { uint4 k = cfgCounts(g); return k.x + k.y + k.z + k.w; }
uint4 ttOf(uint2 g, uint c) {
  uint4 k = cfgCounts(g);
  uint fb = (c > 0u ? k.x : 0u) + (c > 1u ? k.y : 0u) + (c > 2u ? k.z : 0u);
  uint fs = (c > 0u ? k.x * CODE0_SLOTS : 0u) + (c > 1u ? k.y * CODE1_SLOTS : 0u) + (c > 2u ? k.z * CODE2_SLOTS : 0u);
  return uint4(fb, fs, c == 0u ? k.x : (c == 1u ? k.y : (c == 2u ? k.z : k.w)), 0u);
}
uint2 blockAt(uint2 v, uint b) {
  for (uint c = 0u; c < NCODES; c++) { uint4 t = ttOf(v, c); if (b >= t.x && b < t.x + t.z) { return uint2(c, b - t.x); } }
  return uint2(NCODES, 0u);
}
uint fcoefIn(uint r) { return r < PB_REF - 1u ? 1u + r : 2u + r; }
uint whiteOf(uint j, uint c) { return (TAB[WHITE_AT + 20u * j + (c >> 4u)] >> (2u * (c & 15u))) & 3u; }
