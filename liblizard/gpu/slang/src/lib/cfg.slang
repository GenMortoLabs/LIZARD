// A frame's configuration (2026-11-23: its word's four counts, src/fmt.h), two words: x the version << 24 | its 6/7
// blocks << 16 | its 3/4 << 8 | its 2/3 (8 bits each; at version 67 at most 152, 133 and 118), y its 1/2 blocks (at
// most 88). cfgCounts: the four codes' blocks in CODES' order (6/7, 3/4, 2/3, 1/2); cfgOk: a frame (a version of the
// ladder, the blocks within its VERSION_SLOTS data slots, 16 sub-channels of 636: the pilots apart; src/fmt.c
// ob_fmt_layout). Every term an integer; no array indexed at run time.
// Defines (common.mjs CFG_DEFS): CODE0_SLOTS to CODE3_SLOTS (each code's sent bits, uint), CFG_VERSION_MIN,
// CFG_VERSION_MAX, CFG_VERSION_SLOTS (uint).
uint cfgVersion(uint2 g) { return g.x >> 24u; }
uint4 cfgCounts(uint2 g) { return uint4((g.x >> 16u) & 255u, (g.x >> 8u) & 255u, g.x & 255u, g.y); }
bool cfgOk(uint2 g) {
  uint v = cfgVersion(g);
  uint4 k = cfgCounts(g);
  uint used = k.x * CODE0_SLOTS + k.y * CODE1_SLOTS + k.z * CODE2_SLOTS + k.w * CODE3_SLOTS;
  return v >= CFG_VERSION_MIN && v <= CFG_VERSION_MAX && used <= v * CFG_VERSION_SLOTS;
}
