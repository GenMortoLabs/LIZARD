// The squares' fill's points (core/dec/build.cpp Anchors::FILL): whether point p of an axis lies within the band of one
// of the ring's S anchors' centres, in integers of 2 ANCH_FILL times a module (an anchor's centre, the margin, the span
// and a are whole); and where a frame's slices of the fill's sums lie in the lane's fill buffer (ANCH_SUMS writes it,
// ANCH_FIT reads it): frame f's 2 ANCH_GS float4 from 2 ANCH_GS f. (Until 2026-11-25 they lay in anch, frame f's the
// (f + 1)-th run back from its end, found by the buffer's length, which Metal's shaders cannot read.) Defines:
// ANCH_FILL, ANCH_BAND2 (int), ANCH_GS (uint).
bool fillBand(uint r, int p, int sp, int mg, int aa, uint S) {
  bool hit = false;
  for (uint i = 1u; i <= S; i++) {
    int d = 2 * ANCH_FILL * (f2i(knot(r, i)) - mg) - sp * (2 * p + 1);
    hit = hit || abs(d) <= ANCH_BAND2 * ANCH_FILL * aa;
  }
  return hit;
}
uint fillSums(uint f) { return 2u * ANCH_GS * f; }
