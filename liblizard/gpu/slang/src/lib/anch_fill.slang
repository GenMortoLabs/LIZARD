// The squares' fill's points (../../plan/anchors.mjs ANCH_FILL): whether point p of an axis lies within the band of one
// of the ring's S anchors' centres, in integers of 2 ANCH_FILL times a module (an anchor's centre, the margin, the span
// and a are whole); and where a frame's slices of the fill's sums lie in anch: past every frame's anchors, frame f's
// 2 ANCH_GS float4 the (f + 1)-th such run back from the buffer's end (a lane's anch holds (ANCH_K + 2 ANCH_GS) B, so the
// anchors' indexing and the batch size stay out of it). Defines: ANCH_FILL, ANCH_BAND2 (int), ANCH_GS (uint).
bool fillBand(uint r, int p, int sp, int mg, int aa, uint S) {
  bool hit = false;
  for (uint i = 1u; i <= S; i++) {
    int d = 2 * ANCH_FILL * (f2i(knot(r, i)) - mg) - sp * (2 * p + 1);
    hit = hit || abs(d) <= ANCH_BAND2 * ANCH_FILL * aa;
  }
  return hit;
}
uint fillSums(uint f) { uint n, stride; anch.GetDimensions(n, stride); return n - 2u * ANCH_GS * (f + 1u); }
