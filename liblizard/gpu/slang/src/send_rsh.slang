// The sender's RSH, the resampler across and out: a lane one u32 of a frame row, four bytes, the codes symbols side by
// side (FW = codes W pixels a row, W rows, each row padded to RW words: W need not be a multiple of four, as with pxm 1
// in the 256 ring, 546 pixels): inside a symbol's square (q pixels from sq) the same six taps along the row of TMP,
// clamped to 0..1, then (drive 255 + 0.5) truncated and held to 255 as lizard_paint_rgba does; anywhere else the
// border's byte (BORDER, one symbol W x W, the C's own paint: ring, marks, band, word, quiet margin, the anchors'
// squares), so the border and the anchors are the C's to the byte and nothing border-shaped is drawn here. No integer
// division: a row a workgroup y, the symbol by subtraction. A pixel on an anchor's row and column takes BORDER's byte: a
// row's test once a lane (a workgroup is one row, so it is uniform there), a column's only on such a row, each seven
// wrapped differences against the side (integers alone: no device's rounding moves an anchor's edge).
// BORDER_ONLY 1, the border form (the copy form's, where IPIC's frame form writes the squares: ../send_ipic.slang
// FRAME_OUT): the words IPIC writes (a square row's words that hold a pixel of a symbol's square) are left, every
// other word is the border's bytes alone, and TMP is not read.
// Bindings: TMP (ro, f32; unread in the border form), TAPS (ro), BORDER (ro, one symbol's bytes, four a word), FRAME
// (rw, a half's frames FS words apart), the G uniform with the anchors (GA: a = (n, q, sq, W), b = (codes, FW, RW, FS),
// c = (whole, f0, gap, 0), d = the first pixel of anchors 0 to 3 along either axis of a symbol, e = anchors 4 to 6's
// and the square's side; 80 B). Dispatch (ceil(RW / LANES), W, frames).
// Defines (../send.mjs rshSource): BORDER_ONLY (0 or 1), LANES (int, ../../plan/send.mjs RSH_THREADS).
[[vk::binding(0, 0)]] StructuredBuffer<float> TMP;
[[vk::binding(1, 0)]] StructuredBuffer<uint> TAPS;
[[vk::binding(2, 0)]] StructuredBuffer<uint> BORDER;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> FRAME;
struct GBlock { uint4 a; uint4 b; uint4 c; uint4 d; uint4 e; };
[[vk::binding(4, 0)]] ConstantBuffer<GBlock> P;
// whether v, a symbol's pixel along either axis, lies on one of the anchors' seven runs (P.d, P.e.xyz their first
// pixels, P.e.w long)
bool onAnchor(uint v) {
  uint s = P.e.w;
  return (v - P.d.x < s) || (v - P.d.y < s) || (v - P.d.z < s) || (v - P.d.w < s) || (v - P.e.x < s) || (v - P.e.y < s) || (v - P.e.z < s);
}
uint pixel(uint x, uint y, uint f, bool ra) {
  uint q = P.a.y;
  uint sq = P.a.z;
  uint W = P.a.w;
  uint xs = x;
  uint k = f * P.b.x;
  // the codes W wide with a gap of P.c.z between them (../../encoder.mjs GAP_MODULES): a gap pixel takes the margin's
  // byte beside it
  while (xs >= W + P.c.z) { xs = xs - W - P.c.z; k = k + 1u; }
  if (xs >= W) { xs = W - 1u; }
#if !BORDER_ONLY
  if (xs >= sq && xs < sq + q && y >= sq && y < sq + q && !(ra && onAnchor(xs))) {
    uint xx = xs - sq;
    uint base = (k * q + (y - sq)) * P.a.x;
    float v = asfloat(TAPS[6u * q + 6u * xx]) * TMP[base + TAPS[6u * xx]];
    for (uint t = 1u; t < 6u; t = t + 1u) {
      v = v + asfloat(TAPS[6u * q + 6u * xx + t]) * TMP[base + TAPS[6u * xx + t]];
    }
    v = clamp(v, 0.0, 1.0);
    return min(uint(v * 255.0 + 0.5), 255u);
  }
#endif
  uint j = y * W + xs;
  return (BORDER[j >> 2u] >> (8u * (j & 3u))) & 0xffu;
}
[numthreads(LANES, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint wx = id.x;
  uint y = id.y;
  uint f = id.z;
  if (wx >= P.b.z) { return; }
#if BORDER_ONLY
  // IPIC's words: on a square row, from the word of the symbol's first square pixel to that of its last
  if (y >= P.a.z && y < P.a.z + P.a.y) {
    uint x0 = 0u;
    while (4u * wx >= x0 + P.a.w + P.c.z) { x0 = x0 + P.a.w + P.c.z; }
    if (wx >= ((x0 + P.a.z) >> 2u) && wx <= ((x0 + P.a.z + P.a.y - 1u) >> 2u)) { return; }
  }
#endif
  bool ra = onAnchor(y);
#if !BORDER_ONLY
  // a word inside a symbol's square on a row with no anchor: its four pixels' taps alone (a pixel's six samples run on
  // from its first, mod n, as the host lays TAPS' indices), no border and no anchor tests
  uint q = P.a.y;
  uint sq = P.a.z;
  if (!ra && y >= sq && y < sq + q) {
    uint x0 = 0u;
    uint k = f * P.b.x;
    while (4u * wx >= x0 + P.a.w + P.c.z) { x0 = x0 + P.a.w + P.c.z; k = k + 1u; }
    uint xs0 = 4u * wx - x0;
    if (xs0 >= sq && xs0 + 3u < sq + q) {
      uint n = P.a.x;
      uint base = (k * q + (y - sq)) * n;
      uint word = 0u;
      for (uint b = 0u; b < 4u; b = b + 1u) {
        uint xx = xs0 - sq + b;
        uint i = TAPS[6u * xx];
        float v = asfloat(TAPS[6u * q + 6u * xx]) * TMP[base + i];
        for (uint t = 1u; t < 6u; t = t + 1u) {
          i = i + 1u;
          if (i == n) { i = 0u; }
          v = v + asfloat(TAPS[6u * q + 6u * xx + t]) * TMP[base + i];
        }
        v = clamp(v, 0.0, 1.0);
        word = word | (min(uint(v * 255.0 + 0.5), 255u) << (8u * b));
      }
      FRAME[(P.c.y + f) * P.b.w + y * P.b.z + wx] = word;
      return;
    }
  }
#endif
  uint word = 0u;
  for (uint b = 0u; b < 4u; b = b + 1u) {
    uint x = 4u * wx + b;
    if (x < P.b.y) { word = word | (pixel(x, y, f, ra) << (8u * b)); }
  }
  FRAME[(P.c.y + f) * P.b.w + y * P.b.z + wx] = word;
}
