// The sender's resampler onto the frame (RSH): the frame's u32 words, four bytes each, the codes symbols side by side
// (FW = codes W pixels a row, W rows, each row padded to RW words: W need not be a multiple of four, as with pxm 1 in the
// 256 ring, 546 pixels). Inside a symbol's square (q pixels from sq) a pixel is the C's Lanczos-3 resample of the
// picture (src/lizard_phy.c resample, the taps ../../../sim/ob.mjs resampleGeom: the same six taps along both axes, a
// pixel's samples' indices then its weights, the indices wrapped mod n on the host so the picture reads as periodic
// and the q pixels cover it and its guard interval); clamped to 0..1, (drive 255 + 0.5) truncated and held to 255 as
// lizard_paint_rgba does. Anywhere else the border's byte (BORDER, one symbol W x W, the C's own paint: ring, marks,
// band, word, quiet margin, the anchors' squares), so the border and the anchors are the C's to the byte and nothing
// border-shaped is drawn here. A pixel on an anchor's row and column takes BORDER's byte (seven wrapped differences
// against the side: integers alone, so no device's rounding moves an anchor's edge). No integer division: the symbol
// by subtraction.
// The resampling form: a workgroup a strip of LANES words (a lane a word) by a band of BAND rows. Each lane reads its
// four pixels' horizontal taps once, into registers, for every row of the band (a pixel's seven tap words read again
// at every row cost the stage over 40% on the iGPU, 2026-11-24); each square row's vertical pass is made for the
// strip's own samples alone, a code's from the first sample its leftmost pixel takes, into workgroup memory (a strip
// takes at most 2 LANES 4 + 8 samples a code: n at most 2 q), then the band's next row.
// BORDER_ONLY 1, the border form (the copy form's, where IPIC's frame form writes the squares: ../send_ipic.slang
// FRAME_OUT): a lane a word, the words IPIC writes (a square row's words that hold a pixel of a symbol's square) left,
// every other word the border's bytes alone; PIC not read.
// Bindings: PIC (ro, f32, n x n a symbol, row-major; unread in the border form), TAPS (ro: idx, 6 q words, then w, 6 q
// floats' bits), BORDER (ro, one symbol's bytes, four a word), FRAME (rw, a half's frames FS words apart), the G uniform
// with the anchors (GA: a = (n, q, sq, W), b = (codes, FW, RW, FS), c = (whole, f0, gap, 0), d = the first pixel of
// anchors 0 to 3 along either axis of a symbol, e = anchors 4 to 6's and the square's side; 80 B). Dispatch
// (ceil(RW / LANES), ceil(W / BAND), frames); the border form (ceil(RW / LANES), W, frames).
// Defines (core/tx/send_plan.h): BORDER_ONLY (0 or 1), LANES (int, RSH_THREADS), BAND (uint, the
// rows a workgroup, RSH_BAND), STRIP (uint, a code's samples a row, RSH_STRIP), CODES_MAX (uint).
[[vk::binding(0, 0)]] StructuredBuffer<float> PIC;
[[vk::binding(1, 0)]] StructuredBuffer<uint> TAPS;
[[vk::binding(2, 0)]] StructuredBuffer<uint> BORDER;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> FRAME;
struct GBlock { uint4 a; uint4 b; uint4 c; uint4 d; uint4 e; };
[[vk::binding(4, 0)]] ConstantBuffer<GBlock> P;
#if !BORDER_ONLY
// the row's vertical pass over the strip's samples, code kl's from kl STRIP; the strip's leftmost and rightmost square
// pixel of each code (atomics, the band's setup)
groupshared float strip[CODES_MAX * STRIP];
groupshared Atomic<uint> xmin[CODES_MAX];
groupshared Atomic<uint> xmax[CODES_MAX];
#endif
// whether v, a symbol's pixel along either axis, lies on one of the anchors' seven runs (P.d, P.e.xyz their first
// pixels, P.e.w long)
bool onAnchor(uint v) {
  uint s = P.e.w;
  return (v - P.d.x < s) || (v - P.d.y < s) || (v - P.d.z < s) || (v - P.d.w < s) || (v - P.e.x < s) || (v - P.e.y < s) || (v - P.e.z < s);
}
// the border's byte of frame pixel x on row y: the codes W wide with a gap of P.c.z between them (../../encoder.mjs
// GAP_MODULES), a gap pixel the margin's byte beside it
uint border(uint x, uint y) {
  uint W = P.a.w;
  uint xs = x;
  while (xs >= W + P.c.z) { xs = xs - W - P.c.z; }
  if (xs >= W) { xs = W - 1u; }
  uint j = y * W + xs;
  return (BORDER[j >> 2u] >> (8u * (j & 3u))) & 0xffu;
}
[numthreads(LANES, 1, 1)]
void main(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
  uint f = gid.z;
  uint q = P.a.y;
  uint sq = P.a.z;
  uint RW = P.b.z;
#if BORDER_ONLY
  // a lane a word, ceil(RW / LANES) workgroups a row
  uint y = gid.y;
  uint wx = gid.x * LANES + li;
  if (wx >= RW) { return; }
  // IPIC's words: on a square row, from the word of the symbol's first square pixel to that of its last
  if (y >= sq && y < sq + q) {
    uint x0 = 0u;
    while (4u * wx >= x0 + P.a.w + P.c.z) { x0 = x0 + P.a.w + P.c.z; }
    if (wx >= ((x0 + sq) >> 2u) && wx <= ((x0 + sq + q - 1u) >> 2u)) { return; }
  }
  uint word = 0u;
  for (uint b = 0u; b < 4u; b = b + 1u) {
    uint x = 4u * wx + b;
    if (x < P.b.y) { word = word | (border(x, y) << (8u * b)); }
  }
  FRAME[(P.c.y + f) * P.b.w + y * RW + wx] = word;
#else
  uint n = P.a.x;
  uint W = P.a.w;
  uint codes = P.b.x;
  uint wx = gid.x * LANES + li;
  bool live = wx < RW;
  // the lane's four pixels: where each lies (its code, its column in the square or not, on an anchor's column), and a
  // square pixel's taps, read once for the band
  uint kls[4];
  uint xss[4];
  uint xxs[4];
  bool inq[4];
  bool acol[4];
  uint first[4];
  float w[24];
  if (li < CODES_MAX) { xmin[li].store(0xffffffffu); xmax[li].store(0u); }
  GroupMemoryBarrierWithGroupSync();
  [ForceUnroll] for (uint b = 0u; b < 4u; b = b + 1u) {
    uint x = 4u * wx + b;
    uint xs = x;
    uint kl = 0u;
    while (xs >= W + P.c.z) { xs = xs - W - P.c.z; kl = kl + 1u; }
    if (xs >= W) { xs = W - 1u; }
    kls[b] = kl;
    xss[b] = xs;
    inq[b] = live && x < P.b.y && xs >= sq && xs < sq + q;
    acol[b] = onAnchor(xs);
    uint xx = inq[b] ? xs - sq : 0u;
    xxs[b] = xx;
    first[b] = TAPS[6u * xx];
    [ForceUnroll] for (uint t = 0u; t < 6u; t = t + 1u) { w[6u * b + t] = asfloat(TAPS[6u * q + 6u * xx + t]); }
    if (inq[b]) { xmin[kl].min(xx); xmax[kl].max(xx); }
  }
  GroupMemoryBarrierWithGroupSync();
  // each code's strip: from the first sample its leftmost pixel takes, as many as its rightmost's sixth less that, mod n
  uint start[CODES_MAX];
  uint span[CODES_MAX];
  [ForceUnroll] for (uint k = 0u; k < CODES_MAX; k = k + 1u) {
    uint lo = xmin[k].load();
    uint hi = xmax[k].load();
    if (lo <= hi) {
      start[k] = TAPS[6u * lo];
      uint e = TAPS[6u * hi] + (TAPS[6u * hi] < start[k] ? n : 0u);
      span[k] = min(e - start[k] + 6u, STRIP);
    } else { start[k] = 0u; span[k] = 0u; }
  }
  // a square pixel's samples from its offset in its code's strip
  uint off[4];
  [ForceUnroll] for (uint b = 0u; b < 4u; b = b + 1u) {
    uint s0 = start[kls[b] < CODES_MAX ? kls[b] : 0u];
    off[b] = kls[b] * STRIP + (first[b] + (first[b] < s0 ? n : 0u) - s0);
  }
  uint y0 = gid.y * BAND;
  for (uint r = 0u; r < BAND; r = r + 1u) {
    uint y = y0 + r;
    if (y >= W) { break; }
    bool sqRow = y >= sq && y < sq + q;
    bool ra = onAnchor(y);
    if (sqRow) {
      // the vertical pass of this row over each code's strip: its six picture rows and weights the same for every sample
      uint yy = y - sq;
      uint i0 = TAPS[6u * yy], i1 = TAPS[6u * yy + 1u], i2 = TAPS[6u * yy + 2u], i3 = TAPS[6u * yy + 3u], i4 = TAPS[6u * yy + 4u], i5 = TAPS[6u * yy + 5u];
      float v0 = asfloat(TAPS[6u * q + 6u * yy]), v1 = asfloat(TAPS[6u * q + 6u * yy + 1u]), v2 = asfloat(TAPS[6u * q + 6u * yy + 2u]);
      float v3 = asfloat(TAPS[6u * q + 6u * yy + 3u]), v4 = asfloat(TAPS[6u * q + 6u * yy + 4u]), v5 = asfloat(TAPS[6u * q + 6u * yy + 5u]);
      [ForceUnroll] for (uint k = 0u; k < CODES_MAX; k = k + 1u) {
        uint pic = (f * codes + k) * n * n;
        for (uint o = li; o < span[k]; o = o + LANES) {
          uint i = start[k] + o;
          if (i >= n) { i = i - n; }
          float v = v0 * PIC[pic + i0 * n + i];
          v = v + v1 * PIC[pic + i1 * n + i];
          v = v + v2 * PIC[pic + i2 * n + i];
          v = v + v3 * PIC[pic + i3 * n + i];
          v = v + v4 * PIC[pic + i4 * n + i];
          v = v + v5 * PIC[pic + i5 * n + i];
          strip[k * STRIP + o] = v;
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (live) {
      uint word = 0u;
      [ForceUnroll] for (uint b = 0u; b < 4u; b = b + 1u) {
        uint x = 4u * wx + b;
        uint px = 0u;
        if (sqRow && inq[b] && !(ra && acol[b])) {
          uint o = off[b];
          float v = w[6u * b] * strip[o];
          [ForceUnroll] for (uint t = 1u; t < 6u; t = t + 1u) { v = v + w[6u * b + t] * strip[o + t]; }
          v = clamp(v, 0.0, 1.0);
          px = min(uint(v * 255.0 + 0.5), 255u);
        } else if (x < P.b.y) {
          uint j = y * W + xss[b];
          px = (BORDER[j >> 2u] >> (8u * (j & 3u))) & 0xffu;
        }
        word = word | (px << (8u * b));
      }
      FRAME[(P.c.y + f) * P.b.w + y * RW + wx] = word;
    }
    GroupMemoryBarrierWithGroupSync();
  }
#endif
}
