// The sender's IPIC, a workgroup a (quad of rows, symbol), N / 8 lanes. The disc is the spectrum transposed (paint,
// ../../back/disc.mjs sendDiscLayout), so IROWS' rows were its columns and this pass runs along the picture's rows:
// turn_pack's packing takes two rows through one complex inverse (Z(0) = 2 Re A(0) + 2i Re B(0), Z(v) = A + iB,
// Z(N - v) = conj A + i conj B, zero between; zAt), the real part row a and the imaginary row b, clipped to 0.5 +
// clamp(v, -lim, lim) 0.5 / lim (src/lizard_phy.c lizard_encode, turn_pack, clip_row). The inverse: the decoder's
// forward FFT on the conjugate, unscaled, f32, two lanes (../back_transform.mjs firstStage and laterStages).
// Built a picture size each, in two forms:
// - FRAME_OUT 0: the quad's four rows into PIC row-major, a lane a column (whole lines stored), for RSV and RSH.
// - FRAME_OUT 1, the copy form (every pixel's tap the C's on-sample one, frame square pixel x the picture's column
//   (x + TAPS[2]) mod n, rows alike: src/lizard_phy.c resample where the ring's span divides n): the quad's rows
//   straight into the frame's grey bytes, as RSH's copy form made them (clamped to 0..1, (v 255 + 0.5) truncated and
//   held to 255, lizard_paint_rgba), from workgroup memory: each picture row onto the one or two square rows it
//   shows on (the square is q > n pixels: its wrap), every word of those rows that holds a pixel of the symbol's
//   square, a pixel outside the square or on an anchor's square taking BORDER's byte (RSH's rule). RSH's border form
//   writes every other word, so each word has one writer.
// Bindings: Y (ro), the twiddles, PIC (rw, f32 row-major; FRAME_OUT: FRAME, rw, a half's frames FS words apart), the
// size's uniform (a = (0, the disc's rows V, Y a symbol, 0), b = (PIC a symbol in vec4s, ...), c = (the clip level,
// the scale, 0, 0)); FRAME_OUT: RSH's G uniform with the anchors (../send_rsh.slang), BORDER (ro), TAPS (ro).
// Defines (../send.mjs ipicSource): LANES (N / 8, int), PICTURE_N (uint), FRAME_OUT (0 or 1).
[[vk::binding(0, 0)]] StructuredBuffer<float2> yb;
[[vk::binding(1, 0)]] StructuredBuffer<float2> tw;
#if FRAME_OUT
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> FRAME;
#else
[[vk::binding(2, 0)]] RWStructuredBuffer<float> PIC;
#endif
struct SizeParamsBlock { uint4 a; uint4 b; float4 c; };
[[vk::binding(3, 0)]] ConstantBuffer<SizeParamsBlock> P;
#if FRAME_OUT
struct GBlock { uint4 a; uint4 b; uint4 c; uint4 d; uint4 e; };
[[vk::binding(4, 0)]] ConstantBuffer<GBlock> GA;
[[vk::binding(5, 0)]] StructuredBuffer<uint> BORDER;
[[vk::binding(6, 0)]] StructuredBuffer<uint> TAPS;
#endif
static const uint N = PICTURE_N;
static const uint T = PICTURE_N / 8u;
groupshared float2 buf0[N];
groupshared float2 buf1[N];
#include "lib/butterflies.slang"
float2 zAt(uint m, uint yBase, uint xa, uint xb, uint V) {
  if (m == 0u) {
    float2 a = yb[yBase + xa];
    float2 b = yb[yBase + xb];
    return float2(2.0 * a.x, 2.0 * b.x);
  }
  if (m < V) {
    float2 a = yb[yBase + m * 4u + xa];
    float2 b = yb[yBase + m * 4u + xb];
    return float2(a.x - b.y, a.y + b.x);
  }
  if (m > N - V) {
    uint vq = N - m;
    float2 a = yb[yBase + vq * 4u + xa];
    float2 b = yb[yBase + vq * 4u + xb];
    return float2(a.x + b.y, b.x - a.y);
  }
  return float2(0.0);
}
// Registers K of the two lanes: point m = tid + K T of rows (0, 1) and (2, 3), conjugated
#define LOAD(K) \
  { \
    uint m = tid + uint(K) * T; \
    float2 z0 = zAt(m, yBase, 0u, 1u, V); \
    float2 z1 = zAt(m, yBase, 2u, 3u, V); \
    v0[K] = float2(z0.x, -z0.y); \
    v1[K] = float2(z1.x, -z1.y); \
  }
[numthreads(LANES, 1, 1)]
void main(uint3 wg : SV_GroupID, uint tid : SV_GroupIndex) {
  uint r = wg.z;
  uint V = P.a.y;
  uint yBase = r * P.a.z + wg.x * V * 4u;
  float2 v0[8] = { float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0) };
  float2 v1[8] = { float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0) };
  LOAD(0) LOAD(1) LOAD(2) LOAD(3) LOAD(4) LOAD(5) LOAD(6) LOAD(7)
#include "gen/fft"
  // conj out: row a is the real part, row b the imaginary; clip_row's clamp and map to 0..1
  float lim = P.c.x;
  float sc = P.c.y;
#if FRAME_OUT
  // the frame: symbol r is frame r / codes's code r mod codes, its symbol x0 pixels in (RSH's codes side by side, the
  // gap between); a square row y (0 .. q) shows the picture's row (y + off) mod n, off = TAPS[2] (the first pixel's
  // sample, already mod n), so picture row p shows on square rows (p - off) mod n and that + n where it is under q
  uint q = GA.a.y;
  uint sq = GA.a.z;
  uint W = GA.a.w;
  uint codes = GA.b.x;
  uint RW = GA.b.z;
  uint x0 = (r % codes) * (W + GA.c.z);
  uint fbase = (GA.c.y + r / codes) * GA.b.w;
  uint off = TAPS[2];
  uint w0 = (x0 + sq) >> 2u;
  uint nw = ((x0 + sq + q - 1u) >> 2u) + 1u - w0;
  for (uint j = 0u; j < 4u; j = j + 1u) {
    uint p = 4u * wg.x + j;
    uint y = p + N - off;
    if (y >= N) { y = y - N; }
    for (; y < q; y = y + N) {
      uint fy = sq + y;
      // a row of an anchor's square: its pixels on an anchor's column take BORDER's byte (send_rsh.slang onAnchor)
      uint s = GA.e.w;
      bool ra = (fy - GA.d.x < s) || (fy - GA.d.y < s) || (fy - GA.d.z < s) || (fy - GA.d.w < s) || (fy - GA.e.x < s) || (fy - GA.e.y < s) || (fy - GA.e.z < s);
      for (uint k = tid; k < nw; k = k + T) {
        uint wx = w0 + k;
        uint xs0 = 4u * wx - x0;
        uint word = 0u;
        if (!ra && xs0 >= sq && xs0 + 3u < sq + q) {
          // the word inside the square, no anchor on this row: four samples from workgroup memory
          uint c = xs0 - sq + off;
          if (c >= N) { c = c - N; }
          if (c >= N) { c = c - N; }
          for (uint b = 0u; b < 4u; b = b + 1u) {
            float2 o = j < 2u ? buf0[c] : buf1[c];
            float raw = (j & 1u) == 0u ? o.x : -o.y;
            float v = clamp(0.5 + clamp(raw, -lim, lim) * sc, 0.0, 1.0);
            word = word | (min(uint(v * 255.0 + 0.5), 255u) << (8u * b));
            c = c + 1u;
            if (c == N) { c = 0u; }
          }
        } else {
          for (uint b = 0u; b < 4u; b = b + 1u) {
            uint xs = xs0 + b;
            uint byte;
            // an anchor's column (send_rsh.slang onAnchor's runs), read only on an anchor's row
            if (xs >= sq && xs < sq + q && !(ra && ((xs - GA.d.x < s) || (xs - GA.d.y < s) || (xs - GA.d.z < s) || (xs - GA.d.w < s) || (xs - GA.e.x < s) || (xs - GA.e.y < s) || (xs - GA.e.z < s)))) {
              uint c = xs - sq + off;
              if (c >= N) { c = c - N; }
              if (c >= N) { c = c - N; }
              float2 o = j < 2u ? buf0[c] : buf1[c];
              float raw = (j & 1u) == 0u ? o.x : -o.y;
              float v = clamp(0.5 + clamp(raw, -lim, lim) * sc, 0.0, 1.0);
              byte = min(uint(v * 255.0 + 0.5), 255u);
            } else {
              uint at = fy * W + xs;
              byte = (BORDER[at >> 2u] >> (8u * (at & 3u))) & 0xffu;
            }
            word = word | (byte << (8u * b));
          }
        }
        FRAME[fbase + fy * RW + wx] = word;
      }
    }
  }
#else
  // a row's lanes store adjacent columns
  uint row0 = r * 4u * P.b.x + 4u * wg.x * N;
  for (uint x = tid; x < N; x = x + T) {
    float2 o0 = buf0[x];
    float2 o1 = buf1[x];
    float4 v = 0.5 + clamp(float4(o0.x, -o0.y, o1.x, -o1.y), float4(-lim), float4(lim)) * sc;
    PIC[row0 + x] = v.x;
    PIC[row0 + N + x] = v.y;
    PIC[row0 + 2u * N + x] = v.z;
    PIC[row0 + 3u * N + x] = v.w;
  }
#endif
}
