// The sender's IROWS, the first inverse: a workgroup a (disc row, symbol), N / 8 lanes. The disc is the spectrum
// transposed (../../back/disc.mjs sendDiscLayout), so its rows are the picture's columns and this pass runs along the
// picture's y; IPIC then runs along its rows. The row's coefficients at (u + N) mod N, zeros elsewhere, conjugated in,
// the inverse (the decoder's forward FFT on the conjugate, unscaled, f32: gen/fft, ../../glsl/back_transform.mjs firstStage and
// laterStages), conjugated out, the row stored in Y quad-major (Y[x / 4][v][x mod 4], x along this pass: IPIC's
// workgroup reads its quad's rows as one run). Built a picture size each.
// Bindings: S (ro, the disc's compact rows), the twiddles, Y (rw), the rows table (a row's (start, length, first run,
// gap)), the size's uniform (a = (0, the disc's rows, Y a symbol, S a symbol), b, c).
// Defines (../send.mjs irowsSource): LANES (N / 8, int), PICTURE_N (uint).
[[vk::binding(0, 0)]] StructuredBuffer<float2> sb;
[[vk::binding(1, 0)]] StructuredBuffer<float2> tw;
[[vk::binding(2, 0)]] RWStructuredBuffer<float2> yb;
struct RowsBlock { uint4 r[512]; };
[[vk::binding(3, 0)]] ConstantBuffer<RowsBlock> rows;
struct SizeParamsBlock { uint4 a; uint4 b; float4 c; };
[[vk::binding(4, 0)]] ConstantBuffer<SizeParamsBlock> P;
static const uint N = PICTURE_N;
static const uint T = PICTURE_N / 8u;
groupshared float2 buf[N];
#include "lib/butterflies.slang"
// Register K holds x = tid + K T: the row's entry at u = x (x < N / 2) or u = x - N, conjugated; zero off the row.
#define LOAD(K) \
  { \
    uint x = tid + uint(K) * T; \
    float2 val = float2(0.0); \
    if (x < N / 2u) { \
      if (x >= ustart && x < ustart + row.z) { val = sb[sBase + x - ustart]; } \
    } else { \
      uint un = N - x; \
      if (un <= row.w) { val = sb[sBase + row.z + row.w - un]; } \
    } \
    v[K] = float2(val.x, -val.y); \
  }
[numthreads(LANES, 1, 1)]
void main(uint3 wg : SV_GroupID, uint tid : SV_GroupIndex) {
  uint r = wg.z;
  uint vv = wg.x;
  uint4 row = rows.r[vv];
  uint ustart = vv == 0u ? 1u : 0u;
  uint sBase = r * P.a.w + row.x;
  float2 v[8] = { float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0) };
  LOAD(0) LOAD(1) LOAD(2) LOAD(3) LOAD(4) LOAD(5) LOAD(6) LOAD(7)
#include "gen/fft"
  // quad-major: a quad of columns spans the disc's rows (P.a.y) four values each
  uint yBase = r * P.a.z + vv * 4u;
  uint QS = P.a.y * 4u;
  for (uint x = tid; x < N; x = x + T) {
    float2 z = buf[x];
    yb[yBase + (x >> 2u) * QS + (x & 3u)] = float2(z.x, -z.y);
  }
}
