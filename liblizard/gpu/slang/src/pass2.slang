// Pass 2 of the back half's transform: a workgroup of N / 8 a (row, listed frame): row v of one frame along x (row 0
// pass 1's column means), its compact disc row stored to S (the rows table's (start, length, first run, gap)), the
// detrend (reduce's surface) taken off. Built a batch size, a picture size and a precision each; f16 reads Y and writes
// S as half2 (16-bit storage, storage buffers only). Defines (core/dec/build.cpp backHalf): F16 (0 or
// 1), LANES (N / 8, int), PICTURE_N, BATCH, LISTS_ROWS (uint); the row FFT's stages lib/fft_body.slang's, from the picture's size.
#if F16
#define STORE_T half2
#define LD(X) float2(X)
#define ST(X) half2(X)
#else
#define STORE_T float2
#define LD(X) X
#define ST(X) X
#endif
struct ParamsBlock { uint yStride; uint partStride; uint size; float m2; float sx2; float sq2; uint sStride; };
[[vk::binding(9, 0)]] ConstantBuffer<ParamsBlock> P;
[[vk::binding(0, 0)]] StructuredBuffer<uint> lists;
[[vk::binding(1, 0)]] StructuredBuffer<float2> tw;
[[vk::binding(2, 0)]] StructuredBuffer<float4> part;
[[vk::binding(3, 0)]] StructuredBuffer<STORE_T> yb;
[[vk::binding(4, 0)]] StructuredBuffer<float4> coef;
[[vk::binding(5, 0)]] StructuredBuffer<float4> x12;
struct RowsBlock { uint4 r[512]; };
[[vk::binding(6, 0)]] ConstantBuffer<RowsBlock> rows;
[[vk::binding(7, 0)]] RWStructuredBuffer<STORE_T> sb;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> counts;
static const uint N = PICTURE_N;
static const uint T = PICTURE_N / 8u;
static const uint LS = BATCH + 1u;
static const uint ROWS_AT = LISTS_ROWS;
groupshared STORE_T buf[N];
#include "lib/butterflies.slang"
#define FFT_LANES_2 0
#define FFT_BUF0 buf
#include "lib/fft.slang"
[numthreads(LANES, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint tid = lix;
  uint f = lists[P.size * LS + 1u + wg.z];
  uint vv = wg.x;
  if (vv >= lists[ROWS_AT + f]) { return; }
  float2 v[8] = { float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0) };
  if (vv == 0u) {
    for (uint r = 0u; r < 8u; r++) { v[r] = float2(part[f * P.partStride + tid + r * T].w, 0.0); }
  } else {
    uint ybase = f * P.yStride + (vv - 1u) * N;
    for (uint r = 0u; r < 8u; r++) { v[r] = LD(yb[ybase + tid + r * T]); }
  }
#define FFT_V0 v
#include "lib/fft_body.slang"
  uint4 row = rows.r[vv];
  float4 c0 = coef[f * 2u];
  float4 c1 = coef[f * 2u + 1u];
  float4 xv = x12[vv];
  float nf = float(N);
  uint ustart = (vv == 0u) ? 1u : 0u;
  for (uint e = tid; e < row.y; e += T) {
    int u = 0;
    if (e < row.z) { u = int(e + ustart); } else { u = int(e) - int(row.z) - int(row.w); }
    uint m = uint(u + int(N)) % N;
    float4 xu = x12[m];
    float2 d = c0.z * cmul(xu.xy, xv.xy);
    if (vv == 0u) { d += nf * (c0.x * xu.xy + c0.w * xu.zw); }
    if (u == 0) { d += nf * (c0.y * xv.xy + c1.x * xv.zw); }
    float2 s = (LD(buf[m]) - d / nf) / nf;
    sb[f * P.sStride + row.x + e] = ST(s);
  }
  if (tid == 0u) {
    InterlockedAdd(counts[f * 8u + 1u], 1u);
    InterlockedAdd(counts[f * 8u + 2u], row.y);
  }
}
