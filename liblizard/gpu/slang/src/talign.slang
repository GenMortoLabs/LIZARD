// talign: a workgroup a listed frame, its TIER_BLOCKS lanes over the frame's sub-channels (j = t, t + TIER_BLOCKS,
// ...), each sub-channel's two pilots (coefficients 0 and PB_REF) against their whitening, the fit's sums reduced,
// then again about the first answer: the C's pilot_align, each pilot's terms weighted by 1 / N of its sub-channel
// (least squares with the noise each pilot reads), tnoise's, EP's fourth float. Unweighted, a sub-channel a
// glare's blob swells (the innermost, where the blob's spectrum lies) counts most where it reads least.
// Defines (core/dec/build.cpp backHalf): F16 (0 or 1), TIER_BLOCKS, STEP, PILOT_READ, SLOT_COUNT (uint), TWO_PI
// (float); lib/tab.slang's.
#if F16
#define S_T half2
#else
#define S_T float2
#endif
#include "lib/lists.slang"
[[vk::binding(0, 0)]] StructuredBuffer<S_T> S;
[[vk::binding(1, 0)]] StructuredBuffer<uint> UV;
[[vk::binding(2, 0)]] StructuredBuffer<uint> LISTS;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> PILOT;
[[vk::binding(4, 0)]] StructuredBuffer<uint> TAB;
struct ParamsUBlock { uint s; uint uvOff; uint sStride; uint lStride; uint subchMax; uint blocksMax; uint B; uint ucOff; uint n; };
[[vk::binding(5, 0)]] ConstantBuffer<ParamsUBlock> P;
[[vk::binding(6, 0)]] StructuredBuffer<float4> EP;
#include "lib/tab.slang"
groupshared float4 acc0[TIER_BLOCKS];
groupshared float4 acc1[TIER_BLOCKS];
groupshared float2 cur;
groupshared float4 last0;
groupshared float4 last1;
float2 solved(float4 a0, float4 a1) {
  float det = a0.x * a0.z - a0.y * a0.y;
  if (a1.z >= 8.0 && det > 0.0) { return float2((a0.w * a0.z - a1.x * a0.y) / det, (a0.x * a1.x - a0.y * a0.w) / det); }
  return float2(0.0);
}
float2 uvOf(uint w) { return float2(float(int(w) >> 16u), float(w & 0xffffu)); }
float2 turnedBy(float2 y, float th) {
  float cs = cos(th);
  float sn = sin(th);
  return float2(y.x * cs + y.y * sn, y.y * cs - y.x * sn);
}

// The fit's sums over the frame's pilots (a lane over sub-channels), each pilot's terms weighted by 1 / N of its
// sub-channel (EP.w); TURN the pilots turned back by the first answer (TURN_BY) or not (TURN_NONE).
#define TURN_NONE
#define TURN_BY z = turnedBy(z, kk * dot(uv, d));
#define FIT(TURN) \
  s0 = float4(0.0); \
  s1 = float4(0.0); \
  for (uint j = t; j < subch; j += TIER_BLOCKS) { \
    float wj = EP[f * P.subchMax + j].w; \
    for (uint e = 0u; e < 2u; e++) { \
      uint c = e == 0u ? 0u : PB_REF; \
      float2 y = float2(S[f * P.sStride + UV[P.uvOff + 320u * j + c]]); \
      uint w = whiteOf(j, c); \
      float2 x = float2((w & 1u) != 0u ? -1.0 : 1.0, (w & 2u) != 0u ? -1.0 : 1.0); \
      float2 z = float2(y.x * x.x + y.y * x.y, y.y * x.x - y.x * x.y); \
      float2 uv = uvOf(UV[P.ucOff + 320u * j + c]); \
      TURN \
      float cu = kk * z.x * uv.x; \
      float cv = kk * z.x * uv.y; \
      s0 += wj * float4(cu * cu, cu * cv, cv * cv, cu * z.y); \
      s1 += float4(wj * cv * z.y, 0.0, 1.0, 0.0); \
    } \
  }
// The sums reduced over the workgroup, then in lane 0 THEN: KEEP_SOLVED the first answer into cur, KEEP_SUMS the second
// round's sums into last0 and last1.
#define KEEP_SOLVED cur = solved(a0, a1);
#define KEEP_SUMS last0 = a0; last1 = a1;
#define REDUCE(THEN) \
  acc0[t] = s0; \
  acc1[t] = s1; \
  GroupMemoryBarrierWithGroupSync(); \
  if (t < 16u) { \
    float4 a0 = float4(0.0); \
    float4 a1 = float4(0.0); \
    for (uint i = t; i < TIER_BLOCKS; i += 16u) { a0 += acc0[i]; a1 += acc1[i]; } \
    acc0[t] = a0; \
    acc1[t] = a1; \
  } \
  GroupMemoryBarrierWithGroupSync(); \
  if (t == 0u) { \
    float4 a0 = float4(0.0); \
    float4 a1 = float4(0.0); \
    for (uint i = 0u; i < 16u; i++) { a0 += acc0[i]; a1 += acc1[i]; } \
    THEN \
  } \
  GroupMemoryBarrierWithGroupSync();
[numthreads(TIER_BLOCKS, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint t = lix;
  uint count = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  uint f = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  uint2 v = CFG_AT(LISTS, P.B, f);
  uint subch = STEP * cfgVersion(v);
  float kk = TWO_PI / float(P.n);
  float4 s0;
  float4 s1;
  float2 d = float2(0.0);
  FIT(TURN_NONE)
  REDUCE(KEEP_SOLVED)
  d = cur;
  FIT(TURN_BY)
  REDUCE(KEEP_SUMS)
  if (t == 0u) {
    float2 outv = float2(0.0);
    float4 a0 = last0;
    float4 a1 = last1;
    float det = a0.x * a0.z - a0.y * a0.y;
    if (a1.z >= 8.0 && det > 0.0) {
      float2 e = float2((a0.w * a0.z - a1.x * a0.y) / det, (a0.x * a1.x - a0.y * a0.w) / det);
      float2 dd = d + e;
      // no NaN: a NaN's exponent bits all set and its mantissa not zero, tested on the bits (no float optimisation
      // can take the test away)
      if (all((asuint(dd) & uint2(0x7fffffffu)) <= uint2(0x7f800000u))) { outv = dd; }
    }
    PILOT[PILOT_READ * P.B + 2u * f] = outv.x;
    PILOT[PILOT_READ * P.B + 2u * f + 1u] = outv.y;
  }
}
