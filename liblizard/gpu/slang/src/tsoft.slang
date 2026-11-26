// tsoft: a workgroup an 8-sub-channel chunk of a listed frame, 32 lanes a sub-channel, lane l its PAIRS pairs m = l +
// 32 k (k 0 to 4) of data coefficients (2 m, 2 m + 1): the 159 pairs of the sub-channel's 318, and in the last
// (l 31, k 4) its two pilots (coefficients 0 and PB_REF); each coefficient turned back by the shift and whitened
// by its frame slot, scaled by tnoise's EP.y, the soft values stored in data-slot order (L's word 159 j + m: data
// slots 4 (159 j + m) to the next three), and the two pilots' values (the C's pil_c, pil_r: the turned value
// against its whitening, x + y) into EP.yz, once every lane has read EP.y (the barrier), and the scale into EP.w
// (talign read tnoise's 1 / N there before), the pilots' weight in tblocks' readings; EP.x, tnoise's estimate,
// stays for tblocks.
// The pairs' reads, turns and whitening and their stores are unrolled: PAIRS (5) pairs a lane, 32 lanes a sub-channel's
// 320 coefficients, the last lane's last pair the pilots.
// Defines (core/dec/build.cpp backHalf): F16 (0 or 1), CHUNKS, PILOT_READ, DCOEF, SLOT_COUNT (uint), TWO_PI (float);
// lib/tab.slang's.
#if F16
#define S_T half2
#else
#define S_T float2
#endif
#include "lib/lists.slang"
static const int PAIRS = 5;
[[vk::binding(0, 0)]] StructuredBuffer<S_T> S;
[[vk::binding(1, 0)]] StructuredBuffer<uint> UV;
[[vk::binding(2, 0)]] StructuredBuffer<uint> LISTS;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> L;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> EP;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> COUNTS;
[[vk::binding(6, 0)]] StructuredBuffer<float> PILOT;
[[vk::binding(7, 0)]] StructuredBuffer<uint> TAB;
struct ParamsUBlock { uint s; uint uvOff; uint sStride; uint lStride; uint subchMax; uint blocksMax; uint B; uint ucOff; uint n; };
[[vk::binding(8, 0)]] ConstantBuffer<ParamsUBlock> P;
#include "lib/tab.slang"
#include "lib/round.slang"
float2 turned(float2 y, float th) {
  float cs = cos(th);
  float sn = sin(th);
  return float2(y.x * cs + y.y * sn, y.y * cs - y.x * sn);
}

[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint t = lix;
  uint b = wg.x;
  uint count = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  uint f = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  uint2 v = CFG_AT(LISTS, P.B, f);
  if (b >= CHUNKS * cfgVersion(v)) { return; }
  uint g = t >> 5u;
  uint l = t & 31u;
  uint j = 8u * b + g;
  uint ub = P.uvOff + 320u * j;
  uint cb = P.ucOff + 320u * j;
  uint sBase = f * P.sStride;
  float2 sh = float2(PILOT[PILOT_READ * P.B + 2u * f], PILOT[PILOT_READ * P.B + 2u * f + 1u]);
  bool turn = sh.x != 0.0 || sh.y != 0.0;
  float kk = TWO_PI / float(P.n);
  // the lane whose last pair is the sub-channel's pilots
  bool pil = l == 31u;
  // the scale tnoise made from the sub-channel's moments, read by every lane before the barrier in the body, after
  // which the pilot lane writes the pilots' values over it
  float kq = EP[f * P.subchMax + j].y;
  // each pair's two coefficients (the last lane's last pair the pilots), their values, turned back by the shift and
  // whitened by their slots
  uint ca[PAIRS], cbv[PAIRS];
  [ForceUnroll] for (int k = 0; k < PAIRS; k++) {
    if (k < PAIRS - 1) { ca[k] = fcoefIn(2u * (l + 32u * uint(k))); cbv[k] = fcoefIn(2u * (l + 32u * uint(k)) + 1u); }
    else { ca[k] = pil ? 0u : fcoefIn(2u * (l + 32u * uint(k))); cbv[k] = pil ? PB_REF : fcoefIn(2u * (l + 32u * uint(k)) + 1u); }
  }
  float2 y[2 * PAIRS];
  [ForceUnroll] for (int k = 0; k < PAIRS; k++) { y[2 * k] = float2(S[sBase + UV[ub + ca[k]]]); y[2 * k + 1] = float2(S[sBase + UV[ub + cbv[k]]]); }
  if (turn) {
    [ForceUnroll] for (int c = 0; c < 2 * PAIRS; c++) {
      uint w = UV[cb + ((c & 1) != 0 ? cbv[c >> 1] : ca[c >> 1])];
      y[c] = turned(y[c], kk * (float(int(w) >> 16u) * sh.x + float(w & 0xffffu) * sh.y));
    }
  }
  [ForceUnroll] for (int c = 0; c < 2 * PAIRS; c++) {
    uint n = whiteOf(j, (c & 1) != 0 ? cbv[c >> 1] : ca[c >> 1]);
    y[c] = float2((n & 1u) != 0u ? -y[c].x : y[c].x, (n & 2u) != 0u ? -y[c].y : y[c].y);
  }
  GroupMemoryBarrierWithGroupSync();
  if (pil) { EP[f * P.subchMax + j].yzw = float3(y[2 * PAIRS - 2].x + y[2 * PAIRS - 2].y, y[2 * PAIRS - 1].x + y[2 * PAIRS - 1].y, kq); }
  uint lBase = f * P.lStride + DCOEF / 2u * j + l;
  // the soft values, four a word in data-slot order (the pilot lane's last pair none)
  [ForceUnroll] for (int k = 0; k < PAIRS; k++) {
    if (k < PAIRS - 1 || !pil) {
      int2 r0 = int2(roundEven(clamp(kq * y[2 * k], float2(-10.0), float2(10.0)) * 8.0));
      int2 r1 = int2(roundEven(clamp(kq * y[2 * k + 1], float2(-10.0), float2(10.0)) * 8.0));
      L[lBase + 32u * uint(k)] = (uint(r0.x) & 0xffu) | ((uint(r0.y) & 0xffu) << 8u) | ((uint(r1.x) & 0xffu) << 16u) | ((uint(r1.y) & 0xffu) << 24u);
    }
  }
  if (t == 0u) { InterlockedAdd(COUNTS[f * 8u + 6u], 8u); }
}
