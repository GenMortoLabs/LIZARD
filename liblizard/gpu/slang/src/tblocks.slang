// tblocks: a workgroup a listed frame. A lane a block: its estimate, its sub-channels' (EP.x) each by the data slots of
// the block it holds, and its decline. Then every lane over the frame's sub-channels, the frame's two pilot readings as
// the C's lizard_pilot: bit g of the count from the sub-channels j = g mod 2, their counts over the references of every
// sub-channel, each sub-channel by its soft values' scale (EP.w, A / N): r_g = (sum w c_j / W_g) / (sum w ref / W), W_g
// and W the weights' sums; each with its standard error by the delta method (the root of the sum of (w (c_j - r_g
// ref_j))^2, over |the denominator|); NaN where a parity has none; into PILOT[PILOT_READ f]: r, its error, r2, its
// error. The NaN where a parity has none is made at run time (P a uniform). Defines (core/dec/build.cpp backHalf):
// TIER_BLOCKS, STEP, PILOT_READ, SLOT_COUNT (uint); lib/tab.slang's.
#include "lib/lists.slang"
[[vk::binding(0, 0)]] StructuredBuffer<uint> LISTS;
[[vk::binding(1, 0)]] StructuredBuffer<float4> EP;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint2> BLK;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> PILOT;
[[vk::binding(4, 0)]] StructuredBuffer<uint> TAB;
struct ParamsUBlock { uint s; uint uvOff; uint sStride; uint lStride; uint subchMax; uint blocksMax; uint B; uint ucOff; uint n; };
[[vk::binding(5, 0)]] ConstantBuffer<ParamsUBlock> P;
#include "lib/tab.slang"
groupshared float4 acc0[TIER_BLOCKS];
groupshared float4 acc1[TIER_BLOCKS];
groupshared float4 rg;
// a value that is not NaN, tested on its bits (no float optimisation can take the test away)
#define NOT_NAN(X) ((asuint(X) & 0x7fffffffu) <= 0x7f800000u)
void sums(float4 s0, float4 s1, uint t) {
  acc0[t] = s0;
  acc1[t] = s1;
  GroupMemoryBarrierWithGroupSync();
  if (t < 16u) {
    float4 a0 = float4(0.0);
    float4 a1 = float4(0.0);
    for (uint i = t; i < TIER_BLOCKS; i += 16u) { a0 += acc0[i]; a1 += acc1[i]; }
    acc0[t] = a0;
    acc1[t] = a1;
  }
  GroupMemoryBarrierWithGroupSync();
}
[numthreads(TIER_BLOCKS, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint t = lix;
  uint count = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  uint f = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  uint2 v = CFG_AT(LISTS, P.B, f);
  if (t < min(blocksOf(v), TIER_BLOCKS)) {
    uint2 bc = blockAt(v, t);
    if (bc.x < NCODES) {
      uint sl = slotsOf(bc.x);
      uint s0 = ttOf(v, bc.x).y + bc.y * sl;
      float sum = 0.0;
      for (uint j = s0 / SUB_SLOTS; SUB_SLOTS * j < s0 + sl; j++) { sum += EP[f * P.subchMax + j].x * float(min(s0 + sl, SUB_SLOTS * j + SUB_SLOTS) - max(s0, SUB_SLOTS * j)); }
      float be = sum / float(sl);
      BLK[f * P.blocksMax + t] = uint2(asuint(be), be < barOf(bc.x) ? 1u : 0u);
    }
  }
  uint subch = STEP * cfgVersion(v);
  // each sub-channel's terms weighted by its soft values' scale (tsoft's EP.w, the C's sub_w: A / N): (the
  // references' weighted sum, their weights', the even counts', the odd), (the even counts' weights, the odd)
  float4 a = float4(0.0);
  float4 k = float4(0.0);
  for (uint j = t; j < subch; j += TIER_BLOCKS) {
    float4 e = EP[f * P.subchMax + j];
    if (NOT_NAN(e.z)) {
      a.x += e.w * e.z;
      a.y += e.w;
      if ((j & 1u) == 0u) { a.z += e.w * e.y; k.x += e.w; } else { a.w += e.w * e.y; k.y += e.w; }
    }
  }
  sums(a, k, t);
  if (t == 0u) {
    float4 s = float4(0.0);
    float4 q = float4(0.0);
    for (uint i = 0u; i < 16u; i++) { s += acc0[i]; q += acc1[i]; }
    // each parity's denominator, every reference scaling both (the two halves interleave over the same frequencies)
    float sr0 = s.y > 0.0 ? s.x * q.x / s.y : 0.0;
    float sr1 = s.y > 0.0 ? s.x * q.y / s.y : 0.0;
    rg = float4(sr0 != 0.0 ? s.z / sr0 : 0.0, sr1 != 0.0 ? s.w / sr1 : 0.0, sr0, sr1);
  }
  GroupMemoryBarrierWithGroupSync();
  float4 r = rg;
  float4 e2 = float4(0.0);
  for (uint j = t; j < subch; j += TIER_BLOCKS) {
    float4 e = EP[f * P.subchMax + j];
    if (NOT_NAN(e.z)) {
      float d = e.w * (e.y - ((j & 1u) == 0u ? r.x : r.y) * e.z);
      if ((j & 1u) == 0u) { e2.x += d * d; } else { e2.y += d * d; }
    }
  }
  sums(e2, float4(0.0), t);
  if (t == 0u) {
    float4 s = float4(0.0);
    for (uint i = 0u; i < 16u; i++) { s += acc0[i]; }
    float nan = asfloat(0x7fc00000u | (P.n & 0u));
    PILOT[PILOT_READ * f] = r.z != 0.0 ? r.x : nan;
    PILOT[PILOT_READ * f + 1u] = r.z != 0.0 ? sqrt(s.x) / abs(r.z) : nan;
    PILOT[PILOT_READ * f + 2u] = r.w != 0.0 ? r.y : nan;
    PILOT[PILOT_READ * f + 3u] = r.w != 0.0 ? sqrt(s.y) / abs(r.w) : nan;
  }
}
