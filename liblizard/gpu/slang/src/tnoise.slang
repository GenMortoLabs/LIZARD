// tnoise, the rule stage's first: a workgroup an 8-sub-channel chunk of a listed frame (tsoft's grid), 32 lanes a
// sub-channel, lane l the coefficients l + 32 k (k 0 to 9) of its 320: the second and fourth moments of |y|^2 (|y| the
// same turned or whitened or not), reduced over the 32 lanes, and the sub-channel's noise N = m2 - sqrt(max(2 m2^2 -
// m4, 0)), at least 0.02 m2 (the C's pilot_align and sub_scale), and what each stage after takes of it, into EP: x the
// sub-channel's estimate (jinfo(A^2 / N), tblocks'), y its soft values' scale (0.7 x 2 sqrt(2 A^2) / N, tsoft's), w 1 /
// N (talign's weight), each 0 where N is 0; tsoft writes y and z over with the two pilots' values. The moments made
// once, on tsoft's grid (in talign's one workgroup a frame the 320 reads a sub-channel ran a lane's sub-channels one
// after another, 0.4 ms a frame on the iGPU). Defines (back_tiers.mjs tnoiseSource): F16 (0 or 1), CHUNKS, SLOT_COUNT
// (uint); lib/cfg.slang's.
#if F16
#define S_T half2
#else
#define S_T float2
#endif
#include "lib/lists.slang"
[[vk::binding(0, 0)]] StructuredBuffer<S_T> S;
[[vk::binding(1, 0)]] StructuredBuffer<uint> UV;
[[vk::binding(2, 0)]] StructuredBuffer<uint> LISTS;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> EP;
struct ParamsUBlock { uint s; uint uvOff; uint sStride; uint lStride; uint subchMax; uint blocksMax; uint B; uint ucOff; uint n; };
[[vk::binding(4, 0)]] ConstantBuffer<ParamsUBlock> P;
#include "lib/cfg.slang"
groupshared float2 red[256];

float jinfo(float g) {
  if (g <= 0.0) { return 0.0; }
  float base = 1.0 - exp2(-0.3073 * pow(2.0 * sqrt(g), 1.7870));
  if (base <= 0.0) { return 0.0; }
  return pow(base, 1.1064);
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
  uint sBase = f * P.sStride;
  // each coefficient times n (S is T / n^2, f16's range, its largest about 1e-3, so |y|^4 near 1e-13): n S puts
  // |y| near 1, and a product with a value made at run time keeps the squares in f32 on the Adreno 840, whose driver
  // read dot(y, y) of values converted from f16 at half precision, where they underflow (STATUS "The GPU's turn
  // fit weighted as the C's"); the estimate and the weights' ratios do not move with the scale, and the soft
  // values' scale takes it back (times n)
  float sc = float(P.n);
  float2 m = float2(0.0);
  for (uint k = 0u; k < 10u; k++) { float2 y = float2(S[sBase + UV[ub + l + 32u * k]]) * sc; float e = dot(y, y); m += float2(e, e * e); }
  red[t] = m;
  GroupMemoryBarrierWithGroupSync();
  for (uint h = 16u; h > 0u; h = h >> 1u) {
    if (l < h) { red[t] = red[t] + red[t + h]; }
    GroupMemoryBarrierWithGroupSync();
  }
  if (l == 0u) {
    float m2 = red[t].x / 320.0;
    float m4 = red[t].y / 320.0;
    float a2 = sqrt(max(2.0 * m2 * m2 - m4, 0.0));
    float nv = max(m2 - a2, 0.02 * m2);
    float4 e = float4(0.0);
    if (nv > 0.0) { e = float4(jinfo(a2 / nv), 0.7 * 2.0 * sqrt(2.0 * a2) / nv * sc, 0.0, 1.0 / nv); }
    EP[f * P.subchMax + j] = e;
  }
}
