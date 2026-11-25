// F4, the classifier: each kept peak's 32 x 32 patch through the net (cnn-v3) as convolutions tiled as GEMMs, a
// register tile a lane and several patches a workgroup: is it a corner mark, which way is out, how big a module. Its
// float forms (f16 and f32) and its int8 form (int8-v1: per-channel weights, per-layer activation scales, requantised
// by one f32 multiply, so its codes are the trainer's integer forward's), dispatched ceil(cap / PATCHES) workgroups a
// frame. A patch is cut from the pyramid level its size reads at (gen/cut: a lane four samples of a row, bilinear),
// normalised by its mean and spread into X (gen/store), then the plan's layers (gen/convs, gen/fcs:
// ../../plan/classify_gemm.mjs classifyPlan, the weights packed by its packWeights, packWeightsF16, packWeightsInt8;
// the products unrolled for each layer's tile, classify.mjs); the last layer's five outputs are the reading. Every
// workgroup variable is written before the barrier that precedes its reads, so none is zeroed. A bitcast of u32 words
// to f16 is unpackFloat2x16, to f32 asfloat; the packed int8 dot product OpSDot (lib/nets.slang dotPacked4x8); round
// is roundEven. Defines (classify.mjs classifySource): INT8, F16, SUBGROUPS (0 or 1); PATCHES, GROUP (lanes a
// patch, 256 / PATCHES), PATCH (the patch's side), Z_AT (the last layer's outputs in red), BUF_X, BUF_Y, RED, REDI
// (uint); PATCH_KAPPA, Q8_MAX, Q8_INV (float); VEC4_T (half4 or float4).
#if INT8
#elif F16
#endif
#if SUBGROUPS
#endif
#include "lib/nets.slang"
#include "lib/round.slang"
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/level_bindings.slang"
#include "lib/levels.slang"
[[vk::binding(7, 0)]] StructuredBuffer<float4> peaks;
[[vk::binding(8, 0)]] StructuredBuffer<uint> counts;
[[vk::binding(9, 0)]] RWStructuredBuffer<float4> readings;
struct P4Block { uint cap; uint p0; uint p1; uint p2; };
[[vk::binding(10, 0)]] ConstantBuffer<P4Block> P;
[[vk::binding(11, 0)]] StructuredBuffer<uint4> W;
static const float KAPPA = PATCH_KAPPA;
static const float PI = 3.14159265;

#if INT8
static const float QMAX = Q8_MAX;
static const float INV = Q8_INV;
// int8 codes: PATCHES patches a workgroup, X and Y their activations (uvec4s, four codes a word), red and redi scratch
groupshared uint4 bufX[BUF_X];
groupshared uint4 bufY[BUF_Y];
groupshared float red[RED];
groupshared int redi[REDI];
int4 d4(uint4 v, uint w) { return int4(dotPacked4x8(int(v.x), int(w)), dotPacked4x8(int(v.y), int(w)), dotPacked4x8(int(v.z), int(w)), dotPacked4x8(int(v.w), int(w))); }
int4 d4l(uint4 v, uint w) { return int4(0, dotPacked4x8(int(v.y), int(w)), dotPacked4x8(int(v.z), int(w)), dotPacked4x8(int(v.w), int(w))); }
int4 d4r(uint4 v, uint w) { return int4(dotPacked4x8(int(v.x), int(w)), dotPacked4x8(int(v.y), int(w)), dotPacked4x8(int(v.z), int(w)), 0); }
int4 rq(int4 a, float m) { return f2i(clamp(roundEven(float4(a) * m), float4(0.0), float4(QMAX))); }
uint4 packq(int4 c0, int4 c1, int4 c2, int4 c3) {
  return uint4(pack4xI8(int4(c0.x, c1.x, c2.x, c3.x)), pack4xI8(int4(c0.y, c1.y, c2.y, c3.y)), pack4xI8(int4(c0.z, c1.z, c2.z, c3.z)), pack4xI8(int4(c0.w, c1.w, c2.w, c3.w)));
}
#else
// f16 or f32: PATCHES patches a workgroup, X and Y their activations (vec4s), red scratch
groupshared VEC4_T bufX[BUF_X];
groupshared VEC4_T bufY[BUF_Y];
groupshared float red[RED];
#endif

[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.y;
  if (frames[f].valid != 1u) { return; }
  uint n = min(counts[f * 16u + 2u], P.cap);
  uint j0 = wg.x * PATCHES;
  if (j0 >= n) { return; }
  uint cp = li / GROUP;
  uint lq = li % GROUP;
  uint jc = j0 + cp;
  uint ic = select(jc < n, jc, 0u);
  float4 pk = peaks[f * P.cap + ic];
  float u = pk.z / KAPPA;
  uint l = levelFor(u);
  
#include "gen/patches"

  float ssum = 0.0;
  float ssq = 0.0;
  
#include "gen/cut"

  red[li] = ssum;
  red[256u + li] = ssq;
  GroupMemoryBarrierWithGroupSync();
  if (li < 32u) {
    float t1 = 0.0;
    float t2 = 0.0;
    for (uint k = 0u; k < 8u; k++) { t1 += red[li * 8u + k]; t2 += red[256u + li * 8u + k]; }
    red[512u + li] = t1;
    red[544u + li] = t2;
  }
  GroupMemoryBarrierWithGroupSync();
  float tsum = 0.0;
  float tsq = 0.0;
  for (uint k = 0u; k < (GROUP / 8u); k++) { tsum += red[512u + cp * (GROUP / 8u) + k]; tsq += red[544u + cp * (GROUP / 8u) + k]; }
  float mu = tsum / float(PATCH * PATCH);
  float sd = sqrt(max(tsq / float(PATCH * PATCH) - mu * mu, 0.0));
  
#include "gen/store"

  GroupMemoryBarrierWithGroupSync();
  
#include "gen/convs"

  
#include "gen/fcs"

  if (li < PATCHES) {
    uint j = j0 + li;
    if (j < n) {
      uint i = j;
      float4 pw = peaks[f * P.cap + i];
      float uw = pw.z / KAPPA;
      float z0 = red[Z_AT + li * 5u];
      float z1 = red[Z_AT + li * 5u + 1u];
      float z2 = red[Z_AT + li * 5u + 2u];
      float z3 = red[Z_AT + li * 5u + 3u];
      float z4 = red[Z_AT + li * 5u + 4u];
      float t = atan2(z2, z1);
      float pr = 1.0 / (1.0 + exp(-z0));
      uint o3 = 3u * (f * P.cap + i);
      readings[o3] = float4(pw.xy, uw * exp(z3), t - PI / 4.0);
      readings[o3 + 1u] = float4(pr, 0.0, 0.0, 0.0);
      readings[o3 + 2u] = float4(select(z4 > 0.0, 1.0, 0.0), pw.w, float(levelFor(uw)), 1.0);
    }
  }
}
