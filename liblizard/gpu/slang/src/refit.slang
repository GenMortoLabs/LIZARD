// F7's REFIT: the homography over the border's nodes by Gauss-Newton (8 parameters, an 8 x 9 elimination), each node
// weighted by its correlation, a Huber weight past DELTA modules; the map's radial slot left 0 and the lens record
// cleared (ANCH_REFIT fits the lens over the border and the anchors). A workgroup of 64 a frame. Every solve is
// straight-line with literal indices, generated in JS (register.mjs: the S26's Adreno returned zeros from a private
// array indexed at run time). Every workgroup variable is written before it is read.
// Defines (register.mjs): REFIT_ITERS (int), NODES_MAX, LENS_SLOTS, RESID (uint), WMIN, DELTA (float), ZEROS_<n>.
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/map.slang"
[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<float4> nodes;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> sel;
[[vk::binding(3, 0)]] StructuredBuffer<float4> meas;
struct MapRec { float m[16]; };
[[vk::binding(4, 0)]] RWStructuredBuffer<MapRec> maps;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> resid;
struct FitBlock { uint4 count; uint4 base; float4 modules; };
[[vk::binding(6, 0)]] ConstantBuffer<FitBlock> P;
#include "lib/lens.slang"
groupshared float JJ[64][36];
groupshared float Jr[64][8];
groupshared float Ag[44];
groupshared float th[8];
static const int ITERS = REFIT_ITERS;
float2 fwd(float t[8], float2 m, float2 c, float S) {
  float w = t[6] * m.x + t[7] * m.y + 1.0;
  return c + S * float2(t[0] * m.x + t[1] * m.y + t[2], t[3] * m.x + t[4] * m.y + t[5]) / w;
}
[numthreads(64, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x;
  uint4 pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  Frame F = frames[f];
  uint count = P.count[pick.x];
  uint base = P.base[pick.x];
  float Mod = P.modules[pick.x];
  float2 c = float2(0.5 * float(F.w), 0.5 * float(F.h));
  float S = length(c);
  float m0[16] = maps[f].m;
  if (li == 0u) {
    float h[9] = ZEROS_9;
    #include "gen/start"
    #include "gen/theta"
  }
  GroupMemoryBarrierWithGroupSync();
  for (int it = 0; it < ITERS; it++) {
    float t[8] = { th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7] };
    float jj[36] = ZEROS_36;
    float jr[8] = ZEROS_8;
    for (uint n = li; n < count; n += 64u) {
      float4 e = meas[f * NODES_MAX + n];
      if (e.w == 0.0 || e.z < WMIN) { continue; }
      float2 m = nodes[base + n].xy / Mod;
      float2 p = fwd(t, m, c, S);
      float2 r = e.xy - p;
      float du = distance(fwd(t, m + float2(1.0 / Mod, 0.0), c, S), p);
      float rr = length(r);
      float hw = DELTA * du / rr;
      float w = e.z * (rr > DELTA * du ? hw : 1.0);
      #include "gen/jacobian"
      #include "gen/accumulate"
    }
    JJ[li] = jj;
    Jr[li] = jr;
    GroupMemoryBarrierWithGroupSync();
    if (li < 44u) {
      float acc = 0.0;
      if (li < 36u) { for (uint s = 0u; s < 64u; s++) { acc += JJ[s][li]; } }
      else { for (uint s = 0u; s < 64u; s++) { acc += Jr[s][li - 36u]; } }
      Ag[li] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    if (li == 0u) {
      float A[36] = ZEROS_36;
      float g[8] = ZEROS_8;
      #include "gen/sums"
      #include "gen/rhs"
      float N[8][9] = ZEROS_8_9;
      float D[8] = ZEROS_8;
      #include "gen/normal"
      #include "gen/scale"
      #include "gen/scaled"
      bool okSolve = true;
      #include "gen/solve"
      if (okSolve) {
#include "gen/step"
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (li == 0u) {
    float mo[16] = ZEROS_16;
    float t[9] = { th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], 1.0 };
    #include "gen/map"
    // the radial slot, mo[9] to mo[12], stays 0: the map past REFIT is its homography (toImage reads no axis at k 0;
    // ANCH_REFIT fits the lens again over the border and the anchors)
    maps[f].m = mo;
  }
  // the lens record none until ANCH_REFIT's (the anchors are searched through REFIT's homography alone, as the C's
  // first round searches them through the corners')
  if (li < LENS_SLOTS) { resid[f * RESID + li] = float4(0.0); }
}
