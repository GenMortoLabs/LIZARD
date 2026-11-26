// F7's REFIT: the homography over the border's nodes by Gauss-Newton (8 parameters, an 8 x 9 elimination), each node
// weighted by its correlation, a Huber weight past DELTA modules; the map's radial slot left 0 and the lens record
// cleared (ANCH_REFIT fits the lens over the border and the anchors). A workgroup of 64 a frame. Every solve is
// straight-line with literal indices (lib/solve.slang, unrolled: the S26's Adreno returned zeros from a private
// array indexed at run time). Every workgroup variable is written before it is read.
// Defines (core/dec/build.cpp refitRecipe): REFIT_ITERS (int), NODES_MAX, LENS_SLOTS, RESID (uint), WMIN, DELTA (float), ZEROS_<n>.
#include "lib/frame.slang"
#include "lib/solve.slang"
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
    [ForceUnroll] for (int j = 0; j < 3; j++) {
      float s = j < 2 ? Mod : 1.0;
      h[j] = (m0[j] - c.x * m0[6 + j]) / S * s; h[3 + j] = (m0[3 + j] - c.y * m0[6 + j]) / S * s; h[6 + j] = m0[6 + j] * s;
    }
    [ForceUnroll] for (int j = 0; j < 8; j++) th[j] = h[j] / h[8];
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
      // each parameter's derivative by central differences, then the normal equations' sums (jj the upper triangle)
      float2 J[8];
      [ForceUnroll] for (int k = 0; k < 8; k++) {
        float tp[8] = t, tm[8] = t;
        tp[k] = t[k] + 1e-3;
        tm[k] = t[k] - 1e-3;
        J[k] = (fwd(tp, m, c, S) - fwd(tm, m, c, S)) / (2.0 * 1e-3);
      }
      int q = 0;
      [ForceUnroll] for (int a = 0; a < 8; a++) {
        jr[a] += w * dot(J[a], r);
        [ForceUnroll] for (int b = a; b < 8; b++) { jj[q] += w * dot(J[a], J[b]); q++; }
      }
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
      [ForceUnroll] for (int q = 0; q < 36; q++) A[q] = Ag[q];
      [ForceUnroll] for (int a = 0; a < 8; a++) g[a] = Ag[36 + a];
      float N[8][9] = ZEROS_8_9;
      float D[8] = ZEROS_8;
      int q = 0;
      [ForceUnroll] for (int a = 0; a < 8; a++) [ForceUnroll] for (int b = a; b < 8; b++) { N[a][b] = A[q]; N[b][a] = A[q]; q++; }
      [ForceUnroll] for (int a = 0; a < 8; a++) D[a] = 1.0 / sqrt(max(N[a][a], 1e-30));
      [ForceUnroll] for (int a = 0; a < 8; a++) {
        [ForceUnroll] for (int b = 0; b < 8; b++) N[a][b] *= D[a] * D[b];
        N[a][a] = N[a][a] * 1.001 + 1e-9; N[a][8] = g[a] * D[a];
      }
      bool okSolve = true;
      gaussJordan(N, 1e-20, okSolve);
      if (okSolve) {
        [ForceUnroll] for (int a = 0; a < 8; a++) th[a] += D[a] * N[a][8] / N[a][a];
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (li == 0u) {
    float mo[16] = ZEROS_16;
    float t[9] = { th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], 1.0 };
    [ForceUnroll] for (int j = 0; j < 3; j++) {
      float s = j < 2 ? 1.0 / Mod : 1.0;
      mo[j] = (S * t[j] + c.x * t[6 + j]) * s; mo[3 + j] = (S * t[3 + j] + c.y * t[6 + j]) * s; mo[6 + j] = t[6 + j] * s;
    }
    // the radial slot, mo[9] to mo[12], stays 0: the map past REFIT is its homography (toImage reads no axis at k 0;
    // ANCH_REFIT fits the lens again over the border and the anchors)
    maps[f].m = mo;
  }
  // the lens record none until ANCH_REFIT's (the anchors are searched through REFIT's homography alone, as the C's
  // first round searches them through the corners')
  if (li < LENS_SLOTS) { resid[f * RESID + li] = float4(0.0); }
}
