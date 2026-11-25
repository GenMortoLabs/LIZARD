// ANCH_SUMS (2026-11-20): the squares' fill's six sums (the surface's basis 1, u, u^2, w, w^2, u w times the picture at
// each of the ANCH_FILL x ANCH_FILL points in no square), a workgroup an (ANCH_GS-th of the points, frame): each lane
// FILL_PER points of its slice, reduced in workgroup memory, the slice's sums written as two float4 where fillSums puts
// them; ANCH_FIT adds the slices. On ANCH_FIND's layout and group (anch read and written). Its reads through the final
// map, after ANCH_REFIT.
// Defines (anchors.mjs): FILL_PER (uint); lib/anch_head.slang's and lib/anch_fill.slang's.
#include "lib/anch_head.slang"
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> anch;
#include "lib/anch_fill.slang"
groupshared float red[6 * 64];
[numthreads(64, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint li = lix;
  uint g = wid.x;
  uint f = wid.y;
  Frame F = frames[f];
  uint4 pick = sel[f];
  if (F.valid == 0u || pick.y == 0u) { return; }
  uint rgi = pick.x;
  uint S = gridOf(rgi);
  float4 geo = A.k[3u * rgi + 2u];
  float span = geo.z;
  float margin = geo.w;
  float M[16] = maps[f].m;
  Lens lz = lensOf(f);
  int sp = f2i(span);
  int mg = f2i(margin);
  int aa = f2i(geo.y);
  float s0 = 0.0;
  float s1 = 0.0;
  float s2 = 0.0;
  float s3 = 0.0;
  float s4 = 0.0;
  float s5 = 0.0;
  for (uint j = 0u; j < FILL_PER; j++) {
    uint p = g * (64u * FILL_PER) + j * 64u + li;
    int px = int(p % uint(ANCH_FILL));
    int py = int(p / uint(ANCH_FILL));
    if (fillBand(rgi, px, sp, mg, aa, S) && fillBand(rgi, py, sp, mg, aa, S)) { continue; }
    float ux = float(2 * px + 1 - ANCH_FILL) / float(ANCH_FILL);
    float uy = float(2 * py + 1 - ANCH_FILL) / float(ANCH_FILL);
    float2 m = float2(margin + 0.5 * span * (ux + 1.0), margin + 0.5 * span * (uy + 1.0));
    float v = bilin0(f, mapAt(M, lz, m), int(F.w), int(F.h));
    s0 += v;
    s1 += ux * v;
    s2 += ux * ux * v;
    s3 += uy * v;
    s4 += uy * uy * v;
    s5 += ux * uy * v;
  }
  red[li] = s0;
  red[64u + li] = s1;
  red[128u + li] = s2;
  red[192u + li] = s3;
  red[256u + li] = s4;
  red[320u + li] = s5;
  GroupMemoryBarrierWithGroupSync();
  if (li < 6u) {
    float s = 0.0;
    for (uint q = 0u; q < 64u; q++) { s += red[64u * li + q]; }
    red[64u * li] = s;
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    uint o = fillSums(f) + 2u * g;
    anch[o] = float4(red[0], red[64], red[128], red[192]);
    anch[o + 1u] = float4(red[256], red[320], 0.0, 0.0);
  }
}
