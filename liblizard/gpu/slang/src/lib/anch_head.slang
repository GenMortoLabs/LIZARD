// What the anchors' stages (ANCH_FIND, ANCH_SUMS, ANCH_FIT) bind first, 0 to 6, and the map at a module point: REFIT's
// homography and the lens record (none for ANCH_FIND, which runs before ANCH_REFIT writes it: the anchors searched
// through the homography alone). The Anchors uniform: a ring's knots (k, three float4 a ring), the fill's normal
// matrix's inverse (g, nine a ring) and (its grid a side, the anchors its geometry takes) (s); ../../plan/anchors.mjs
// anchUniform. Defines: RING_COUNT, LATTICE_VECS, RESID (lib/lens.slang).
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/map.slang"
[[vk::binding(0, 0)]] Texture2DArray<float> img;
[[vk::binding(1, 0)]] StructuredBuffer<Frame> frames;
struct MapRec { float m[16]; };
[[vk::binding(2, 0)]] StructuredBuffer<MapRec> maps;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> sel;
[[vk::binding(4, 0)]] StructuredBuffer<float4> resid;
struct LatticeBlock { uint4 at[RING_COUNT]; float4 lat[LATTICE_VECS]; };
[[vk::binding(5, 0)]] ConstantBuffer<LatticeBlock> LT;
struct AnchorsBlock { float4 k[3 * RING_COUNT]; float4 g[9 * RING_COUNT]; float4 s[RING_COUNT]; };
[[vk::binding(6, 0)]] ConstantBuffer<AnchorsBlock> A;
float knot(uint r, uint i) { return A.k[3u * r + (i >> 2u)][i & 3u]; }
// ring r's grid a side (3, 5, 7) and the anchors its geometry takes (plan/anchors.mjs anchMin)
uint gridOf(uint r) { return f2u(A.s[r].x); }
uint gridMin(uint r) { return f2u(A.s[r].y); }
#include "lib/lens.slang"
#include "lib/lum0.slang"
float2 mapAt(float M[16], Lens lz, float2 m) {
  float2 p = toImage(M, m);
  return p + lensAt(lz, p);
}
