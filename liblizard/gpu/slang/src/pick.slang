// F6b: the frame's answer, one workgroup a frame.
// Defines (core/dec/build.cpp Finder): HYP, K, ORIENT, RING_COUNT, ZEROS_16, ZEROS_32, ZEROS_9.
#include "lib/frame.slang"
#include "lib/homography.slang"
#include "lib/bend.slang"

[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<float4> quads;
[[vk::binding(2, 0)]] StructuredBuffer<float> score;
struct MapRec { float m[16]; };
[[vk::binding(3, 0)]] RWStructuredBuffer<MapRec> maps;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint4> sel;
struct ResultRec { float r[32]; };
[[vk::binding(5, 0)]] RWStructuredBuffer<ResultRec> result;
struct PickBlock { float tau; uint p0; uint p1; uint p2; float4 modules; };
[[vk::binding(6, 0)]] ConstantBuffer<PickBlock> P;
[numthreads(1, 1, 1)]
void main(uint3 wid : SV_GroupID) {
  uint3 wg = wid;
  uint f = wg.x;
  if (frames[f].valid > 1u) { return; }
  float best = -2.0;
  uint bh = 0u;
  for (uint h = 0u; h < HYP; h++) {
    float s = score[f * HYP + h];
    if (s > best) { best = s; bh = h; }
  }
  bool found = frames[f].valid != 0u && best >= P.tau;
  uint bh0 = bh % (K * ORIENT * RING_COUNT);
  uint bendBit = bh / (K * ORIENT * RING_COUNT);
  uint qk = bh0 / (ORIENT * RING_COUNT);
  uint o = (bh0 / RING_COUNT) % ORIENT;
  uint rg = bh0 % RING_COUNT;
  { float z16[16] = ZEROS_16; maps[f].m = z16; }
  { float z32[32] = ZEROS_32; result[f].r = z32; }
  result[f].r[1] = best;
  if (found) {
    uint base = 5u * (f * K + qk);
    float4 quad[4] = { quads[base], quads[base + 1u], quads[base + 2u], quads[base + 3u] };
    // a bent winner: H of the unbent marks, the bend in the map's radial slot (toImage reads it)
    float kb = bendBit == 0u ? 0.0 : quads[base + 4u].z;
    float3 ax = bendAxis(frames[f]);
    float4 qu[4] = quad;
    if (kb != 0.0) { for (int k = 0; k < 4; k++) { qu[k].xy = unbend(qu[k].xy, ax, kb); } }
    float H[9] = hypH(qu, o, P.modules[rg]);
    if (kb != 0.0) { maps[f].m[9] = kb; maps[f].m[10] = ax.x; maps[f].m[11] = ax.y; maps[f].m[12] = ax.z; }
    maps[f].m[0] = H[0]; result[f].r[8] = H[0];
    maps[f].m[1] = H[1]; result[f].r[9] = H[1];
    maps[f].m[2] = H[2]; result[f].r[10] = H[2];
    maps[f].m[3] = H[3]; result[f].r[11] = H[3];
    maps[f].m[4] = H[4]; result[f].r[12] = H[4];
    maps[f].m[5] = H[5]; result[f].r[13] = H[5];
    maps[f].m[6] = H[6]; result[f].r[14] = H[6];
    maps[f].m[7] = H[7]; result[f].r[15] = H[7];
    maps[f].m[8] = H[8]; result[f].r[16] = H[8];
    for (uint k = 0u; k < 4u; k++) {
      float4 p = atSlot(quad, slotOf(o, k));
      result[f].r[17u + 2u * k] = p.x;
      result[f].r[18u + 2u * k] = p.y;
      result[f].r[25u + k] = p.z;
    }
    result[f].r[0] = 1.0;
    result[f].r[2] = float(rg);
    result[f].r[3] = float(o);
    result[f].r[4] = float(qk);
  }
  sel[f] = uint4(rg, found ? 1u : 0u, 0u, 0u);
}
