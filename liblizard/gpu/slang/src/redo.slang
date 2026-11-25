// The finder's second round's frame records and its frames' resets, and its grids.
// Defines (finder.mjs): K, R2_N, RING_COUNT.
#include "lib/frame.slang"

[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> sel;
[[vk::binding(2, 0)]] RWStructuredBuffer<Frame> frames2;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> readings;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> cand;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> quads;
struct P5Block { uint cap; uint gx; uint gy; uint p0; float4 modules; };
[[vk::binding(7, 0)]] ConstantBuffer<P5Block> P;

[[vk::binding(8, 0)]] RWStructuredBuffer<uint> args2;
struct R2TBlock { uint4 d[int(R2_N)]; };
[[vk::binding(9, 0)]] ConstantBuffer<R2TBlock> R2;
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x;
  Frame F = frames[f];
  bool again = F.valid == 1u && sel[f].y == 0u;
  if (li == 0u) { frames2[f] = Frame(F.w, F.h, F.axis, again ? 1u : (F.valid != 0u ? 2u : 0u)); }
  if (!again) { return; }
  if (li < R2_N) {
    uint4 t = R2.d[li];
    for (uint c = 0u; c < 3u; c++) {
      if (t.w > 2u) { break; }
      // the other frames' workgroups take the maximum of the same word: the store atomic too
      uint old;
      if (c == t.w) { InterlockedMax(args2[3u * li + c], t[c] * (f + 1u)); } else { InterlockedExchange(args2[3u * li + c], t[c], old); }
    }
  }
  uint na = RING_COUNT * P.gx * P.gy;
  for (uint i = li; i < na; i += 256u) { acc[f * na + i] = 0u; }
  uint nr = 3u * P.cap;
  for (uint i = li; i < nr; i += 256u) { readings[f * nr + i] = float4(0.0); }
  if (li < K) { cand[f * K + li] = float4(0.0); }
  if (li < (5u * K)) { quads[f * (5u * K) + li] = float4(0.0); }
}
