// F5a, VOTE: each classified reading votes along its inward diagonal for the symbol's centre, a ray a ring.
// Dispatched (ceil(cap / VOTE_READS), frames).
// Defines (finder.mjs): CELL, CENTRE0, CENTRE1, RAY_HI, RAY_LO, RAY_SPREAD, RING_COUNT, SMIN, VOTE_READS.
#include "lib/sync.slang"
#include "lib/frame.slang"
#include "lib/convert.slang"
static const float PI = 3.14159265;

[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<float4> readings;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;
struct P5Block { uint cap; uint gx; uint gy; uint p0; float4 modules; };
[[vk::binding(3, 0)]] ConstantBuffer<P5Block> P;
static const float CENTRE[2] = { CENTRE0, CENTRE1 };

static const uint CHUNK = 256u;
groupshared uint vlist[int(VOTE_READS)];
groupshared Atomic<uint> vcount;
groupshared uint nvote;
groupshared float dk[CHUNK];
groupshared uint nk;
groupshared float dnext;
[numthreads(64, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.y;
  if (li == 0u) { vcount.store(0u); }
  GroupMemoryBarrierWithGroupSync();
  uint i0 = wg.x * VOTE_READS + li;
  if (li < VOTE_READS && i0 < P.cap && frames[f].valid == 1u) {
    uint o0 = 3u * (f * P.cap + i0);
    if (!(readings[o0 + 2u].w == 0.0 || readings[o0 + 1u].x < SMIN)) { vlist[vcount.add(1u)] = i0; }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) { nvote = vcount.load(); }
  UNIFORM_LOAD(uint nv, nvote)
  for (uint e = 0u; e < nv; e++) {
    uint i = vlist[e];
    uint o = 3u * (f * P.cap + i);
    float4 r0 = readings[o];
    float4 r2 = readings[o + 2u];
    float p = readings[o + 1u].x;
    float beta = r0.w + PI / 4.0;
    float2 inward = -float2(cos(beta), sin(beta));
    float2 perp = float2(-inward.y, inward.x);
    float c = CENTRE[f2u(r2.x)];
    uint w = f2u(clamp(p, 0.0, 1.0) * 256.0);
    int gxf = int((frames[f].w + (CELL - 1u)) / CELL);
    int gyf = int((frames[f].h + (CELL - 1u)) / CELL);
    float far = length(max(r0.xy, float2(float(gxf), float(gyf)) * float(CELL) - r0.xy)) + float(CELL);
    for (uint m = 0u; m < RING_COUNT; m++) {
      float reach = (P.modules[m] - 2.0 * c) / sqrt(2.0) * r0.z;
      float d0 = RAY_LO * reach;
      float d1 = min(RAY_HI * reach, far);
      int spread = f2i(clamp(ceil(RAY_SPREAD * reach / float(CELL)), 1.0, 6.0));
      uint S = uint(2 * spread + 1);
      GroupMemoryBarrierWithGroupSync();
      if (li == 0u) { dnext = d0; }
      for (;;) {
        GroupMemoryBarrierWithGroupSync();
        if (li == 0u) {
          float d = dnext;
          uint k = 0u;
          for (; k < CHUNK && d <= d1; k++) { dk[k] = d; d += float(CELL); }
          nk = k;
          dnext = d;
        }
        UNIFORM_LOAD(uint n, nk)
        if (n == 0u) { break; }
        for (uint t2 = li; t2 < n * S; t2 += 64u) {
          int q = int(t2 % S) - spread;
          float2 at = (r0.xy + dk[t2 / S] * inward) / float(CELL);
          float2 cell = at + float(q) * perp;
          int x = f2i(floor(cell.x));
          int y = f2i(floor(cell.y));
          if (x < 0 || y < 0 || x >= gxf || y >= gyf) { continue; }
          float t = 1.0 - float(abs(q)) / float(spread + 1);
          InterlockedAdd(acc[((f * RING_COUNT + m) * P.gy + uint(y)) * P.gx + uint(x)], f2u(float(w) * t));
        }
        GroupMemoryBarrierWithGroupSync();
      }
    }
  }
}
