// F5b: every peak of a frame's merged rings, a lane a cell. Dispatched (ceil(gx gy / 256), frames).
// Defines (core/dec/build.cpp Finder): CELL, PEAK_LIST, RING_COUNT, SUPPRESS.
#include "lib/frame.slang"

[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<uint> acc;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> vlist;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> vcount;
struct P5Block { uint cap; uint gx; uint gy; uint p0; float4 modules; };
[[vk::binding(4, 0)]] ConstantBuffer<P5Block> P;

uint accAt(uint f, uint m, int x, int y, int gw, int gh) {
  if (x < 0 || y < 0 || x >= gw || y >= gh) { return 0u; }
  return acc[((f * RING_COUNT + m) * P.gy + uint(y)) * P.gx + uint(x)];
}
uint merged(uint f, int x, int y, int gw, int gh) {
  uint v = 0u;
  for (uint m = 0u; m < RING_COUNT; m++) { v = max(v, accAt(f, m, x, y, gw, gh)); }
  return v;
}
bool outranks(uint f, int x, int y, uint v, int dx, int dy, int gw, int gh) {
  uint q = merged(f, x + dx, y + dy, gw, gh);
  return q > v || (q == v && (dy < 0 || (dy == 0 && dx < 0)));
}
bool tallest(uint f, int x, int y, uint v, int r, int gw, int gh) {
  for (int d = 1; d <= r; d++) {
    for (int k = -d; k <= d; k++) {
      if (outranks(f, x, y, v, k, -d, gw, gh) || outranks(f, x, y, v, k, d, gw, gh) || outranks(f, x, y, v, -d, k, gw, gh) || outranks(f, x, y, v, d, k, gw, gh)) { return false; }
    }
  }
  return true;
}

[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.y;
  if (frames[f].valid > 1u) { return; }
  int gw = int((frames[f].w + (CELL - 1u)) / CELL);
  int gh = int((frames[f].h + (CELL - 1u)) / CELL);
  uint c = wg.x * 256u + li;
  if (c >= uint(gw * gh)) { return; }
  int x = int(c % uint(gw));
  int y = int(c / uint(gw));
  uint v = merged(f, x, y, gw, gh);
  if (v == 0u || !tallest(f, x, y, v, int(SUPPRESS), gw, gh)) { return; }
  uint ring = 0u;
  for (uint m = RING_COUNT; m > 0u; m--) { if (accAt(f, m - 1u, x, y, gw, gh) == v) { ring = m - 1u; } }
  uint k;
  InterlockedAdd(vcount[f], 1u, k);
  if (k < PEAK_LIST) { vlist[f * PEAK_LIST + k] = float4((float(x) + 0.5) * float(CELL), (float(y) + 0.5) * float(CELL), float(ring), float(v)); }
}
