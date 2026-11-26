// F5b's selection and F5c's quads, a workgroup a frame.
// Defines (core/dec/build.cpp Finder): CELL, CENTRE0, CENTRE1, GROUP, K, PEAK_LIST, RAY_HI, RAY_LO, RAY_SPREAD, RING_COUNT, SMIN.
#include "lib/frame.slang"
#include "lib/round.slang"
#include "lib/convert.slang"
static const float PI = 3.14159265;

[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<float4> readings;
[[vk::binding(2, 0)]] StructuredBuffer<uint> acc;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> cand;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> quads;
struct P5Block { uint cap; uint gx; uint gy; uint p0; float4 modules; };
[[vk::binding(5, 0)]] ConstantBuffer<P5Block> P;

[[vk::binding(6, 0)]] StructuredBuffer<float4> vlist;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> vcount;
static const float CENTRE[2] = { CENTRE0, CENTRE1 };

groupshared float4 list[int(PEAK_LIST)];
groupshared Atomic<uint> nlist;
groupshared float4 peak[int(K)];
groupshared Atomic<uint> leadK[int(4u * K)];
groupshared Atomic<uint> slotK[int(16u * K)];
bool ahead(float4 a, float4 b) {
  return a.w > b.w || (a.w == b.w && (a.y < b.y || (a.y == b.y && a.x < b.x)));
}
uint4 keyOf(uint f, uint i) {
  uint o = 3u * (f * P.cap + i);
  float4 r0 = readings[o];
  uint w = f2u(floor(clamp(readings[o + 1u].x, 0.0, 1.0) * 1000.0));
  uint rb = asuint(readings[o + 2u].y) & 0x7fffffffu;
  uint yb = asuint(max(r0.y, 0.0)) & 0x7fffffffu;
  uint xb = asuint(max(r0.x, 0.0)) & 0x7fffffffu;
  return uint4((w << 13u) | (rb >> 19u), ((rb & 0x7ffffu) << 13u) | (yb >> 19u), ((yb & 0x7ffffu) << 13u) | (xb >> 19u),
               ((xb & 0x7ffffu) << 13u) | i);
}
bool leadHas(uint4 k, uint pk, uint j) {
  bool ok = true;
  for (uint t = 0u; t < j; t++) { ok = ok && k[t] == leadK[4u * pk + t].load(); }
  return ok;
}
bool slotHas(uint4 k, uint pk, uint q, uint j) {
  bool ok = true;
  for (uint t = 0u; t < j; t++) { ok = ok && k[t] == slotK[16u * pk + 4u * q + t].load(); }
  return ok;
}
bool support(uint f, uint i, float4 c) {
  uint o = 3u * (f * P.cap + i);
  float4 r0 = readings[o];
  float4 r2 = readings[o + 2u];
  if (r2.w == 0.0 || readings[o + 1u].x < SMIN) { return false; }
  float edge = CENTRE[f2u(r2.x)];
  float near = (P.modules[0] - 2.0 * edge) / sqrt(2.0) * r0.z;
  float far = (P.modules[int(RING_COUNT - 1u)] - 2.0 * edge) / sqrt(2.0) * r0.z;
  float beta = r0.w + PI / 4.0;
  float2 inward = -float2(cos(beta), sin(beta));
  float2 d = c.xy - r0.xy;
  float along = dot(d, inward);
  float across = abs(d.x * inward.y - d.y * inward.x);
  if (along < RAY_LO * near || along > RAY_HI * far) { return false; }
  return across <= max(2.0 * float(CELL), RAY_SPREAD * along + float(CELL));
}
uint cornerOf(uint f, uint i, float4 c, float u, float alpha) {
  if (!support(f, i, c)) { return 4u; }
  float4 r = readings[3u * (f * P.cap + i)];
  if (r.z < 0.2 * u || r.z > 5.0 * u) { return 4u; }
  float a = atan2(r.y - c.y, r.x - c.x) - alpha - PI / 4.0;
  return uint(f2i(roundEven(a / (PI / 2.0))) & 3);
}
float2 slotAt(uint f, uint pk, uint q) { return readings[3u * (f * P.cap + (slotK[16u * pk + 4u * q + 3u].load() & 8191u))].xy; }
float2 centreOf(uint f, uint pk, bool p02, bool p13) {
  float2 a = slotAt(f, pk, 0u);
  float2 b = slotAt(f, pk, 2u);
  float2 e = slotAt(f, pk, 1u);
  float2 g = slotAt(f, pk, 3u);
  float2 r = b - a;
  float2 s = g - e;
  float den = r.x * s.y - r.y * s.x;
  if (p02 && p13 && abs(den) > 1e-3) { return a + ((e.x - a.x) * s.y - (e.y - a.y) * s.x) / den * r; }
  float2 mid = p02 ? 0.5 * (a + b) : 0.5 * (e + g);
  float2 A = p02 ? a : e;
  float2 d = p02 ? r : s;
  uint q1 = 4u;
  for (uint q = p02 ? 1u : 0u; q < 4u; q += 2u) { if (slotK[16u * pk + 4u * q].load() != 0u) { q1 = q; } }
  if (q1 == 4u) { return mid; }
  float4 rq = readings[3u * (f * P.cap + (slotK[16u * pk + 4u * q1 + 3u].load() & 8191u))];
  float beta = rq.w + PI / 4.0;
  float2 inward = -float2(cos(beta), sin(beta));
  float dn = inward.x * d.y - inward.y * d.x;
  if (abs(dn) < 1e-3 * length(d)) { return mid; }
  float t = ((A.x - rq.x) * d.y - (A.y - rq.y) * d.x) / dn;
  return (t > 0.0) ? rq.xy + t * inward : mid;
}
void gather(uint f, uint pk, float4 c, uint lane) {
  bool live = frames[f].valid != 0u && c.w > 0.0;
  for (uint j = 0u; j < 4u; j++) {
    for (uint i = lane; i < P.cap; i += GROUP) {
      if (!live) { break; }
      uint4 k = keyOf(f, i);
      if (leadHas(k, pk, j) && support(f, i, c)) { leadK[4u * pk + j].max(k[j]); }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  bool lead = leadK[4u * pk].load() != 0u;
  float4 lr = readings[3u * (f * P.cap + (leadK[4u * pk + 3u].load() & 8191u))];
  float alpha = lr.w;
  float4 at = c;
  bool refined = false;
  for (uint rnd = 0u; rnd < 2u; rnd++) {
    if (rnd == 1u) {
      bool p02 = slotK[16u * pk].load() != 0u && slotK[16u * pk + 8u].load() != 0u;
      bool p13 = slotK[16u * pk + 4u].load() != 0u && slotK[16u * pk + 12u].load() != 0u;
      refined = lead && (p02 || p13);
      if (refined) { at = float4(centreOf(f, pk, p02, p13), c.z, c.w); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (rnd == 1u && refined && lane < 16u) { slotK[16u * pk + lane].store(0u); }
    GroupMemoryBarrierWithGroupSync();
    bool go = lead && (rnd == 0u || refined);
    for (uint j = 0u; j < 4u; j++) {
      uint4 firsts = uint4(slotK[16u * pk].load(), slotK[16u * pk + 4u].load(), slotK[16u * pk + 8u].load(), slotK[16u * pk + 12u].load());
      for (uint i = lane; i < P.cap; i += GROUP) {
        if (!go) { break; }
        uint4 k = keyOf(f, i);
        if (j > 0u && !any((firsts == uint4(k.x)))) { continue; }
        uint q = cornerOf(f, i, at, lr.z, alpha);
        if (q < 4u && slotHas(k, pk, q, j)) { slotK[16u * pk + 4u * q + j].max(k[j]); }
      }
      GroupMemoryBarrierWithGroupSync();
    }
  }
  if (lane == 0u) {
    uint base = 5u * (f * K + pk);
    uint n = 0u;
    float4 pts[4] = { float4(0.0), float4(0.0), float4(0.0), float4(0.0) };
    for (uint q = 0u; q < 4u; q++) {
      if (!lead || slotK[16u * pk + 4u * q].load() == 0u) { pts[q] = float4(0.0); continue; }
      uint o = 3u * (f * P.cap + (slotK[16u * pk + 4u * q + 3u].load() & 8191u));
      pts[q] = float4(readings[o].xy, CENTRE[f2u(readings[o + 2u].x)], max(readings[o].z, 1e-3));   // w: the mark's own module size (> 0: the corner is there)
      n++;
    }
    float u = 0.0;
    if (n == 4u) {
      uint m = f2u(c.z);
      u = 0.25 * (distance(pts[0].xy, pts[1].xy) + distance(pts[1].xy, pts[2].xy) + distance(pts[2].xy, pts[3].xy) + distance(pts[3].xy, pts[0].xy)) / (P.modules[m] - 2.0 * pts[0].z);
    }
    for (uint q = 0u; q < 4u; q++) { quads[base + q] = pts[q]; }
    quads[base + 4u] = float4(c.z, u, 0.0, float(n));   // z: the quad's bend (TRACE), 0 until traced
  }
}
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x;
  if (frames[f].valid > 1u) { return; }
  // the peaks' count PEAKS made (the dispatch before)
  uint np = min(vcount[f], PEAK_LIST);
  for (uint k = li; k < np; k += 256u) { list[k] = vlist[f * PEAK_LIST + k]; }
  if (li == 0u) { nlist.store(np); }
  // the quads' keys from zero (the workgroup's memory is not zeroed for it)
  for (uint k = li; k < 16u * K; k += 256u) { slotK[k].store(0u); if (k < 4u * K) { leadK[k].store(0u); } }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    vcount[f] = 0u;
    uint n = min(nlist.load(), PEAK_LIST);
    for (uint k = 0u; k < K; k++) {
      uint b = k;
      for (uint q = k + 1u; q < n; q++) { if (ahead(list[q], list[b])) { b = q; } }
      if (k < n) { float4 e = list[b]; list[b] = list[k]; list[k] = e; cand[f * K + k] = e; peak[k] = e; } else { peak[k] = float4(0.0); }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  gather(f, li / GROUP, peak[li / GROUP], li % GROUP);
}
