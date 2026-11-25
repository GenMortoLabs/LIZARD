// F6a: every hypothesis through the track read, a workgroup each.
// Defines (finder.mjs): HYP, K, NEAR, ORIENT, RING_COUNT, ZEROS_9.
#include "lib/sync.slang"
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/level_bindings.slang"
#include "lib/levels.slang"
#include "lib/homography.slang"
#include "lib/bend.slang"

[[vk::binding(7, 0)]] StructuredBuffer<float4> quads;
[[vk::binding(8, 0)]] StructuredBuffer<float4> plan;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> score;
struct PlanBlock { uint4 at[int(RING_COUNT)]; };
[[vk::binding(10, 0)]] ConstantBuffer<PlanBlock> PL;
groupshared float4 sums[64];
groupshared float H[9];
bool near(float2 m, float M) {
  float along = (m.x < 7.0 || m.x > M - 7.0) ? m.y : m.x;
  return min(along, M - along) <= float(NEAR);
}
groupshared uint ok;
[numthreads(64, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x / HYP;
  if (frames[f].valid > 1u) { return; }
  uint hyp = wg.x % HYP;
  uint h0 = hyp % (K * ORIENT * RING_COUNT);
  uint bendBit = hyp / (K * ORIENT * RING_COUNT);
  uint qk = h0 / (ORIENT * RING_COUNT);
  uint o = (h0 / RING_COUNT) % ORIENT;
  uint rg = h0 % RING_COUNT;
  uint base = 5u * (f * K + qk);
  float4 qm = quads[base + 4u];
  uint4 at = PL.at[rg];
  float M = float(at.w);
  // the bend bit: 0 the marks' homography alone, 1 the quad's bend (TRACE), the marks unbent for H and every read
  // bent back
  float kb = bendBit == 0u ? 0.0 : qm.z;
  float3 ax = bendAxis(frames[f]);
  if (li == 0u) {
    ok = (frames[f].valid != 0u && (qm.w == 4.0 || qm.w == 5.0)) ? 1u : 0u;
    if (ok == 1u) {
      float4 c4[4] = { quads[base], quads[base + 1u], quads[base + 2u], quads[base + 3u] };
      if (kb != 0.0) { for (int k = 0; k < 4; k++) { c4[k].xy = unbend(c4[k].xy, ax, kb); } }
      float h[9] = hypH(c4, o, M);
      for (int j = 0; j < 9; j++) { H[j] = h[j]; }
      if (h[8] == 0.0) { ok = 0u; }
    }
  }
  UNIFORM_LOAD(uint live, ok)
  if (live == 0u) {
    if (li == 0u) { score[f * HYP + hyp] = -1.0; }
    return;
  }
  float Hl[9] = ZEROS_9;
  for (int j = 0; j < 9; j++) { Hl[j] = H[j]; }
  float u = distance(applyH(Hl, float2(0.5 * M, 0.5 * M)), applyH(Hl, float2(0.5 * M + 1.0, 0.5 * M)));
  uint l = levelFor(u);
  float4 acc = float4(0.0);
  for (uint s = li; s < at.y; s += 64u) {
    uint b = at.x + 4u * s;
    if (!near(plan[b].xy, M)) { continue; }
    float2 q0 = bend(applyH(Hl, plan[b].xy), ax, kb);
    float2 q1 = bend(applyH(Hl, plan[b + 1u].xy), ax, kb);
    float2 q2 = bend(applyH(Hl, plan[b + 2u].xy), ax, kb);
    float2 q3 = bend(applyH(Hl, plan[b + 3u].xy), ax, kb);
    if (!(onFrame(f, q0) && onFrame(f, q1) && onFrame(f, q2) && onFrame(f, q3))) { continue; }
    acc += float4(look(l, f, q0), look(l, f, q1), look(l, f, q2) + look(l, f, q3), 1.0);
  }
  sums[li] = acc;
  GroupMemoryBarrierWithGroupSync();
  for (uint st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sums[li] += sums[li + st]; }
    GroupMemoryBarrierWithGroupSync();
  }
  float4 lev = sums[0] / max(sums[0].w, 1.0);
  float thr = 0.5 * lev.z;
  float con = lev.y - lev.x;
  GroupMemoryBarrierWithGroupSync();
  float q = 0.0;
  float nq = 0.0;
  for (uint j = li; j < at.z; j += 64u) {
    float4 pt = plan[at.x + 4u * at.y + j];
    if (!near(pt.xy, M)) { continue; }
    float2 qp = bend(applyH(Hl, pt.xy), ax, kb);
    if (!onFrame(f, qp)) { continue; }
    q += pt.z * clamp((thr - look(l, f, qp)) / max(con, 1e-3), -0.5, 0.5);
    nq += 1.0;
  }
  sums[li] = float4(q, nq, 0.0, 0.0);
  GroupMemoryBarrierWithGroupSync();
  for (uint st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sums[li] += sums[li + st]; }
    GroupMemoryBarrierWithGroupSync();
  }
  if (li == 0u) { score[f * HYP + hyp] = (con >= 0.08) ? sums[0].x / max(sums[0].y, 1.0) : -1.0; }
}
