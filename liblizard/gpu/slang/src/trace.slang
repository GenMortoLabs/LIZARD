// Defines (finder.mjs): K, TR_L, TR_MAX, TR_STEP.
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/level_bindings.slang"
#include "lib/levels.slang"
#include "lib/bend.slang"

[[vk::binding(7, 0)]] RWStructuredBuffer<float4> quads;
groupshared float2 tp[8][int(TR_MAX)];
groupshared uint tn[8];
groupshared float prof[8][49];
groupshared float nc[8][25];
groupshared uint alive[8];
groupshared float2 best[66];
groupshared float4 qmS;   // the quad's own slot, read once for the workgroup (main)
groupshared uint anyAlive;   // one lane's OR of alive, for the stations loop's end (main)
float tmpl(uint k) { float a = -2.0 + 0.25 * float(k) + 0.125; return (a >= 0.0 && a < 2.0) ? -4.0 / 3.0 : 2.0 / 3.0; }
float ncc(uint tr, int o) {
  float m = 0.0;
  for (uint k = 0u; k < 24u; k++) { m += prof[tr][uint(int(k) + 12 + o)]; }
  m /= 24.0;
  float sv = 0.0;
  float sx = 0.0;
  for (uint k = 0u; k < 24u; k++) { float d = prof[tr][uint(int(k) + 12 + o)] - m; sv += d * d; sx += d * tmpl(k); }
  return sv > 1e-9 ? sx / (sqrt(sv) * 4.6188022) : 0.0;
}
float plumb(float k, float3 ax) {
  float e = 0.0;
  uint sides = 0u;
  for (uint sd = 0u; sd < 4u; sd++) {
    uint t0 = 2u * sd;
    uint n0 = tn[t0];
    uint n1 = tn[t0 + 1u];
    if (n0 + n1 < 4u) { continue; }
    float2 mu = float2(0.0);
    for (uint i = 0u; i < n0; i++) { mu += unbend(tp[t0][i], ax, k); }
    for (uint i = 0u; i < n1; i++) { mu += unbend(tp[t0 + 1u][i], ax, k); }
    mu /= float(n0 + n1);
    float3 S = float3(0.0);
    for (uint i = 0u; i < n0; i++) { float2 d = unbend(tp[t0][i], ax, k) - mu; S += float3(d.x * d.x, d.y * d.y, d.x * d.y); }
    for (uint i = 0u; i < n1; i++) { float2 d = unbend(tp[t0 + 1u][i], ax, k) - mu; S += float3(d.x * d.x, d.y * d.y, d.x * d.y); }
    float tr = S.x + S.y;
    float rt = sqrt(max(0.25 * tr * tr - (S.x * S.y - S.z * S.z), 0.0));
    float l1 = 0.5 * tr + rt;
    if (l1 > 0.0) { e += (0.5 * tr - rt) / l1; }
    sides++;
  }
  return sides >= 2u ? e : 3.0e38;
}
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint wq = wid.x;
  uint f = wq / K;
  uint base = 5u * wq;
  uint li = lix;
  uint tr = li / TR_L;
  uint j = li % TR_L;
  Frame F = frames[f];
  // the quad's slot read by one lane and handed on through workgroup memory, so the early return is one the whole
  // workgroup takes together, uniform control flow for the barriers below; the slot is this workgroup's own
  if (li == 0u) { qmS = quads[base + 4u]; }
  GroupMemoryBarrierWithGroupSync();
  float4 qm = qmS;
  if (F.valid > 1u || F.valid == 0u || !(qm.w == 4.0 || qm.w == 5.0)) { return; }
  float3 ax = bendAxis(F);
  uint side = tr >> 1u;
  float4 a4 = quads[base + side];
  float4 b4 = quads[base + ((side + 1u) & 3u)];
  if ((tr & 1u) == 1u) { float4 t4 = a4; a4 = b4; b4 = t4; }
  float2 A = a4.xy;
  float2 B = b4.xy;
  float2 g = 0.25 * (quads[base].xy + quads[base + 1u].xy + quads[base + 2u].xy + quads[base + 3u].xy);
  float dc = a4.z;
  float ua = a4.w;
  float ub = b4.w;
  float len = distance(A, B);
  float2 U = (B - A) / max(len, 1e-6);
  float2 n = float2(-U.y, U.x);
  if (dot(n, 0.5 * (A + B) - g) < 0.0) { n = -n; }
  float2 p0 = A + U * (16.0 - dc) * ua + n * (dc - 3.0) * ua;
  uint np = 0u;
  uint miss = 0u;
  bool live = ua > 0.0;
  for (uint st = 0u; st < TR_MAX; st++) {
    float2 P = float2(0.0);
    float2 tv = float2(0.0);
    float2 v = float2(0.0);
    if (live) {
      float t = 16.0 + float(TR_STEP) * float(st);
      if (np < 2u) {
        P = A + U * (t - dc) * ua + n * (dc - 3.0) * ua;
        tv = U;
        if (np == 1u) { P += tp[tr][0] - p0; }
      } else if (np == 2u) {
        P = 2.0 * tp[tr][1] - tp[tr][0];
        tv = tp[tr][1] - tp[tr][0];
      } else {
        P = 3.0 * tp[tr][np - 1u] - 3.0 * tp[tr][np - 2u] + tp[tr][np - 3u];
        tv = P - tp[tr][np - 1u];
      }
      float tl = length(tv);
      if (!(tl > 0.0)) { live = false; } else {
        tv /= tl;
        v = float2(-tv.y, tv.x);
        if (dot(v, n) < 0.0) { v = -v; }
        if (dot(P - A, U) > 0.5 * len + 3.0 * ua) { live = false; }
        if (P.x < 2.0 || P.y < 2.0 || P.x > float(F.w) - 3.0 || P.y > float(F.h) - 3.0) { live = false; }
      }
    }
    if (live) {
      float u = ua + (ub - ua) * clamp(dot(P - A, U) / len, 0.0, 1.0);
      for (uint k = j; k < 49u; k += TR_L) {
        float a = -5.0 + 0.25 * float(k);
        prof[tr][k] = look(0u, f, P + (v * a - tv) * u) + look(0u, f, P + v * a * u) + look(0u, f, P + (v * a + tv) * u);
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (live && j < 25u) { nc[tr][j] = ncc(tr, int(j) - 12); }
    GroupMemoryBarrierWithGroupSync();
    if (live) {
      float bc = -2.0;
      int bo = 0;
      for (int o = -12; o <= 12; o++) { float c = nc[tr][o + 12]; if (c > bc) { bc = c; bo = o; } }
      if (bc < 0.6 || bo == -12 || bo == 12) { miss++; if (miss > 2u) { live = false; } } else {
        miss = 0u;
        float l = nc[tr][bo + 11];
        float r = nc[tr][bo + 13];
        float dd = l - 2.0 * bc + r;
        float off = float(bo) + ((dd < 0.0) ? 0.5 * (l - r) / dd : 0.0);
        float u = ua + (ub - ua) * clamp(dot(P - A, U) / len, 0.0, 1.0);
        if (j == 0u) { tp[tr][np] = P + v * (0.25 * off) * u; }
        np++;
      }
    }
    if (j == 0u) { alive[tr] = live ? 1u : 0u; }
    GroupMemoryBarrierWithGroupSync();
    // one lane's OR of the tracks' words handed on through workgroup memory, so the loop's end is one the whole
    // workgroup takes together, uniform control flow for the barriers that follow
    if (li == 0u) { uint a = 0u; for (uint t = 0u; t < 8u; t++) { a |= alive[t]; } anyAlive = a; }
    GroupMemoryBarrierWithGroupSync();
    uint alive0 = anyAlive;
    if (alive0 == 0u) { break; }
  }
  if (j == 0u) { tn[tr] = np; }
  GroupMemoryBarrierWithGroupSync();
  if (li < 66u) { float k = -0.3 + 0.02 * float(li); best[li] = float2(plumb(k, ax), k); }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    float2 b = best[0];
    for (uint i = 1u; i < 66u; i++) { if (best[i].x < b.x) { b = best[i]; } }
    best[0] = float2(b.x, b.x < 3.0e38 ? b.y : 0.0);
  }
  GroupMemoryBarrierWithGroupSync();
  float k0 = best[0].y;
  GroupMemoryBarrierWithGroupSync();
  if (li < 21u) { float k = k0 - 0.02 + 0.002 * float(li); best[li] = float2(plumb(k, ax), k); }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    float2 b = float2(3.0e38, k0);
    for (uint i = 0u; i < 21u; i++) { if (best[i].x < b.x) { b = best[i]; } }
    quads[base + 4u] = float4(qm.x, qm.y, b.x < 3.0e38 ? b.y : 0.0, qm.w);
  }
}
