// ANCH_REFIT and ANCH_PREDICT (PREDICT 1), a workgroup of 64 a frame. ANCH_REFIT: the lens about the camera's axis (the
// model of 8 + k terms, k 0 to 3 by AICc) fitted over the border's nodes and the found anchors, robust over the border
// (up to four rounds, a node past max(half a module, three robust spreads of the residuals) dropped), the C's
// border_fill_anchors; over the border alone where fewer than the ring's minimum anchors are found; then each anchor's
// offset with the lens taken out (the mesh's knots). ANCH_PREDICT: the fit over the anchors alone (its radial terms no
// more than the anchors past 8 allow), each border node's place written for the border's second search (w 2); nothing
// under the minimum. wcnt, wnba and wchg, whose atomics start from zero, are zeroed first; everything else is
// written before it is read. The solves are unrolled (every index a constant). Defines (core/dec/build.cpp
// Anchors::refit): PREDICT (0 or 1), ANCH_K, ANCH_SIDE, NODES_MAX, LENS_P, RESID (uint), WMIN, TAU_SPREAD (float),
// RING_COUNT, LATTICE_VECS, ZEROS_<n>.
#include "lib/sync.slang"
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/map.slang"
#if PREDICT
#define UNDER_MIN return;
#define FIRST (count + li)
#define KMAX (wu >= LENS_P ? 3u : (wu > 8u ? wu - 8u : 0u))
#define WEIGHT pw[i]
#else
#define UNDER_MIN if (li < NK) { ak[li] = 0u; }
#define FIRST li
#define KMAX 3u
#define WEIGHT (i < count && pd[min(i, NODES_MAX - 1u)] == 1u) ? 0.0 : pw[i]
#endif
// The lens model's rows: parameter a's coefficient in a point's x and its y, d.x, d.y, 1, the
// projective pair, then the radial terms t, t^2, t^3 along q; LENS_HX and LENS_HY whether it has one in each (a
// parameter with none takes no term in a sum).
static const bool LENS_HX[11] = { true, true, false, false, true, false, true, true, true, true, true };
static const bool LENS_HY[11] = { false, false, true, true, false, true, true, true, true, true, true };
float lensRx(int a, float2 d, float2 q, float t, float t2, float t3) {
  return a == 0 ? d.x : a == 1 ? d.y : a == 4 ? 1.0 : a == 6 ? (-d.x * d.x) : a == 7 ? (-d.y * d.x) : a == 8 ? (t * q.x) : a == 9 ? (t2 * q.x) : (t3 * q.x);
}
float lensRy(int a, float2 d, float2 q, float t, float t2, float t3) {
  return a == 2 ? d.x : a == 3 ? d.y : a == 5 ? 1.0 : a == 6 ? (-d.x * d.y) : a == 7 ? (-d.y * d.y) : a == 8 ? (t * q.y) : a == 9 ? (t2 * q.y) : (t3 * q.y);
}
// the packed upper triangle's entry (a, b), b >= a
int lensIdx(int a, int b) { return a * LENS_P - (a * (a - 1)) / 2 + (b - a); }
// A point's terms in the lens fit (p, the frame's LF, the axis LA, lir): d its place normalised over the points' box, q
// about the axis over the frame's farthest corner, t the radial argument and its powers.
#define LENS_POINT \
  float2 d = (p - LF.ctr) * LF.sig; \
  float2 q = (p - LA) * lir; \
  float t = (dot(q, q) - LF.smid) * LF.ish; \
  float t2 = t * t; \
  float t3 = t2 * t;
[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<float4> nodes;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> sel;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> meas;
struct MapRec { float m[16]; };
[[vk::binding(4, 0)]] RWStructuredBuffer<MapRec> maps;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> resid;
struct FitBlock { uint4 count; uint4 base; float4 modules; };
[[vk::binding(6, 0)]] ConstantBuffer<FitBlock> P;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> anch;
struct LatticeBlock { uint4 at[RING_COUNT]; float4 lat[LATTICE_VECS]; };
[[vk::binding(8, 0)]] ConstantBuffer<LatticeBlock> LAT;
struct AnchorsBlock { float4 k[3 * RING_COUNT]; float4 g[9 * RING_COUNT]; float4 s[RING_COUNT]; };
[[vk::binding(9, 0)]] ConstantBuffer<AnchorsBlock> A;
float knot(uint r, uint i) { return A.k[3u * r + (i >> 2u)][i & 3u]; }
#include "lib/lens.slang"
struct LensFrame { float2 ctr; float sig; float smid; float ish; uint kmax; uint nw; uint on; };
static const uint NK = ANCH_K;
static const uint SIDE = ANCH_SIDE;
groupshared float4 pp[NODES_MAX + ANCH_K];
groupshared float pw[NODES_MAX + ANCH_K];
groupshared float pr[NODES_MAX];
groupshared uint pd[NODES_MAX];
groupshared uint ak[ANCH_K];
groupshared LensFrame LF;
groupshared float LS[16][77];
groupshared float LT[4][LENS_P];
groupshared uint LK[4];
groupshared float4 LR[64];
groupshared float4 LQ[64];
groupshared float LP[77];
groupshared Lens LZ;
groupshared uint wu;
groupshared uint wok;
groupshared float wmed;
groupshared uint wnb;
groupshared float wtau;
groupshared Atomic<uint> wcnt;
groupshared Atomic<uint> wnba;
groupshared Atomic<uint> wchg;
groupshared float4 LR8[8];
groupshared float4 LQ8[8];
[numthreads(64, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x;
  uint4 pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  Frame F = frames[f];
  uint rgi = pick.x;
  uint count = P.count[rgi];
  uint base = P.base[rgi];
  // the counters from zero (the workgroup's memory is not zeroed for it)
  if (li == 0u) { wcnt.store(0u); wnba.store(0u); wchg.store(0u); }
  GroupMemoryBarrierWithGroupSync();
  if (li < NK) {
    bool on = anch[NK * f + li].w > 0.5;
    ak[li] = on ? 1u : 0u;
    if (on) { wcnt.add(1u); }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) { wu = wcnt.load(); }
  // under the anchors' minimum ANCH_PREDICT has nothing to predict from; ANCH_REFIT fits over the border alone (the
  // anchors off, no mesh: the C's border_fill_anchors with the anchors cleared)
  { UNIFORM_LOAD(uint wu0, wu) if (wu0 < f2u(A.s[rgi].y)) { UNDER_MIN } }
  float M[16] = maps[f].m;
  Lens L0 = lensOf(f);
#if PREDICT
  for (uint n = li; n < count; n += 64u) { pp[n] = float4(toImage(M, nodes[base + n].xy), 0.0, 0.0); }
#else
  for (uint n = li; n < count; n += 64u) {
    float4 e = meas[f * NODES_MAX + n];
    bool ok = e.w != 0.0 && e.z >= WMIN;
    float2 p = toImage(M, nodes[base + n].xy);
    float2 pe = e.xy - p;
    pp[n] = float4(p, ok ? pe : float2(0.0));
    pw[n] = ok ? e.z : 0.0;
    pd[n] = 0u;
    if (ok) { wnba.add(1u); }
  }
#endif
  if (li < NK) {
    float4 e = anch[NK * f + li];
    float2 X = float2(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    float2 p = toImage(M, X);
    float2 h = p;
    bool on = ak[li] == 1u;
    float2 pe = p + e.xy + lensAt(L0, p) - h;
    pp[count + li] = float4(h, on ? pe : float2(0.0));
    pw[count + li] = on ? 1.0 : 0.0;
  }
  STORAGE_BARRIER
  GroupMemoryBarrierWithGroupSync();
  uint npts = count + NK;
  uint aw = F.axis;
  float2 LA = float2(0.5 * float(F.w), 0.5 * float(F.h));
  if (aw != 0u) { LA = float2(0.5 * float(aw & 0xffffu) - 16384.0, 0.5 * float(aw >> 16u) - 16384.0); }
  float2 fwh = float2(float(F.w), float(F.h));
  float lR = max(max(length(LA), length(float2(fwh.x, 0.0) - LA)), max(length(float2(0.0, fwh.y) - LA), length(fwh - LA)));
  float lir = 1.0 / max(lR, 1.0);
  {
    float2 lo = float2(3.0e38);
    float2 hi = float2(-3.0e38);
    float smin = 3.0e38;
    float smax = 0.0;
    uint nw = 0u;
    for (uint i = FIRST; i < npts; i += 64u) {
      if (!(pw[i] > 0.0)) { continue; }
      float2 p = pp[i].xy;
      lo = min(lo, p);
      hi = max(hi, p);
      float2 q = (p - LA) * lir;
      float s = dot(q, q);
      smin = min(smin, s);
      smax = max(smax, s);
      nw += 1u;
    }
    LR[li] = float4(lo, hi);
    LQ[li] = float4(smin, smax, float(nw), 0.0);
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < 8u) {
    float2 lo = LR[8u * li].xy;
    float2 hi = LR[8u * li].zw;
    float smin = LQ[8u * li].x;
    float smax = LQ[8u * li].y;
    float nw = LQ[8u * li].z;
    for (uint s = 8u * li + 1u; s < 8u * li + 8u; s++) {
      lo = min(lo, LR[s].xy);
      hi = max(hi, LR[s].zw);
      smin = min(smin, LQ[s].x);
      smax = max(smax, LQ[s].y);
      nw += LQ[s].z;
    }
    LR8[li] = float4(lo, hi);
    LQ8[li] = float4(smin, smax, nw, 0.0);
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    float2 lo = LR8[0].xy;
    float2 hi = LR8[0].zw;
    float smin = LQ8[0].x;
    float smax = LQ8[0].y;
    uint nw = f2u(LQ8[0].z);
    for (uint s = 1u; s < 8u; s++) {
      lo = min(lo, LR8[s].xy);
      hi = max(hi, LR8[s].zw);
      smin = min(smin, LQ8[s].x);
      smax = max(smax, LQ8[s].y);
      nw += f2u(LQ8[s].z);
    }
    // the radial terms: ANCH_PREDICT's fit over the anchors alone takes no more than anchors where they are fewer than
    // the model's terms (the 64 ring's 3 x 3 grid: the C's acquire.c lens_cap, which seeds the border's search);
    // ANCH_REFIT's over the border and the anchors every term
    LF = LensFrame(0.5 * (lo + hi), 2.0 / max(max(hi.x - lo.x, hi.y - lo.y), 1.0), 0.5 * (smin + smax),
      2.0 / max(smax - smin, 1e-6), KMAX, nw, nw > 0u ? 1u : 0u);
    wok = 0u;
#if !PREDICT
    wnb = wnba.load();
    uint nq = (count + 4u) / 4u;
    float px = 0.0;
    float md = 0.0;
    for (uint c = 0u; c < 4u; c++) {
      uint a = (c * (nq - 1u)) % count;
      uint b = (((c + 1u) % 4u) * (nq - 1u)) % count;
      px += distance(pp[a].xy, pp[b].xy);
      md += distance(nodes[base + a].xy, nodes[base + b].xy);
    }
    wtau = 0.5 * px / max(md, 1.0);
#endif
  }
  GroupMemoryBarrierWithGroupSync();
#if PREDICT
  {
#else
  for (uint rd = 0u; rd < 4u; rd++) {
#endif
    if (li < 16u) {
      float acc[77] = ZEROS_77;
      for (uint i = FIRST; i < npts; i += 16u) {
        float w = WEIGHT;
        if (!(w > 0.0)) { continue; }
        float2 p = pp[i].xy;
        float2 r = pp[i].zw;
        LENS_POINT
        // the normal equations' sums, each over the terms the pair has, then the right-hand sides
        [ForceUnroll] for (int a = 0; a < int(LENS_P); a++) [ForceUnroll] for (int b = a; b < int(LENS_P); b++) {
          const bool hx = LENS_HX[a] && LENS_HX[b], hy = LENS_HY[a] && LENS_HY[b];
          if (hx && hy) acc[lensIdx(a, b)] += w * (lensRx(a, d, q, t, t2, t3) * lensRx(b, d, q, t, t2, t3) + lensRy(a, d, q, t, t2, t3) * lensRy(b, d, q, t, t2, t3));
          else if (hx) acc[lensIdx(a, b)] += w * (lensRx(a, d, q, t, t2, t3) * lensRx(b, d, q, t, t2, t3));
          else if (hy) acc[lensIdx(a, b)] += w * (lensRy(a, d, q, t, t2, t3) * lensRy(b, d, q, t, t2, t3));
        }
        [ForceUnroll] for (int a = 0; a < int(LENS_P); a++) {
          if (LENS_HX[a] && LENS_HY[a]) acc[66 + a] += w * (lensRx(a, d, q, t, t2, t3) * r.x + lensRy(a, d, q, t, t2, t3) * r.y);
          else if (LENS_HX[a]) acc[66 + a] += w * (lensRx(a, d, q, t, t2, t3) * r.x);
          else acc[66 + a] += w * (lensRy(a, d, q, t, t2, t3) * r.y);
        }
      }
      LS[li] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint e = li; e < 77u; e += 64u) {
      float a = 0.0;
      for (uint s = 0u; s < 16u; s++) { a += LS[s][e]; }
      LP[e] = a;
    }
    GroupMemoryBarrierWithGroupSync();
    if (li == 0u) {
      for (uint k = 0u; k < 4u; k++) { LK[k] = 0u; }
      // the normal equations unpacked, scaled to a unit diagonal (a ridge of 1e-4 and 1e-9), factored by Cholesky (okD
      // the first column whose pivot fell under 1e-7), then each model of 8 + K terms solved where its columns factored
      float Ns[11][11] = ZEROS_11_11;
      float bs[11] = ZEROS_11;
      float Dg[11] = ZEROS_11;
      float Ln[11][11] = ZEROS_11_11;
      float z[11] = ZEROS_11;
      float Ap[77] = ZEROS_77;
      [ForceUnroll] for (int i = 0; i < 77; i++) Ap[i] = LP[i];
      [ForceUnroll] for (int a = 0; a < int(LENS_P); a++) [ForceUnroll] for (int b = a; b < int(LENS_P); b++) { Ns[a][b] = Ap[lensIdx(a, b)]; Ns[b][a] = Ap[lensIdx(a, b)]; }
      [ForceUnroll] for (int a = 0; a < int(LENS_P); a++) Dg[a] = 1.0 / sqrt(max(Ns[a][a], 1e-30));
      [ForceUnroll] for (int a = 0; a < int(LENS_P); a++) {
        [ForceUnroll] for (int b = 0; b < int(LENS_P); b++) Ns[a][b] *= Dg[a] * Dg[b];
        Ns[a][a] = Ns[a][a] * 1.0001 + 1e-9; bs[a] = Ap[66 + a] * Dg[a];
      }
      uint okD = 11u;
      [ForceUnroll] for (int j = 0; j < int(LENS_P); j++) {
        { float v = Ns[j][j]; [ForceUnroll] for (int k = 0; k < j; k++) v = v - Ln[j][k] * Ln[j][k];
          uint om = min(okD, uint(j)); okD = !(v > 1e-7) ? om : okD; Ln[j][j] = sqrt(max(v, 1e-7)); }
        [ForceUnroll] for (int i = j + 1; i < int(LENS_P); i++) {
          float e = Ns[i][j]; [ForceUnroll] for (int k = 0; k < j; k++) e = e - Ln[i][k] * Ln[j][k];
          Ln[i][j] = (e) / Ln[j][j];
        }
      }
      [ForceUnroll] for (int i = 0; i < int(LENS_P); i++) {
        float e = bs[i]; [ForceUnroll] for (int k = 0; k < i; k++) e = e - Ln[i][k] * z[k];
        z[i] = (e) / Ln[i][i];
      }
      [ForceUnroll] for (int K = 0; K <= 3; K++) {
        if (uint(8 + K) <= okD && uint(K) <= LF.kmax) {
          float y[11] = ZEROS_11;
          [ForceUnroll] for (int i = 8 + K - 1; i >= 0; i--) {
            float e = z[i]; [ForceUnroll] for (int k = i + 1; k < 8 + K; k++) e = e - Ln[k][i] * y[k];
            y[i] = (e) / Ln[i][i];
          }
          [ForceUnroll] for (int a = 0; a < int(LENS_P); a++) LT[K][a] = a < 8 + K ? Dg[a] * y[a] : 0.0;
          LK[K] = 1u;
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
    {
      float4 rs = float4(0.0);
      uint nu = 0u;
      for (uint i = FIRST; i < npts; i += 64u) {
        float w = WEIGHT;
        if (!(w > 0.0)) { continue; }
        nu += 1u;
        float2 p = pp[i].xy;
        float2 r = pp[i].zw;
        LENS_POINT
        // each model fitted: its residual's square, weighted
        [ForceUnroll] for (int K = 0; K < 4; K++) {
          if (LK[K] != 0u) {
            float mx = 0.0, my = 0.0;
            bool ax = false, ay = false;
            [ForceUnroll] for (int a = 0; a < 8 + K; a++) if (LENS_HX[a]) { const float u = LT[K][a] * lensRx(a, d, q, t, t2, t3); mx = ax ? mx + u : u; ax = true; }
            [ForceUnroll] for (int a = 0; a < 8 + K; a++) if (LENS_HY[a]) { const float u = LT[K][a] * lensRy(a, d, q, t, t2, t3); my = ay ? my + u : u; ay = true; }
            float2 e = r - float2(mx, my);
            rs[K] += w * dot(e, e);
          }
        }
      }
      LR[li] = rs;
      LQ[li] = float4(float(nu), 0.0, 0.0, 0.0);
    }
    GroupMemoryBarrierWithGroupSync();
    if (li < 8u) {
      float4 rss = LR[8u * li];
      float nu = LQ[8u * li].x;
      for (uint s = 8u * li + 1u; s < 8u * li + 8u; s++) { rss += LR[s]; nu += LQ[s].x; }
      LR8[li] = rss;
      LQ8[li] = float4(nu, 0.0, 0.0, 0.0);
    }
    GroupMemoryBarrierWithGroupSync();
    if (li == 0u) {
      float4 rss = float4(0.0);
      uint nu = 0u;
      for (uint s = 0u; s < 8u; s++) { rss += LR8[s]; nu += f2u(LQ8[s].x); }
      uint kb = 4u;
      float best = 3.0e38;
      float nobs = 2.0 * float(nu);
      for (uint k = 0u; k < 4u; k++) {
        float pk = 8.0 + float(k);
        if (LK[k] == 0u || nobs - pk - 1.0 <= 0.0) { continue; }
        float score = nobs * log(max(rss[k], 1e-30) / nobs) + 2.0 * pk + 2.0 * pk * (pk + 1.0) / (nobs - pk - 1.0);
        if (score < best) { best = score; kb = k; }
      }
      float tk[LENS_P] = ZEROS_11;
      [ForceUnroll] for (int K = 0; K < 4; K++) { if (kb == uint(K)) { [ForceUnroll] for (int a = 0; a < int(LENS_P); a++) tk[a] = LT[K][a]; } }
      if (kb < 4u) {
        LZ = Lens(float4(LF.ctr, LF.sig, LF.smid), float4(LA, lir, LF.ish), float4(tk[0], tk[1], tk[2], tk[3]),
          float4(tk[4], tk[5], tk[6], tk[7]), float4(tk[8], tk[9], tk[10], 0.0));
        wok = 1u;
      }
      wu = kb < 4u ? 1u : 0u;
    }
#if PREDICT
  }
#else
    { UNIFORM_LOAD(uint wu1, wu) if (wu1 == 0u || rd == 3u) { break; } }
    Lens lz = LZ;
    for (uint n = li; n < count; n += 64u) { float dn = length(pp[n].zw - lensAt(lz, pp[n].xy)); pr[n] = pw[n] > 0.0 ? dn : 0.0; }
    if (li == 0u) { wmed = 0.0; wchg.store(0u); }
    GroupMemoryBarrierWithGroupSync();
    uint nb = wnb;
    for (uint n = li; n < count; n += 64u) {
      if (!(pw[n] > 0.0)) { continue; }
      float v = pr[n];
      uint rank = 0u;
      for (uint j = 0u; j < count; j++) { float u = pr[j]; rank += (pw[j] > 0.0 && (u < v || (u == v && j < n))) ? 1u : 0u; }
      if (rank == nb / 2u) { wmed = v; }
    }
    GroupMemoryBarrierWithGroupSync();
    float tau = max(wtau, TAU_SPREAD * wmed);
    for (uint n = li; n < count; n += 64u) {
      uint d = (pw[n] > 0.0 && pr[n] > tau) ? 1u : 0u;
      if (d != pd[n]) { wchg.add(1u); }
      pd[n] = d;
    }
    GroupMemoryBarrierWithGroupSync();
    if (li == 0u) { wu = wchg.load(); }
    { UNIFORM_LOAD(uint wu2, wu) if (wu2 == 0u) { break; } }
  }
#endif
  { UNIFORM_LOAD(uint wok0, wok) if (wok0 == 0u) { return; } }
  Lens lz = LZ;
#if PREDICT
  for (uint n = li; n < count; n += 64u) { meas[f * NODES_MAX + n] = float4(pp[n].xy + lensAt(lz, pp[n].xy), 0.0, 2.0); }
#else
  if (li == 0u) {
    uint b = f * RESID;
    resid[b] = lz.c;
    resid[b + 1u] = lz.a;
    resid[b + 2u] = lz.t0;
    resid[b + 3u] = lz.t1;
    resid[b + 4u] = lz.t2;
  }
  // the border's nodes reach the map through the fit alone (2026-11-24, STATUS "The GPU's registration as the C's one
  // model"): the C keeps its measured border nodes in its node mesh, but here a band of their residuals (a lattice
  // step inward) read 0.35% fewer blocks over the published lenses and 0.18% fewer in the stress spectrum (0.07% and
  // 0.44% more on the judge's front and angles), and the Coons patch of the sides' curves before it 1.1% fewer on both
  STORAGE_BARRIER
  GroupMemoryBarrierWithGroupSync();
  if (li < NK && ak[li] == 1u) {
    float2 X = float2(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    float4 e = anch[NK * f + li];
    float4 q = pp[count + li];
    anch[NK * f + li] = float4(q.zw - lensAt(lz, q.xy), e.zw);
  }
#endif
}
