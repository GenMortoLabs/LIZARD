// ANCH_FIT: a workgroup of 64 a frame: the mesh's knots (each found anchor's offset; a missing one the mean of its
// found neighbours, ANCH_KNOT_ROUNDS rounds; all zero where fewer than the ring's minimum are found), the squares' fill
// (the shading's least-squares surface over the picture's points in no square, from ANCH_SUMS' slices), the picture's
// gain from the signal's swing (SPEC 6.10), and the record pass 1 reads; the count found into result word 31. swH,
// whose atomics start from zero, is zeroed first; everything else is written before it is
// read. Defines (core/dec/build.cpp Anchors): ANCH_K, ANCH_SIDE, ANCH_REC, ANCH_SURF, ANCH_GAIN, ANCH_KNOT_ROUNDS, SW_T, SW_R, SW_LAG,
// SW_MIN, SW_BINS (uint), SPREADS3 (float); lib/anch_head.slang's and lib/anch_fill.slang's.
#include "lib/sync.slang"
#include "lib/solve.slang"
#include "lib/anch_head.slang"
struct Result { float r[32]; };
[[vk::binding(7, 0)]] StructuredBuffer<float4> anch;
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> arec;
[[vk::binding(9, 0)]] RWStructuredBuffer<Result> result;
[[vk::binding(10, 0)]] StructuredBuffer<float4> fill;
#include "lib/anch_fill.slang"
static const uint NK = ANCH_K;
static const uint SIDE = ANCH_SIDE;
groupshared float2 kv[ANCH_K];
groupshared uint kn[ANCH_K];
groupshared uint kn2[ANCH_K];
groupshared float red[6 * 64];
groupshared uint wfound;
// the gain's tiles (SW_T^2): each its swing, its centre (u, w), its state (0 kept, 1 dropped, 2 too few pairs) and
// residual; the fit's sums a lane, its coefficients, the round's word (go on), the median's bracket (its low end, its
// width, the rank sought within it) and the histogram
groupshared float swA[SW_T * SW_T];
groupshared float2 swC[SW_T * SW_T];
groupshared uint swD[SW_T * SW_T];
groupshared float swR[SW_T * SW_T];
groupshared float swS[28 * 64];
groupshared float swAl[6];
groupshared uint swGo;
groupshared float4 swLH;
groupshared Atomic<uint> swH[SW_BINS];
// a module coordinate within the band of one of the ring's S anchors' centres (the fill's band, fillBand's rule)
bool inBand(uint r, float m, float aa, uint S) {
  bool hit = false;
  for (uint i = 1u; i <= S; i++) { hit = hit || abs(knot(r, i) - m) <= 0.5 * float(ANCH_BAND2) * aa; }
  return hit;
}
// a knot's found 4-neighbours within the ring's S x S grid (indices j SIDE + i; those past it never read)
float3 around(uint q, uint S) {
  uint i = q % SIDE;
  uint j = q / SIDE;
  float2 s = float2(0.0);
  float c = 0.0;
  if (i > 0u && kn[q - 1u] == 1u) { s += kv[q - 1u]; c += 1.0; }
  if (i + 1u < S && kn[q + 1u] == 1u) { s += kv[q + 1u]; c += 1.0; }
  if (j > 0u && kn[q - SIDE] == 1u) { s += kv[q - SIDE]; c += 1.0; }
  if (j + 1u < S && kn[q + SIDE] == 1u) { s += kv[q + SIDE]; c += 1.0; }
  return float3(s, c);
}
[numthreads(64, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x;
  Frame F = frames[f];
  uint4 pick = sel[f];
  uint ab = f * ANCH_REC;
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { arec[ab] = float4(0.0); result[f].r[31] = 0.0; }
    return;
  }
  uint rgi = pick.x;
  uint S = gridOf(rgi);
  // the swing's histogram from zero (the workgroup's memory is not zeroed for it; the fits put it back to zero after)
  for (uint i = li; i < SW_BINS; i += 64u) { swH[i].store(0u); }
  if (li < NK) {
    float4 e = anch[NK * f + li];
    bool ok = e.w > 0.5;
    kv[li] = ok ? e.xy : float2(0.0);
    kn[li] = ok ? 1u : 0u;
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    uint c = 0u;
    for (uint q = 0u; q < NK; q++) { c += kn[q]; }
    wfound = c;
  }
  UNIFORM_LOAD(uint found, wfound)
  if (li == 0u) { result[f].r[31] = float(found); }
  if (found < gridMin(rgi) && li < NK) { kv[li] = float2(0.0); kn[li] = 1u; }
  for (uint rd = 0u; rd < ANCH_KNOT_ROUNDS; rd++) {
    GroupMemoryBarrierWithGroupSync();
    if (li < NK) {
      uint now = kn[li];
      if (now == 0u) {
        float3 nb = around(li, S);
        if (nb.z > 0.0) { kv[li] = nb.xy / nb.z; now = 1u; }
      }
      kn2[li] = now;
    }
    GroupMemoryBarrierWithGroupSync();
    if (li < NK) { kn[li] = kn2[li]; }
  }
  float4 geo = A.k[3u * rgi + 2u];
  float span = geo.z;
  float margin = geo.w;
  float M[16] = maps[f].m;
  Lens lz = lensOf(f);
  int aa = f2i(geo.y);
  // the squares' fill (SPEC 6.13; core/dec/build.cpp Anchors::FILL): the shading's least-squares surface over the
  // picture's points in no square, its normal matrix's inverse the ring's (A.g, the C's hole_inv); its six sums
  // ANCH_SUMS' slices added (a lane a sum)
  if (li < 6u) {
    uint o = fillSums(f) + (li >> 2u);
    float s = 0.0;
    for (uint g = 0u; g < ANCH_GS; g++) { s += fill[o + 2u * g][li & 3u]; }
    red[64u * li] = s;
  }
  GroupMemoryBarrierWithGroupSync();
  // The gain (SPEC 6.10, the C's lizard_finish_ext): the signal's swing in SW_T x SW_T tiles of the picture, the root
  // of half the mean squared difference of neighbours on runs of reads span / SW_LAG modules apart along SW_R rows a
  // tile, outside the squares' bands (lib/anch_fill.slang); the fill's basis fitted to the tiles' swings at their centres,
  // robustly (a tile past three robust spreads dropped and the surface refitted, three rounds; the median from two
  // histograms); pass 1 multiplies each sample by the surface's mean over its value (pass1.slang). A
  // falloff scales the swing, an added light does not.
  {
    const uint RUN = SW_LAG / SW_T;
    float lagm = span / float(SW_LAG);
    float tw = span / float(SW_T);
    float aaf = float(aa);
    for (uint t = li; t < SW_T * SW_T; t += 64u) {
      uint tx = t % SW_T;
      uint ty = t / SW_T;
      float D = 0.0;
      uint cnt = 0u;
      for (uint r = 0u; r < SW_R; r++) {
        float my = margin + tw * (float(ty) + (float(r) + 0.5) / float(SW_R));
        float mx0 = margin + tw * float(tx) + 0.5 * lagm;
        bool yb = inBand(rgi, my, aaf, S);
        // the run's points through the map at its ends and middle, quadratic between (P(s) = P0 + s (Bq + s Cq), s in
        // [0, 1])
        float2 P0 = mapAt(M, lz, float2(mx0, my));
        float2 Pm = mapAt(M, lz, float2(mx0 + 0.5 * float(RUN - 1u) * lagm, my));
        float2 P1 = mapAt(M, lz, float2(mx0 + float(RUN - 1u) * lagm, my));
        float2 Bq = 4.0 * Pm - 3.0 * P0 - P1;
        float2 Cq = 2.0 * (P0 + P1) - 4.0 * Pm;
        float vp = bilin0(f, P0, int(F.w), int(F.h));
        bool bp = yb && inBand(rgi, mx0, aaf, S);
        for (uint j = 1u; j < RUN; j++) {
          float sj = float(j) / float(RUN - 1u);
          float v = bilin0(f, P0 + sj * (Bq + sj * Cq), int(F.w), int(F.h));
          bool bj = yb && inBand(rgi, mx0 + float(j) * lagm, aaf, S);
          if (!(bp || bj)) { D += (v - vp) * (v - vp); cnt++; }
          vp = v;
          bp = bj;
        }
      }
      swA[t] = cnt >= SW_MIN ? sqrt(D / (2.0 * float(cnt))) : 0.0;
      swD[t] = cnt >= SW_MIN ? 0u : 2u;
      swC[t] = float2((2.0 * float(tx) + 1.0) / float(SW_T) - 1.0, (2.0 * float(ty) + 1.0) / float(SW_T) - 1.0);
    }
    if (li == 0u) { swAl[0] = 1.0; swAl[1] = 0.0; swAl[2] = 0.0; swAl[3] = 0.0; swAl[4] = 0.0; swAl[5] = 0.0; }
    for (uint rd = 0u; rd < 4u; rd++) {
      GroupMemoryBarrierWithGroupSync();
      float q[28] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      for (uint t = li; t < SW_T * SW_T; t += 64u) {
        if (swD[t] != 0u) { continue; }
        float2 c = swC[t];
        float av = swA[t];
        float b0 = 1.0;
        float b1 = c.x;
        float b2 = c.x * c.x;
        float b3 = c.y;
        float b4 = c.y * c.y;
        float b5 = c.x * c.y;
        q[0] += b0 * b0;
        q[1] += b0 * b1;
        q[2] += b0 * b2;
        q[3] += b0 * b3;
        q[4] += b0 * b4;
        q[5] += b0 * b5;
        q[6] += b1 * b1;
        q[7] += b1 * b2;
        q[8] += b1 * b3;
        q[9] += b1 * b4;
        q[10] += b1 * b5;
        q[11] += b2 * b2;
        q[12] += b2 * b3;
        q[13] += b2 * b4;
        q[14] += b2 * b5;
        q[15] += b3 * b3;
        q[16] += b3 * b4;
        q[17] += b3 * b5;
        q[18] += b4 * b4;
        q[19] += b4 * b5;
        q[20] += b5 * b5;
        q[21] += b0 * av;
        q[22] += b1 * av;
        q[23] += b2 * av;
        q[24] += b3 * av;
        q[25] += b4 * av;
        q[26] += b5 * av;
        q[27] += 1.0;
      }
      swS[0u + li] = q[0];
      swS[64u + li] = q[1];
      swS[128u + li] = q[2];
      swS[192u + li] = q[3];
      swS[256u + li] = q[4];
      swS[320u + li] = q[5];
      swS[384u + li] = q[6];
      swS[448u + li] = q[7];
      swS[512u + li] = q[8];
      swS[576u + li] = q[9];
      swS[640u + li] = q[10];
      swS[704u + li] = q[11];
      swS[768u + li] = q[12];
      swS[832u + li] = q[13];
      swS[896u + li] = q[14];
      swS[960u + li] = q[15];
      swS[1024u + li] = q[16];
      swS[1088u + li] = q[17];
      swS[1152u + li] = q[18];
      swS[1216u + li] = q[19];
      swS[1280u + li] = q[20];
      swS[1344u + li] = q[21];
      swS[1408u + li] = q[22];
      swS[1472u + li] = q[23];
      swS[1536u + li] = q[24];
      swS[1600u + li] = q[25];
      swS[1664u + li] = q[26];
      swS[1728u + li] = q[27];
      GroupMemoryBarrierWithGroupSync();
      if (li < 28u) {
        float sm = 0.0;
        for (uint i = 0u; i < 64u; i++) { sm += swS[li * 64u + i]; }
        swS[li * 64u] = sm;
      }
      GroupMemoryBarrierWithGroupSync();
      if (li == 0u) {
        float N[6][7];
        N[0][0] = swS[0];
        N[0][1] = swS[64];
        N[0][2] = swS[128];
        N[0][3] = swS[192];
        N[0][4] = swS[256];
        N[0][5] = swS[320];
        N[0][6] = swS[1344];
        N[1][0] = swS[64];
        N[1][1] = swS[384];
        N[1][2] = swS[448];
        N[1][3] = swS[512];
        N[1][4] = swS[576];
        N[1][5] = swS[640];
        N[1][6] = swS[1408];
        N[2][0] = swS[128];
        N[2][1] = swS[448];
        N[2][2] = swS[704];
        N[2][3] = swS[768];
        N[2][4] = swS[832];
        N[2][5] = swS[896];
        N[2][6] = swS[1472];
        N[3][0] = swS[192];
        N[3][1] = swS[512];
        N[3][2] = swS[768];
        N[3][3] = swS[960];
        N[3][4] = swS[1024];
        N[3][5] = swS[1088];
        N[3][6] = swS[1536];
        N[4][0] = swS[256];
        N[4][1] = swS[576];
        N[4][2] = swS[832];
        N[4][3] = swS[1024];
        N[4][4] = swS[1152];
        N[4][5] = swS[1216];
        N[4][6] = swS[1600];
        N[5][0] = swS[320];
        N[5][1] = swS[640];
        N[5][2] = swS[896];
        N[5][3] = swS[1088];
        N[5][4] = swS[1216];
        N[5][5] = swS[1280];
        N[5][6] = swS[1664];
        bool ok = swS[27 * 64] >= 12.0;
        gaussJordan(N, 1e-12, ok);
        float a0 = N[0][6] / N[0][0];
        float a1 = N[1][6] / N[1][1];
        float a2 = N[2][6] / N[2][2];
        float a3 = N[3][6] / N[3][3];
        float a4 = N[4][6] / N[4][4];
        float a5 = N[5][6] / N[5][5];
        ok = ok && a0 > 0.0 && a0 + (a2 + a4) / 3.0 > 0.0;
        if (ok) { swAl[0] = a0; swAl[1] = a1; swAl[2] = a2; swAl[3] = a3; swAl[4] = a4; swAl[5] = a5; }
        swGo = (ok && rd < 3u) ? 1u : 0u;
      }
      UNIFORM_LOAD(uint go, swGo)
      if (go == 0u) { break; }
      // the residuals, the median of the kept ones (the C's sorted[kept / 2]) from two histograms, the drops
      float rmax = 0.0;
      float kept = 0.0;
      for (uint t = li; t < SW_T * SW_T; t += 64u) {
        float2 c = swC[t];
        float fv = swAl[0] + c.x * (swAl[1] + c.x * swAl[2]) + c.y * (swAl[3] + swAl[5] * c.x + c.y * swAl[4]);
        swR[t] = abs(swA[t] - fv);
        if (swD[t] == 0u) { rmax = max(rmax, swR[t]); kept += 1.0; }
      }
      swS[li] = rmax;
      swS[64u + li] = kept;
      GroupMemoryBarrierWithGroupSync();
      if (li == 0u) {
        float mx = 0.0;
        float kk = 0.0;
        for (uint i = 0u; i < 64u; i++) { mx = max(mx, swS[i]); kk += swS[64u + i]; }
        // the bracket [0, the largest residual], a hair wider so the largest falls in the last bin; the rank sought
        swLH = float4(0.0, max(mx, 1e-30) * (1.0 + 1.0 / 1024.0), floor(0.5 * kk) + 1.0, 0.0);
      }
      for (uint lv = 0u; lv < 2u; lv++) {
        UNIFORM_LOAD(float4 lh, swLH)
        for (uint t = li; t < SW_T * SW_T; t += 64u) {
          float q = (swR[t] - lh.x) * float(SW_BINS) / lh.y;
          if (swD[t] == 0u && q >= 0.0 && q < float(SW_BINS)) { swH[min(uint(q), SW_BINS - 1u)].add(1u); }
        }
        GroupMemoryBarrierWithGroupSync();
        if (li == 0u) {
          float below = 0.0;
          uint b = 0u;
          for (uint i = 0u; i < SW_BINS; i++) {
            float h = float(swH[i].load());
            swH[i].store(0u);
            if (b == i && below + h < lh.z) { below += h; b = i + 1u; }
          }
          b = min(b, SW_BINS - 1u);
          float w = lh.y / float(SW_BINS);
          swLH = float4(lh.x + float(b) * w, w, lh.z - below, 0.0);
        }
      }
      UNIFORM_LOAD(float4 med, swLH)
      float tau = SPREADS3 * (med.x + 0.5 * med.y);
      float changed = 0.0;
      for (uint t = li; t < SW_T * SW_T; t += 64u) {
        if (swD[t] == 2u) { continue; }
        uint bad = swR[t] > tau ? 1u : 0u;
        changed += bad != swD[t] ? 1.0 : 0.0;
        swD[t] = bad;
      }
      swS[li] = changed;
      GroupMemoryBarrierWithGroupSync();
      if (li == 0u) {
        float ch = 0.0;
        for (uint i = 0u; i < 64u; i++) { ch += swS[i]; }
        swGo = ch > 0.0 ? 1u : 0u;
      }
      UNIFORM_LOAD(uint go2, swGo)
      if (go2 == 0u) { break; }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (li == 0u) {
    float b0 = A.g[9u * rgi + 0u][0] * red[0] + A.g[9u * rgi + 0u][1] * red[64] + A.g[9u * rgi + 0u][2] * red[128] + A.g[9u * rgi + 0u][3] * red[192] + A.g[9u * rgi + 1u][0] * red[256] + A.g[9u * rgi + 1u][1] * red[320];
    float b1 = A.g[9u * rgi + 1u][2] * red[0] + A.g[9u * rgi + 1u][3] * red[64] + A.g[9u * rgi + 2u][0] * red[128] + A.g[9u * rgi + 2u][1] * red[192] + A.g[9u * rgi + 2u][2] * red[256] + A.g[9u * rgi + 2u][3] * red[320];
    float b2 = A.g[9u * rgi + 3u][0] * red[0] + A.g[9u * rgi + 3u][1] * red[64] + A.g[9u * rgi + 3u][2] * red[128] + A.g[9u * rgi + 3u][3] * red[192] + A.g[9u * rgi + 4u][0] * red[256] + A.g[9u * rgi + 4u][1] * red[320];
    float b3 = A.g[9u * rgi + 4u][2] * red[0] + A.g[9u * rgi + 4u][3] * red[64] + A.g[9u * rgi + 5u][0] * red[128] + A.g[9u * rgi + 5u][1] * red[192] + A.g[9u * rgi + 5u][2] * red[256] + A.g[9u * rgi + 5u][3] * red[320];
    float b4 = A.g[9u * rgi + 6u][0] * red[0] + A.g[9u * rgi + 6u][1] * red[64] + A.g[9u * rgi + 6u][2] * red[128] + A.g[9u * rgi + 6u][3] * red[192] + A.g[9u * rgi + 7u][0] * red[256] + A.g[9u * rgi + 7u][1] * red[320];
    float b5 = A.g[9u * rgi + 7u][2] * red[0] + A.g[9u * rgi + 7u][3] * red[64] + A.g[9u * rgi + 8u][0] * red[128] + A.g[9u * rgi + 8u][1] * red[192] + A.g[9u * rgi + 8u][2] * red[256] + A.g[9u * rgi + 8u][3] * red[320];
    arec[ab] = float4(1.0, float(S), 0.0, float(found));
    arec[ab + 1u] = A.k[3u * rgi];
    arec[ab + 2u] = A.k[3u * rgi + 1u];
    arec[ab + 3u] = geo;
    arec[ab + ANCH_SURF] = float4(b0, b1, b2, b3);
    arec[ab + (ANCH_SURF + 1u)] = float4(b4, b5, 0.0, 0.0);
    arec[ab + ANCH_GAIN] = float4(swAl[0], swAl[1], swAl[2], swAl[3]);
    arec[ab + (ANCH_GAIN + 1u)] = float4(swAl[4], swAl[5], 0.0, 0.0);
  }
  if (li < ANCH_SURF - 4u) {
    uint q = 2u * li;
    float2 kq = kv[min(q + 1u, NK - 1u)];
    arec[ab + 4u + li] = float4(kv[q], q + 1u < NK ? kq : float2(0.0));
  }
}
