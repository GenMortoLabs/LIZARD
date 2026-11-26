// Pass 1 of the back half's transform, fused: a workgroup of N / 8 a (quad of four picture columns, listed frame): each
// frame sampled through its map on a node grid (a node every 8 rows, samples linear between; the map, the lens and the
// anchors' mesh in the nodes), the squares filled with the anchors' shading surface and the gain applied (ANCH_FIT's
// record), the columns' partial sums for the detrend, then the column FFT of the four columns (two complex lanes), the
// half spectrum stored to Y. Built a batch size, a picture size and a precision each. Every workgroup variable is
// written before the barrier that precedes its reads, so none is zeroed. Every listed frame's anchors record is on
// (ANCH_FIT writes it for every frame it finds), so the anchors' mesh, erase and gain are read on every one. The
// arithmetic: ../../back/DESIGN.md 1 to 3 and 10. Defines (core/dec/build.cpp backHalf): F16 (0 or 1), LANES (N / 8,
// int), PICTURE_N, BATCH, LISTS_HEAD, LISTS_ROWS, LISTS_RING, ANCH_SIDE, ANCH_ROWS, ANCH_REC, ANCH_SURF, ANCH_GAIN
// (uint), ANCH_ERASE (int), RING_COUNT, LATTICE_VECS, RESID, LUMH_BINDING; the column FFT's stages lib/fft_body.slang's,
// from the picture's size.
#if F16
#define STORE_T half2
#define LD(X) float2(X)
#define ST(X) half2(X)
#define YB_T uint4
#else
#define STORE_T float2
#define LD(X) X
#define ST(X) X
#define YB_T float4
#endif
struct ParamsBlock { uint yStride; uint partStride; uint size; float m2; float sx2; float sq2; uint sStride; };
[[vk::binding(6, 0)]] ConstantBuffer<ParamsBlock> P;
[[vk::binding(0, 0)]] Texture2DArray<float> img;
[[vk::binding(1, 0)]] StructuredBuffer<uint> lists;
[[vk::binding(2, 0)]] StructuredBuffer<float2> tw;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> part;
[[vk::binding(4, 0)]] RWStructuredBuffer<YB_T> yb;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> counts;
struct MapRec { float m[16]; };
[[vk::binding(7, 0)]] StructuredBuffer<MapRec> maps;
[[vk::binding(8, 0)]] StructuredBuffer<float4> resid;
static const uint N = PICTURE_N;
static const uint T = PICTURE_N / 8u;
static const uint LS = BATCH + 1u;
static const uint ROWS_AT = LISTS_ROWS;
groupshared STORE_T buf0[N];
groupshared STORE_T buf1[N];
groupshared float4 red[T];
groupshared float4 nodeP[T + 1u];
// the picture's node grid: per ring (g0, step, nodes a side, first lattice coordinate), then every ring's lattice
struct PictureBlock { float4 ring[RING_COUNT]; float4 lat[LATTICE_VECS]; };
[[vk::binding(9, 0)]] ConstantBuffer<PictureBlock> PIC;
#include "lib/convert.slang"
#include "lib/map.slang"
#include "lib/lumh.slang"
#include "lib/lens.slang"
#include "lib/pack.slang"
// The anchors: their record (ANCH_FIT's), the mesh's Hermite weights along an axis, the erased runs, the mesh's offsets
// at four columns.
[[vk::binding(10, 0)]] StructuredBuffer<float4> arec;
groupshared float2 anE[4u * ANCH_ROWS];
groupshared float anK[ANCH_SIDE + 2u];
groupshared int anLo[ANCH_SIDE];
groupshared int anHi[ANCH_SIDE];
groupshared uint anCol[4];
struct AnchHerm { int b; float4 w; uint i; };
AnchHerm herm(int b, float4 w, uint i) { AnchHerm h; h.b = b; h.w = w; h.i = i; return h; }
float aknot(uint ab, uint i) { return arec[ab + 1u + (i >> 2u)][i & 3u]; }
// S, the ring's grid a side (ANCH_FIT's record, arec[ab].y): S + 2 knots an axis, anK[0] to anK[S + 1]
AnchHerm anchHerm(float x, uint S) {
  if (!(x >= anK[0] && x <= anK[S + 1u])) { return herm(0, float4(0.0), 9u); }
  uint i = 0u;
  for (uint j = 1u; j < S + 1u; j++) { i += (x >= anK[j]) ? 1u : 0u; }
  float ta = anK[i];
  float tb = anK[i + 1u];
  float hc = tb - ta;
  float s = (x - ta) / hc;
  float s2 = s * s;
  float s3 = s2 * s;
  float ha = (s3 - 2.0 * s2 + s) * hc;
  float hb = (s3 - s2) * hc;
  float4 w = float4(0.0, 2.0 * s3 - 3.0 * s2 + 1.0, -2.0 * s3 + 3.0 * s2, 0.0);
  if (i > 0u) { float d = tb - anK[i - 1u]; w.x -= ha / d; w.z += ha / d; } else { w.y -= ha / hc; w.z += ha / hc; }
  if (i < S) { float d = anK[i + 2u] - ta; w.y -= hb / d; w.w += hb / d; } else { w.y -= hb / hc; w.z += hb / hc; }
  return herm(int(i) - 1, w, i);
}
bool anchIn(int x, uint S) {
  bool hit = false;
  for (uint i = 0u; i < S; i++) { hit = hit || (x >= anLo[i] && x <= anHi[i]); }
  return hit;
}
float2 aval(uint ab, int c, uint r, uint S) {
  if (c < 1 || c > int(S)) { return float2(0.0); }
  uint q = ANCH_SIDE * (r - 1u) + uint(c - 1);
  float4 v = arec[ab + 4u + (q >> 1u)];
  return ((q & 1u) == 1u) ? v.zw : v.xy;
}
float2 anchE(uint ab, float x, uint r, uint S) {
  AnchHerm h = anchHerm(x, S);
  return h.w.x * aval(ab, h.b, r, S) + h.w.y * aval(ab, h.b + 1, r, S) + h.w.z * aval(ab, h.b + 2, r, S) + h.w.w * aval(ab, h.b + 3, r, S);
}
float2 anchMesh(AnchHerm h, uint col) {
  uint o = ANCH_ROWS * col + uint(h.b + 1);
  return h.w.x * anE[o] + h.w.y * anE[o + 1u] + h.w.z * anE[o + 2u] + h.w.w * anE[o + 3u];
}
#include "lib/butterflies.slang"
#define FFT_LANES_2 1
#define FFT_BUF0 buf0
#define FFT_BUF1 buf1
#include "lib/fft.slang"
// Row y = tid + R T of the quad's four columns: read between its two nodes, the squares filled and the gain applied,
// into z[R] and the columns' sums.
#define ROW(R) \
    { \
      uint y = tid + uint(R) * T; \
      float4 nA = nodeP[y / 8u]; \
      float4 nB = nodeP[y / 8u + 1u]; \
      float fy = float(y % 8u) * 0.125; \
      float2 pL = lerp(nA.xy, nB.xy, fy); \
      float2 pR = lerp(nA.zw, nB.zw, fy); \
      float4 zz = float4(bilinH(f, pL, hhi, hinv), bilinH(f, lerp(pL, pR, 1.0 / 3.0), hhi, hinv), bilinH(f, lerp(pL, pR, 2.0 / 3.0), hhi, hinv), bilinH(f, pR, hhi, hinv)); \
      bool ay = anchIn(int(y), aS); \
      float suy = (g0 + float(y) * gstep) * sk + so; \
      float gG = gm / max(ga + suy * (gb + suy * sgb.x), gfl); \
      zz = select(bool4(aix.x && ay, aix.y && ay, aix.z && ay, aix.w && ay), fa + suy * (fb + suy * sfb.x), zz) * gG; \
      z[R] = zz; \
      float c = float(2 * int(y) + 1 - int(N)) / float(N); \
      s1 += zz; \
      sc += zz * c; \
      sq += zz * (c * c - P.m2); \
    }
// A lane's PART summed over the workgroup into SUM (eight partial sums, then their total in every lane)
#define REDUCE(PART, SUM) \
    red[tid] = PART; \
    GroupMemoryBarrierWithGroupSync(); \
    if (tid < 8u) { \
      float4 a = float4(0.0); \
      for (uint i = tid; i < T; i += 8u) { a += red[i]; } \
      red[tid] = a; \
    } \
    GroupMemoryBarrierWithGroupSync(); \
    for (uint i = 0u; i < 8u; i++) { SUM += red[i]; }
[numthreads(LANES, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint tid = lix;
  uint f = lists[P.size * LS + 1u + wg.z];
  uint rowsF = lists[ROWS_AT + f];
  uint x0 = wg.x * 4u;
  float2 v0[8] = { float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0) };
  float2 v1[8] = { float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0), float2(0.0) };
  int fw = int(lists[LISTS_HEAD + 2u * f]);
  int fh = int(lists[LISTS_HEAD + 2u * f + 1u]);
  float M[16] = maps[f].m;
  float4 pr = PIC.ring[lists[LISTS_RING + f]];
  float g0 = pr.x;
  float gstep = pr.y;
  Lens lz = lensOf(f);
  LUMH_HEAD(fw, fh)
  // the squares' fill (ANCH_FIT's surface, core/dec/plan.h ANCH_SURF): u = m sk + so of a module
  // coordinate m
  uint ab = f * ANCH_REC;
  float4 ahd = arec[ab];
  uint aS = f2u(ahd.y);
  float4 sfa = arec[ab + ANCH_SURF];
  float2 sfb = arec[ab + (ANCH_SURF + 1u)].xy;
  float4 sga = arec[ab + ANCH_GAIN];
  float2 sgb = arec[ab + (ANCH_GAIN + 1u)].xy;
  float gm = sga.x + (sga.z + sgb.x) / 3.0;
  float gfl = 0.1 * gm;
  float4 sgeo = arec[ab + 3u];
  float sk = 2.0 / sgeo.z;
  float so = -2.0 * sgeo.w / sgeo.z - 1.0;
  {
    uint xq = x0 + 0u;
    float mx0 = g0 + float(xq + 0u) * gstep;
    float mx1 = g0 + float(xq + 1u) * gstep;
    float mx2 = g0 + float(xq + 2u) * gstep;
    float mx3 = g0 + float(xq + 3u) * gstep;
    // the fill along this quad's columns: fa its terms in u alone, fb the coefficient of w (b4 w^2 apart); the gain's
    // surface (ANCH_GAIN) at the quad's middle, ga and gb alike: it moves under 0.2% across four neighbouring samples
    float4 sux = float4(mx0, mx1, mx2, mx3) * sk + so;
    float4 fa = sfa.x + sux * (sfa.y + sux * sfa.z);
    float4 fb = sfa.w + sfb.y * sux;
    float gx = 0.5 * (sux.y + sux.z);
    float ga = sga.x + gx * (sga.y + gx * sga.z);
    float gb = sga.w + sgb.y * gx;
    if (tid < ANCH_SIDE + 2u) { anK[tid] = aknot(ab, tid); }
    // the squares' samples as the C erases them (src/lizard_phy.c, SPEC 6.13): floor((m0 - margin) n / span) - E to
    // ceil((m0 + a - margin) n / span) - 1 + E on either axis, E = ANCH_ERASE; n (m0 + a - margin) stays under 2^20
    if (tid < aS) {
      float4 ag = arec[ab + 3u];
      int ia = f2i(ag.y);
      int isp = f2i(ag.z);
      int img = f2i(ag.w);
      int im0 = f2i(aknot(ab, tid + 1u)) - ia / 2;
      anLo[tid] = max(int(N) * (im0 - img) / isp - ANCH_ERASE, 0);
      anHi[tid] = min((int(N) * (im0 - img + ia) + isp - 1) / isp + (ANCH_ERASE - 1), int(N) - 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < 4u) { anCol[tid] = anchIn(int(xq + tid), aS) ? 1u : 0u; }
    for (uint ai = tid; ai < 4u * ANCH_ROWS; ai += T) {
      uint arow = ai % ANCH_ROWS;
      float2 aev = float2(0.0);
      if (arow >= 2u && arow <= aS + 1u) { aev = anchE(ab, g0 + float(xq + ai / ANCH_ROWS) * gstep, arow - 1u, aS); }
      anE[ai] = aev;
    }
    GroupMemoryBarrierWithGroupSync();
    bool4 aix = bool4(anCol[0] != 0u, anCol[1] != 0u, anCol[2] != 0u, anCol[3] != 0u);
    for (uint j = tid; j <= T; j += T) {
      float my = g0 + float(8u * j) * gstep;
      AnchHerm ah = anchHerm(my, aS);
      float2 m0 = anchMesh(ah, 0u);
      float2 m3 = anchMesh(ah, 3u);
      float2 p0 = toImage(M, float2(mx0, my));
      float2 p3 = toImage(M, float2(mx3, my));
      nodeP[j] = float4(p0 + lensAt(lz, p0) + m0, p3 + lensAt(lz, p3) + m3);
    }
    GroupMemoryBarrierWithGroupSync();
    float4 z[8] = { float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0) };
    float4 s1 = float4(0.0);
    float4 sc = float4(0.0);
    float4 sq = float4(0.0);
    ROW(0) ROW(1) ROW(2) ROW(3) ROW(4) ROW(5) ROW(6) ROW(7)
    float4 S1 = float4(0.0);
    float4 SC = float4(0.0);
    float4 SQ = float4(0.0);
    REDUCE(s1, S1)
    GroupMemoryBarrierWithGroupSync();
    REDUCE(sc, SC)
    GroupMemoryBarrierWithGroupSync();
    REDUCE(sq, SQ)
    float4 mean = S1 / float(N);
    if (tid == 0u) {
      part[f * P.partStride + xq + 0u] = float4(S1[0], SC[0], SQ[0], mean[0]);
      part[f * P.partStride + xq + 1u] = float4(S1[1], SC[1], SQ[1], mean[1]);
      part[f * P.partStride + xq + 2u] = float4(S1[2], SC[2], SQ[2], mean[2]);
      part[f * P.partStride + xq + 3u] = float4(S1[3], SC[3], SQ[3], mean[3]);
    }
    v0[0] = z[0].xy - mean.xy;
    v1[0] = z[0].zw - mean.zw;
    v0[1] = z[1].xy - mean.xy;
    v1[1] = z[1].zw - mean.zw;
    v0[2] = z[2].xy - mean.xy;
    v1[2] = z[2].zw - mean.zw;
    v0[3] = z[3].xy - mean.xy;
    v1[3] = z[3].zw - mean.zw;
    v0[4] = z[4].xy - mean.xy;
    v1[4] = z[4].zw - mean.zw;
    v0[5] = z[5].xy - mean.xy;
    v1[5] = z[5].zw - mean.zw;
    v0[6] = z[6].xy - mean.xy;
    v1[6] = z[6].zw - mean.zw;
    v0[7] = z[7].xy - mean.xy;
    v1[7] = z[7].zw - mean.zw;
#define FFT_V0 v0
#define FFT_V1 v1
#include "lib/fft_body.slang"
    // the half spectrum: each complex lane two real columns, split by symmetry, stored to Y four columns a row
    for (uint vv = 1u + tid; vv < rowsF; vv += T) {
      float2 z0 = LD(buf0[vv]);
      float2 m0 = LD(buf0[N - vv]);
      float2 z1 = LD(buf1[vv]);
      float2 m1 = LD(buf1[N - vv]);
      float2 a0 = float2(z0.x + m0.x, z0.y - m0.y) * (0.5 / float(N));
      float2 b0 = float2(z0.y + m0.y, m0.x - z0.x) * (0.5 / float(N));
      float2 a1 = float2(z1.x + m1.x, z1.y - m1.y) * (0.5 / float(N));
      float2 b1 = float2(z1.y + m1.y, m1.x - z1.x) * (0.5 / float(N));
      uint e = f * P.yStride + (vv - 1u) * N + xq;
#if F16
      yb[e >> 2u] = uint4(packHalf2x16(a0), packHalf2x16(b0), packHalf2x16(a1), packHalf2x16(b1));
#else
      yb[e >> 1u] = float4(a0, b0);
      yb[(e >> 1u) + 1u] = float4(a1, b1);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (tid == 0u) { InterlockedAdd(counts[f * 8u + 0u], 4u); }
}
