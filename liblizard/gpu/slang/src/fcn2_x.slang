// F2's X pass (stage x): each pixel of a pyramid level standardised over its own BOX x BOX box (the window's mean c
// taken off first) into the plane, XB0 x XB1 pixels a workgroup: the window read into poolB (from the level, or with
// TEX from the frame texture: level 0, the capture itself), the box row sums into poolA, the column pass into
// registers (X_RR rows a lane), then each X into the plane (in f32, as halves, or as int8 codes, Xq). No workgroup
// variable is zeroed: each is written before a barrier that precedes its reads. Bindings: 0 the level (or the frame
// texture), 1 frames, 2 the plane, 3 the Bank uniform (level; stride and rows, the level's; slots, slotBase, tilesX the
// net's). Defines (the host's plan, core/dec/build.cpp Fcn2::xRecipe): TEX, INT8, HALVES (0 or 1: the plane's X as
// int8 codes, as pairs of halves, else f32), PLANE_T (the plane's element), POOL_A, POOL_B (the pools' uvec4s), BOX
// (the box's side, odd), BANK_HALO, XB0, XB1, X_RR (int), BANK_KEEP, BANK_LOCAL, PLANE_XE (uint), BANK_KAPPA,
// BANK_NOMINAL, BANK_EPS, BANK_FLOOR, Q8_MAX, Q8_INV (float).
#include "lib/nets.slang"
#include "lib/round.slang"
#include "lib/frame.slang"
#if TEX
[[vk::binding(0, 0)]] Texture2DArray<float> img;
#else
[[vk::binding(0, 0)]] StructuredBuffer<float> lev;
#endif
[[vk::binding(1, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(2, 0)]] RWStructuredBuffer<PLANE_T> xpl;
struct BankBlock { uint level; uint stride; uint rows; uint slots; uint slotBase; uint tilesX; };
[[vk::binding(3, 0)]] ConstantBuffer<BankBlock> P;
static const int HALO = BANK_HALO;
static const uint KEEP = BANK_KEEP;
static const uint LOCAL = BANK_LOCAL;
static const float KAPPA = BANK_KAPPA;
static const float NOMINAL = BANK_NOMINAL;
static const float EPS = BANK_EPS;
static const float FLOOR = BANK_FLOOR;
#if INT8
static const float QMAX = Q8_MAX;
static const float INV = Q8_INV;
#endif
groupshared uint4 poolA[POOL_A];
groupshared uint4 poolB[POOL_B];
static const uint XE = PLANE_XE;
static const uint KBOX = uint(BOX);
// the box's reach, the window's elements a row (VE: XE and the reach's NE), its rows, the column pass's items
static const int R = (BOX - 1) / 2;
static const int NE = (BOX + 6) / 4;
static const uint VE = XE - 1u + uint(NE);
static const int VH = XB1 + 2 * R;
static const int NV = VH * int(VE);
static const int WG = 256;
static const int NLD = (NV + WG - 1) / WG;
static const uint HOFF = uint(VH) * XE;
static const int NCI = int(XE) * ((XB1 + X_RR - 1) / X_RR);
static const int NCL = (NCI + WG - 1) / WG;
groupshared float cst[16];

// The plane: frame f's row y at (f rows + y) SP; SP even so a pair of halves never straddles two rows, int8 a multiple
// of four so a word of four Xq never does.
#if INT8
uint planeRow(uint f, int y) { return (f * P.rows + uint(y)) * ((P.stride + 3u) & ~3u); }
// Xq of X columns xo0 + 4g .. + 3 (xo0 a multiple of 4: one word) of block row j into the plane, on the level; past its
// right edge the word repeats the edge's X.
void planePut(uint j, uint g, float4 xv, int xo0, int yo0, int w, int h, uint f) {
  int x = xo0 + int(g) * 4;
  int y = yo0 + int(j);
  if (x >= w || y >= h) { return; }
  xpl[(planeRow(f, y) + uint(x)) >> 2u] = pack4xI8(int4(clamp(roundEven(xv * INV), float4(-QMAX), float4(QMAX))));
}
#else
uint planeRow(uint f, int y) { return (f * P.rows + uint(y)) * ((P.stride + 1u) & ~1u); }
// X columns xo0 + 4g .. + 3 (xo0 a multiple of 4, so two whole pairs) of block row j into the plane, on the level.
void planePut(uint j, uint g, float4 xv, int xo0, int yo0, int w, int h, uint f) {
  int x = xo0 + int(g) * 4;
  int y = yo0 + int(j);
  if (x >= w || y >= h) { return; }
  uint i = planeRow(f, y) + uint(x);
#if HALVES
  xpl[i >> 1u] = packHalf2x16(xv.xy);
  if (x + 2 < w) { xpl[(i >> 1u) + 1u] = packHalf2x16(xv.zw); }
#else
  for (int k = 0; k < 4; k++) { if (x + k < w) { xpl[i + uint(k)] = xv[k]; } }
#endif
}
#endif

float vGet(uint j, uint i) { return asfloat(poolB[j * VE + (i >> 2u)][i & 3u]); }
// The row sums of box row r for X columns 4g .. 4g + 3: one element each where no column is clamped, else a column at
// a time, each column's box being the one centred on its pixel clamped into the level (the spec's step 4).
struct HS { float4 a; float4 b; };
HS hrow(uint r, uint g, uint4 cx, bool inner) {
  if (inner) { return HS(asfloat(poolA[r * XE + g]), asfloat(poolA[HOFF + r * XE + g])); }
  HS o = HS(float4(0.0), float4(0.0));
  for (uint k = 0u; k < 4u; k++) {
    uint e = r * XE + (cx[k] >> 2u);
    o.a[k] = asfloat(poolA[e][cx[k] & 3u]);
    o.b[k] = asfloat(poolA[HOFF + e][cx[k] & 3u]);
  }
  return o;
}
// X from a box's sums of v - c and (v - c)^2 and its centre's window row vy.
float4 xval(float4 s1, float4 s2, uint vy, uint4 cx) {
  float4 mu = s1 / float(BOX * BOX);
  float4 sd = sqrt(max(s2 / float(BOX * BOX) - mu * mu, float4(0.0)));
  float4 vc = float4(vGet(vy, cx.x + uint(R)), vGet(vy, cx.y + uint(R)), vGet(vy, cx.z + uint(R)), vGet(vy, cx.w + uint(R)));
  return (vc - mu) / (sd + float4(EPS));
}

[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.z;
  Frame F = frames[f];
  int w = int((F.w + (1u << P.level) - 1u) >> P.level);
  int h = int((F.h + (1u << P.level) - 1u) >> P.level);
  int xo0 = int(wg.x) * XB0;
  int yo0 = int(wg.y) * XB1;
  if (F.valid != 1u || xo0 >= w || yo0 >= h) { return; }
  // the window, four pixels an element, its sum for c
  float s0 = 0.0;
  float4 v[NLD];
#if TEX
#define X_LOAD(r) { \
      int yy = clamp(yo0 - R + int(q / VE), 0, h - 1); \
      int4 xs = clamp(int4(xo0 - R + 4 * int(q % VE)) + int4(0, 1, 2, 3), int4(0), int4(w - 1)); \
      v[r] = float4(img.Load(int4(xs.x, yy, int(f), 0)), img.Load(int4(xs.y, yy, int(f), 0)), img.Load(int4(xs.z, yy, int(f), 0)), img.Load(int4(xs.w, yy, int(f), 0))); \
      s0 += (v[r].x + v[r].y) + (v[r].z + v[r].w); }
#else
#define X_LOAD(r) { \
      uint row = (f * P.rows + uint(clamp(yo0 - R + int(q / VE), 0, h - 1))) * P.stride; \
      uint4 xs = uint4(clamp(int4(xo0 - R + 4 * int(q % VE)) + int4(0, 1, 2, 3), int4(0), int4(w - 1))); \
      v[r] = float4(lev[row + xs.x], lev[row + xs.y], lev[row + xs.z], lev[row + xs.w]); \
      s0 += (v[r].x + v[r].y) + (v[r].z + v[r].w); }
#endif
  [ForceUnroll] for (int r = 0; r < NLD; r++) {
    v[r] = float4(0.0);
    uint q = li + uint(r * WG);
    // the last load's lanes past the window guarded
    if ((r + 1) * WG <= NV) X_LOAD(r) else if (q < uint(NV)) X_LOAD(r)
  }
#undef X_LOAD
  // c, the window's mean: partial sums in two rounds of sixteen, every lane adding the last sixteen in one order.
  poolA[li >> 2u][li & 3u] = asuint(s0);
  GroupMemoryBarrierWithGroupSync();
  if (li < 16u) {
    float t = 0.0;
    for (uint k = 0u; k < 16u; k++) { t += asfloat(poolA[(li * 16u + k) >> 2u][(li * 16u + k) & 3u]); }
    cst[li] = t;
  }
  // cst to every lane: a barrier, the load, a barrier
  GroupMemoryBarrierWithGroupSync();
  float gs[16] = cst;
  GroupMemoryBarrierWithGroupSync();
  float tc = 0.0;
  for (uint k = 0u; k < 16u; k++) { tc += gs[k]; }
  float c = tc / float(4 * NV);
  [ForceUnroll] for (int r = 0; r < NLD; r++) {
    uint q = li + uint(r * WG);
    if ((r + 1) * WG <= NV) { poolB[q] = asuint(v[r] - float4(c)); } else if (q < uint(NV)) { poolB[q] = asuint(v[r] - float4(c)); }
  }
  GroupMemoryBarrierWithGroupSync();
  // the row pass: each window row's box sums of v - c and its square for four X columns an item
  for (uint it = li; it < HOFF; it += 256u) {
    uint j = it / XE;
    uint g = it % XE;
    float4 b[NE];
    [ForceUnroll] for (int n = 0; n < NE; n++) { b[n] = asfloat(poolB[j * VE + g + uint(n)]); }
    float a[BOX + 3], q[BOX + 3];
    [ForceUnroll] for (int n = 0; n < BOX + 3; n++) { a[n] = b[n >> 2][n & 3]; q[n] = a[n] * a[n]; }
    float s1[4], s2[4];
    s1[0] = a[0];
    [ForceUnroll] for (int n = 1; n < BOX; n++) { s1[0] = s1[0] + a[n]; }
    s2[0] = q[0];
    [ForceUnroll] for (int n = 1; n < BOX; n++) { s2[0] = s2[0] + q[n]; }
    [ForceUnroll] for (int k = 1; k < 4; k++) { s1[k] = s1[k - 1] + a[BOX + k - 1] - a[k - 1]; s2[k] = s2[k - 1] + q[BOX + k - 1] - q[k - 1]; }
    poolA[j * XE + g] = asuint(float4(s1[0], s1[1], s1[2], s1[3]));
    poolA[HOFF + j * XE + g] = asuint(float4(s2[0], s2[1], s2[2], s2[3]));
  }
  GroupMemoryBarrierWithGroupSync();
  // the column pass: an item X_RR rows of four X columns, the box's sums slid down the rows
  float4 xs[NCL][X_RR];
  [ForceUnroll] for (int ic = 0; ic < NCL; ic++) {
    [ForceUnroll] for (int k = 0; k < X_RR; k++) { xs[ic][k] = float4(0.0); }
    uint ci = li + uint(ic * WG);
    if (ci < uint(NCI)) {
      uint g = ci % XE;
      uint j0 = (ci / XE) * uint(X_RR);
      int x0 = xo0 + int(g) * 4;
      uint4 cx = uint4(clamp(int4(x0) + int4(0, 1, 2, 3), int4(0), int4(w - 1)) - int4(xo0));
      bool inner = x0 >= 0 && x0 + 3 < w;
      float4 s1 = float4(0.0);
      float4 s2 = float4(0.0);
      uint r0 = uint(clamp(yo0 + int(j0), 0, h - 1) - yo0);
      for (uint r = 0u; r < KBOX; r++) { HS hh = hrow(r0 + r, g, cx, inner); s1 += hh.a; s2 += hh.b; }
      xs[ic][0] = xval(s1, s2, r0 + uint(R), cx);
      [ForceUnroll] for (int k = 1; k < X_RR; k++) {
        if (j0 + uint(k) < uint(XB1)) {
          uint rn = uint(clamp(yo0 + int(j0 + uint(k)), 0, h - 1) - yo0);
          if (rn != r0) { HS hi = hrow(r0 + KBOX, g, cx, inner); HS lo = hrow(r0, g, cx, inner); s1 += hi.a - lo.a; s2 += hi.b - lo.b; r0 = rn; }
          xs[ic][k] = xval(s1, s2, r0 + uint(R), cx);
        }
      }
    }
  }
  [ForceUnroll] for (int ic = 0; ic < NCL; ic++) {
    uint ci = li + uint(ic * WG);
    uint g = ci % XE;
    uint j0 = (ci / XE) * uint(X_RR);
    [ForceUnroll] for (int k = 0; k < X_RR; k++) {
      if (ci < uint(NCI) && j0 + uint(k) < uint(XB1)) { planePut(j0 + uint(k), g, xs[ic][k], xo0, yo0, w, h, f); }
    }
  }
}
