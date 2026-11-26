// F2's network (stage net): the mark score and size a pixel from the plane, a peak where a pixel beats its eight
// neighbours and the file's floor (placed by a parabola each axis), each tile's KEEP strongest into the raw slots, by
// rank. A workgroup walks WALK regions of RW x RH pixels along a row, TX x TY tiles each: the region's X (or Xq) read
// into poolB, the layers (the weights constants of the module: the file's values as the recipe states them, f32, f16,
// f32 with halves stored, or int8 codes), the map in poolB, the peaks into poolA's tile lists (nfound), then the tiles'
// slots. The next region's X is fetched into registers before the current one's convolutions, so the loads' latency
// hides behind them. Its small fixed loops unrolled, its pixel loops and the walk [loop]. No workgroup variable is
// zeroed: each is written before a barrier that precedes its reads (nfound by its store at a region's start).
// Bindings: 0 the plane, 1 frames, 2 raw, 3 counts, 4 the Bank uniform (level; stride and rows, the level's; slots: a
// frame's slots over every level; slotBase: where this level's tiles start in them; tilesX: tiles a row at this level).
// Defines (the host's plan, core/dec/build.cpp Fcn2::netRecipe): F16, INT8, HALVES (0 or 1: f16 or f32 with halves
// stored keep a pixel's channels as halves, eight an element), PLANE_T, POOL_T (the pools' elements), POOL_A, POOL_B,
// BANK_HALO, NET_RW, NET_RH, NET_TX, NET_TY, NET_XH (the window's rows), NET_G1 (conv1's four-pixel groups a row),
// NET_MW (the map's row), NET_CONVS (2 to 4) (int), BANK_KEEP, BANK_LOCAL, NET_WALK, NET_XE, NET_RS, NET_NT (uint),
// BANK_KAPPA, BANK_NOMINAL, BANK_EPS, BANK_FLOOR, Q8_MAX, Q8_INV (float); a conv's geometry NET_C<k> (dilation, in
// channels, out channels, in width, out width, out height) and its values in the file's order: NET_W<k> (weights, out x
// in x 3 x 3), NET_B<k> (biases); int8: codes and NET_M<k>, its requantisation's multipliers; the head's NET_HW
// (2 x channels) and NET_HB (2).
#include "lib/nets.slang"
#include "lib/round.slang"
#include "lib/frame.slang"
[[vk::binding(0, 0)]] StructuredBuffer<PLANE_T> xpl;
[[vk::binding(1, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> raw;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> counts;
struct BankBlock { uint level; uint stride; uint rows; uint slots; uint slotBase; uint tilesX; };
[[vk::binding(4, 0)]] ConstantBuffer<BankBlock> P;
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
groupshared POOL_T poolA[POOL_A];
groupshared POOL_T poolB[POOL_B];
static const int RW = NET_RW;
static const uint WALK = NET_WALK;
static const int RH = NET_RH;
static const uint XE = NET_XE;
static const uint RS = NET_RS;
groupshared Atomic<uint> nfound[NET_NT];
static const int WG = 256;
static const int TILE = 16;
// the frame's counter slot for peaks past a tile's LOCAL (decoder's counters 0 to 3 named)
static const uint OVERFLOW = 4u;
// the window's X items (four pixels each) and a lane's share of them
static const int NXI = NET_XH * int(XE);
static const int NXL = (NXI + WG - 1) / WG;

#if INT8
static const int W1[] = { NET_W1 };
static const int B1[] = { NET_B1 };
static const float M1[] = { NET_M1 };
static const int W2[] = { NET_W2 };
static const int B2[] = { NET_B2 };
static const float M2[] = { NET_M2 };
#if NET_CONVS > 2
static const int W3[] = { NET_W3 };
static const int B3[] = { NET_B3 };
static const float M3[] = { NET_M3 };
#endif
#if NET_CONVS > 3
static const int W4[] = { NET_W4 };
static const int B4[] = { NET_B4 };
static const float M4[] = { NET_M4 };
#endif
static const float HW[] = { NET_HW };
static const float HB[] = { NET_HB };
// a conv's code, bias and multiplier by its index (K 2 to 4)
int wq<let K : int>(int i) { if (K == 2) { return W2[i]; }
#if NET_CONVS > 2
  if (K == 3) { return W3[i]; }
#endif
#if NET_CONVS > 3
  if (K == 4) { return W4[i]; }
#endif
  return 0; }
int bq<let K : int>(int i) { if (K == 2) { return B2[i]; }
#if NET_CONVS > 2
  if (K == 3) { return B3[i]; }
#endif
#if NET_CONVS > 3
  if (K == 4) { return B4[i]; }
#endif
  return 0; }
float mq<let K : int>(int i) { if (K == 2) { return M2[i]; }
#if NET_CONVS > 2
  if (K == 3) { return M3[i]; }
#endif
#if NET_CONVS > 3
  if (K == 4) { return M4[i]; }
#endif
  return 0.0; }
// four codes as a weight word (the low byte the first), as the int OpSDot takes
int word4(int a, int b, int c, int d) { return int(pack4xI8(int4(a, b, c, d))); }
#else
#if F16
typealias T = half;
typealias V4 = half4;
#else
typealias T = float;
typealias V4 = float4;
#endif
static const T W1[] = { NET_W1 };
static const T B1[] = { NET_B1 };
static const T W2[] = { NET_W2 };
static const T B2[] = { NET_B2 };
#if NET_CONVS > 2
static const T W3[] = { NET_W3 };
static const T B3[] = { NET_B3 };
#endif
#if NET_CONVS > 3
static const T W4[] = { NET_W4 };
static const T B4[] = { NET_B4 };
#endif
static const T HW[] = { NET_HW };
static const float HB[] = { NET_HB };
// a conv's weight and bias by its index (K 2 to 4)
T wf<let K : int>(int i) { if (K == 2) { return W2[i]; }
#if NET_CONVS > 2
  if (K == 3) { return W3[i]; }
#endif
#if NET_CONVS > 3
  if (K == 4) { return W4[i]; }
#endif
  return T(0.0); }
T bf<let K : int>(int i) { if (K == 2) { return B2[i]; }
#if NET_CONVS > 2
  if (K == 3) { return B3[i]; }
#endif
#if NET_CONVS > 3
  if (K == 4) { return B4[i]; }
#endif
  return T(0.0); }
#endif

// The plane: frame f's row y at (f rows + y) SP, SP even so a pair of halves never straddles two rows; int8: SP a
// multiple of four, so a word of four Xq never straddles two rows.
#if INT8
uint planeRow(uint f, int y) { return (f * P.rows + uint(y)) * ((P.stride + 3u) & ~3u); }
// the Xq byte of level pixel (x, y), as its bits
uint planeByte(uint f, int x, int y) { uint i = planeRow(f, y) + uint(x); return (xpl[i >> 2u] >> (8u * (i & 3u))) & 0xffu; }
#else
uint planeRow(uint f, int y) { return (f * P.rows + uint(y)) * ((P.stride + 1u) & ~1u); }
#if HALVES
float planeAt(uint f, int x, int y) { uint i = planeRow(f, y) + uint(x); return unpackHalf2x16(xpl[i >> 1u])[i & 1u]; }
#else
float planeAt(uint f, int x, int y) { return xpl[planeRow(f, y) + uint(x)]; }
#endif
#endif

#if INT8
// The X window's Xq words, four pixels a u32, two an element of B.
void xw(uint i, uint v) { poolB[i >> 1u][i & 1u] = v; }
uint xr(uint i) { return poolB[i >> 1u][i & 1u]; }
// A peak (x, y, sigma, response) is two elements of A.
void peakPut(uint i, float4 v) { poolA[2u * i] = asuint(v.xy); poolA[2u * i + 1u] = asuint(v.zw); }
float4 peakGet(uint i) { return float4(asfloat(poolA[2u * i]), asfloat(poolA[2u * i + 1u])); }
float peakW(uint i) { return asfloat(poolA[2u * i + 1u].y); }
// The contract's requantisation (scripts/gpu/cnn/q8.py) of four channels' accumulators, the bias in: one f32 multiply,
// rounded half to even, clamped to the codes 0 .. QMAX.
int4 rq(int4 a, float4 m) { return int4(clamp(roundEven(float4(a) * m), float4(0.0), float4(QMAX))); }
// The map: one 8-byte element a pixel.
void mapPut(uint i, float z, float s) { poolB[i] = uint2(asuint(z), asuint(s)); }
float mapP(uint i) { return asfloat(poolB[i].x); }
float mapS(uint i) { return asfloat(poolB[i].y); }
// An item is one word of four Xq. The window starts HALO before a region (a multiple of four), so where HALO is not
// one either a word inside the level is cut from two plane words.
static const uint CUT = 8u * uint(((-HALO % 4) + 4) % 4);
uint xfetch(uint it, int xo0, int yo0, int w, int h, uint f) {
  uint jr = it / XE;
  uint g = it % XE;
  int x0 = xo0 + int(g) * 4;
  int y = clamp(yo0 + int(jr), 0, h - 1);
  if (x0 >= 0 && x0 + 3 < w) {
    uint i = (planeRow(f, y) + uint(x0)) >> 2u;
    if (CUT == 0u) { return xpl[i]; }
    return (xpl[i] >> CUT) | (xpl[i + 1u] << (32u - CUT));
  }
  uint v = 0u;
  [unroll] for (int k = 0; k < 4; k++) { v |= planeByte(f, clamp(x0 + k, 0, w - 1), y) << (8u * uint(k)); }
  return v;
}
// X columns xo0 + 4g .. + 3 of window row jr, as Xq, into the X pool (RS words a row, the last ones zero).
void xPut(uint jr, uint g, uint xv, int xo0, int yo0, int w, int h, uint f) {
  xw(jr * RS + g, xv);
  if (g == XE - 1u) { [unroll] for (uint e = XE; e < RS; e++) { xw(jr * RS + e, 0u); } }
}
#define PEAK_PUT(I, V) peakPut(I, V)
#define PEAK_GET(I) peakGet(I)
#define PEAK_W(I) peakW(I)
typealias XP = uint;
#else
#if F16
// halves to and from an f16 vector
half4 u2h(uint2 w) { return half4(unpackFloat2x16(w.x), unpackFloat2x16(w.y)); }
uint2 h2u(half4 h) { return uint2(packFloat2x16(h.xy), packFloat2x16(h.zw)); }
#define HALF2(A) h2u(A)
#define GRP(E) u2h(E)
#elif HALVES
// f32 through halves
uint2 pk(float4 v) { return uint2(packHalf2x16(v.xy), packHalf2x16(v.zw)); }
float4 upk(uint2 p) { return float4(unpackHalf2x16(p.x), unpackHalf2x16(p.y)); }
#define HALF2(A) pk(A)
#define GRP(E) upk(E)
#endif
void mapPut(uint i, float z, float s) { poolB[i >> 1u][(i & 1u) * 2u] = asuint(z); poolB[i >> 1u][(i & 1u) * 2u + 1u] = asuint(s); }
float mapP(uint i) { return asfloat(poolB[i >> 1u][(i & 1u) * 2u]); }
float mapS(uint i) { return asfloat(poolB[i >> 1u][(i & 1u) * 2u + 1u]); }
// The X window from the plane, four pixels an item, clamped to the level; a group of four wholly in the level is two
// pairs (the window's origin rx0 - HALO is even), else each pixel is read at its clamped place.
float4 xfetch(uint it, int xo0, int yo0, int w, int h, uint f) {
  uint jr = it / XE;
  uint g = it % XE;
  int x0 = xo0 + int(g) * 4;
  int y = clamp(yo0 + int(jr), 0, h - 1);
  float4 xv = float4(0.0);
  if (x0 >= 0 && x0 + 3 < w) {
#if HALVES
    uint i = (planeRow(f, y) + uint(x0)) >> 1u;
    xv = float4(unpackHalf2x16(xpl[i]), unpackHalf2x16(xpl[i + 1u]));
#else
    uint i = planeRow(f, y) + uint(x0);
    xv = float4(xpl[i], xpl[i + 1u], xpl[i + 2u], xpl[i + 3u]);
#endif
  } else {
    [unroll] for (int k = 0; k < 4; k++) { xv[k] = planeAt(f, clamp(x0 + k, 0, w - 1), y); }
  }
  return xv;
}
// X columns xo0 + 4g .. + 3 of window row jr into the X pool (RS elements a row, the last ones zero).
void xPut(uint jr, uint g, float4 xv, int xo0, int yo0, int w, int h, uint f) {
  poolB[jr * RS + g] = asuint(xv);
  if (g == XE - 1u) { [unroll] for (uint e = XE; e < RS; e++) { poolB[jr * RS + e] = uint4(0u); } }
}
#define PEAK_PUT(I, V) poolA[I] = asuint(V)
#define PEAK_GET(I) asfloat(poolA[I])
#define PEAK_W(I) asfloat(poolA[I].w)
typealias XP = float4;
#endif

// A peak of the probability map at map index cc from its 3 x 3 neighbourhood nb (row major): above the floor and its
// eight neighbours, ties to the earlier one. Placed by a parabola each axis, into tile t's list.
void peakAt(float nb[9], uint cc, uint t, int lx, int ly, int w, int h) {
  float v = nb[4];
  bool peak = v > FLOOR && lx < w && ly < h;
  [unroll] for (uint k = 0u; k < 9u; k++) {
    if (k != 4u) { peak = peak && (k < 4u ? v >= nb[k] : v > nb[k]); }
  }
  if (peak) {
    float ex = nb[5] + nb[3] - 2.0 * v;
    float ey = nb[7] + nb[1] - 2.0 * v;
    float ox = ex < 0.0 ? clamp(0.5 * (nb[3] - nb[5]) / ex, -0.5, 0.5) : 0.0;
    float oy = ey < 0.0 ? clamp(0.5 * (nb[1] - nb[7]) / ey, -0.5, 0.5) : 0.0;
    float kk = float(1u << P.level);
    float uu = kk * NOMINAL * exp(mapS(cc));
    uint at = nfound[t].add(1u);
    if (at < LOCAL) { PEAK_PUT(t * LOCAL + at, float4((float(lx) + ox + 0.5) * kk, (float(ly) + oy + 0.5) * kk, KAPPA * uu, v)); }
  }
}

#if INT8
// conv1 (one input channel, dilation 1): four output pixels a lane. A tap row's word t[j] is X pixels x0 + j .. x0 + j +
// 3, cut from the row's words gk and gk + 1; its weight word holds the row's three taps in bytes 0 to 2 and 0 in byte 3.
void conv1<let DIL : int, let IN_C : int, let OUT_C : int, let IN_W : int, let OUT_W : int, let OUT_H : int>(uint li) {
  [loop] for (uint i = li; i < uint(NET_G1 * OUT_H); i += uint(WG)) {
    uint gk = i % uint(NET_G1);
    uint y = i / uint(NET_G1);
    uint x0 = gk * 4u;
    int a[4][8];
    [ForceUnroll] for (int j = 0; j < 4; j++) { [ForceUnroll] for (int o = 0; o < OUT_C; o++) { a[j][o] = B1[o]; } }
    [ForceUnroll] for (int ky = 0; ky < 3; ky++) {
      uint w0 = xr((y + uint(ky)) * RS + gk);
      uint w1 = xr((y + uint(ky)) * RS + gk + 1u);
      uint t[4];
      t[0] = w0;
      t[1] = (w0 >> 8u) | (w1 << 24u);
      t[2] = (w0 >> 16u) | (w1 << 16u);
      t[3] = (w0 >> 24u) | (w1 << 8u);
      [ForceUnroll] for (int j = 0; j < 4; j++) {
        [ForceUnroll] for (int o = 0; o < OUT_C; o++) {
          const int W = word4(W1[o * 9 + ky * 3], W1[o * 9 + ky * 3 + 1], W1[o * 9 + ky * 3 + 2], 0);
          if (W != 0) { a[j][o] += dotPacked4x8(int(t[j]), W); }
        }
      }
    }
    [ForceUnroll] for (int j = 0; j < 4; j++) {
      if (x0 + uint(j) < uint(OUT_W)) {
        int4 q0 = rq(int4(a[j][0], a[j][1], a[j][2], a[j][3]), float4(M1[0], M1[1], M1[2], M1[3]));
        int4 q1 = rq(int4(a[j][4], a[j][5], a[j][6], a[j][7]), float4(M1[4], M1[5], M1[6], M1[7]));
        poolA[y * uint(OUT_W) + x0 + uint(j)] = uint2(pack4xI8(q0), pack4xI8(q1));
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
}
// a later conv's taps: a pixel a lane, one load a tap, each output channel two dot products (input channels 0 to 3,
// then 4 to 7), each added to the channel's accumulator in a statement of its own; a tap with no word left out
#define CONV_TAPS \
  [ForceUnroll] for (int ky = 0; ky < 3; ky++) { [ForceUnroll] for (int kx = 0; kx < 3; kx++) { \
    bool any = false; \
    [ForceUnroll] for (int o = 0; o < OUT_C; o++) { [ForceUnroll] for (int g = 0; g < 2; g++) { \
      any = any || word4(wq<K>(((o * IN_C + 4 * g) * 3 + ky) * 3 + kx), wq<K>(((o * IN_C + 4 * g + 1) * 3 + ky) * 3 + kx), wq<K>(((o * IN_C + 4 * g + 2) * 3 + ky) * 3 + kx), wq<K>(((o * IN_C + 4 * g + 3) * 3 + ky) * 3 + kx)) != 0; } } \
    if (any) { \
      uint2 e = poolA[(y + uint(ky * DIL)) * uint(IN_W) + x + uint(kx * DIL)]; \
      [ForceUnroll] for (int o = 0; o < OUT_C; o++) { [ForceUnroll] for (int g = 0; g < 2; g++) { \
        const int W = word4(wq<K>(((o * IN_C + 4 * g) * 3 + ky) * 3 + kx), wq<K>(((o * IN_C + 4 * g + 1) * 3 + ky) * 3 + kx), wq<K>(((o * IN_C + 4 * g + 2) * 3 + ky) * 3 + kx), wq<K>(((o * IN_C + 4 * g + 3) * 3 + ky) * 3 + kx)); \
        if (W != 0) { a[o] += dotPacked4x8(int(g == 0 ? e.x : e.y), W); } } } \
    } } }
// a middle conv: requantised into registers, written over the input after a barrier (staged)
void convMid<let DIL : int, let IN_C : int, let OUT_C : int, let IN_W : int, let OUT_W : int, let OUT_H : int, let K : int>(uint li) {
  const int N = OUT_W * OUT_H, NR = (N + WG - 1) / WG;
  uint2 o[NR];
  [ForceUnroll] for (int r = 0; r < NR; r++) { o[r] = uint2(0u); }
  [loop] for (uint r = 0u; r < uint(NR); r++) {
    uint i = li + r * uint(WG);
    if (i < uint(N)) {
      uint x = i % uint(OUT_W);
      uint y = i / uint(OUT_W);
      int a[8];
      [ForceUnroll] for (int c = 0; c < OUT_C; c++) { a[c] = bq<K>(c); }
      CONV_TAPS
      int4 q0 = rq(int4(a[0], a[1], a[2], a[3]), float4(mq<K>(0), mq<K>(1), mq<K>(2), mq<K>(3)));
      int4 q1 = rq(int4(a[4], a[5], a[6], a[7]), float4(mq<K>(4), mq<K>(5), mq<K>(6), mq<K>(7)));
      uint2 st = uint2(pack4xI8(q0), pack4xI8(q1));
      [ForceUnroll] for (int rq = 0; rq < NR; rq++) { if (r == uint(rq)) { o[rq] = st; } }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  [ForceUnroll] for (int r = 0; r < NR; r++) { if (li + uint(r * WG) < uint(N)) { poolA[li + uint(r * WG)] = o[r]; } }
  GroupMemoryBarrierWithGroupSync();
}
// the last conv and the head, fused: the head from its bias, channel 0 first, one fma a channel (q8.py head64's
// order, in f32)
void convLast<let DIL : int, let IN_C : int, let OUT_C : int, let IN_W : int, let OUT_W : int, let OUT_H : int, let K : int>(uint li) {
  [loop] for (uint i = li; i < uint(OUT_W * OUT_H); i += uint(WG)) {
    uint x = i % uint(OUT_W);
    uint y = i / uint(OUT_W);
    int a[8];
    [ForceUnroll] for (int c = 0; c < OUT_C; c++) { a[c] = bq<K>(c); }
    CONV_TAPS
    float4 h[2];
    h[0] = max(float4(int4(a[0], a[1], a[2], a[3])) * float4(mq<K>(0), mq<K>(1), mq<K>(2), mq<K>(3)), float4(0.0));
    h[1] = max(float4(int4(a[4], a[5], a[6], a[7])) * float4(mq<K>(4), mq<K>(5), mq<K>(6), mq<K>(7)), float4(0.0));
    // (one expression a row, the last channel's term outermost: eight channels, the int8 form's pixel)
#define HEAD(R) fma(HW[R * 8 + 7], h[1].w, fma(HW[R * 8 + 6], h[1].z, fma(HW[R * 8 + 5], h[1].y, fma(HW[R * 8 + 4], h[1].x, \
    fma(HW[R * 8 + 3], h[0].w, fma(HW[R * 8 + 2], h[0].z, fma(HW[R * 8 + 1], h[0].y, fma(HW[R * 8], h[0].x, HB[R]))))))))
    float zl = HEAD(0);
    float sv = HEAD(1);
#undef HEAD
    mapPut(i, 1.0 / (1.0 + exp(-zl)), sv);
  }
  GroupMemoryBarrierWithGroupSync();
}
#else
// a pixel's channels' accumulators as its elements (ReLU first): halves, two groups of four an element, else a group
#if HALVES
#define PACK_PIXEL(SINK, ACC, OG) \
  [ForceUnroll] for (int e = 0; e < (OG + 1) / 2; e++) { \
    if (2 * e + 1 < OG) { SINK(e, uint4(HALF2(max(ACC[2 * e], V4(0.0))), HALF2(max(ACC[min(2 * e + 1, OG - 1)], V4(0.0))))) } \
    else { SINK(e, uint4(HALF2(max(ACC[2 * e], V4(0.0))), uint2(0u))) } }
#define EPP(C) (((C) + 7) / 8)
#define GROUP(G) GRP(G % 2 == 1 ? e[G >> 1].zw : e[G >> 1].xy)
#else
#define PACK_PIXEL(SINK, ACC, OG) [ForceUnroll] for (int e = 0; e < OG; e++) { SINK(e, asuint(max(ACC[e], V4(0.0)))) }
#define EPP(C) ((C) / 4)
#define GROUP(G) asfloat(e[G])
#endif
// conv1: four output pixels a lane, each tap row read as two elements of X (four pixels an element)
void conv1<let DIL : int, let IN_C : int, let OUT_C : int, let IN_W : int, let OUT_W : int, let OUT_H : int>(uint li) {
  const int OG = OUT_C / 4;
  [loop] for (uint i = li; i < uint(NET_G1 * OUT_H); i += uint(WG)) {
    uint gk = i % uint(NET_G1);
    uint y = i / uint(NET_G1);
    uint x0 = gk * 4u;
    V4 a[4][OG];
    [ForceUnroll] for (int j = 0; j < 4; j++) { [ForceUnroll] for (int go = 0; go < OG; go++) { a[j][go] = V4(B1[go * 4], B1[go * 4 + 1], B1[go * 4 + 2], B1[go * 4 + 3]); } }
    [ForceUnroll] for (int ky = 0; ky < 3; ky++) {
      float4 r0 = asfloat(poolB[(y + uint(ky * DIL)) * RS + gk]);
      float4 r1 = asfloat(poolB[(y + uint(ky * DIL)) * RS + gk + 1u]);
      T cv[8] = T[8](T(r0.x), T(r0.y), T(r0.z), T(r0.w), T(r1.x), T(r1.y), T(r1.z), T(r1.w));
      [ForceUnroll] for (int j = 0; j < 4; j++) { [ForceUnroll] for (int kx = 0; kx < 3; kx++) { [ForceUnroll] for (int go = 0; go < OG; go++) {
        const int at = go * 36 + ky * 3 + kx;
        a[j][go] = fma(V4(W1[at], W1[at + 9], W1[at + 18], W1[at + 27]), V4(cv[j + kx * DIL]), a[j][go]);
      } } }
    }
    [ForceUnroll] for (int j = 0; j < 4; j++) {
      if (x0 + uint(j) < uint(OUT_W)) {
        uint p = y * uint(OUT_W) + x0 + uint(j);
#define TO_POOL(E, V) poolA[p * uint(EPP(OUT_C)) + uint(E)] = V;
        PACK_PIXEL(TO_POOL, a[j], OG)
#undef TO_POOL
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
}
// a later conv's taps: a pixel a lane, its input's elements a tap, the output channels four a vector
#define CONV_TAPS \
  [ForceUnroll] for (int ky = 0; ky < 3; ky++) { [ForceUnroll] for (int kx = 0; kx < 3; kx++) { \
    uint q = ((y + uint(ky * DIL)) * uint(IN_W) + x + uint(kx * DIL)) * uint(EPP(IN_C)); \
    uint4 e[EPP(IN_C)]; \
    [ForceUnroll] for (int n = 0; n < EPP(IN_C); n++) { e[n] = poolA[q + uint(n)]; } \
    V4 iv[IN_C / 4]; \
    [ForceUnroll] for (int gi = 0; gi < IN_C / 4; gi++) { iv[gi] = GROUP(gi); } \
    [ForceUnroll] for (int go = 0; go < OUT_C / 4; go++) { [ForceUnroll] for (int gi = 0; gi < IN_C / 4; gi++) { [ForceUnroll] for (int j = 0; j < 4; j++) { \
      const int at = (((go * 4) * IN_C + gi * 4 + j) * 3 + ky) * 3 + kx; \
      acc[go] = fma(V4(wf<K>(at), wf<K>(at + IN_C * 9), wf<K>(at + 2 * IN_C * 9), wf<K>(at + 3 * IN_C * 9)), V4(iv[gi][j]), acc[go]); \
    } } } \
  } }
// a middle conv: its pixel's elements into registers for round r, written over the input after a barrier (staged)
void convMid<let DIL : int, let IN_C : int, let OUT_C : int, let IN_W : int, let OUT_W : int, let OUT_H : int, let K : int>(uint li) {
  const int N = OUT_W * OUT_H, NR = (N + WG - 1) / WG, OG = OUT_C / 4, ES = EPP(OUT_C);
  uint4 o[NR][ES];
  [ForceUnroll] for (int r = 0; r < NR; r++) { [ForceUnroll] for (int e = 0; e < ES; e++) { o[r][e] = uint4(0u); } }
  [loop] for (uint r = 0u; r < uint(NR); r++) {
    uint i = li + r * uint(WG);
    if (i < uint(N)) {
      uint x = i % uint(OUT_W);
      uint y = i / uint(OUT_W);
      V4 acc[OG];
      [ForceUnroll] for (int go = 0; go < OG; go++) { acc[go] = V4(bf<K>(go * 4), bf<K>(go * 4 + 1), bf<K>(go * 4 + 2), bf<K>(go * 4 + 3)); }
      CONV_TAPS
      uint4 st[ES];
#define TO_ST(E, V) st[E] = V;
      PACK_PIXEL(TO_ST, acc, OG)
#undef TO_ST
      [ForceUnroll] for (int rq = 0; rq < NR; rq++) { if (r == uint(rq)) { [ForceUnroll] for (int e = 0; e < ES; e++) { o[rq][e] = st[e]; } } }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  [ForceUnroll] for (int r = 0; r < NR; r++) {
    if (li + uint(r * WG) < uint(N)) { [ForceUnroll] for (int e = 0; e < ES; e++) { poolA[(li + uint(r * WG)) * uint(ES) + uint(e)] = o[r][e]; } }
  }
  GroupMemoryBarrierWithGroupSync();
}
// the last conv and the head, fused; the map keeps the probability, so the non-maximum test reads it nine times
// without nine exps
void convLast<let DIL : int, let IN_C : int, let OUT_C : int, let IN_W : int, let OUT_W : int, let OUT_H : int, let K : int>(uint li) {
  const int OG = OUT_C / 4;
  [loop] for (uint i = li; i < uint(OUT_W * OUT_H); i += uint(WG)) {
    uint x = i % uint(OUT_W);
    uint y = i / uint(OUT_W);
    V4 acc[OG];
    [ForceUnroll] for (int go = 0; go < OG; go++) { acc[go] = V4(bf<K>(go * 4), bf<K>(go * 4 + 1), bf<K>(go * 4 + 2), bf<K>(go * 4 + 3)); }
    CONV_TAPS
    float z[2];
    [ForceUnroll] for (int row = 0; row < 2; row++) {
      T d = dot(V4(HW[row * OUT_C], HW[row * OUT_C + 1], HW[row * OUT_C + 2], HW[row * OUT_C + 3]), max(acc[0], V4(0.0)));
      [ForceUnroll] for (int go = 1; go < OG; go++) { const int at = row * OUT_C + go * 4; d = d + dot(V4(HW[at], HW[at + 1], HW[at + 2], HW[at + 3]), max(acc[go], V4(0.0))); }
      z[row] = HB[row] + float(d);
    }
    mapPut(i, 1.0 / (1.0 + exp(-z[0])), z[1]);
  }
  GroupMemoryBarrierWithGroupSync();
}
#endif

[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.z;
  Frame F = frames[f];
  int w = int((F.w + (1u << P.level) - 1u) >> P.level);
  int h = int((F.h + (1u << P.level) - 1u) >> P.level);
  int ry0 = int(wg.y) * RH;
  int yo0 = ry0 - HALO;
  // A whole workgroup returns together, so the early exit keeps control flow uniform.
  if (F.valid != 1u || int(wg.x * WALK) * RW >= w || ry0 >= h) { return; }
  // the first region's X (or Xq) into registers, a lane's share of the window
  XP xp[NXL];
  [ForceUnroll] for (int r = 0; r < NXL; r++) { xp[r] = XP(0); }
  {
    int xo0 = int(wg.x * WALK) * RW - HALO;
    [ForceUnroll] for (int r = 0; r < NXL; r++) { if (li + uint(r * WG) < uint(NXI)) { xp[r] = xfetch(li + uint(r * WG), xo0, yo0, w, h, f); } }
  }
  [loop] for (uint rr = 0u; rr < WALK; rr++) {
    uint rxi = wg.x * WALK + rr;
    int rx0 = int(rxi) * RW;
    int xo0 = rx0 - HALO;
    // The same for every lane (the frame's width and the loop count), so the loop's barriers stay uniform.
    if (rx0 >= w) { break; }
    if (li < NET_NT) { nfound[li].store(0u); }
    [ForceUnroll] for (int r = 0; r < NXL; r++) { if (li + uint(r * WG) < uint(NXI)) { xPut((li + uint(r * WG)) / XE, (li + uint(r * WG)) % XE, xp[r], xo0, yo0, w, h, f); } }
    GroupMemoryBarrierWithGroupSync();
    if (rr + 1u < WALK && rx0 + RW < w) {
      [ForceUnroll] for (int r = 0; r < NXL; r++) { if (li + uint(r * WG) < uint(NXI)) { xp[r] = xfetch(li + uint(r * WG), xo0 + RW, yo0, w, h, f); } }
    }
    conv1<NET_C1>(li);
#if NET_CONVS == 2
    convLast<NET_C2, 2>(li);
#elif NET_CONVS == 3
    convMid<NET_C2, 2>(li);
    convLast<NET_C3, 3>(li);
#else
    convMid<NET_C2, 2>(li);
    convMid<NET_C3, 3>(li);
    convLast<NET_C4, 4>(li);
#endif
    // the non-maximum test: a lane takes a column run of PPL pixels and reads the PPL + 2 map rows round it once,
    // where the region's width divides the workgroup; else a pixel a round
    const int PPL = (RW * RH) / WG;
    if (WG % RW == 0 && (WG / RW) * PPL == RH) {
      uint cx = li % uint(RW);
      uint y0 = (li / uint(RW)) * uint(PPL);
      float3 m[PPL + 2];
      [ForceUnroll] for (int k = 0; k < PPL + 2; k++) { m[k] = float3(mapP((y0 + uint(k)) * uint(NET_MW) + cx), mapP((y0 + uint(k)) * uint(NET_MW) + cx + 1u), mapP((y0 + uint(k)) * uint(NET_MW) + cx + 2u)); }
      [ForceUnroll] for (int p = 0; p < PPL; p++) {
        peakAt(float[9](m[p].x, m[p].y, m[p].z, m[p + 1].x, m[p + 1].y, m[p + 1].z, m[p + 2].x, m[p + 2].y, m[p + 2].z), (y0 + uint(p + 1)) * uint(NET_MW) + cx + 1u, ((y0 + uint(p)) / uint(TILE)) * uint(NET_TX) + cx / uint(TILE), rx0 + int(cx), ry0 + int(y0) + p, w, h);
      }
    } else {
      [unroll] for (uint r = 0u; r < uint(PPL); r++) {
        uint q = li + r * uint(WG);
        uint px = q % uint(RW) + 1u;
        uint py = q / uint(RW) + 1u;
        uint cc = py * uint(NET_MW) + px;
        float nb[9] = float[9](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        [unroll] for (uint k = 0u; k < 9u; k++) { nb[k] = mapP(cc + (k / 3u) * uint(NET_MW) + (k % 3u) - uint(NET_MW) - 1u); }
        peakAt(nb, cc, ((py - 1u) / uint(TILE)) * uint(NET_TX) + (px - 1u) / uint(TILE), rx0 + int(px) - 1, ry0 + int(py) - 1, w, h);
      }
    }
    GroupMemoryBarrierWithGroupSync();
    // Each tile's KEEP strongest by rank (ties by position, so ranks are distinct) into its slots; empty slots stay as
    // the clear left them, response 0. An item is (tile, place in its list).
    [unroll] for (uint it = li; it < NET_NT * LOCAL; it += 256u) {
      uint t = it / LOCAL;
      uint j = it % LOCAL;
      uint tx = rxi * uint(NET_TX) + t % uint(NET_TX);
      uint ty = wg.y * uint(NET_TY) + t / uint(NET_TX);
      uint n = nfound[t].load();
      uint kept = min(n, LOCAL);
      if (j == 0u && n > 0u) { InterlockedAdd(counts[f * 16u + 1u], n); }
      // Peaks past LOCAL were never stored, so which LOCAL a full tile keeps is the lanes' order; count them.
      if (j == 0u && n > LOCAL) { InterlockedAdd(counts[f * 16u + OVERFLOW], n - LOCAL); }
      if (j < kept && tx < P.tilesX) {
        float4 e = PEAK_GET(t * LOCAL + j);
        uint rank = 0u;
        for (uint q = 0u; q < kept; q++) {
          float o = PEAK_W(t * LOCAL + q);
          if (o > e.w || (o == e.w && q < j)) { rank++; }
        }
        if (rank < KEEP) { raw[f * P.slots + P.slotBase + (ty * P.tilesX + tx) * KEEP + rank] = e; }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
}
