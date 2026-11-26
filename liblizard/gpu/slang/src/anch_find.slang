// ANCH_FIND (SPEC 6.13): a workgroup of 128 an (anchor, frame): the ring's k x k squares (3, 5, 7 a side; an index past
// the ring's grid writes an empty record and returns) found by normalised correlation from integral images, through
// REFIT's homography: a reach of one square side each way, ANCH_TS points a square side; coarse at every second point
// and step, then every point and step within FD of the coarse best; a square's point the mean of up to 4 x 4 reads
// where a step spans more than 2 image px. Writes (the anchor's offset from where the map puts it, its peak, found).
// Defines (core/dec/build.cpp Anchors): ANCH_TS, ANCH_K, ANCH_SIDE (uint), ANCH_PEAK (float); the windows WC, W1C, OC (coarse) and WF,
// W1F, OF (fine) (uint), RC (int); each window's square pattern PC_* and PF_*; lib/anch_head.slang's.
#include "lib/anch_head.slang"
#define AL 128u
// The square's score at offset (px, py) of a window, from the window's integral images I1 (values) and I2 (squares), W1
// a row: normalised correlation with the square as painted (dark ring, light centre, light rim; its runs I25 + N25 and
// I6 + N6 of the TS_ points a side, AREA = TS_^2, TM1 = 1 - its mean, TN its norm; core/dec/build.cpp Anchors' pattern).
#define BOX(I, X, Y, SZ, W1) (I[(Y + SZ) * W1 + X + SZ] - I[Y * W1 + X + SZ] - I[(Y + SZ) * W1 + X] + I[Y * W1 + X])
#define SCORE(NAME, I1, I2, W1, TS_, AREA, I25, N25, I6, N6, TM1, TN) \
float NAME(uint px, uint py) { \
  float s1 = BOX(I1, px, py, TS_, W1); \
  float s2 = BOX(I2, px, py, TS_, W1); \
  float vr = s2 - s1 * s1 / AREA; \
  float s25 = BOX(I1, (px + I25), (py + I25), N25, W1); \
  float s6 = BOX(I1, (px + I6), (py + I6), N6, W1); \
  float st = 2.0 * (s25 - s6) + s1 * TM1; \
  float sc = st / (sqrt(vr) * TN); \
  return vr > VAR_MIN ? sc : 0.0; \
}
// A window's integral images in place, I1 and I2 of W x W values at rows of W1 (row and column 0 zero): rows, then
// columns, a lane a row or column of one image.
#define INTEGRAL(I1, I2, W, W1) \
  if (li < 2u * W) { \
    uint r = (li % W + 1u) * W1; \
    float a = 0.0; \
    if (li < W) { for (uint x = 1u; x <= W; x++) { a += I1[r + x]; I1[r + x] = a; } } \
    else { for (uint x = 1u; x <= W; x++) { a += I2[r + x]; I2[r + x] = a; } } \
  } \
  GroupMemoryBarrierWithGroupSync(); \
  if (li < 2u * W) { \
    uint c = li % W + 1u; \
    float a = 0.0; \
    if (li < W) { for (uint y = 1u; y <= W; y++) { a += I1[y * W1 + c]; I1[y * W1 + c] = a; } } \
    else { for (uint y = 1u; y <= W; y++) { a += I2[y * W1 + c]; I2[y * W1 + c] = a; } } \
  } \
  GroupMemoryBarrierWithGroupSync();
// A window's W x W reads at its module coordinates (wcx, wry) through the map and the lens, into I1 and I2 (values and
// squares) past their zero row and column; READ the read: READ_POINT one bilinear read, READ_AREA (the fine window,
// 2026-11-07) where a template step spans more than 2 image px (ms > 1: a square close up, 4 to 8 modules at 6 to 12 px
// a module, 24 to 96 px, its 16 points 1.5 to 6 px apart) the mean of ms x ms bilinear reads spread over the point's
// own cell (ex, ey: a step along each axis in image px at the square's middle), so the sharp pattern is not aliased by
// reading single pixels a step apart (read so, squares of 8 modules painted at 8 to 12 px a module were located 1 to
// 1.5 px off, which the mesh then bent the picture by).
#define READ_POINT v = bilin0(f, pq, fw, fh);
#define READ_AREA \
    if (ms == 1u) { v = bilin0(f, pq, fw, fh); } else { \
      float sv = 0.0; \
      for (uint b = 0u; b < ms; b++) for (uint c = 0u; c < ms; c++) { \
        sv += bilin0(f, pq + ex * ((float(c) + 0.5) / float(ms) - 0.5) + ey * ((float(b) + 0.5) / float(ms) - 0.5), fw, fh); \
      } \
      v = sv / float(ms * ms); \
    }
#define SAMPLES(I1, I2, W, W1, READ) \
  for (uint t = li; t < W * W; t += AL) { \
    uint wx = t % W; \
    uint wy = t / W; \
    float2 p0 = toImage(M, float2(wcx[wx], wry[wy])); \
    float2 pq = p0 + lensAt(lz, p0); \
    float v; \
    READ \
    uint o = (wy + 1u) * W1 + wx + 1u; \
    I1[o] = v; \
    I2[o] = v * v; \
  } \
  GroupMemoryBarrierWithGroupSync();
// A window's module coordinates: W points along each axis from (LOX, LOY) template steps of h modules off the square's
// corner (x0, y0), SCALE steps apart, at the middle of each (HALF = SCALE / 2).
#define COORDS(W, LOX, LOY, SCALE, HALF) \
  for (uint t = li; t < 2u * W; t += AL) { \
    if (t < W) { \
      float mx = x0 + (float(LOX) + SCALE * float(t) + HALF) * h; \
      wcx[t] = mx; \
    } else { \
      uint s = t - W; \
      float my = y0 + (float(LOY) + SCALE * float(s) + HALF) * h; \
      wry[s] = my; \
    } \
  }
// A score and its offset in one word for atomicMax: the higher score, then the lower offset.
#define PACK(S, O) (f2u(clamp(S + 1.0, 0.0, 1.999) * 1048575.0) << 11u) | (2047u - O)
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> anch;
static const uint TS = ANCH_TS;
static const int RR = int(ANCH_TS);
static const int FD = 3;
static const float VAR_MIN = 1e-6;
groupshared float IC1[W1C * W1C];
groupshared float IC2[W1C * W1C];
groupshared float IF1[W1F * W1F];
groupshared float IF2[W1F * W1F];
groupshared float CF[OF * OF];
groupshared float wcx[WC];
groupshared float wry[WC];
groupshared Atomic<uint> bestC;
groupshared Atomic<uint> bestF;
SCORE(scoreC, IC1, IC2, W1C, PC_TS, PC_AREA, PC_I25, PC_N25, PC_I6, PC_N6, PC_TM1, PC_TN)
SCORE(scoreF, IF1, IF2, W1F, PF_TS, PF_AREA, PF_I25, PF_N25, PF_I6, PF_N6, PF_TM1, PF_TN)
[numthreads(128, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.y;
  uint k = wg.x;
  if (k >= ANCH_K) { return; }
  Frame F = frames[f];
  uint4 pick = sel[f];
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { anch[ANCH_K * f + k] = float4(0.0); }
    return;
  }
  uint rgi = pick.x;
  uint S = gridOf(rgi);
  if (k % ANCH_SIDE >= S || k / ANCH_SIDE >= S) {
    if (li == 0u) { anch[ANCH_K * f + k] = float4(0.0); }
    return;
  }
  float a = A.k[3u * rgi + 2u].y;
  float h = a / float(TS);
  int rr = RR;
  float x0 = knot(rgi, 1u + k % ANCH_SIDE) - 0.5 * a;
  float y0 = knot(rgi, 1u + k / ANCH_SIDE) - 0.5 * a;
  float M[16] = maps[f].m;
  Lens lz = lensOf(f);
  int fw = int(F.w);
  int fh = int(F.h);
  COORDS(WC, -RR, -RR, 2.0, 1.0)
  for (uint t = li; t < W1C; t += AL) { IC1[t] = 0.0; IC2[t] = 0.0; IC1[t * W1C] = 0.0; IC2[t * W1C] = 0.0; }
  for (uint t = li; t < W1F; t += AL) { IF1[t] = 0.0; IF2[t] = 0.0; IF1[t * W1F] = 0.0; IF2[t * W1F] = 0.0; }
  if (li == 0u) { bestC.store(0u); bestF.store(0u); }
  GroupMemoryBarrierWithGroupSync();
  SAMPLES(IC1, IC2, WC, W1C, READ_POINT)
  INTEGRAL(IC1, IC2, WC, W1C)
  for (uint o = li; o < OC * OC; o += AL) { bestC.max(PACK(scoreC(o % OC, o / OC), o)); }
  GroupMemoryBarrierWithGroupSync();
  uint oc = 2047u - (bestC.load() & 2047u);
  int cx = clamp(2 * (int(oc % OC) - RC), FD - rr, rr - FD);
  int cy = clamp(2 * (int(oc / OC) - RC), FD - rr, rr - FD);
  COORDS(WF, cx - FD, cy - FD, 1.0, 0.5)
  // a fine step in image px at the square's middle, and the reads a point's side takes (area, above)
  float2 xm = float2(x0 + 0.5 * a, y0 + 0.5 * a);
  float2 pm = mapAt(M, lz, xm);
  float2 ex = mapAt(M, lz, xm + float2(h, 0.0)) - pm;
  float2 ey = mapAt(M, lz, xm + float2(0.0, h)) - pm;
  uint ms = clamp(f2u(ceil(max(length(ex), length(ey)) * 0.5)), 1u, 4u);
  GroupMemoryBarrierWithGroupSync();
  SAMPLES(IF1, IF2, WF, W1F, READ_AREA)
  INTEGRAL(IF1, IF2, WF, W1F)
  for (uint o = li; o < OF * OF; o += AL) {
    float s = scoreF(o % OF, o / OF);
    CF[o] = s;
    bestF.max(PACK(s, o));
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    uint po = 2047u - (bestF.load() & 2047u);
    float pk = CF[po];
    int px = int(po % OF);
    int py = int(po / OF);
    int bx = cx - FD + px;
    int by = cy - FD + py;
    float sx = 0.0;
    float sy = 0.0;
    if (px > 0 && px < 2 * FD) { float cl = CF[po - 1u]; float cr = CF[po + 1u]; float d = cl - 2.0 * pk + cr; if (d < 0.0) { sx = 0.5 * (cl - cr) / d; } }
    if (py > 0 && py < 2 * FD) { float cl = CF[po - OF]; float cr = CF[po + OF]; float d = cl - 2.0 * pk + cr; if (d < 0.0) { sy = 0.5 * (cl - cr) / d; } }
    bool ok = pk >= ANCH_PEAK && bx > -rr && bx < rr && by > -rr && by < rr;
    float2 X = float2(x0 + 0.5 * a, y0 + 0.5 * a);
    float2 D = float2((float(bx) + sx) * h, (float(by) + sy) * h);
    float2 rk = mapAt(M, lz, X + D) - mapAt(M, lz, X);
    anch[ANCH_K * f + k] = float4(rk, pk, ok ? 1.0 : 0.0);
  }
}
