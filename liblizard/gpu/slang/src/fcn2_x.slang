// F2's X pass (stage x): each pixel of a pyramid level standardised over its own BOX x BOX box (the window's mean c
// taken off first) into the plane, XB0 x XB1 pixels a workgroup: the window read into poolB (from the level, or with
// TEX from the frame texture: level 0, the capture itself), the box row sums into poolA, the column pass into
// registers, then each X into the plane (in f32, as halves, or as int8 codes, Xq). The row and column passes and the
// plane's store are generated for the layout (bank_fcn2.mjs, ../../plan/bank_fcn2.mjs fcn2Layout): gen/plane_fns,
// gen/stats_fns, gen/plane_put, gen/stats. No workgroup variable is zeroed: each is written before a barrier that
// precedes its reads. Bindings: 0 the level (or the frame texture), 1 frames, 2 the plane, 3 the Bank uniform (level;
// stride and rows, the level's; slots, slotBase, tilesX the net's). Defines (bank_fcn2.mjs fcn2Source): TEX, INT8 (0 or
// 1), PLANE_T (the plane's element), POOL_A, POOL_B (the pools' uvec4s), BANK_HALO, XB0, XB1 (int), BANK_KEEP,
// BANK_LOCAL, PLANE_XE (uint), BANK_KAPPA, BANK_NOMINAL, BANK_EPS, BANK_FLOOR, Q8_MAX, Q8_INV (float).
#if TEX
#endif
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
// the form, its box and its workgroup memory (read by ../../plan/bank_fcn2.mjs fcn2Bytes):
#include "gen/about"
groupshared uint4 poolA[POOL_A];
groupshared uint4 poolB[POOL_B];
static const uint XE = PLANE_XE;
#if INT8
#endif
#include "gen/plane_fns"
#include "gen/stats_fns"
#include "gen/plane_put"

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
#include "gen/stats"
}
