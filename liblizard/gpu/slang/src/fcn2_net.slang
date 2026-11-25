// F2's network (stage net): the mark score and size a pixel from the plane, a peak where a pixel beats its eight
// neighbours and the file's floor (placed by a parabola each axis), each tile's KEEP strongest into the raw slots, by
// rank. A workgroup walks WALK regions of RW x RH pixels along a row, TX x TY tiles each: the region's X (or Xq) read
// into poolB, the layers (the weights' literals in the code, f32, f16, f32 with halves stored, or int8), the map in
// poolB, the peaks into poolA's tile lists (nfound), then the tiles' slots. Generated for the weights and the layout
// (bank_fcn2.mjs, ../../plan/bank_fcn2.mjs fcn2Layout): gen/plane_fns, gen/net_fns, gen/fetch (the first region's X
// into registers), gen/net (the walk); its small fixed loops [unroll], its pixel loops and the walk [loop].
// No workgroup variable is zeroed: each is written before a barrier that precedes its reads (nfound by its store at a
// region's start).
// Bindings: 0 the plane, 1 frames, 2 raw, 3 counts, 4 the Bank uniform (level; stride and rows, the level's; slots: a
// frame's slots over every level; slotBase: where this level's tiles start in them; tilesX: tiles a row at this level).
// Defines (bank_fcn2.mjs fcn2Source): F16, INT8 (0 or 1), PLANE_T, POOL_T (the pools' elements), POOL_A, POOL_B,
// BANK_HALO, NET_RW, NET_RH (int), BANK_KEEP, BANK_LOCAL, NET_WALK, NET_XE, NET_RS, NET_NT (uint), BANK_KAPPA,
// BANK_NOMINAL, BANK_EPS, BANK_FLOOR, Q8_MAX, Q8_INV (float).
#if F16
#endif
#if INT8
#endif
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
// the form, its box and its workgroup memory (read by ../../plan/bank_fcn2.mjs fcn2Bytes):
#include "gen/about"
groupshared POOL_T poolA[POOL_A];
groupshared POOL_T poolB[POOL_B];
static const int RW = NET_RW;
static const uint WALK = NET_WALK;
static const int RH = NET_RH;
static const uint XE = NET_XE;
static const uint RS = NET_RS;
groupshared Atomic<uint> nfound[NET_NT];
#if INT8
#endif
#include "gen/plane_fns"
#include "gen/net_fns"

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
#include "gen/fetch"
#include "gen/net"
}
