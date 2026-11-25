// F1, the pyramid: level l + 1 the 2 x 2 mean of level l, levels 1 to 4 from one 64 x 64 block of a frame's layer a
// workgroup, each mean summed from 0 in one order, so every device's levels are the same bit for bit (../pyramid.mjs:
// the block, plan/pyramid.mjs PYRAMID_BLOCK, is 64).
// Bindings: 0 the frame texture (r8unorm 2D array, read by Load without a sampler: SPIR-V's OpImageFetch on a sampled
// image), 1 the
// frames, 2 to 5 levels 1 to 4 (f32), 6 the levels' (stride, rows). Every workgroup variable is written before the
// barrier that precedes its reads, so none is zeroed. Dispatched (ceil(W / 64), ceil(H / 64), frames).
#include "lib/frame.slang"
[[vk::binding(0, 0)]] Texture2DArray<float> img;
[[vk::binding(1, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> lv1;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> lv2;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> lv3;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> lv4;
struct LevelDimsBlock { uint4 d[5]; };
[[vk::binding(6, 0)]] ConstantBuffer<LevelDimsBlock> LD;
groupshared float t1[1024];   // the block's level 1, 32 x 32
groupshared float t2[256];    // level 2, 16 x 16
groupshared float t3[64];     // level 3, 8 x 8
uint dimAt(uint v, uint level) { return (v + (1u << level) - 1u) >> level; }
float mean4(float tl, float tr, float bl, float br) {
  float s = 0.0;
  s += tl;
  s += tr;
  s += bl;
  s += br;
  return 0.25 * s;
}
float tex(uint f, uint x, uint y, uint w, uint h) {
  return img.Load(int4(int(min(x, w - 1u)), int(min(y, h - 1u)), int(f), 0));
}
// The mean of the 2 x 2 under pixel (x, y) of level l + 1 from tile T (side S, uint) of level l, the block's own pixel
// (lx, ly) of it: the right column and the lower row clamp onto the pixel beside them past the frame's edge (W x H,
// level l's).
#define DOWN(T, S, W, H) \
    uint c0 = 2u * lx; uint c1 = 2u * x + 1u >= W ? 2u * lx : 2u * lx + 1u; \
    uint r0 = 2u * ly * S; uint r1 = 2u * y + 1u >= H ? 2u * ly * S : (2u * ly + 1u) * S; \
    float v = mean4(T[r0 + c0], T[r0 + c1], T[r1 + c0], T[r1 + c1]);
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.z;
  Frame F = frames[f];
  bool live = F.valid != 0u;
  uint w1 = dimAt(F.w, 1u); uint h1 = dimAt(F.h, 1u);
  uint w2 = dimAt(F.w, 2u); uint h2 = dimAt(F.h, 2u);
  uint w3 = dimAt(F.w, 3u); uint h3 = dimAt(F.h, 3u);
  uint w4 = dimAt(F.w, 4u); uint h4 = dimAt(F.h, 4u);
  // Level 1: pixel (lx, ly) of the block's 32 x 32 at i = li + 256 k, a row of 32 every 32 lanes.
  for (uint k = 0u; k < 4u; k++) {
    uint i = li + 256u * k;
    uint lx = i & 31u;
    uint ly = i >> 5u;
    uint x = 32u * wg.x + lx;
    uint y = 32u * wg.y + ly;
    float v = 0.0;
    if (live) {
      float p00 = tex(f, 2u * x, 2u * y, F.w, F.h);
      float p01 = tex(f, 2u * x + 1u, 2u * y, F.w, F.h);
      float p10 = tex(f, 2u * x, 2u * y + 1u, F.w, F.h);
      float p11 = tex(f, 2u * x + 1u, 2u * y + 1u, F.w, F.h);
      v = mean4(p00, p01, p10, p11);
      if (x < w1 && y < h1) { lv1[(f * LD.d[1].y + y) * LD.d[1].x + x] = v; }
    }
    t1[i] = v;
  }
  GroupMemoryBarrierWithGroupSync();
  {
    uint lx = li & 15u;
    uint ly = li >> 4u;
    uint x = 16u * wg.x + lx;
    uint y = 16u * wg.y + ly;
    DOWN(t1, 32u, w1, h1)
    if (live && x < w2 && y < h2) { lv2[(f * LD.d[2].y + y) * LD.d[2].x + x] = v; }
    t2[li] = v;
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < 64u) {
    uint lx = li & 7u;
    uint ly = li >> 3u;
    uint x = 8u * wg.x + lx;
    uint y = 8u * wg.y + ly;
    DOWN(t2, 16u, w2, h2)
    if (live && x < w3 && y < h3) { lv3[(f * LD.d[3].y + y) * LD.d[3].x + x] = v; }
    t3[li] = v;
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < 16u) {
    uint lx = li & 3u;
    uint ly = li >> 2u;
    uint x = 4u * wg.x + lx;
    uint y = 4u * wg.y + ly;
    DOWN(t3, 8u, w3, h3)
    if (live && x < w4 && y < h4) { lv4[(f * LD.d[4].y + y) * LD.d[4].x + x] = v; }
  }
}
