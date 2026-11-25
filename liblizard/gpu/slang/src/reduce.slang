// Reduce, the back half's transform between its passes: a listed frame's five surface coefficients (the detrend pass 2
// takes off) from pass 1's column partials. A workgroup of 256 a listed frame. red is written whole before its first
// read. Built a batch size and a picture size each.
// Defines (back_transform.mjs reduceSource): PICTURE_N, BATCH (uint).
struct ParamsBlock { uint yStride; uint partStride; uint size; float m2; float sx2; float sq2; uint sStride; };
[[vk::binding(3, 0)]] ConstantBuffer<ParamsBlock> P;
[[vk::binding(0, 0)]] StructuredBuffer<uint> lists;
[[vk::binding(1, 0)]] StructuredBuffer<float4> part;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> coef;
static const uint N = PICTURE_N;
static const uint LS = BATCH + 1u;
groupshared float4 red[512];
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint tid = lix;
  uint f = lists[P.size * LS + 1u + wg.z];
  float4 s1 = float4(0.0);
  float s2 = 0.0;
  for (uint x = tid; x < N; x += 256u) {
    float4 p = part[f * P.partStride + x];
    float c = float(2 * int(x) + 1 - int(N)) / float(N);
    float q = c * c - P.m2;
    s1 += float4(c * p.x, p.y, c * p.y, q * p.x);
    s2 += p.z;
  }
  red[tid] = s1;
  red[256u + tid] = float4(s2, 0.0, 0.0, 0.0);
  GroupMemoryBarrierWithGroupSync();
  if (tid < 32u) {
    float4 a = float4(0.0);
    float4 b = float4(0.0);
    for (uint i = tid; i < 256u; i += 32u) { a += red[i]; b += red[256u + i]; }
    red[tid] = a;
    red[256u + tid] = b;
  }
  GroupMemoryBarrierWithGroupSync();
  if (tid == 0u) {
    float4 a = float4(0.0);
    float4 b = float4(0.0);
    for (uint i = 0u; i < 32u; i++) { a += red[i]; b += red[256u + i]; }
    float nf = float(N);
    coef[f * 2u] = float4(a.x / (nf * P.sx2), a.y / (nf * P.sx2), a.z / (P.sx2 * P.sx2), a.w / (nf * P.sq2));
    coef[f * 2u + 1u] = float4(b.x / (nf * P.sq2), 0.0, 0.0, 0.0);
  }
}
