// ai: Stream files packed on the GPU (gpu_zstd.h), stage 2: each 128 KiB block of each square's residuals (zstd's
// ai: largest block) counted, a workgroup a block: hist[(q nb + b) 256 + v] the residuals of value v.
// ai: a: squares, n, wps, nb
struct P { uint4 a; };
[[vk::binding(0, 0)]] StructuredBuffer<uint> res;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> hist;
[[vk::binding(2, 0)]] ConstantBuffer<P> p;
groupshared uint h[256];
static const uint BLOCK = 131072u;
[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint t : SV_GroupIndex) {
  let b = wg.x; let q = wg.y;
  h[t] = 0u;
  GroupMemoryBarrierWithGroupSync();
  let start = b * BLOCK; let end = min(p.a.y, start + BLOCK); let base = q * p.a.z;
  for (uint i = start + t * 4u; i < end; i += 1024u) {
    let w = res[base + (i >> 2u)];
    for (uint j = 0u; j < 4u && i + j < end; j++) InterlockedAdd(h[(w >> (8u * j)) & 0xffu], 1u);
  }
  GroupMemoryBarrierWithGroupSync();
  hist[(q * p.a.w + b) * 256u + t] = h[t];
}
