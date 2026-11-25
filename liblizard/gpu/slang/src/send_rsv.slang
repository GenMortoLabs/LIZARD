// The sender's RSV, the resampler down the picture: TMP[y][k] = sum over t < 6 of w[6 y + t] pic[idx[6 y + t]][k],
// y < q, k < n, a lane four adjacent k (float4), the taps the C's (../../../sim/ob.mjs resampleGeom), idx its first
// samples wrapped mod n on the host: the picture read as periodic, so the q pixels cover the picture and its guard
// interval, which is the picture's own wrap.
// Bindings: PIC (ro, float4), TAPS (ro: idx, 6 q words, then w, 6 q floats' bits), TMP (rw, float4), the G uniform (a = (n,
// q, sq, W)). Dispatch (ceil(n / 4 / LANES), q, symbols).
// Defines (../send.mjs rsvSource): LANES (int, ../../plan/send.mjs RSV_THREADS).
[[vk::binding(0, 0)]] StructuredBuffer<float4> PIC;
[[vk::binding(1, 0)]] StructuredBuffer<uint> TAPS;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> TMP;
struct GBlock { uint4 a; uint4 b; uint4 c; };
[[vk::binding(3, 0)]] ConstantBuffer<GBlock> P;
[numthreads(LANES, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint n4 = P.a.x / 4u;
  uint q = P.a.y;
  uint k4 = id.x;
  uint y = id.y;
  uint r = id.z;
  if (k4 >= n4) { return; }
  uint base = r * P.a.x * n4 + k4;
  float4 v = asfloat(TAPS[6u * q + 6u * y]) * PIC[base + TAPS[6u * y] * n4];
  for (uint t = 1u; t < 6u; t = t + 1u) {
    v = v + asfloat(TAPS[6u * q + 6u * y + t]) * PIC[base + TAPS[6u * y + t] * n4];
  }
  TMP[(r * q + y) * n4 + k4] = v;
}
