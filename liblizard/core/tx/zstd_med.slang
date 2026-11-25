// ai: Stream files packed on the GPU (gpu_zstd.h, 2026-11-18), stage 1: each square of each frame in the sender's ring
// ai: (grey, four pixels a u32, rows rw words apart; frame i at slot slot[i], frameWords words a slot; code k's square
// ai: at x0 = k step pixels, side x side) as the residuals of JPEG-LS's median predictor (rx/lizfile.h: of a, b, c, the
// ai: neighbours left, above and above-left, a neighbour outside the square 0), each the pixel less its prediction mod
// ai: 256, in rows, four a u32 (the residual i in word i / 4, byte i mod 4), a square wps words from word q wps.
// ai: a: squares, side, rw, frameWords; b: codes, step, n (side x side), wps
struct P { uint4 a; uint4 b; };
[[vk::binding(0, 0)]] StructuredBuffer<uint> ring;
[[vk::binding(1, 0)]] StructuredBuffer<uint> slots;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> res;
[[vk::binding(3, 0)]] ConstantBuffer<P> p;
int px(uint base, uint y, uint x) { return int((ring[base + y * p.a.z + (x >> 2u)] >> ((x & 3u) * 8u)) & 0xffu); }
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  let q = gid.y; let wi = gid.x;
  if (q >= p.a.x || wi >= p.b.w) return;
  let side = p.a.y; let base = slots[q / p.b.x] * p.a.w; let x0 = (q % p.b.x) * p.b.y;
  uint o = 0u;
  for (uint j = 0u; j < 4u; j++) {
    let i = wi * 4u + j;
    if (i >= p.b.z) break;
    let y = i / side; let x = i - y * side;
    let a = x > 0u ? px(base, y, x0 + x - 1u) : 0;
    let b = y > 0u ? px(base, y - 1u, x0 + x) : 0;
    let c = x > 0u && y > 0u ? px(base, y - 1u, x0 + x - 1u) : 0;
    let mx = max(a, b); let mn = min(a, b);
    let pred = c >= mx ? mn : (c <= mn ? mx : a + b - c);
    o |= (uint(px(base, y, x0 + x) - pred) & 0xffu) << (8u * j);
  }
  res[q * p.b.w + wi] = o;
}
