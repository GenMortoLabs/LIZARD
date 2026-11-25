// ai: The presenter's Lanczos-3 stretch of the grey picture, its second pass, down: scale_h.slang's dw x sh floats (0 to
// ai: 255) to the dw x dh RGBA8 picture the presenter copies 1:1 to the screen, the grey along the line from the code's
// ai: ink to its paper (0xRRGGBB each; black and white the grey itself), as the expand kernel draws it; each destination
// ai: row's taps from the table's second part (from yo), the same for the whole row; the edge rows repeated; clamped to
// ai: 0..1 after the kernel's overshoot. Each output reads its rows straight from the buffer (consecutive threads,
// ai: consecutive columns: one row's read one line of memory); a tile of the rows in shared memory measured slower on
// ai: the iGPU (3.0 against 1.65 ms at 1840 x 910).
struct P { int sw; int sh; int dw; int dh; int tx; int ty; int yo; int rw; uint ink; uint paper; };
[[vk::push_constant]] ConstantBuffer<P> p;
[[vk::binding(1, 0)]] StructuredBuffer<float> m;
[[vk::binding(2, 0)]] StructuredBuffer<float> t;
[[vk::binding(3, 0)]] [[vk::image_format("rgba8")]] WTexture2D<float4> dst;
float3 rgbOf(uint c) { return float3(float((c >> 16u) & 0xffu), float((c >> 8u) & 0xffu), float(c & 0xffu)) / 255.0; }
[numthreads(64, 4, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int2 o = int2(gid.xy);
  if (o.x >= p.dw || o.y >= p.dh) return;
  int b = p.yo + o.y * (p.ty + 1);
  int lo = int(t[b]);
  float acc = 0.0;
  for (int k = 0; k < p.ty; k++) acc += t[b + 1 + k] * m[clamp(lo + k, 0, p.sh - 1) * p.dw + o.x];
  dst.Store(o, float4(lerp(rgbOf(p.ink), rgbOf(p.paper), clamp(acc / 255.0, 0.0, 1.0)), 1.0));
}
