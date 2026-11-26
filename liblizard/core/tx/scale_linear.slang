// ai: The Metal presenter's Bilinear (lizard-desktop/native/presenter_metal.mm): the slot's picture (the expand kernel's
// ai: RGBA) stretched to its place on the drawable by the texture unit's bilinear filter, the stretch the Vulkan
// ai: presenter makes with vkCmdBlitImage's VK_FILTER_LINEAR (Metal's blit encoder copies 1:1 alone). Place pixel o of
// ai: the dw x dh place from (ox, oy) samples the picture at its centre, ((o + 0.5) / (dw, dh)), clamped to the edge.
// ai: Metal's alone: gen.mjs compiles it to MSL and to no SPIR-V (MSL_ONLY).
struct P { int ox; int oy; int dw; int dh; };
[[vk::push_constant]] ConstantBuffer<P> p;
[[vk::binding(0, 0)]] Texture2D<float4> src;
[[vk::binding(1, 0)]] SamplerState lin;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] WTexture2D<float4> dst;
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int2 o = int2(gid.xy);
  if (o.x >= p.dw || o.y >= p.dh) return;
  float2 uv = (float2(o) + 0.5) / float2(float(p.dw), float(p.dh));
  dst.Store(o + int2(p.ox, p.oy), src.SampleLevel(lin, uv, 0.0));
}
