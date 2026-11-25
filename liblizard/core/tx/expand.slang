// ai: A frame of the sender's ring (grey, four pixels a u32, rows `rw` words apart from word `base`) expanded into an
// ai: RGBA8 image, opaque, the desktop presenter's picture to blit on screen (2026-10-07) and the phone's ring's
// ai: buffer: no host copy on the way from the painter to the swapchain. The grey painted along the line from the
// ai: code's ink to its paper (0xRRGGBB each, each channel apart; 2026-11-16, Send's Colours: black and white the grey
// ai: itself), as Sender::take does on the CPU; rgba8's store rounds back to the byte.
struct P { uint base; uint rw; uint w; uint h; uint ink; uint paper; };
[[vk::push_constant]] ConstantBuffer<P> p;
[[vk::binding(0, 0)]] StructuredBuffer<uint> frame;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] WTexture2D<float4> pic;
float3 rgbOf(uint c) { return float3(float((c >> 16u) & 0xffu), float((c >> 8u) & 0xffu), float(c & 0xffu)) / 255.0; }
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint2 q = gid.xy;
  if (q.x >= p.w || q.y >= p.h) return;
  uint word = frame[p.base + q.y * p.rw + (q.x >> 2u)];
  float g = float((word >> (8u * (q.x & 3u))) & 0xffu) / 255.0;
  pic.Store(int2(q), float4(lerp(rgbOf(p.ink), rgbOf(p.paper), g), 1.0));
}
