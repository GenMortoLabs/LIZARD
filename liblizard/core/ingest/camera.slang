// ai: The camera's frame into a ring layer: its Y at each pixel of the crop, read
// ai: through the YCbCr sampler the host builds for the camera's format (RGB_IDENTITY: no model or range conversion,
// ai: so G is the stored Y), nearest at the texel's centre, which is the byte itself, and stored into the ring's r8
// ai: layer, which rounds it back to that byte: the camera's own Y, as the web's CPU path reads it (plane 0 as is).
// ai: cam a combined image sampler (Sampler2D), the YCbCr conversion the host's immutable sampler.
struct P { int2 origin; int2 size; float2 inv; int layer; int pad; };
[[vk::push_constant]] ConstantBuffer<P> p;
[[vk::binding(0, 0)]] Sampler2D cam;
[[vk::binding(1, 0)]] [[vk::image_format("r8")]] WTexture2DArray<float4> ring;
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int2 q = int2(gid.xy);
  if (q.x >= p.size.x || q.y >= p.size.y) return;
  float2 uv = (float2(p.origin + q) + 0.5) * p.inv;
  ring.Store(int3(q, p.layer), float4(cam.SampleLevel(uv, 0.0).g));
}
