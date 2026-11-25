// The web sender's PRESENT, its fragment stage: canvas pixel (x, y) shows the frame's byte (x / whole, y / whole) as
// the code's colours: the grey along the line from the ink (G.c.y, 0xRRGGBB) to the paper (G.c.w), as src/wasm.c
// lizard_tx_rgba and core/tx/expand.comp paint it.
// Bindings: FRAME (ro, one frame: bound at its offset in the batch's), the G uniform (a = (n, q, sq, W), b = (codes, FW,
// RW, FS), c = (whole, the ink, gap, the paper)).
[[vk::binding(0, 0)]] StructuredBuffer<uint> FRAME;
struct GBlock { uint4 a; uint4 b; uint4 c; };
[[vk::binding(1, 0)]] ConstantBuffer<GBlock> P;
float3 rgbOf(uint c) {
  return float3(float((c >> 16u) & 0xffu), float((c >> 8u) & 0xffu), float(c & 0xffu)) / 255.0;
}
[shader("fragment")]
float4 main(float4 pos : SV_Position) : SV_Target {
  uint whole = max(P.c.x, 1u);
  uint x = min(uint(pos.x) / whole, P.b.y - 1u);
  uint y = min(uint(pos.y) / whole, P.a.w - 1u);
  uint i = 4u * P.b.z * y + x;
  float g = float((FRAME[i >> 2u] >> (8u * (i & 3u))) & 0xffu) / 255.0;
  return float4(lerp(rgbOf(P.c.y), rgbOf(P.c.w), g), 1.0);
}
