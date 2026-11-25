// Level 0's bilinear read by the texture unit (lum0's bilin0 in one fetch): the frame texture through the device's
// linear sampler (clamp to the edge, normalised coordinates; natively the layout's immutable sampler, core/wg
// Device::linearSampler), the weights in the device's fixed point (8 bits a texel on the 4090, RADV and the Adreno
// 840). p is clamped to the frame's outer texel centres, [0.5, w - 0.5] x [0.5, h - 0.5]: there the far texel's weight
// is exactly 0, so a read past the frame takes its edge texel as lum0's clamp does (the layer may be wider and taller
// than the frame; the sampler clamps only at the layer's edge). hi = (w, h) - 0.5 and inv = one over the layer's size,
// the caller's, made once (LUMH_HEAD). A stage including it binds the frame texture as `img` (Texture2DArray<float>)
// and the sampler at LUMH_BINDING (its define).
[[vk::binding(LUMH_BINDING, 0)]] SamplerState smp;
float bilinH(uint f, float2 p, float2 hi, float2 inv) {
  return img.SampleLevel(smp, float3(clamp(p, float2(0.5), hi) * inv, float(f)), 0.0);
}
// the layer's size (the texture's level 0)
float2 layerSize() {
  uint w, h, e, m;
  img.GetDimensions(0, w, h, e, m);
  return float2(float(w), float(h));
}
// hi and inv for bilinH, a frame of w x h in the bound layer (int w, h): hhi and hinv
#define LUMH_HEAD(W, H) float2 hhi = float2(float(W), float(H)) - float2(0.5); float2 hinv = float2(1.0) / layerSize();
