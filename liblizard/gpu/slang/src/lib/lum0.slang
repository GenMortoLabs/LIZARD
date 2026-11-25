// Level 0's bilinear read: pixel centres at i + 0.5, clamped to the frame's own edge. A stage including it binds the
// frame texture as `img` (Texture2DArray<float>, read by Load without a sampler) and includes convert.slang.
float lum0(uint f, int x, int y, int w, int h) {
  return img.Load(int4(clamp(x, 0, w - 1), clamp(y, 0, h - 1), int(f), 0));
}
float bilin0(uint f, float2 p, int w, int h) {
  float2 q = p - float2(0.5);
  float2 q0 = floor(q);
  float2 a = q - q0;
  int2 i = f2i(q0);
  float t = lerp(lum0(f, i.x, i.y, w, h), lum0(f, i.x + 1, i.y, w, h), a.x);
  float b = lerp(lum0(f, i.x, i.y + 1, w, h), lum0(f, i.x + 1, i.y + 1, w, h), a.x);
  return lerp(t, b, a.y);
}
