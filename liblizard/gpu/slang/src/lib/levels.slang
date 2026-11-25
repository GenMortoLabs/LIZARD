float pix(uint l, uint f, int x, int y) {
  Frame F = frames[f];
  int w = int((F.w + (1u << l) - 1u) >> l);
  int h = int((F.h + (1u << l) - 1u) >> l);
  int xc = clamp(x, 0, w - 1);
  int yc = clamp(y, 0, h - 1);
  if (l == 0u) { return img.Load(int4(xc, yc, int(f), 0)); }
  uint4 s = LD.d[l];
  uint i = (f * s.y + uint(yc)) * s.x + uint(xc);
  switch (l) {
    case 1u: { return lv1[i]; }
    case 2u: { return lv2[i]; }
    case 3u: { return lv3[i]; }
    default: { return lv4[i]; }
  }
}
float look(uint l, uint f, float2 p) {
  float2 q = p / float(1u << l) - float2(0.5);
  float2 q0 = floor(q);
  float2 a = q - q0;
  int2 i = f2i(q0);
  return lerp(lerp(pix(l, f, i.x, i.y), pix(l, f, i.x + 1, i.y), a.x), lerp(pix(l, f, i.x, i.y + 1), pix(l, f, i.x + 1, i.y + 1), a.x), a.y);
}
bool onFrame(uint f, float2 p) { Frame F = frames[f]; return p.x >= 0.0 && p.y >= 0.0 && p.x < float(F.w) && p.y < float(F.h); }
uint levelFor(float u) { return f2u(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }
float2 dir(float a) { return float2(cos(a), sin(a)); }
