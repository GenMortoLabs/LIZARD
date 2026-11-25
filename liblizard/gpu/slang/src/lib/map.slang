// A frame's map from module coordinates to image pixels: 16 floats, the homography, then the radial term about a
// centre (k1 = 0 a plain homography). A buffer of maps is `float maps[][16]` (64 B a map in std430).
float2 toImage(float M[16], float2 m) {
  float w = M[6] * m.x + M[7] * m.y + M[8];
  float2 u = float2(M[0] * m.x + M[1] * m.y + M[2], M[3] * m.x + M[4] * m.y + M[5]) / w;
  if (M[9] == 0.0) { return u; }
  float2 c = float2(M[10], M[11]);
  float2 e = u - c;
  float g = M[9] * dot(e, e) * M[12];
  float s = 1.0;
  for (int i = 0; i < 3; i++) { float gs = g * s * s; s = s - (s * (1.0 + gs) - 1.0) / (1.0 + 3.0 * gs); }
  return c + e * s;
}
