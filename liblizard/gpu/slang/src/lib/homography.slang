// A homography from four point pairs, closed form, no private array indexed at a run-time value (mul3's nine products
// written out): solveH, squareToQuad, adjugate, mul3, applyH; a quad's corners under an orientation (cornerAt, slotOf,
// atSlot) and its hypothesis' map (hypH). Defines: ZEROS_9.
float[9] squareToQuad(float2 q0, float2 q1, float2 q2, float2 q3) {
  float2 d1 = q1 - q2;
  float2 d2 = q3 - q2;
  float2 d3 = q0 - q1 + q2 - q3;
  float det = d1.x * d2.y - d2.x * d1.y;
  if (abs(det) < 1e-12) { return ZEROS_9; }
  float g = (d3.x * d2.y - d2.x * d3.y) / det;
  float h = (d1.x * d3.y - d3.x * d1.y) / det;
  return { q1.x - q0.x + g * q1.x, q3.x - q0.x + h * q3.x, q0.x,
                  q1.y - q0.y + g * q1.y, q3.y - q0.y + h * q3.y, q0.y, g, h, 1.0 };
}
float[9] adjugate(float m[9]) {
  return { m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
                  m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
                  m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3] };
}
float[9] mul3(float a[9], float b[9]) {
  float r[9] = ZEROS_9;
  r[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];
  r[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];
  r[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];
  r[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6];
  r[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7];
  r[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];
  r[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6];
  r[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7];
  r[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8];
  return r;
}
float[9] solveH(float2 src[4], float2 dst[4], float M) {
  float2 c = 0.25 * (dst[0] + dst[1] + dst[2] + dst[3]);
  float sc = max(0.25 * (distance(dst[0], c) + distance(dst[1], c) + distance(dst[2], c) + distance(dst[3], c)), 1e-3);
  float S[9] = squareToQuad(src[0] / M, src[1] / M, src[2] / M, src[3] / M);
  float D[9] = squareToQuad((dst[0] - c) / sc, (dst[1] - c) / sc, (dst[2] - c) / sc, (dst[3] - c) / sc);
  float hs[9] = mul3(D, adjugate(S));
  if (S[8] == 0.0 || D[8] == 0.0 || abs(hs[8]) < 1e-12) { return ZEROS_9; }
  float hn[9] = ZEROS_9;
  for (int j = 0; j < 9; j++) { hn[j] = hs[j] / hs[8]; }
  float H[9] = ZEROS_9;
  for (int j = 0; j < 3; j++) {
    float s = (j < 2) ? 1.0 / M : 1.0;
    H[j] = (sc * hn[j] + c.x * hn[6 + j]) * s;
    H[3 + j] = (sc * hn[3 + j] + c.y * hn[6 + j]) * s;
    H[6 + j] = hn[6 + j] * s;
  }
  return H;
}
float2 applyH(float H[9], float2 m) {
  float w = H[6] * m.x + H[7] * m.y + H[8];
  return float2(H[0] * m.x + H[1] * m.y + H[2], H[3] * m.x + H[4] * m.y + H[5]) / w;
}
float2 cornerAt(uint k, float c, float M) {
  float x = (k == 1u || k == 2u) ? M - c : c;
  float y = (k >= 2u) ? M - c : c;
  return float2(x, y);
}
uint slotOf(uint o, uint k) { return (o >= 4u) ? ((o + 4u - k) & 3u) : ((o + k) & 3u); }
float4 atSlot(float4 quad[4], uint s) {
  return (s >= 2u) ? ((s == 3u) ? quad[3] : quad[2]) : ((s == 1u) ? quad[1] : quad[0]);
}
float[9] hypH(float4 quad[4], uint o, float M) {
  float4 p0 = atSlot(quad, slotOf(o, 0u));
  float4 p1 = atSlot(quad, slotOf(o, 1u));
  float4 p2 = atSlot(quad, slotOf(o, 2u));
  float4 p3 = atSlot(quad, slotOf(o, 3u));
  float2 src[4] = { cornerAt(0u, p0.z, M), cornerAt(1u, p1.z, M), cornerAt(2u, p2.z, M), cornerAt(3u, p3.z, M) };
  float2 dst[4] = { p0.xy, p1.xy, p2.xy, p3.xy };
  return solveH(src, dst, M);
}
