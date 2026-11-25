// The transforms' complex helpers and butterflies (the decoder's transform and the sender's inverses): a function-scope
// array of eight points taken inout.
float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
float2 mneg_i(float2 a) { return float2(a.y, -a.x); }
static const float C8 = 0.70710678118654752;
static const float S3 = 0.86602540378443864676;
void dft2(inout float2 p[8], uint o) {
  float2 a = p[o]; float2 b = p[o + 1u];
  p[o] = a + b; p[o + 1u] = a - b;
}
void dft4(inout float2 p[8], uint o) {
  float2 p0 = p[o]; float2 p1 = p[o + 1u]; float2 p2 = p[o + 2u]; float2 p3 = p[o + 3u];
  float2 b0 = p0 + p2; float2 b2 = p0 - p2; float2 b1 = p1 + p3; float2 b3 = mneg_i(p1 - p3);
  p[o] = b0 + b1; p[o + 1u] = b2 + b3; p[o + 2u] = b0 - b1; p[o + 3u] = b2 - b3;
}
void dft8(inout float2 p[8]) {
  float2 a0 = p[0] + p[4]; float2 a4 = p[0] - p[4];
  float2 a1 = p[1] + p[5]; float2 d1 = p[1] - p[5];
  float2 a2 = p[2] + p[6]; float2 a6 = mneg_i(p[2] - p[6]);
  float2 a3 = p[3] + p[7]; float2 d3 = p[3] - p[7];
  float2 a5 = float2(C8 * (d1.x + d1.y), C8 * (d1.y - d1.x));
  float2 a7 = float2(C8 * (d3.y - d3.x), -C8 * (d3.x + d3.y));
  float2 b0 = a0 + a2; float2 b2 = a0 - a2; float2 b1 = a1 + a3; float2 b3 = mneg_i(a1 - a3);
  float2 c0 = a4 + a6; float2 c2 = a4 - a6; float2 c1 = a5 + a7; float2 c3 = mneg_i(a5 - a7);
  p[0] = b0 + b1; p[4] = b0 - b1; p[2] = b2 + b3; p[6] = b2 - b3;
  p[1] = c0 + c1; p[5] = c0 - c1; p[3] = c2 + c3; p[7] = c2 - c3;
}
