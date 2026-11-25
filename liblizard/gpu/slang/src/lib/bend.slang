// A quad's bend about the camera's axis in the map's own radial form (lib/map.slang
// toImage): an image point p is, undistorted, c + e (1 + k |e|^2 iR2), e = p - c, c the frame's axis (its dims'
// third word, the centre where 0) and R its farthest corner from c; bend() the inverse, toImage's Newton steps.
float3 bendAxis(Frame F) {
  float2 LA = float2(0.5 * float(F.w), 0.5 * float(F.h));
  if (F.axis != 0u) { LA = float2(0.5 * float(F.axis & 0xffffu) - 16384.0, 0.5 * float(F.axis >> 16u) - 16384.0); }
  float2 fwh = float2(float(F.w), float(F.h));
  float R = max(max(length(LA), length(float2(fwh.x, 0.0) - LA)), max(length(float2(0.0, fwh.y) - LA), length(fwh - LA)));
  return float3(LA, 1.0 / max(R * R, 1.0));
}
float2 unbend(float2 p, float3 ax, float k) { float2 e = p - ax.xy; return ax.xy + e * (1.0 + k * dot(e, e) * ax.z); }
float2 bend(float2 u, float3 ax, float k) {
  if (k == 0.0) { return u; }
  float2 e = u - ax.xy;
  float g = k * dot(e, e) * ax.z;
  float s = 1.0;
  for (int i = 0; i < 3; i++) { float gs = g * s * s; s = s - (s * (1.0 + gs) - 1.0) / (1.0 + 3.0 * gs); }
  return ax.xy + e * s;
}
