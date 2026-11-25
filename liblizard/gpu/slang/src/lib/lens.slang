// The lens about the camera's axis: a frame's lens record, RESID float4 a frame in `resid` (its first five: the centre
// and scale c, the axis a, the terms t0 to t2), and its displacement at image point p. A stage including it binds
// `resid` first and defines RESID (uint).
struct Lens { float4 c; float4 a; float4 t0; float4 t1; float4 t2; };
Lens lensOf(uint f) {
  uint b = f * RESID;
  Lens L;
  L.c = resid[b]; L.a = resid[b + 1u]; L.t0 = resid[b + 2u]; L.t1 = resid[b + 3u]; L.t2 = resid[b + 4u];
  return L;
}
float2 lensAt(Lens L, float2 p) {
  float2 d = (p - L.c.xy) * L.c.z;
  float2 q = (p - L.a.xy) * L.a.z;
  float t = (dot(q, q) - L.c.w) * L.a.w;
  return float2(L.t0.x * d.x + L.t0.y * d.y + L.t1.x, L.t0.z * d.x + L.t0.w * d.y + L.t1.y)
    - (L.t1.z * d.x + L.t1.w * d.y) * d + (t * (L.t2.x + t * (L.t2.y + t * L.t2.z))) * q;
}
