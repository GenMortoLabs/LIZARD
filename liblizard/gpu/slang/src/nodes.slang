// F7's node search (NODES; the centred form, NODES_CENTRED 1, the border's second search about the anchors' prediction,
// ANCH_ROUND): each border node found by normalised cross-correlation of the border as painted, over offsets about the
// finder's map (the centred form: about the node's place the anchors predicted, a node not at w 2 left as it is), on
// one lattice of samples and offsets (../plan/register.mjs latticeSide), whose values are read once a node and gathered
// by each offset. A workgroup of 256 a (node, frame). Every workgroup variable is written before it is read.
// Defines (register.mjs nodesSource): NODES_CENTRED (0 or 1), NODES_CHUNK, NODES_ITEMS, NODES_GRID (the lattice's
// side), NODES_MAX, LIGHT_VECS (uint); NODES_SPACING (a template sample's spacing in modules), NODES_RATIO (lattice
// steps a sample step, uint), HALF_WINDOW, SUB_F (float).
#include "lib/sync.slang"
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/round.slang"
#include "lib/level_bindings.slang"
// Registration's own level reader: a level's four taps in one switch.
struct Lvl { uint l; int w; int h; uint base; uint stride; float scale; };
Lvl lvlOf(uint l, uint f) {
  Frame F = frames[f];
  Lvl L = Lvl(l, int((F.w + (1u << l) - 1u) >> l), int((F.h + (1u << l) - 1u) >> l), 0u, 0u, 1.0 / float(1u << l));
  if (l > 0u) { uint4 s = LD.d[l]; L.base = f * s.y * s.x; L.stride = s.x; }
  return L;
}
float4 tap4(Lvl L, uint f, int x0, int x1, int y0, int y1) {
  switch (L.l) {
    case 0u: {
      return float4(img.Load(int4(x0, y0, int(f), 0)), img.Load(int4(x1, y0, int(f), 0)),
        img.Load(int4(x0, y1, int(f), 0)), img.Load(int4(x1, y1, int(f), 0)));
    }
    case 1u: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return float4(lv1[r0 + uint(x0)], lv1[r0 + uint(x1)], lv1[r1 + uint(x0)], lv1[r1 + uint(x1)]); }
    case 2u: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return float4(lv2[r0 + uint(x0)], lv2[r0 + uint(x1)], lv2[r1 + uint(x0)], lv2[r1 + uint(x1)]); }
    case 3u: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return float4(lv3[r0 + uint(x0)], lv3[r0 + uint(x1)], lv3[r1 + uint(x0)], lv3[r1 + uint(x1)]); }
    default: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return float4(lv4[r0 + uint(x0)], lv4[r0 + uint(x1)], lv4[r1 + uint(x0)], lv4[r1 + uint(x1)]); }
  }
}
float bilinear(Lvl L, uint f, float2 p) {
  float2 q = p * L.scale - float2(0.5);
  float2 q0 = floor(q);
  float2 a = q - q0;
  int2 i = f2i(q0);
  int x0 = clamp(i.x, 0, L.w - 1);
  int x1 = clamp(i.x + 1, 0, L.w - 1);
  int y0 = clamp(i.y, 0, L.h - 1);
  int y1 = clamp(i.y + 1, 0, L.h - 1);
  float4 v = tap4(L, f, x0, x1, y0, y1);
  return lerp(lerp(v.x, v.y, a.x), lerp(v.z, v.w, a.x), a.y);
}
uint levelFor(float u) { return f2u(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }
#include "lib/map.slang"
[[vk::binding(7, 0)]] StructuredBuffer<float4> nodes;
[[vk::binding(8, 0)]] StructuredBuffer<float4> tmpl;
struct MapRec { float m[16]; };
[[vk::binding(9, 0)]] StructuredBuffer<MapRec> maps;
[[vk::binding(10, 0)]] StructuredBuffer<uint4> sel;
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> meas;
struct RoundBlock { uint steps; uint stride; uint p0; uint p1; float step; float p2; float p3; float p4; uint4 count; uint4 base; uint4 light[LIGHT_VECS]; };
[[vk::binding(12, 0)]] ConstantBuffer<RoundBlock> R;
static const uint CHUNK = NODES_CHUNK;
static const uint ITEMS = NODES_ITEMS;
static const uint WG = 256u;
static const uint GRID = NODES_GRID;
groupshared uint sp[CHUNK];
groupshared float gv[GRID * GRID];
groupshared float psv[ITEMS];
groupshared float psv2[ITEMS];
groupshared float psev[ITEMS];
groupshared float ncc[225];
groupshared uint best;
#if NODES_CENTRED
groupshared float4 wm0;
#endif
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.y;
  uint n = wg.x;
  uint4 pick = sel[f];
  uint count = R.count[pick.x];
  if (frames[f].valid == 0u || pick.y == 0u || n >= count) { return; }
  float4 node = nodes[R.base[pick.x] + n];
  float M[16] = maps[f].m;
#if NODES_CENTRED
  if (li == 0u) { wm0 = meas[f * NODES_MAX + n]; }
  UNIFORM_LOAD(float4 m0, wm0)
  if (m0.w != 2.0) { return; }
  float2 cm = toImage(M, node.xy);
  float2 sh = m0.xy - cm;
  float2 c = cm + sh;
  float u = 0.5 * (distance(toImage(M, node.xy + float2(1.0, 0.0)), cm) + distance(toImage(M, node.xy + float2(0.0, 1.0)), cm));
#else
  float2 c = toImage(M, node.xy);
  float u = 0.5 * (distance(toImage(M, node.xy + float2(1.0, 0.0)), c) + distance(toImage(M, node.xy + float2(0.0, 1.0)), c));
#endif
  Frame F0 = frames[f];
  if (c.x < 2.0 * u || c.y < 2.0 * u || c.x > float(F0.w) - 2.0 * u || c.y > float(F0.h) - 2.0 * u) {
    if (li == 0u) { meas[f * NODES_MAX + n] = float4(c, 0.0, 0.0); }
    return;
  }
  Lvl L = lvlOf(levelFor(u), f);
  uint first = f2u(node.z);
  uint pts = f2u(node.w);
  uint P = (pts + R.stride - 1u) / R.stride;
  int side = int(R.steps);
  uint gridW = uint(2 * side + 1);
  uint O = gridW * gridW;
  uint G = max(1u, ITEMS / O);
  uint items = O * G;
  float spacing = NODES_SPACING;
  uint ratioN = NODES_RATIO;
  float2 origin = floor(node.xy - HALF_WINDOW) + float2(0.5 / SUB_F);
  if (li == 0u) { best = 0u; }
  uint nq = R.base[pick.x] + n;
  float tse = float(R.light[nq >> 2u][nq & 3u]);
  float tse2 = tse;
  for (uint q = li; q < GRID * GRID; q += WG) {
    float2 g = float2(float(q % GRID), float(q / GRID));
#if NODES_CENTRED
    gv[q] = bilinear(L, f, toImage(M, origin + (g - float(side)) * R.step) + sh);
#else
    gv[q] = bilinear(L, f, toImage(M, origin + (g - float(side)) * R.step));
#endif
  }
  for (uint item = li; item < items; item += WG) { psv[item] = 0.0; psv2[item] = 0.0; psev[item] = 0.0; }
  for (uint c0 = 0u; c0 < P; c0 += CHUNK) {
    uint c1 = min(P, c0 + CHUNK);
    for (uint j = c0 + li; j < c1; j += WG) {
      float4 t = tmpl[first + j * R.stride];
      uint2 g = f2u(roundEven((t.xy - origin) / spacing)) * ratioN;
      sp[j - c0] = g.x | (g.y << 8u) | (f2u(t.z) << 16u);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint item = li; item < items; item += WG) {
      uint o = item % O;
      uint g = item / O;
      uint jl = g + G * ((c0 - g + G - 1u) / G);
      uint j = c0 > g ? jl : g;
      float sv = 0.0;
      float sv2 = 0.0;
      float sev = 0.0;
        uint d = (o % gridW) + (o / gridW) * GRID;
        for (; j < c1; j += G) {
          uint s = sp[j - c0];
          float v = gv[(s & 255u) + ((s >> 8u) & 255u) * GRID + d];
          sv += v; sv2 += v * v; sev += ((s >> 16u) & 1u) != 0u ? v : 0.0;
        }
      psv[item] += sv;
      psv2[item] += sv2;
      psev[item] += sev;
    }
    GroupMemoryBarrierWithGroupSync();
  }
  for (uint o = li; o < O; o += WG) {
    float sv = 0.0;
    float sv2 = 0.0;
    float sev = 0.0;
    for (uint g = 0u; g < G; g++) { uint i = g * O + o; sv += psv[i]; sv2 += psv2[i]; sev += psev[i]; }
    float k = float(P);
    float cov = sev - tse * sv / k;
    float den = sqrt(max((tse2 - tse * tse / k) * (sv2 - sv * sv / k), 1e-12));
    ncc[o] = cov / den;
    InterlockedMax(best, (f2u(clamp(ncc[o] + 1.0, 0.0, 1.999) * 4000000.0) << 8u) | o);
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    uint b = best & 255u;
    int bx = int(b % gridW);
    int by = int(b / gridW);
    float v = ncc[b];
    float dx = 0.0;
    float dy = 0.0;
    if (bx > 0 && bx < 2 * side) {
      float a = ncc[b - 1u];
      float z = ncc[b + 1u];
      float e = a + z - 2.0 * v;
      if (e < 0.0) { dx = clamp(0.5 * (a - z) / e, -0.5, 0.5); }
    }
    if (by > 0 && by < 2 * side) {
      float a = ncc[b - gridW];
      float z = ncc[b + gridW];
      float e = a + z - 2.0 * v;
      if (e < 0.0) { dy = clamp(0.5 * (a - z) / e, -0.5, 0.5); }
    }
    float2 off = R.step * float2(float(bx - side) + dx, float(by - side) + dy);
#if NODES_CENTRED
    meas[f * NODES_MAX + n] = float4(toImage(M, node.xy + off) + sh, v, 3.0);
#else
    meas[f * NODES_MAX + n] = float4(toImage(M, node.xy + off), v, 1.0);
#endif
  }
}
