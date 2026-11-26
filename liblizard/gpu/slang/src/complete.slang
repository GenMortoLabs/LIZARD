// F5d, COMPLETE: a quad of two or three corners on a frame with no quad of four, completed from the border along the
// sides its corners hold. A workgroup a (frame, quad), 64 lanes: two sides traced at once, 32 stations each, in one
// round or two (2026-11-13, research/11-corner-completion.md, research/build/complete2-2026-11-13):
// - three corners a, o, b about the missing m: the sides a -> m and b -> m;
// - two adjacent corners: the sides leaving them;
// - two diagonal corners: the two sides from each (round 0 from a, round 1 from o). A side's pass 0 reads the ring
// beside its mark, modules ALONG0 along from the corner (the mark's own ring and the band's reserve: the ring is the
// only line with its profile there; past it the band 4 modules in has the same profile and takes the stations from a
// first direction a few degrees off), or ALONG1 where the frame cuts that stretch, decided from the stations' places
// before any is read; the later passes about the fitted line. The ruler (SPEC 4.6, 4.7): from a corner the band's cells
// lie at modules 15 + 2k, whichever end the side is read from, and its track cells are a hash of the side. Along the
// traced side the 1D projective map t = s d / (1 + s d / D) (d modules from the mark's centre, s px a module there, D
// the distance to the vanishing point; 0 parallel): s and the reading's offset along by the correlation of the cells'
// values with the track over its 8 readings (side 0 to 3, from its start or its end), first over RULER_CELLS0 cells at
// scale steps of half a module at the last cell, then about the best over every cell on the frame. The band's depth
// from the traced line: the darkest row of the band's region (RULER_DEPTHS), averaged along it. The band read once into
// workgroup memory (bprof). A side is ruled where the model needs it to place its far side: both of two adjacent
// corners' sides (their crossing the vanishing point), the one traced side of three corners whose other side the frame
// does not show (its crossing with the side parallel to it). Three corners with both sides traced and two diagonal ones
// with four are held by their marks and lines (a ruler there only added its own scale error under strong perspective,
// 2026-11-13). The quad is one model fitted to every measurement: the inverse homography G, image to modules (8
// parameters, G33 1), by weighted linear least squares over each traced ring's line (its stations' fit, at its ends:
// the ring's centre 2 modules in from its side), each found mark's centre (c modules in from its corner) and each ruled
// side's band cells (cell k at 16 + 2k along from its corner). The corners measured stay as the classifier placed them,
// the missing ones are the model's (the model's own estimate at a found corner read fewer blocks than the classifier's
// place there, 2026-11-13). A system that leaves a direction free (its least pivot under FIT_PIVOT of its largest
// diagonal: no ruler where the far side needs one) completes nothing; a completed quad is kept when convex, and SCORE
// judges it as every quad. A quad the same as one before it in its frame is left: that one is completed. Defines
// (core/dec/build.cpp Finder): LANES (2 STATIONS, int), ALONG0_HI, ALONG0_LO, ALONG1_HI, ALONG1_LO, FIT_LINE, FIT_MARK, FIT_PIVOT,
// FIT_ROWS, FIT_RULER_CELL, K, NCC_MIN, PROFILE, PROF_LEN, REACH0, RULER_CELLS0, RULER_DEPTH0, RULER_DEPTH_STEP,
// RULER_DMAX, RULER_HI, RULER_LOG_LO, RULER_N0, RULER_N1, RULER_ND, RULER_NDEP, RULER_NP, RULER_STEP0, RULER_STEP0C,
// RULER_STEP1, STATIONS; the later passes' PASS0_EXT, PASS1_EXT (modules of the corner's) and PASS0_REACH, PASS1_REACH,
// PASS2_REACH (Finder's PASS_*).
#include "lib/sync.slang"
#include "lib/frame.slang"
#include "lib/round.slang"
#include "lib/convert.slang"
#include "lib/level_bindings.slang"
#include "lib/levels.slang"
// A later pass about the side's fitted line: its stations over EXT (modules of the corner's own module size from the
// first station, at most the run, or the whole run) at a reach of REACH modules (Finder's PASS_*).
#define PASS_AFTER(EXT, REACH) \
{ \
    float3 hit = float3(0.0); \
    bool has = kept[job] > 0u; \
    if (span && has) { \
      float t = t0 + EXT * (float(j) + 0.5) / float(STATIONS); \
      float uu = lerp(u0, um, t / L); \
      float4 ln = line[job]; \
      float2 nn = float2(-ln.w, ln.z); \
      if (dot(nn, n0) < 0.0) { nn = -nn; } \
      float2 dl = (dot(ln.zw, ja.zw) < 0.0) ? -ln.zw : ln.zw; \
      float2 Bp = ln.xy + dot(P0 - ln.xy, ln.zw) * ln.zw + t * dl; \
      hit = ringAt(f, Bp, nn, uu, REACH); \
    } \
    st[li] = float4(hit.xy, hit.z, 0.0); \
    GroupMemoryBarrierWithGroupSync(); \
    if (j == 0u && has) { fitSide(side, job, 0.5 * (u0 + um), EXT, 6u); } \
    GroupMemoryBarrierWithGroupSync(); \
  }

[[vk::binding(7, 0)]] StructuredBuffer<float4> readings;
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> quads;
struct P5Block { uint cap; uint gx; uint gy; uint p0; float4 modules; };
[[vk::binding(9, 0)]] ConstantBuffer<P5Block> P;

groupshared uint go;
groupshared Atomic<uint> ri[4];
groupshared uint kase;
groupshared uint nround;
groupshared uint far[4];
groupshared float4 JA[4];
groupshared float4 JB[4];
groupshared float4 JC[4];
groupshared uint JD[4];
groupshared float4 st[int(2u * STATIONS)];
groupshared float4 line[4];
groupshared uint kept[4];
groupshared uint rjob[2];
groupshared float4 RA[2];
groupshared float4 RB[2];
groupshared float4 RC[2];
groupshared float4 RS[2];
groupshared float dm[int(2u * RULER_NDEP)];
groupshared uint pk[int(RULER_CELLS0)];
groupshared float4 rbest[int(2u * STATIONS)];
groupshared float4 RT[2];
groupshared float2 lext[4];
groupshared uint src[4];
groupshared float NE[72];
float tmpl(uint s) { float r = -3.25 + 0.5 * float(s); return (r >= -1.0 && r < 1.0) ? -1.0 : 1.0; }
// A search's profile read once: its windows (PROFILE samples half a module apart, centred a quarter module a step
// from -(steps + 1) to steps + 1) share one lattice of quarter modules, X + m u n / 4 for m from -steps - 12 to
// steps + 14, so each point is read once into the lane's own slice (PROF_MAX points; off the frame -1, as no
// bilinear read of the levels is) and every window reads its samples there.
static const int PROF_MAX = PROF_LEN;
groupshared float prof[int(2u * STATIONS) * PROF_MAX];
groupshared float bprof[int(2u * RULER_NP)];
float ringNcc(uint at) {
  float k = 0.0;
  float sv = 0.0;
  float sv2 = 0.0;
  float sumT = 0.0;
  float sumT2 = 0.0;
  float svt = 0.0;
  bool bad = false;
  for (uint s = 0u; s < PROFILE; s++) {
    float v = prof[at - 2u * s];
    bool inside = v >= 0.0;
    bad = bad || (!inside && s >= 5u);
    float w = inside ? 1.0 : 0.0;
    float t = tmpl(s);
    k += w; sv += w * v; sv2 += w * v * v; sumT += w * t; sumT2 += w * t * t; svt += w * v * t;
  }
  if (bad || k < 8.0) { return -2.0; }
  float cov = svt - sv * sumT / k;
  float vv = sv2 - sv * sv / k;
  float tv = sumT2 - sumT * sumT / k;
  return cov / sqrt(max(vv * tv, 1e-12));
}
// a 3 x 3 matrix's inverse: its adjugate over its determinant
float3x3 inverse3(float3x3 m) {
  float3 r0 = cross(m[1], m[2]);
  float3 r1 = cross(m[2], m[0]);
  float3 r2 = cross(m[0], m[1]);
  float det = dot(m[0], r0);
  return transpose(float3x3(r0, r1, r2)) / det;
}
// the invocation's index in the workgroup, main's to set: ringAt reads its own slice of prof by it
static uint lane;
float3 ringAt(uint f, float2 X, float2 n, float u, float reach) {
  uint l = levelFor(u);
  int steps = f2i(roundEven(reach * 4.0));
  uint base = lane * uint(PROF_MAX);
  for (int m = -steps - 12; m <= steps + 14; m++) {
    float2 p = X + (0.25 * float(m)) * u * n;
    prof[base + uint(m + steps + 12)] = onFrame(f, p) ? look(l, f, p) : -1.0;
  }
  // the window centred at X + i u n / 4 has its sample s at m = i + 13 - 2 s: its first at base + i + steps + 25
  uint at0 = base + uint(steps) + 25u;
  float a = ringNcc(uint(int(at0) - steps - 1));
  float b = ringNcc(uint(int(at0) - steps));
  float best = -2.0;
  float at = 0.0;
  for (int i = -steps; i <= steps; i++) {
    float c = ringNcc(uint(int(at0) + i + 1));
    if (b >= NCC_MIN && b >= a && b >= c) {
      float dt = 0.0;
      float e = a + c - 2.0 * b;
      if (a > -2.0 && c > -2.0 && e < 0.0) { dt = clamp(0.5 * (a - c) / e, -0.5, 0.5); }
      best = b;
      at = 0.25 * (float(i) + dt);
    }
    a = b; b = c;
  }
  return float3(X + at * u * n, (best >= NCC_MIN) ? best : 0.0);
}
// the stations kept, a bit each of one word (STATIONS 32): on the Adreno 16 registers and 4,495 instructions
// against a private array's 24 and 5,648. A side's stations st[side 32 + j], its line and count line[job], kept[job].
bool fitSide(uint side, uint job, float u, float span, uint need) {
  uint ok = 0u;
  for (uint j = 0u; j < STATIONS; j++) { ok |= (st[side * STATIONS + j].z > 0.0) ? (1u << j) : 0u; }
  float2 o = float2(0.0);
  float2 d = float2(1.0, 0.0);
  for (uint ps = 0u; ps < 2u; ps++) {
    float sw = 0.0;
    float2 m = float2(0.0);
    for (uint j = 0u; j < STATIONS; j++) { float4 p = st[side * STATIONS + j]; if ((ok & (1u << j)) != 0u) { sw += p.z; m += p.z * p.xy; } }
    if (sw <= 0.0) { return false; }
    m /= sw;
    float cxx = 0.0; float cxy = 0.0; float cyy = 0.0;
    for (uint j = 0u; j < STATIONS; j++) {
      float4 p = st[side * STATIONS + j];
      if ((ok & (1u << j)) == 0u) { continue; }
      float2 e = p.xy - m;
      cxx += p.z * e.x * e.x; cxy += p.z * e.x * e.y; cyy += p.z * e.y * e.y;
    }
    float th = 0.5 * atan2(2.0 * cxy, cxx - cyy);
    o = m;
    d = float2(cos(th), sin(th));
    if (ps == 0u) {
      for (uint j = 0u; j < STATIONS; j++) {
        float2 e = st[side * STATIONS + j].xy - o;
        if ((ok & (1u << j)) != 0u && abs(e.x * d.y - e.y * d.x) > u) { ok &= ~(1u << j); }
      }
    }
  }
  uint n = 0u;
  float tmin = 1e30;
  float tmax = -1e30;
  for (uint j = 0u; j < STATIONS; j++) {
    if ((ok & (1u << j)) == 0u) { continue; }
    n++;
    float t = dot(st[side * STATIONS + j].xy - o, d);
    tmin = min(tmin, t); tmax = max(tmax, t);
  }
  if (n < max(need, 6u) || tmax - tmin < 0.25 * span) { return false; }
  line[job] = float4(o, d);
  lext[job] = float2(tmin, tmax);
  kept[job] = n;
  return true;
}
float2 rot(float2 v, float a) { float c = cos(a); float s = sin(a); return float2(c * v.x - s * v.y, s * v.x + c * v.y); }
float turn(float2 a, float2 b, float2 c) { float2 e = b - a; float2 g = c - b; return e.x * g.y - e.y * g.x; }
bool convex(float2 c0, float2 c1, float2 c2, float2 c3) {
  float4 tn = float4(turn(c0, c1, c2), turn(c1, c2, c3), turn(c2, c3, c0), turn(c3, c0, c1));
  return all((tn > float4(0.0))) || all((tn < float4(0.0)));
}
// two lines' crossing for a vanishing point: w 0 only where they are parallel to a part in 10^4
float3 vanish(float4 A, float4 B) {
  float den = A.z * B.w - A.w * B.z;
  if (abs(den) < 1e-4) { return float3(0.0); }
  float s = ((B.x - A.x) * B.w - (B.y - A.y) * B.z) / den;
  return float3(A.xy + s * A.zw, 1.0);
}
// a side's job: from the mark at P0 (its reading r), the direction guesses g0 and g1, the outward reference nref
// (a normal of the side points outward where it has a positive part along it), its module size u0, the far
// corner's um, the run's length L, the mark's form c; pass 0's place from the stations' places before any is read
void job(uint k, float2 P0, float2 g0, float2 g1, float2 nref, float u0, float um, float L, float c, uint f, uint fs, uint ss) {
  JA[k] = float4(P0, g0);
  JB[k] = float4(g1, nref);
  JC[k] = float4(u0, um, L, c);
  float2 n = float2(-g0.y, g0.x);
  if (dot(n, nref) < 0.0) { n = -n; }
  uint on = 0u;
  for (uint i = 0u; i < 8u; i++) {
    float t = (float(ALONG0_LO) - c + float(ALONG0_HI - ALONG0_LO) * (float(i) + 0.5) / 8.0) * u0;
    float2 X = P0 + t * g0 + (c - 2.0) * u0 * n;
    on += (onFrame(f, X + 3.0 * u0 * n) && onFrame(f, X - 4.0 * u0 * n)) ? 1u : 0u;
  }
  JD[k] = (on < 4u) ? 1u : 0u;
  far[k] = fs;
  src[k] = ss;
  kept[k] = 0u;
}
bool trackDark(uint sd, uint j) {
  uint v = (sd * 4099u + (j >> 1u)) * 2654435761u + 40503u;
  v = v ^ (v >> 15u);
  v = v * 2246822519u;
  v = v ^ (v >> 13u);
  return ((v ^ j) & 1u) == 1u;
}
// band cell k from a corner, read from the side's start (rev 0) or its end (rev 1): a track cell, and dark
bool2 cellOf(uint k, uint B, uint sd, uint rev) {
  uint b = (rev == 1u) ? B - 1u - k : k;
  if (((b >> 2u) & 1u) == 1u) { return bool2(false); }
  return bool2(true, trackDark(sd, 4u * (b >> 3u) + (b & 3u)));
}
// the band's value at t px along the ruler's line (bprof, half h's): -1 where it was off the frame
float bandAt(uint h, float t, float step) {
  float x = t / step;
  if (x < 0.0 || x >= float(RULER_NP - 1u)) { return -1.0; }
  uint i0 = uint(x);
  float a = bprof[h * RULER_NP + i0];
  float b = bprof[h * RULER_NP + i0 + 1u];
  if (a < 0.0 || b < 0.0) { return -1.0; }
  return lerp(a, b, x - float(i0));
}
// t px along the line for d modules from the mark's centre (D 0: no vanishing point); -1 past it
float tOf(float s, float d, float D) {
  float den = (D == 0.0) ? 1.0 : 1.0 + s * d / D;
  return (den > 0.2) ? s * d / den : -1.0;
}
// the fit's module frame: slot q's outer corner (0, 0), (S, 0), (S, S), (0, S) in the quad's cyclic order (a
// homography holds either winding), its mark's centre c in on both axes, the ring's line of the side from q to its
// neighbour qn (x: the axis, 0 X = y or 1 Y = y; 2 modules in), the side's inward unit normal
float2 outerMod(uint q, float S) { return float2((q == 1u || q == 2u) ? S : 0.0, (q >= 2u) ? S : 0.0); }
float2 markMod(uint q, float S, float c) { float2 o = outerMod(q, S); return float2((o.x == 0.0) ? c : S - c, (o.y == 0.0) ? c : S - c); }
float2 ringMod(uint q, uint qn, float S) {
  float2 a = outerMod(q, S);
  float2 b = outerMod(qn, S);
  return (a.y == b.y) ? float2(1.0, (a.y == 0.0) ? 2.0 : S - 2.0) : float2(0.0, (a.x == 0.0) ? 2.0 : S - 2.0);
}
float2 inMod(float2 a, float2 b) { return (a.y == b.y) ? float2(0.0, (a.y == 0.0) ? 1.0 : -1.0) : float2((a.x == 0.0) ? 1.0 : -1.0, 0.0); }
// the fit's rows in prof (free once the passes are done): row r at 10 r, its 8 coefficients of G (G11 G12 G13 G21
// G22 G23 G31 G32; G33 1), its right side and its weight; FIT_ROWS rows, a row of weight 0 adds nothing: marks 0 to
// 7 (two a slot), line ends 8 to 15 (two a job), a ruled cell's two from 16 + 2 li (lane li, half li / 32)
void putRow(uint r, float4 r0, float4 r1, float rhs, float w) {
  uint b = 10u * r;
  prof[b] = r0.x; prof[b + 1u] = r0.y; prof[b + 2u] = r0.z; prof[b + 3u] = r0.w;
  prof[b + 4u] = r1.x; prof[b + 5u] = r1.y; prof[b + 6u] = r1.z; prof[b + 7u] = r1.w;
  prof[b + 8u] = rhs; prof[b + 9u] = w;
}
void noRows(uint r, uint n) { for (uint i = 0u; i < n; i++) { putRow(r + i, float4(0.0), float4(0.0), 0.0, 0.0); } }
// an image point p at module point u (both normalised): G1 p = u.x G3 p, G2 p = u.y G3 p
void rowPoint(uint r, float2 p, float2 u, float w) {
  putRow(r, float4(p, 1.0, 0.0), float4(0.0, 0.0, -u.x * p.x, -u.x * p.y), u.x, w);
  putRow(r + 1u, float4(0.0, 0.0, 0.0, p.x), float4(p.y, 1.0, -u.y * p.x, -u.y * p.y), u.y, w);
}
// an image point p on a module line (axis 0: X = v, 1: Y = v; normalised)
void rowLine(uint r, float2 p, float axis, float v, float w) {
  if (axis == 0.0) { putRow(r, float4(p, 1.0, 0.0), float4(0.0, 0.0, -v * p.x, -v * p.y), v, w); }
  else { putRow(r, float4(0.0, 0.0, 0.0, p.x), float4(p.y, 1.0, -v * p.x, -v * p.y), v, w); }
}
[numthreads(LANES, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  lane = lix;
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x / K;
  uint qk = wg.x % K;
  uint base = 5u * wg.x;
  if (li == 0u) {
    bool four = false;
    for (uint q = 0u; q < K; q++) { four = four || quads[5u * (f * K + q) + 4u].w == 4.0; }
    float nq = quads[base + 4u].w;
    bool dup = false;
    for (uint q = 0u; q < qk; q++) {
      uint b2 = 5u * (f * K + q);
      bool same = quads[b2 + 4u].w == nq;
      for (uint c = 0u; c < 4u; c++) { same = same && all((quads[b2 + c].xy == quads[base + c].xy)) && ((quads[b2 + c].w > 0.0) == (quads[base + c].w > 0.0)); }
      dup = dup || same;
    }
    go = (frames[f].valid == 1u && !four && !dup && (nq == 3.0 || nq == 2.0)) ? 1u : 0u;
    for (uint q = 0u; q < 4u; q++) { ri[q].store(0u); }
    rjob[0] = 4u;
    rjob[1] = 4u;
    RT[0] = float4(-2.0);
    RT[1] = float4(-2.0);
  }
  UNIFORM_LOAD(uint go0, go)
  if (go0 == 0u) { return; }
  for (uint i = li; i < P.cap; i += (2u * STATIONS)) {
    float4 r = readings[3u * (f * P.cap + i)];
    for (uint q = 0u; q < 4u; q++) {
      float4 c = quads[base + q];
      if (c.w > 0.0 && r.x == c.x && r.y == c.y && r.z > 0.0) { ri[q].max(i + 1u); }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    uint pm = 0u;
    bool okr = true;
    for (uint q = 0u; q < 4u; q++) {
      if (quads[base + q].w > 0.0) { pm |= 1u << q; okr = okr && ri[q].load() != 0u; }
    }
    uint n = uint(countbits(pm));
    if (!okr) { go = 0u; }
    else if (n == 3u) {
      uint m = uint(firstbitlow(~pm & 15u));
      uint a = (m + 1u) % 4u;
      uint b = (m + 3u) % 4u;
      uint o = (m + 2u) % 4u;
      float4 ra = readings[3u * (f * P.cap + ri[a].load() - 1u)];
      float4 rb = readings[3u * (f * P.cap + ri[b].load() - 1u)];
      float4 ro = readings[3u * (f * P.cap + ri[o].load() - 1u)];
      float2 Pm0 = ra.xy + rb.xy - ro.xy;
      float um = clamp(ra.z + rb.z - ro.z, 0.5 * min(ra.z, rb.z), 2.0 * max(ra.z, rb.z));
      float c = quads[base + a].z;
      float2 Q = 0.25 * (ra.xy + rb.xy + ro.xy + Pm0);
      for (uint sd = 0u; sd < 2u; sd++) {
        float4 r0 = (sd == 1u) ? rb : ra;
        float2 k = normalize(ro.xy - r0.xy);
        float2 e = -dir(r0.w + 0.25 * 3.14159265);
        job(sd, r0.xy, 2.0 * dot(k, e) * e - k, normalize(Pm0 - r0.xy), r0.xy - Q, r0.z, um, distance(r0.xy, Pm0), c, f, m, (sd == 1u) ? b : a);
      }
      kase = 3u; nround = 1u;
    } else if (n == 2u) {
      uint s0 = uint(firstbitlow(pm));
      uint s1 = uint(firstbithigh(pm));
      if (s1 - s0 == 2u) {
        // diagonal: a and o; from each, the sides toward a + 1 and a + 3 (a + 1 lies where the turn from a to o to it
        // is negative, GATHER's slots running with the angle about the centre): first guesses the mark's diagonal
        // and the quad's, each turned 45 degrees
        uint a = s0;
        uint o = s1;
        float4 ra = readings[3u * (f * P.cap + ri[a].load() - 1u)];
        float4 ro = readings[3u * (f * P.cap + ri[o].load() - 1u)];
        float c = quads[base + a].z;
        float2 Q = 0.5 * (ra.xy + ro.xy);
        float L = distance(ra.xy, ro.xy) * 0.70710678;
        float um = 0.5 * (ra.z + ro.z);
        float2 ka = normalize(ro.xy - ra.xy);
        float2 ko = -ka;
        float2 ea = -dir(ra.w + 0.25 * 3.14159265);
        float2 eo = -dir(ro.w + 0.25 * 3.14159265);
        float q4 = 0.25 * 3.14159265;
        job(0u, ra.xy, rot(ea, -q4), rot(ka, -q4), ra.xy - Q, ra.z, um, L, c, f, (a + 1u) % 4u, a);
        job(1u, ra.xy, rot(ea, q4), rot(ka, q4), ra.xy - Q, ra.z, um, L, c, f, (a + 3u) % 4u, a);
        job(2u, ro.xy, rot(eo, q4), rot(ko, q4), ro.xy - Q, ro.z, um, L, c, f, (a + 1u) % 4u, o);
        job(3u, ro.xy, rot(eo, -q4), rot(ko, -q4), ro.xy - Q, ro.z, um, L, c, f, (a + 3u) % 4u, o);
        kase = 4u; nround = 2u;
      } else {
        // adjacent: a then b = a + 1; the sides leaving a (toward a + 3) and b (toward b + 1), first guesses the known
        // side reflected across the mark's diagonal, and its normal on the mark's side
        uint a = (s1 - s0 == 1u) ? s0 : s1;
        uint b = (a + 1u) % 4u;
        float4 ra = readings[3u * (f * P.cap + ri[a].load() - 1u)];
        float4 rb = readings[3u * (f * P.cap + ri[b].load() - 1u)];
        float c = quads[base + a].z;
        float L = distance(ra.xy, rb.xy) * RULER_HI;
        for (uint sd = 0u; sd < 2u; sd++) {
          float4 r0 = (sd == 1u) ? rb : ra;
          float4 r1 = (sd == 1u) ? ra : rb;
          float2 k = normalize(r1.xy - r0.xy);
          float2 e = -dir(r0.w + 0.25 * 3.14159265);
          float2 pp = float2(-k.y, k.x);
          if (dot(pp, e) < 0.0) { pp = -pp; }
          job(sd, r0.xy, 2.0 * dot(k, e) * e - k, pp, r0.xy - r1.xy, r0.z, r0.z, L, c, f, (sd == 1u) ? (b + 1u) % 4u : (a + 3u) % 4u, (sd == 1u) ? b : a);
        }
        kase = 2u; nround = 1u;
      }
    } else { go = 0u; }
  }
  UNIFORM_LOAD(uint go1, go)
  if (go1 == 0u) { return; }
  UNIFORM_LOAD(uint nr, nround)
  uint side = li / STATIONS;
  uint j = li % STATIONS;
  for (uint r = 0u; r < nr; r++) {
    uint job = 2u * r + side;
    float4 ja = JA[job];
    float4 jb = JB[job];
    float4 jc = JC[job];
    bool past = JD[job] == 1u;
    float2 P0 = ja.xy;
    float u0 = jc.x;
    float um = jc.y;
    float L = jc.z;
    float c = jc.w;
    float2 n0 = float2(-ja.w, ja.z);
    if (dot(n0, jb.zw) < 0.0) { n0 = -n0; }
    float t0 = (13.0 - c) * u0;
    float t1 = L - (13.0 - c) * um;
    bool span = t1 > t0 + 8.0 * u0;
    float a0 = past ? float(ALONG1_LO) : float(ALONG0_LO);
    float a1 = past ? float(ALONG1_HI) : float(ALONG0_HI);
    for (uint g = 0u; g < 2u; g++) {
      float2 dd = (g == 1u) ? jb.xy : ja.zw;
      float2 n = float2(-dd.y, dd.x);
      if (dot(n, n0) < 0.0) { n = -n; }
      float3 hit = float3(0.0);
      if (span) {
        float t = (a0 - c + (a1 - a0) * (float(j) + 0.5) / float(STATIONS)) * u0;
        float uu = lerp(u0, um, max(t, 0.0) / L);
        hit = ringAt(f, P0 + t * dd + (c - 2.0) * uu * n, n, uu, REACH0);
      }
      st[li] = float4(hit.xy, hit.z, 0.0);
      GroupMemoryBarrierWithGroupSync();
      if (j == 0u) { fitSide(side, job, 0.5 * (u0 + um), (a1 - a0) * u0, kept[job] + 1u); }
      GroupMemoryBarrierWithGroupSync();
    }
    
    // the later passes (Finder's PASS_*): over PASS0_EXT and PASS1_EXT modules of the corner's from the first
    // station (at most the run), then the whole run
    PASS_AFTER(min(PASS0_EXT * u0, t1 - t0), PASS0_REACH)
    PASS_AFTER(min(PASS1_EXT * u0, t1 - t0), PASS1_REACH)
    PASS_AFTER((t1 - t0), PASS2_REACH)

  }
  if (li == 0u) {
    uint k = kase;
    float ring = quads[base + 4u].x;
    float S = P.modules[f2u(ring)];
    if (k == 3u) {
      // a side traced alone ruled, its vanishing point its crossing with the side parallel to it (o to the other mark)
      uint m = far[0];
      uint a = (m + 1u) % 4u;
      uint b = (m + 3u) % 4u;
      uint o = (m + 2u) % 4u;
      float2 Pa = quads[base + a].xy;
      float2 Pb = quads[base + b].xy;
      float2 Po = quads[base + o].xy;
      float c = JC[0].w;
      if ((kept[0] > 0u) != (kept[1] > 0u)) {
        uint jb0 = (kept[0] > 0u) ? 0u : 1u;
        float2 Pn = (jb0 == 0u) ? Pb : Pa;
        float2 Pk = (jb0 == 0u) ? Pa : Pb;
        rjob[0] = jb0;
        float3 V = vanish(line[jb0], float4(Po, normalize(Pn - Po)));
        RB[0] = float4(V.xy, V.z, distance(Pk, Po) / (S - 2.0 * c));
      }
    } else if (k == 2u) {
      if (kept[0] > 0u && kept[1] > 0u) {
        rjob[0] = 0u;
        rjob[1] = 1u;
        float3 V = vanish(line[0], line[1]);
        float w = distance(JA[0].xy, JA[1].xy) / (S - 2.0 * JC[0].w);
        RB[0] = float4(V.xy, V.z, w);
        RB[1] = float4(V.xy, V.z, w);
      }
    }
    // each ruler half's line: Ra the mark's centre on the line, d toward the far corner, ni inward; D the vanishing
    // point's distance along d (0 where none)
    for (uint h = 0u; h < 2u; h++) {
      uint jj = rjob[h];
      if (jj == 4u) { continue; }
      float4 ln = line[jj];
      float2 P0 = JA[jj].xy;
      float2 Ra = ln.xy + dot(P0 - ln.xy, ln.zw) * ln.zw;
      float2 d = (dot(ln.zw, JA[jj].zw) < 0.0) ? -ln.zw : ln.zw;
      float2 ni = float2(-d.y, d.x);
      if (dot(ni, JB[jj].zw) > 0.0) { ni = -ni; }
      float4 rb = RB[h];
      float D = (rb.z > 0.0) ? dot(rb.xy - Ra, d) : 0.0;
      if (abs(D) > 1e6) { D = 0.0; }
      float w = rb.w;
      float c = JC[jj].w;
      float tmax = (S - c) * w * RULER_HI * 1.5;
      RA[h] = float4(Ra, d);
      RB[h] = float4(ni, w, D);
      RC[h] = float4(c, S, max(1.0, tmax / float(RULER_NP)), 0.0);
    }
  }
  UNIFORM_LOAD(uint rj0, rjob[0])
  UNIFORM_LOAD(uint rj1, rjob[1])
  if (rj0 != 4u || rj1 != 4u) {
    uint h = side;
    bool on = (h == 0u) ? rj0 != 4u : rj1 != 4u;
    float4 ra = RA[h];
    float4 rb = RB[h];
    float4 rc = RC[h];
    float2 Ra = ra.xy;
    float2 d = ra.zw;
    float2 ni = rb.xy;
    float w = rb.z;
    float D = rb.w;
    float c = rc.x;
    float S = rc.y;
    float step = rc.z;
    uint B = f2u((S - 30.0) * 0.5);
    uint l = levelFor(w);
    // the band's depth: the darkest of RULER_NDEP rows across the line, each averaged over the first 64 cells' run
    if (on && j < RULER_NDEP) {
      float o = RULER_DEPTH0 + float(j) * RULER_DEPTH_STEP;
      float acc = 0.0;
      float cnt = 0.0;
      for (uint i = 0u; i < 64u; i++) {
        float t = w * (16.0 + 2.0 * float(i) - c);
        float fall = (D == 0.0) ? 1.0 : 1.0 - t / D;
        float2 p = Ra + t * d + o * w * fall * ni;
        if (fall > 0.2 && onFrame(f, p)) { acc += look(l, f, p); cnt += 1.0; }
      }
      dm[h * RULER_NDEP + j] = (cnt >= 16.0) ? acc / cnt : 2.0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (on && j == 0u) {
      uint jm = 0u;
      for (uint i = 1u; i < RULER_NDEP; i++) { if (dm[h * RULER_NDEP + i] < dm[h * RULER_NDEP + jm]) { jm = i; } }
      float dt = 0.0;
      if (jm > 0u && jm < (RULER_NDEP - 1u)) {
        float a = dm[h * RULER_NDEP + jm - 1u];
        float b = dm[h * RULER_NDEP + jm];
        float cc = dm[h * RULER_NDEP + jm + 1u];
        float e = a + cc - 2.0 * b;
        if (e > 0.0) { dt = clamp(0.5 * (a - cc) / e, -0.5, 0.5); }
      }
      RC[h].w = (dm[h * RULER_NDEP + jm] < 2.0) ? RULER_DEPTH0 + (float(jm) + dt) * RULER_DEPTH_STEP : -1.0;
    }
    // the first RULER_CELLS0 cells' track bits for the 8 readings (bit h: dark; 256 a track cell read from the
    // start, 512 from the end); the same for both halves (one ring)
    if (li < RULER_CELLS0) {
      uint bits = 0u;
      for (uint sd = 0u; sd < 4u; sd++) {
        bool2 cf = cellOf(li, B, sd, 0u);
        bool2 cr = cellOf(li, B, sd, 1u);
        bits |= (cf.y ? (1u << (2u * sd)) : 0u) | (cr.y ? (1u << (2u * sd + 1u)) : 0u);
        if (sd == 0u) { bits |= (cf.x ? 256u : 0u) | (cr.x ? 512u : 0u); }
      }
      pk[li] = bits;
    }
    GroupMemoryBarrierWithGroupSync();
    float depth = RC[h].w;
    bool go2 = on && depth > 0.0;
    // the band read once along the line at its depth, the depth's scale falling as 1 - t / D toward the vanishing point
    for (uint i = j; i < RULER_NP; i += STATIONS) {
      float t = float(i) * step;
      float fall = (D == 0.0) ? 1.0 : 1.0 - t / D;
      float2 p = Ra + t * d + depth * w * fall * ni;
      bprof[h * RULER_NP + i] = (go2 && fall > 0.2 && onFrame(f, p)) ? look(l, f, p) : -1.0;
    }
    GroupMemoryBarrierWithGroupSync();
    // stage 0: scale and offset over the first RULER_CELLS0 cells at RULER_STEP0C, the 4 readings h = 2 side + (read
    // from the end) the side's direction allows: a capture is never mirrored, so the quad's slots run clockwise as
    // the symbol's corners (SPEC 4.1: sides 0 top, 1 right, 2 bottom, 3 left, each counted left to right or top to
    // bottom); a side traced toward the next slot is read from its start on sides 0 and 1 and from its end on 2 and 3,
    // toward the slot before the other way round
    uint jr = (h == 0u) ? rj0 : rj1;
    uint4 hs4 = (on && far[jr] == (src[jr] + 1u) % 4u) ? uint4(0u, 2u, 5u, 7u) : uint4(1u, 3u, 4u, 6u);
    bool4 odd = ((hs4 & uint4(1u)) == uint4(1u));
    float4 best = float4(-2.0, 0.0, 0.0, 0.0);
    for (uint p = j; go2 && p < (RULER_N0 * RULER_ND); p += STATIONS) {
      float s = w * exp(RULER_LOG_LO + float(p / RULER_ND) * RULER_STEP0C);
      float delta = -RULER_DMAX + 0.5 * float(p % RULER_ND);
      float nF = 0.0; float sF = 0.0; float qF = 0.0; float nR = 0.0; float sR = 0.0; float qR = 0.0;
      float4 sxy = float4(0.0);
      float4 sy = float4(0.0);
      for (uint k = 0u; k < RULER_CELLS0; k++) {
        float t = tOf(s, 16.0 + 2.0 * float(k) + delta - c, D);
        if (t < 0.0) { continue; }
        float g = bandAt(h, t, step);
        if (g < 0.0) { continue; }
        float x = -g;
        uint bits = pk[k];
        bool tF = (bits & 256u) != 0u;
        bool tR = (bits & 512u) != 0u;
        if (tF) { nF += 1.0; sF += x; qF += x * x; }
        if (tR) { nR += 1.0; sR += x; qR += x * x; }
        float4 trk = select(odd, float4(tR ? 1.0 : 0.0), float4(tF ? 1.0 : 0.0));
        float4 y = select(((uint4(bits) >> hs4) & uint4(1u)) != uint4(0u), float4(1.0), float4(-1.0));
        sxy += trk * y * x;
        sy += trk * y;
      }
      float4 N = select(odd, float4(nR), float4(nF));
      float4 SX = select(odd, float4(sR), float4(sF));
      float4 QX = select(odd, float4(qR), float4(qF));
      float4 Nq = max(N, float4(1.0));
      float4 vx = QX - SX * SX / Nq;
      float4 vy = N - sy * sy / Nq;
      float4 r = (sxy - SX * sy / Nq) / sqrt(max(vx * vy, float4(1e-12)));
      for (uint i = 0u; i < 4u; i++) {
        if (N[i] >= 12.0 && r[i] > best.x) { best = float4(r[i], s, delta, float(hs4[i])); }
      }
    }
    rbest[li] = best;
    GroupMemoryBarrierWithGroupSync();
    if (j == 0u) {
      float4 bb = rbest[h * STATIONS];
      for (uint i = 1u; i < STATIONS; i++) { float4 v = rbest[h * STATIONS + i]; if (v.x > bb.x) { bb = v; } }
      RS[h] = bb;
    }
    GroupMemoryBarrierWithGroupSync();
    // stage 1: about stage 0's best, its reading alone, over every cell on the frame
    float4 s0b = RS[h];
    bool go3 = go2 && s0b.x > -1.0;
    uint hb = f2u(s0b.w);
    uint sdb = hb >> 1u;
    uint revb = hb & 1u;
    best = float4(-2.0, 0.0, 0.0, 0.0);
    for (uint p = j; go3 && p < (RULER_N1 * 5u); p += STATIONS) {
      float s = s0b.y * exp(-3.0 * RULER_STEP0 + float(p / 5u) * RULER_STEP1);
      float delta = s0b.z - 0.5 + 0.25 * float(p % 5u);
      float n = 0.0; float sx = 0.0; float qx = 0.0; float sxy = 0.0; float sy = 0.0;
      for (uint k = 0u; k < B; k++) {
        bool2 cl = cellOf(k, B, sdb, revb);
        if (!cl.x) { continue; }
        float t = tOf(s, 16.0 + 2.0 * float(k) + delta - c, D);
        if (t < 0.0) { continue; }
        float g = bandAt(h, t, step);
        if (g < 0.0) { continue; }
        float x = -g;
        float y = cl.y ? 1.0 : -1.0;
        n += 1.0; sx += x; qx += x * x; sxy += y * x; sy += y;
      }
      if (n >= 12.0) {
        float r = (sxy - sx * sy / n) / sqrt(max((qx - sx * sx / n) * (n - sy * sy / n), 1e-12));
        if (r > best.x) { best = float4(r, s, delta, 0.0); }
      }
    }
    rbest[li] = best;
    GroupMemoryBarrierWithGroupSync();
    if (j == 0u && go3) {
      float4 bb = rbest[h * STATIONS];
      for (uint i = 1u; i < STATIONS; i++) { float4 v = rbest[h * STATIONS + i]; if (v.x > bb.x) { bb = v; } }
      // the ruler's map for the fit: its correlation, scale, offset and vanishing distance
      if (bb.x > -1.0) { RT[h] = float4(bb.x, bb.y, bb.z, D); }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  GroupMemoryBarrierWithGroupSync();
  // the fit: every lane its rows, then 44 lanes an entry each of the normal equations (36 of the symmetric 8 x 8,
  // 8 of the right side), then one lane the solve
  float ring = quads[base + 4u].x;
  float S = P.modules[f2u(ring)];
  float c = JC[0].w;
  float um = JC[0].y;
  float hs = 0.5 * S;
  // image coordinates about the found marks' mean in half the symbol's side; modules about the centre in half it
  float2 ci = float2(0.0);
  float nm = 0.0;
  for (uint q = 0u; q < 4u; q++) { if (quads[base + q].w > 0.0) { ci += quads[base + q].xy; nm += 1.0; } }
  ci /= nm;
  float si = hs * um;
  if (li < 4u) {
    float4 mk = quads[base + li];
    if (mk.w > 0.0) { rowPoint(2u * li, (mk.xy - ci) / si, (markMod(li, S, c) - hs) / hs, FIT_MARK); } else { noRows(2u * li, 2u); }
  } else if (li < 12u) {
    uint jb = (li - 4u) / 2u;
    if (jb < 2u * nround && kept[jb] > 0u) {
      float4 ln = line[jb];
      float2 ml = ringMod(src[jb], far[jb], S);
      float te = ((li & 1u) == 0u) ? lext[jb].x : lext[jb].y;
      rowLine(li + 4u, (ln.xy + te * ln.zw - ci) / si, ml.x, (ml.y - hs) / hs, FIT_LINE * 0.5 * float(kept[jb]));
    } else { noRows(li + 4u, 1u); }
  }
  {
    uint hh = li / STATIONS;
    uint jj = rjob[hh];
    float4 rt = RT[hh];
    bool none = true;
    if (jj != 4u && rt.x > 0.0) {
      uint B = f2u((S - 30.0) * 0.5);
      uint stepR = max(1u, (B + (STATIONS - 1u)) / STATIONS);
      uint kk = (li % STATIONS) * stepR;
      if (kk < B) {
        float t = tOf(rt.y, 16.0 + 2.0 * float(kk) + rt.z - c, rt.w);
        if (t >= 0.0 && bandAt(hh, t, RC[hh].z) >= 0.0) {
          float2 O1 = outerMod(src[jj], S);
          float2 O2 = outerMod(far[jj], S);
          float2 X = RA[hh].xy + t * RA[hh].zw;
          float2 U = O1 + (16.0 + 2.0 * float(kk)) * (O2 - O1) / S + 2.0 * inMod(O1, O2);
          rowPoint(16u + 2u * li, (X - ci) / si, (U - hs) / hs, FIT_RULER_CELL * rt.x * float(stepR));
          none = false;
        }
      }
    }
    if (none) { noRows(16u + 2u * li, 2u); }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < 44u) {
    uint i = 0u;
    uint jc = 8u;
    if (li < 36u) {
      uint rem = li;
      while (rem >= 8u - i) { rem -= 8u - i; i++; }
      jc = i + rem;
    } else { i = li - 36u; }
    float acc = 0.0;
    for (uint r = 0u; r < FIT_ROWS; r++) { uint b = 10u * r; acc += prof[b + 9u] * prof[b + i] * prof[b + jc]; }
    if (jc == 8u) { NE[64u + i] = acc; } else { NE[8u * i + jc] = acc; NE[8u * jc + i] = acc; }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    // the normal equations solved in place by elimination with partial pivoting; a pivot under FIT_PIVOT of the
    // largest diagonal leaves the model free in some direction: nothing completed
    float dmax = 0.0;
    for (uint i = 0u; i < 8u; i++) { dmax = max(dmax, NE[9u * i]); }
    bool ok = dmax > 0.0;
    for (uint col = 0u; ok && col < 8u; col++) {
      uint pr = col;
      for (uint r = col + 1u; r < 8u; r++) { if (abs(NE[8u * r + col]) > abs(NE[8u * pr + col])) { pr = r; } }
      if (abs(NE[8u * pr + col]) < FIT_PIVOT * dmax) { ok = false; break; }
      if (pr != col) {
        for (uint cc = 0u; cc < 8u; cc++) { float t = NE[8u * col + cc]; NE[8u * col + cc] = NE[8u * pr + cc]; NE[8u * pr + cc] = t; }
        float t = NE[64u + col]; NE[64u + col] = NE[64u + pr]; NE[64u + pr] = t;
      }
      float pv = NE[8u * col + col];
      for (uint r = 0u; r < 8u; r++) {
        if (r == col) { continue; }
        float fr = NE[8u * r + col] / pv;
        for (uint cc = col; cc < 8u; cc++) { NE[8u * r + cc] -= fr * NE[8u * col + cc]; }
        NE[64u + r] -= fr * NE[64u + col];
      }
    }
    if (ok) {
      // the homography's rows (h0 h1 h2 / h3 h4 h5 / h6 h7 1): Slang's float3x3 is filled row by row
      float3x3 G = float3x3(NE[64] / NE[0], NE[65] / NE[9], NE[66] / NE[18], NE[67] / NE[27], NE[68] / NE[36], NE[69] / NE[45], NE[70] / NE[54], NE[71] / NE[63], 1.0);
      float3x3 Hm = inverse3(G);
      float3 v0 = mul(Hm, float3((markMod(0u, S, c) - hs) / hs, 1.0));
      float3 v1 = mul(Hm, float3((markMod(1u, S, c) - hs) / hs, 1.0));
      float3 v2 = mul(Hm, float3((markMod(2u, S, c) - hs) / hs, 1.0));
      float3 v3 = mul(Hm, float3((markMod(3u, S, c) - hs) / hs, 1.0));
      ok = abs(v0.z) > 1e-6 && abs(v1.z) > 1e-6 && abs(v2.z) > 1e-6 && abs(v3.z) > 1e-6;
      if (ok) {
        float2 c0 = ci + si * v0.xy / v0.z;
        float2 c1 = ci + si * v1.xy / v1.z;
        float2 c2 = ci + si * v2.xy / v2.z;
        float2 c3 = ci + si * v3.xy / v3.z;
        if (convex(c0, c1, c2, c3)) {
          float4 m0 = quads[base];
          float4 m1 = quads[base + 1u];
          float4 m2 = quads[base + 2u];
          float4 m3 = quads[base + 3u];
          // the found corners kept, the missing ones the model's
          quads[base] = (m0.w > 0.0) ? m0 : float4(c0, c, um);
          quads[base + 1u] = (m1.w > 0.0) ? m1 : float4(c1, c, um);
          quads[base + 2u] = (m2.w > 0.0) ? m2 : float4(c2, c, um);
          quads[base + 3u] = (m3.w > 0.0) ? m3 : float4(c3, c, um);
          float4 qm = quads[base + 4u];
          float uq = 0.25 * (distance(c0, c1) + distance(c1, c2) + distance(c2, c3) + distance(c3, c0)) / (P.modules[f2u(qm.x)] - 2.0 * c);
          quads[base + 4u] = float4(qm.x, uq, qm.z, 5.0);
        }
      }
    }
  }
}
