// The anchors in GLSL: the WGSL anchors.mjs (archived) statement for statement (its comments say what each stage does, SPEC 6.13):
// ANCH_FIND (a workgroup of 128 an (anchor, frame): the squares found by normalised correlation from integral images,
// coarse then fine), ANCH_FIT (the knots, the fill, the squares' shading, the record), and anchRefitSource: ANCH_REFIT
// (the lens about the axis over the border and the anchors, robust over the border, the sides' curves) and
// ANCH_PREDICT (the fit over the anchors alone, the border's places for its second search).
// ai: Bindings, workgroup sizes and layouts are the WGSL's. Workgroup memory is not zeroed but where the WGSL reads it
// ai: before writing it: ANCH_REFIT's three counters (wcnt, wnba, wchg), whose atomics start from WGSL's zero, are
// ai: null-initialised (GL_EXT_null_initializer: the same OpConstantNull initializer naga writes for every workgroup
// ai: variable).
import { VERSION, SAMPLERLESS, NULL_INIT, ATOMICS, DIMS, CONVERT, MAP, LUM0, lit, zeros, uniformLoad, STORAGE_BARRIER, wgAdd, wgLoad, wgStore } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { LATTICE_MAX, ANCH_SIDE, ANCH_K, ANCH_REC, ANCH_SURF, ANCH_GAIN } from "../plan/back_transform.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { NODES_MAX, RESID, WMIN, LENS_P } from "../plan/register.mjs";
import { ANCH_TS, ANCH_PEAK, ANCH_ROUNDS, ANCH_FILL, ANCH_BAND2, SW_T, SW_R, SW_LAG, SW_MIN, SW_BINS } from "../plan/anchors.mjs";
import { lensNode, lensAccumulate, lensModel, lensNormalSolve, SIDE_FITS, SIDE_DECL, gaussJordan } from "./register.mjs";

const RR = ANCH_TS, TC = ANCH_TS / 2, RC = RR / 2, WC = TC + 2 * RC, W1C = WC + 1, OC = 2 * RC + 1;
const FD = 3, WF = ANCH_TS + 2 * FD, W1F = WF + 1, OF = 2 * FD + 1;
const pattern = (ts) => {
  const dist = (t) => Math.abs((t + 0.5) / (ts / 2) - 1);
  const span1 = (lim) => { const q = [...Array(ts).keys()].filter((t) => dist(t) < lim); return [q[0], q.at(-1)]; };
  const [i25, j25] = span1(0.25), [i6, j6] = span1(0.6);
  const dark = (j6 - i6 + 1) ** 2 - (j25 - i25 + 1) ** 2, light = ts * ts - dark;
  const tm = (light - dark) / (ts * ts);
  return { ts, i25, j25, i6, j6, tm, tn: Math.fround(Math.sqrt(light * (1 - tm) ** 2 + dark * (1 + tm) ** 2)) };
};

// ai: What both kernels bind first, 0 to 6 (the WGSL anchors.mjs (archived) HEAD), and the final map at a module point.
const HEAD = DIMS + CONVERT + MAP + /* glsl */ `
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 2, std430) readonly buffer Maps { float maps[][16]; };
layout(set = 0, binding = 3, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 4, std430) readonly buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 5, std140) uniform Lattice { uvec4 at[${RING_COUNT}]; vec4 lat[${(RING_COUNT * LATTICE_MAX) / 4}]; } LT;
layout(set = 0, binding = 6, std140) uniform Anchors { vec4 k[${3 * RING_COUNT}]; vec4 g[${9 * RING_COUNT}]; vec4 s[${RING_COUNT}]; } A;
float latAt(uint i) { return LT.lat[i >> 2u][i & 3u]; }
float knot(uint r, uint i) { return A.k[3u * r + (i >> 2u)][i & 3u]; }
// ai: ring r's grid a side (3, 5, 7) and the anchors its geometry takes (plan/anchors.mjs anchMin)
uint gridOf(uint r) { return f2u(A.s[r].x); }
uint gridMin(uint r) { return f2u(A.s[r].y); }
` + COONS + LUM0 + /* glsl */ `
vec2 mapAt(uint f, float M[16], Lens lz, uint first, uint nn, vec2 m) {
  vec2 p = toImage(M, m);
  return p + lensAt(lz, p) + coons(f, first, nn, m);
}
`;

const AL = 128;
const box = (I) => (x, y, sz, w1) => `(${I}[(${y} + ${sz}) * ${w1} + ${x} + ${sz}] - ${I}[${y} * ${w1} + ${x} + ${sz}] - ${I}[(${y} + ${sz}) * ${w1} + ${x}] + ${I}[${y} * ${w1} + ${x}])`;
const scoreFn = (name, I1, I2, w1, { ts, i25, j25, i6, j6, tm, tn }) => /* glsl */ `
float ${name}(uint px, uint py) {
  float s1 = ${box(I1)("px", "py", `${ts}u`, `${w1}u`)};
  float s2 = ${box(I2)("px", "py", `${ts}u`, `${w1}u`)};
  float vr = s2 - s1 * s1 / ${lit(ts * ts)};
  float s25 = ${box(I1)(`(px + ${i25}u)`, `(py + ${i25}u)`, `${j25 - i25 + 1}u`, `${w1}u`)};
  float s6 = ${box(I1)(`(px + ${i6}u)`, `(py + ${i6}u)`, `${j6 - i6 + 1}u`, `${w1}u`)};
  float st = 2.0 * (s25 - s6) + s1 * ${lit(1 - tm)};
  float sc = st / (sqrt(vr) * ${lit(tn)});
  return vr > VAR_MIN ? sc : 0.0;
}`;
const integral = (I1, I2, w, w1) => /* glsl */ `
  if (li < ${2 * w}u) {
    uint r = (li % ${w}u + 1u) * ${w1}u;
    float a = 0.0;
    if (li < ${w}u) { for (uint x = 1u; x <= ${w}u; x++) { a += ${I1}[r + x]; ${I1}[r + x] = a; } }
    else { for (uint x = 1u; x <= ${w}u; x++) { a += ${I2}[r + x]; ${I2}[r + x] = a; } }
  }
  barrier();
  if (li < ${2 * w}u) {
    uint c = li % ${w}u + 1u;
    float a = 0.0;
    if (li < ${w}u) { for (uint y = 1u; y <= ${w}u; y++) { a += ${I1}[y * ${w1}u + c]; ${I1}[y * ${w1}u + c] = a; } }
    else { for (uint y = 1u; y <= ${w}u; y++) { a += ${I2}[y * ${w1}u + c]; ${I2}[y * ${w1}u + c] = a; } }
  }
  barrier();`;
// ai: area (the fine window, 2026-11-07): where a template step spans more than 2 image px (ms > 1: a square close up,
// ai: 4 to 8 modules at 6 to 12 px a module, 24 to 96 px, its 16 points 1.5 to 6 px apart), each point
// ai: is the mean of ms x ms bilinear reads spread over its own cell (ex, ey: a step along each axis in image px at the
// ai: square's middle), so the sharp pattern is not aliased by reading single pixels a step apart (read so, squares of
// ai: 8 modules painted at 8 to 12 px a module were located 1 to 1.5 px off, which the mesh then bent the picture by); at
// ai: 2 px or less a single read as before
const samples = (I1, I2, w, w1, area = false) => /* glsl */ `
  for (uint t = li; t < ${w * w}u; t += ${AL}u) {
    uint wx = t % ${w}u;
    uint wy = t / ${w}u;
    vec2 p0 = toImage(M, vec2(wcx[wx], wry[wy]));
    vec2 pq = p0 + lensAt(lz, p0) + coonsMix(wcu[wx], wrv[wy], wct[wx], wcb[wx], wrl[wy], wrr[wy], cn);
    float v;
    ${area ? `if (ms == 1u) { v = bilin0(f, pq, fw, fh); } else {
      float sv = 0.0;
      for (uint b = 0u; b < ms; b++) for (uint c = 0u; c < ms; c++) {
        sv += bilin0(f, pq + ex * ((float(c) + 0.5) / float(ms) - 0.5) + ey * ((float(b) + 0.5) / float(ms) - 0.5), fw, fh);
      }
      v = sv / float(ms * ms);
    }` : `v = bilin0(f, pq, fw, fh);`}
    uint o = (wy + 1u) * ${w1}u + wx + 1u;
    ${I1}[o] = v;
    ${I2}[o] = v * v;
  }
  barrier();`;
const coords = (w, lo, scale) => /* glsl */ `
  for (uint t = li; t < ${2 * w}u; t += ${AL}u) {
    if (t < ${w}u) {
      float mx = x0 + (float(${lo.x}) + ${lit(scale)} * float(t) + ${lit(scale / 2)}) * h;
      wcx[t] = mx;
      wcu[t] = coonsT(mx, first, nn);
      wct[t] = curve(f, 0u, mx, first, nn);
      wcb[t] = curve(f, 2u, mx, first, nn);
    } else {
      uint s = t - ${w}u;
      float my = y0 + (float(${lo.y}) + ${lit(scale)} * float(s) + ${lit(scale / 2)}) * h;
      wry[s] = my;
      wrv[s] = coonsT(my, first, nn);
      wrl[s] = curve(f, 3u, my, first, nn);
      wrr[s] = curve(f, 1u, my, first, nn);
    }
  }`;
const pack = (s, o) => `(f2u(clamp(${s} + 1.0, 0.0, 1.999) * 1048575.0) << 11u) | (2047u - ${o})`;
export const ANCH_FIND = VERSION + SAMPLERLESS + HEAD + /* glsl */ `
layout(local_size_x = ${AL}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 7, std430) buffer Anch { vec4 anch[]; };
const uint TS = ${ANCH_TS}u;
const int RR = ${RR};
const int FD = ${FD};
const float VAR_MIN = 1e-6;
shared float IC1[${W1C * W1C}];
shared float IC2[${W1C * W1C}];
shared float IF1[${W1F * W1F}];
shared float IF2[${W1F * W1F}];
shared float CF[${OF * OF}];
shared float wcx[${WC}];
shared float wcu[${WC}];
shared vec2 wct[${WC}];
shared vec2 wcb[${WC}];
shared float wry[${WC}];
shared float wrv[${WC}];
shared vec2 wrl[${WC}];
shared vec2 wrr[${WC}];
shared uint bestC;
shared uint bestF;
${scoreFn("scoreC", "IC1", "IC2", W1C, pattern(TC))}
${scoreFn("scoreF", "IF1", "IF2", W1F, pattern(ANCH_TS))}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.y;
  uint k = wg.x;
  if (k >= ${ANCH_K}u) { return; }
  Frame F = frames[f];
  uvec4 pick = sel[f];
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { anch[${ANCH_K}u * f + k] = vec4(0.0); }
    return;
  }
  uint rgi = pick.x;
  uint S = gridOf(rgi);
  if (k % ${ANCH_SIDE}u >= S || k / ${ANCH_SIDE}u >= S) {
    if (li == 0u) { anch[${ANCH_K}u * f + k] = vec4(0.0); }
    return;
  }
  float a = A.k[3u * rgi + 2u].y;
  float h = a / float(TS);
  int rr = RR;
  float x0 = knot(rgi, 1u + k % ${ANCH_SIDE}u) - 0.5 * a;
  float y0 = knot(rgi, 1u + k / ${ANCH_SIDE}u) - 0.5 * a;
  float M[16] = maps[f];
  Lens lz = lensOf(f);
  uvec4 at = LT.at[rgi];
  uint first = at.x;
  uint nn = at.y;
  Corners cn = corners(f, nn);
  int fw = int(F.w);
  int fh = int(F.h);
  ${coords(WC, { x: -RR, y: -RR }, 2)}
  for (uint t = li; t < ${W1C}u; t += ${AL}u) { IC1[t] = 0.0; IC2[t] = 0.0; IC1[t * ${W1C}u] = 0.0; IC2[t * ${W1C}u] = 0.0; }
  for (uint t = li; t < ${W1F}u; t += ${AL}u) { IF1[t] = 0.0; IF2[t] = 0.0; IF1[t * ${W1F}u] = 0.0; IF2[t * ${W1F}u] = 0.0; }
  if (li == 0u) { bestC = 0u; bestF = 0u; }
  barrier();
  ${samples("IC1", "IC2", WC, W1C)}
  ${integral("IC1", "IC2", WC, W1C)}
  for (uint o = li; o < ${OC * OC}u; o += ${AL}u) { atomicMax(bestC, ${pack("scoreC(o % " + OC + "u, o / " + OC + "u)", "o")}); }
  barrier();
  uint oc = 2047u - (bestC & 2047u);
  int cx = clamp(2 * (int(oc % ${OC}u) - ${RC}), FD - rr, rr - FD);
  int cy = clamp(2 * (int(oc / ${OC}u) - ${RC}), FD - rr, rr - FD);
  ${coords(WF, { x: "cx - FD", y: "cy - FD" }, 1)}
  // ai: a fine step in image px at the square's middle, and the reads a point's side takes (area, above)
  vec2 xm = vec2(x0 + 0.5 * a, y0 + 0.5 * a);
  vec2 pm = mapAt(f, M, lz, first, nn, xm);
  vec2 ex = mapAt(f, M, lz, first, nn, xm + vec2(h, 0.0)) - pm;
  vec2 ey = mapAt(f, M, lz, first, nn, xm + vec2(0.0, h)) - pm;
  uint ms = clamp(f2u(ceil(max(length(ex), length(ey)) * 0.5)), 1u, 4u);
  barrier();
  ${samples("IF1", "IF2", WF, W1F, true)}
  ${integral("IF1", "IF2", WF, W1F)}
  for (uint o = li; o < ${OF * OF}u; o += ${AL}u) {
    float s = scoreF(o % ${OF}u, o / ${OF}u);
    CF[o] = s;
    atomicMax(bestF, ${pack("s", "o")});
  }
  barrier();
  if (li == 0u) {
    uint po = 2047u - (bestF & 2047u);
    float pk = CF[po];
    int px = int(po % ${OF}u);
    int py = int(po / ${OF}u);
    int bx = cx - FD + px;
    int by = cy - FD + py;
    float sx = 0.0;
    float sy = 0.0;
    if (px > 0 && px < 2 * FD) { float cl = CF[po - 1u]; float cr = CF[po + 1u]; float d = cl - 2.0 * pk + cr; if (d < 0.0) { sx = 0.5 * (cl - cr) / d; } }
    if (py > 0 && py < 2 * FD) { float cl = CF[po - ${OF}u]; float cr = CF[po + ${OF}u]; float d = cl - 2.0 * pk + cr; if (d < 0.0) { sy = 0.5 * (cl - cr) / d; } }
    bool ok = pk >= ${lit(ANCH_PEAK)} && bx > -rr && bx < rr && by > -rr && by < rr;
    vec2 X = vec2(x0 + 0.5 * a, y0 + 0.5 * a);
    vec2 D = vec2((float(bx) + sx) * h, (float(by) + sy) * h);
    vec2 rk = mapAt(f, M, lz, first, nn, X + D) - mapAt(f, M, lz, first, nn, X);
    anch[${ANCH_K}u * f + k] = vec4(rk, pk, ok ? 1.0 : 0.0);
  }
}
`;

export const ANCH_FIT = VERSION + SAMPLERLESS + NULL_INIT + ATOMICS + HEAD + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct Result { float r[32]; };
layout(set = 0, binding = 7, std430) readonly buffer Anch { vec4 anch[]; };
layout(set = 0, binding = 8, std430) buffer Arec { vec4 arec[]; };
layout(set = 0, binding = 9, std430) buffer Results { Result result[]; };
const uint NK = ${ANCH_K}u;
const uint SIDE = ${ANCH_SIDE}u;
shared vec2 kv[${ANCH_K}];
shared uint kn[${ANCH_K}];
shared uint kn2[${ANCH_K}];
shared float red[${6 * 64}];
shared uint wfound;
// ai: the gain's tiles (SW_T^2): each its swing, its centre (u, w), its state (0 kept, 1 dropped, 2 too few pairs) and
// ai: residual; the fit's sums a lane, its coefficients, the round's word (go on), the median's bracket (its low end, its
// ai: width, the rank sought within it) and the histogram
shared float swA[${SW_T * SW_T}];
shared vec2 swC[${SW_T * SW_T}];
shared uint swD[${SW_T * SW_T}];
shared float swR[${SW_T * SW_T}];
shared float swS[${28 * 64}];
shared float swAl[6];
shared uint swGo;
shared vec4 swLH;
shared uint swH[${SW_BINS}] = {};
// ai: a module coordinate within the band of one of the ring's S anchors' centres (the fill's band, fillBand's rule)
bool inBand(uint r, float m, float aa, uint S) {
  bool hit = false;
  for (uint i = 1u; i <= S; i++) { hit = hit || abs(knot(r, i) - m) <= ${lit(ANCH_BAND2 / 2)} * aa; }
  return hit;
}
// ai: a knot's found 4-neighbours within the ring's S x S grid (indices j SIDE + i; those past it never read)
vec3 around(uint q, uint S) {
  uint i = q % SIDE;
  uint j = q / SIDE;
  vec2 s = vec2(0.0);
  float c = 0.0;
  if (i > 0u && kn[q - 1u] == 1u) { s += kv[q - 1u]; c += 1.0; }
  if (i + 1u < S && kn[q + 1u] == 1u) { s += kv[q + 1u]; c += 1.0; }
  if (j > 0u && kn[q - SIDE] == 1u) { s += kv[q - SIDE]; c += 1.0; }
  if (j + 1u < S && kn[q + SIDE] == 1u) { s += kv[q + SIDE]; c += 1.0; }
  return vec3(s, c);
}
// ai: the fill's points (plan/anchors.mjs ANCH_FILL): whether point p of an axis lies within the band of one of the
// ai: ring's S anchors' centres, in integers of 2 ANCH_FILL times a module (an anchor's centre, the margin, the span and
// ai: a are whole)
bool fillBand(uint r, int p, int sp, int mg, int aa, uint S) {
  bool hit = false;
  for (uint i = 1u; i <= S; i++) {
    int d = ${2 * ANCH_FILL} * (f2i(knot(r, i)) - mg) - sp * (2 * p + 1);
    hit = hit || abs(d) <= ${ANCH_BAND2 * ANCH_FILL} * aa;
  }
  return hit;
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  Frame F = frames[f];
  uvec4 pick = sel[f];
  uint ab = f * ${ANCH_REC}u;
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { arec[ab] = vec4(0.0); result[f].r[31] = 0.0; }
    return;
  }
  uint rgi = pick.x;
  uint S = gridOf(rgi);
  if (li < NK) {
    vec4 e = anch[NK * f + li];
    bool ok = e.w > 0.5;
    kv[li] = ok ? e.xy : vec2(0.0);
    kn[li] = ok ? 1u : 0u;
  }
  barrier();
  if (li == 0u) {
    uint c = 0u;
    for (uint q = 0u; q < NK; q++) { c += kn[q]; }
    wfound = c;
  }
  ${uniformLoad("uint found", "wfound")}
  if (li == 0u) { result[f].r[31] = float(found); }
  if (found < gridMin(rgi) && li < NK) { kv[li] = vec2(0.0); kn[li] = 1u; }
  for (uint rd = 0u; rd < ${ANCH_ROUNDS}u; rd++) {
    barrier();
    if (li < NK) {
      uint now = kn[li];
      if (now == 0u) {
        vec3 nb = around(li, S);
        if (nb.z > 0.0) { kv[li] = nb.xy / nb.z; now = 1u; }
      }
      kn2[li] = now;
    }
    barrier();
    if (li < NK) { kn[li] = kn2[li]; }
  }
  vec4 geo = A.k[3u * rgi + 2u];
  float span = geo.z;
  float margin = geo.w;
  float M[16] = maps[f];
  Lens lz = lensOf(f);
  uvec4 at = LT.at[rgi];
  // ai: the squares' fill (SPEC 6.13; plan/anchors.mjs ANCH_FILL): the shading's least-squares surface over the
  // ai: picture's points in no square, its normal matrix's inverse the ring's (A.g, the C's hole_inv)
  int sp = f2i(span);
  int mg = f2i(margin);
  int aa = f2i(geo.y);
  float s0 = 0.0;
  float s1 = 0.0;
  float s2 = 0.0;
  float s3 = 0.0;
  float s4 = 0.0;
  float s5 = 0.0;
  for (uint p = li; p < ${ANCH_FILL * ANCH_FILL}u; p += 64u) {
    int px = int(p % ${ANCH_FILL}u);
    int py = int(p / ${ANCH_FILL}u);
    if (fillBand(rgi, px, sp, mg, aa, S) && fillBand(rgi, py, sp, mg, aa, S)) { continue; }
    float ux = float(2 * px + 1 - ${ANCH_FILL}) / ${lit(ANCH_FILL)};
    float uy = float(2 * py + 1 - ${ANCH_FILL}) / ${lit(ANCH_FILL)};
    vec2 m = vec2(margin + 0.5 * span * (ux + 1.0), margin + 0.5 * span * (uy + 1.0));
    float v = bilin0(f, mapAt(f, M, lz, at.x, at.y, m), int(F.w), int(F.h));
    s0 += v;
    s1 += ux * v;
    s2 += ux * ux * v;
    s3 += uy * v;
    s4 += uy * uy * v;
    s5 += ux * uy * v;
  }
  red[li] = s0;
  red[64u + li] = s1;
  red[128u + li] = s2;
  red[192u + li] = s3;
  red[256u + li] = s4;
  red[320u + li] = s5;
  barrier();
  if (li < 6u) {
    float s = 0.0;
    for (uint q = 0u; q < 64u; q++) { s += red[64u * li + q]; }
    red[64u * li] = s;
  }
  barrier();
  // ai: The gain (SPEC 6.10, the C's lizard_finish_ext): the signal's swing in SW_T x SW_T tiles of the picture, the root of
  // ai: half the mean squared difference of neighbours on runs of reads span / SW_LAG modules apart along SW_R rows a tile,
  // ai: outside the squares' bands (plan/anchors.mjs); the fill's basis fitted to the tiles' swings at their centres,
  // ai: robustly (a tile past three robust spreads dropped and the surface refitted, three rounds; the median from two
  // ai: histograms); pass 1 multiplies each sample by the surface's mean over its value (glsl/back_transform.mjs). A
  // ai: falloff scales the swing, an added light does not.
  {
    const uint RUN = ${SW_LAG / SW_T}u;
    float lagm = span / ${lit(SW_LAG)};
    float tw = span / ${lit(SW_T)};
    float aaf = float(aa);
    for (uint t = li; t < ${SW_T * SW_T}u; t += 64u) {
      uint tx = t % ${SW_T}u;
      uint ty = t / ${SW_T}u;
      float D = 0.0;
      uint cnt = 0u;
      for (uint r = 0u; r < ${SW_R}u; r++) {
        float my = margin + tw * (float(ty) + (float(r) + 0.5) / ${lit(SW_R)});
        float mx0 = margin + tw * float(tx) + 0.5 * lagm;
        bool yb = inBand(rgi, my, aaf, S);
        // ai: the run's points through the map at its ends and middle, quadratic between (P(s) = P0 + s (Bq + s Cq), s in [0, 1])
        vec2 P0 = mapAt(f, M, lz, at.x, at.y, vec2(mx0, my));
        vec2 Pm = mapAt(f, M, lz, at.x, at.y, vec2(mx0 + 0.5 * float(RUN - 1u) * lagm, my));
        vec2 P1 = mapAt(f, M, lz, at.x, at.y, vec2(mx0 + float(RUN - 1u) * lagm, my));
        vec2 Bq = 4.0 * Pm - 3.0 * P0 - P1;
        vec2 Cq = 2.0 * (P0 + P1) - 4.0 * Pm;
        float vp = bilin0(f, P0, int(F.w), int(F.h));
        bool bp = yb && inBand(rgi, mx0, aaf, S);
        for (uint j = 1u; j < RUN; j++) {
          float sj = float(j) / float(RUN - 1u);
          float v = bilin0(f, P0 + sj * (Bq + sj * Cq), int(F.w), int(F.h));
          bool bj = yb && inBand(rgi, mx0 + float(j) * lagm, aaf, S);
          if (!(bp || bj)) { D += (v - vp) * (v - vp); cnt++; }
          vp = v;
          bp = bj;
        }
      }
      swA[t] = cnt >= ${SW_MIN}u ? sqrt(D / (2.0 * float(cnt))) : 0.0;
      swD[t] = cnt >= ${SW_MIN}u ? 0u : 2u;
      swC[t] = vec2((2.0 * float(tx) + 1.0) / ${lit(SW_T)} - 1.0, (2.0 * float(ty) + 1.0) / ${lit(SW_T)} - 1.0);
    }
    if (li == 0u) { swAl[0] = 1.0; swAl[1] = 0.0; swAl[2] = 0.0; swAl[3] = 0.0; swAl[4] = 0.0; swAl[5] = 0.0; }
    for (uint rd = 0u; rd < 4u; rd++) {
      barrier();
      float q[28] = float[28](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      for (uint t = li; t < ${SW_T * SW_T}u; t += 64u) {
        if (swD[t] != 0u) { continue; }
        vec2 c = swC[t];
        float av = swA[t];
        float b0 = 1.0;
        float b1 = c.x;
        float b2 = c.x * c.x;
        float b3 = c.y;
        float b4 = c.y * c.y;
        float b5 = c.x * c.y;
        q[0] += b0 * b0;
        q[1] += b0 * b1;
        q[2] += b0 * b2;
        q[3] += b0 * b3;
        q[4] += b0 * b4;
        q[5] += b0 * b5;
        q[6] += b1 * b1;
        q[7] += b1 * b2;
        q[8] += b1 * b3;
        q[9] += b1 * b4;
        q[10] += b1 * b5;
        q[11] += b2 * b2;
        q[12] += b2 * b3;
        q[13] += b2 * b4;
        q[14] += b2 * b5;
        q[15] += b3 * b3;
        q[16] += b3 * b4;
        q[17] += b3 * b5;
        q[18] += b4 * b4;
        q[19] += b4 * b5;
        q[20] += b5 * b5;
        q[21] += b0 * av;
        q[22] += b1 * av;
        q[23] += b2 * av;
        q[24] += b3 * av;
        q[25] += b4 * av;
        q[26] += b5 * av;
        q[27] += 1.0;
      }
      swS[0u + li] = q[0];
      swS[64u + li] = q[1];
      swS[128u + li] = q[2];
      swS[192u + li] = q[3];
      swS[256u + li] = q[4];
      swS[320u + li] = q[5];
      swS[384u + li] = q[6];
      swS[448u + li] = q[7];
      swS[512u + li] = q[8];
      swS[576u + li] = q[9];
      swS[640u + li] = q[10];
      swS[704u + li] = q[11];
      swS[768u + li] = q[12];
      swS[832u + li] = q[13];
      swS[896u + li] = q[14];
      swS[960u + li] = q[15];
      swS[1024u + li] = q[16];
      swS[1088u + li] = q[17];
      swS[1152u + li] = q[18];
      swS[1216u + li] = q[19];
      swS[1280u + li] = q[20];
      swS[1344u + li] = q[21];
      swS[1408u + li] = q[22];
      swS[1472u + li] = q[23];
      swS[1536u + li] = q[24];
      swS[1600u + li] = q[25];
      swS[1664u + li] = q[26];
      swS[1728u + li] = q[27];
      barrier();
      if (li < 28u) {
        float sm = 0.0;
        for (uint i = 0u; i < 64u; i++) { sm += swS[li * 64u + i]; }
        swS[li * 64u] = sm;
      }
      barrier();
      if (li == 0u) {
        float N[6][7];
        N[0][0] = swS[0];
        N[0][1] = swS[64];
        N[0][2] = swS[128];
        N[0][3] = swS[192];
        N[0][4] = swS[256];
        N[0][5] = swS[320];
        N[0][6] = swS[1344];
        N[1][0] = swS[64];
        N[1][1] = swS[384];
        N[1][2] = swS[448];
        N[1][3] = swS[512];
        N[1][4] = swS[576];
        N[1][5] = swS[640];
        N[1][6] = swS[1408];
        N[2][0] = swS[128];
        N[2][1] = swS[448];
        N[2][2] = swS[704];
        N[2][3] = swS[768];
        N[2][4] = swS[832];
        N[2][5] = swS[896];
        N[2][6] = swS[1472];
        N[3][0] = swS[192];
        N[3][1] = swS[512];
        N[3][2] = swS[768];
        N[3][3] = swS[960];
        N[3][4] = swS[1024];
        N[3][5] = swS[1088];
        N[3][6] = swS[1536];
        N[4][0] = swS[256];
        N[4][1] = swS[576];
        N[4][2] = swS[832];
        N[4][3] = swS[1024];
        N[4][4] = swS[1152];
        N[4][5] = swS[1216];
        N[4][6] = swS[1600];
        N[5][0] = swS[320];
        N[5][1] = swS[640];
        N[5][2] = swS[896];
        N[5][3] = swS[1088];
        N[5][4] = swS[1216];
        N[5][5] = swS[1280];
        N[5][6] = swS[1664];
        bool ok = swS[${27 * 64}] >= 12.0;
        ${gaussJordan("N", 6, 7, "1e-12", "ok")}
        float a0 = N[0][6] / N[0][0];
        float a1 = N[1][6] / N[1][1];
        float a2 = N[2][6] / N[2][2];
        float a3 = N[3][6] / N[3][3];
        float a4 = N[4][6] / N[4][4];
        float a5 = N[5][6] / N[5][5];
        ok = ok && a0 > 0.0 && a0 + (a2 + a4) / 3.0 > 0.0;
        if (ok) { swAl[0] = a0; swAl[1] = a1; swAl[2] = a2; swAl[3] = a3; swAl[4] = a4; swAl[5] = a5; }
        swGo = (ok && rd < 3u) ? 1u : 0u;
      }
      ${uniformLoad("uint go", "swGo")}
      if (go == 0u) { break; }
      // ai: the residuals, the median of the kept ones (the C's sorted[kept / 2]) from two histograms, the drops
      float rmax = 0.0;
      float kept = 0.0;
      for (uint t = li; t < ${SW_T * SW_T}u; t += 64u) {
        vec2 c = swC[t];
        float fv = swAl[0] + c.x * (swAl[1] + c.x * swAl[2]) + c.y * (swAl[3] + swAl[5] * c.x + c.y * swAl[4]);
        swR[t] = abs(swA[t] - fv);
        if (swD[t] == 0u) { rmax = max(rmax, swR[t]); kept += 1.0; }
      }
      swS[li] = rmax;
      swS[64u + li] = kept;
      barrier();
      if (li == 0u) {
        float mx = 0.0;
        float kk = 0.0;
        for (uint i = 0u; i < 64u; i++) { mx = max(mx, swS[i]); kk += swS[64u + i]; }
        // ai: the bracket [0, the largest residual], a hair wider so the largest falls in the last bin; the rank sought
        swLH = vec4(0.0, max(mx, 1e-30) * ${lit(1 + 1 / 1024)}, floor(0.5 * kk) + 1.0, 0.0);
      }
      for (uint lv = 0u; lv < 2u; lv++) {
        ${uniformLoad("vec4 lh", "swLH")}
        for (uint t = li; t < ${SW_T * SW_T}u; t += 64u) {
          float q = (swR[t] - lh.x) * ${lit(SW_BINS)} / lh.y;
          if (swD[t] == 0u && q >= 0.0 && q < ${lit(SW_BINS)}) { ${wgAdd(`swH[min(uint(q), ${SW_BINS - 1}u)]`, "1u")}; }
        }
        barrier();
        if (li == 0u) {
          float below = 0.0;
          uint b = 0u;
          for (uint i = 0u; i < ${SW_BINS}u; i++) {
            float h = float(${wgLoad("swH[i]")});
            ${wgStore("swH[i]", "0u")};
            if (b == i && below + h < lh.z) { below += h; b = i + 1u; }
          }
          b = min(b, ${SW_BINS - 1}u);
          float w = lh.y / ${lit(SW_BINS)};
          swLH = vec4(lh.x + float(b) * w, w, lh.z - below, 0.0);
        }
      }
      ${uniformLoad("vec4 med", "swLH")}
      float tau = ${lit(3 * 1.4826)} * (med.x + 0.5 * med.y);
      float changed = 0.0;
      for (uint t = li; t < ${SW_T * SW_T}u; t += 64u) {
        if (swD[t] == 2u) { continue; }
        uint bad = swR[t] > tau ? 1u : 0u;
        changed += bad != swD[t] ? 1.0 : 0.0;
        swD[t] = bad;
      }
      swS[li] = changed;
      barrier();
      if (li == 0u) {
        float ch = 0.0;
        for (uint i = 0u; i < 64u; i++) { ch += swS[i]; }
        swGo = ch > 0.0 ? 1u : 0u;
      }
      ${uniformLoad("uint go2", "swGo")}
      if (go2 == 0u) { break; }
    }
    barrier();
  }
  if (li == 0u) {
    float b0 = A.g[9u * rgi + 0u][0] * red[0] + A.g[9u * rgi + 0u][1] * red[64] + A.g[9u * rgi + 0u][2] * red[128] + A.g[9u * rgi + 0u][3] * red[192] + A.g[9u * rgi + 1u][0] * red[256] + A.g[9u * rgi + 1u][1] * red[320];
    float b1 = A.g[9u * rgi + 1u][2] * red[0] + A.g[9u * rgi + 1u][3] * red[64] + A.g[9u * rgi + 2u][0] * red[128] + A.g[9u * rgi + 2u][1] * red[192] + A.g[9u * rgi + 2u][2] * red[256] + A.g[9u * rgi + 2u][3] * red[320];
    float b2 = A.g[9u * rgi + 3u][0] * red[0] + A.g[9u * rgi + 3u][1] * red[64] + A.g[9u * rgi + 3u][2] * red[128] + A.g[9u * rgi + 3u][3] * red[192] + A.g[9u * rgi + 4u][0] * red[256] + A.g[9u * rgi + 4u][1] * red[320];
    float b3 = A.g[9u * rgi + 4u][2] * red[0] + A.g[9u * rgi + 4u][3] * red[64] + A.g[9u * rgi + 5u][0] * red[128] + A.g[9u * rgi + 5u][1] * red[192] + A.g[9u * rgi + 5u][2] * red[256] + A.g[9u * rgi + 5u][3] * red[320];
    float b4 = A.g[9u * rgi + 6u][0] * red[0] + A.g[9u * rgi + 6u][1] * red[64] + A.g[9u * rgi + 6u][2] * red[128] + A.g[9u * rgi + 6u][3] * red[192] + A.g[9u * rgi + 7u][0] * red[256] + A.g[9u * rgi + 7u][1] * red[320];
    float b5 = A.g[9u * rgi + 7u][2] * red[0] + A.g[9u * rgi + 7u][3] * red[64] + A.g[9u * rgi + 8u][0] * red[128] + A.g[9u * rgi + 8u][1] * red[192] + A.g[9u * rgi + 8u][2] * red[256] + A.g[9u * rgi + 8u][3] * red[320];
    arec[ab] = vec4(1.0, float(S), 0.0, float(found));
    arec[ab + 1u] = A.k[3u * rgi];
    arec[ab + 2u] = A.k[3u * rgi + 1u];
    arec[ab + 3u] = geo;
    arec[ab + ${ANCH_SURF}u] = vec4(b0, b1, b2, b3);
    arec[ab + ${ANCH_SURF + 1}u] = vec4(b4, b5, 0.0, 0.0);
    arec[ab + ${ANCH_GAIN}u] = vec4(swAl[0], swAl[1], swAl[2], swAl[3]);
    arec[ab + ${ANCH_GAIN + 1}u] = vec4(swAl[4], swAl[5], 0.0, 0.0);
  }
  if (li < ${ANCH_SURF - 4}u) {
    uint q = 2u * li;
    vec2 kq = kv[min(q + 1u, NK - 1u)];
    arec[ab + 4u + li] = vec4(kv[q], q + 1u < NK ? kq : vec2(0.0));
  }
}
`;

// ai: ANCH_REFIT and ANCH_PREDICT (the WGSL anchors.mjs (archived) anchRefitSource, its comments).
const NPTS = NODES_MAX + ANCH_K;
const TAU_SPREAD = String(Math.fround(3.0 * 1.4826));   // ai: the WGSL's 3.0 * 1.4826, folded in abstract precision then rounded to f32
export const anchRefitSource = ({ predict = false } = {}) => VERSION + NULL_INIT + DIMS + CONVERT + MAP + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Nodes { vec4 nodes[]; };
layout(set = 0, binding = 2, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 3, std430) buffer Meas { vec4 meas[]; };
layout(set = 0, binding = 4, std430) buffer Maps { float maps[][16]; };
layout(set = 0, binding = 5, std430) buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 6, std140) uniform Fit { uvec4 count; uvec4 base; vec4 modules; } P;
layout(set = 0, binding = 7, std430) buffer Anch { vec4 anch[]; };
layout(set = 0, binding = 8, std140) uniform Lattice { uvec4 at[${RING_COUNT}]; vec4 lat[${(RING_COUNT * LATTICE_MAX) / 4}]; } LAT;
layout(set = 0, binding = 9, std140) uniform Anchors { vec4 k[${3 * RING_COUNT}]; vec4 g[${9 * RING_COUNT}]; vec4 s[${RING_COUNT}]; } A;
float latAt(uint i) { return LAT.lat[i >> 2u][i & 3u]; }
float knot(uint r, uint i) { return A.k[3u * r + (i >> 2u)][i & 3u]; }
` + COONS + /* glsl */ `
struct LensFrame { vec2 ctr; float sig; float smid; float ish; uint kmax; uint nw; uint on; };
const uint NK = ${ANCH_K}u;
const uint SIDE = ${ANCH_SIDE}u;
shared vec4 pp[${NPTS}];
shared float pw[${NPTS}];
shared float pr[${NODES_MAX}];
shared uint pd[${NODES_MAX}];
shared uint ak[${ANCH_K}];
shared LensFrame LF;
shared float LS[16][77];
shared float LT[4][${LENS_P}];
shared uint LK[4];
shared vec4 LR[64];
shared vec4 LQ[64];
shared float LP[77];
shared Lens LZ;
shared vec4 rw[${NODES_MAX}];
${SIDE_DECL}shared uint wu;
shared uint wok;
shared float wmed;
shared uint wnb;
shared float wtau;
shared uint wcnt = {};
shared uint wnba = {};
shared uint wchg = {};
shared vec4 LR8[8];
shared vec4 LQ8[8];
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  uvec4 pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  Frame F = frames[f];
  uint rgi = pick.x;
  uint count = P.count[rgi];
  uint base = P.base[rgi];
  if (li < NK) {
    bool on = anch[NK * f + li].w > 0.5;
    ak[li] = on ? 1u : 0u;
    if (on) { atomicAdd(wcnt, 1u); }
  }
  barrier();
  if (li == 0u) { wu = wcnt; }
  { ${uniformLoad("uint wu0", "wu")} if (wu0 < f2u(A.s[rgi].y)) { return; } }
  float M[16] = maps[f];
  Lens L0 = lensOf(f);
  uvec4 at = LAT.at[rgi];
  uint first = at.x;
  uint nl = at.y;
${predict ? `  for (uint n = li; n < count; n += 64u) { pp[n] = vec4(toImage(M, nodes[base + n].xy), 0.0, 0.0); }` : `  for (uint n = li; n < count; n += 64u) {
    vec4 e = meas[f * ${NODES_MAX}u + n];
    bool ok = e.w != 0.0 && e.z >= ${WMIN};
    vec2 p = toImage(M, nodes[base + n].xy);
    vec2 pe = e.xy - p;
    pp[n] = vec4(p, ok ? pe : vec2(0.0));
    pw[n] = ok ? e.z : 0.0;
    pd[n] = 0u;
    if (ok) { atomicAdd(wnba, 1u); }
  }`}
  if (li < NK) {
    vec4 e = anch[NK * f + li];
    vec2 X = vec2(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    vec2 p = toImage(M, X);
    vec2 h = p;
    bool on = ak[li] == 1u;
    vec2 pe = p + e.xy + lensAt(L0, p) + coons(f, first, nl, X) - h;
    pp[count + li] = vec4(h, on ? pe : vec2(0.0));
    pw[count + li] = on ? 1.0 : 0.0;
  }
  ${STORAGE_BARRIER}
  barrier();
  uint npts = count + NK;
  uint aw = F.size;
  vec2 LA = vec2(0.5 * float(F.w), 0.5 * float(F.h));
  if (aw != 0u) { LA = vec2(0.5 * float(aw & 0xffffu) - 16384.0, 0.5 * float(aw >> 16u) - 16384.0); }
  vec2 fwh = vec2(float(F.w), float(F.h));
  float lR = max(max(length(LA), length(vec2(fwh.x, 0.0) - LA)), max(length(vec2(0.0, fwh.y) - LA), length(fwh - LA)));
  float lir = 1.0 / max(lR, 1.0);
  {
    vec2 lo = vec2(3.0e38);
    vec2 hi = vec2(-3.0e38);
    float smin = 3.0e38;
    float smax = 0.0;
    uint nw = 0u;
    for (uint i = ${predict ? "count + li" : "li"}; i < npts; i += 64u) {
      if (!(pw[i] > 0.0)) { continue; }
      vec2 p = pp[i].xy;
      lo = min(lo, p);
      hi = max(hi, p);
      vec2 q = (p - LA) * lir;
      float s = dot(q, q);
      smin = min(smin, s);
      smax = max(smax, s);
      nw += 1u;
    }
    LR[li] = vec4(lo, hi);
    LQ[li] = vec4(smin, smax, float(nw), 0.0);
  }
  barrier();
  if (li < 8u) {
    vec2 lo = LR[8u * li].xy;
    vec2 hi = LR[8u * li].zw;
    float smin = LQ[8u * li].x;
    float smax = LQ[8u * li].y;
    float nw = LQ[8u * li].z;
    for (uint s = 8u * li + 1u; s < 8u * li + 8u; s++) {
      lo = min(lo, LR[s].xy);
      hi = max(hi, LR[s].zw);
      smin = min(smin, LQ[s].x);
      smax = max(smax, LQ[s].y);
      nw += LQ[s].z;
    }
    LR8[li] = vec4(lo, hi);
    LQ8[li] = vec4(smin, smax, nw, 0.0);
  }
  barrier();
  if (li == 0u) {
    vec2 lo = LR8[0].xy;
    vec2 hi = LR8[0].zw;
    float smin = LQ8[0].x;
    float smax = LQ8[0].y;
    uint nw = f2u(LQ8[0].z);
    for (uint s = 1u; s < 8u; s++) {
      lo = min(lo, LR8[s].xy);
      hi = max(hi, LR8[s].zw);
      smin = min(smin, LQ8[s].x);
      smax = max(smax, LQ8[s].y);
      nw += f2u(LQ8[s].z);
    }
    // ai: the radial terms: ANCH_PREDICT's fit over the anchors alone takes no more than anchors where they are fewer than
    // ai: the model's terms (the 64 ring's 3 x 3 grid: the C's acquire.c lens_cap, which seeds the border's search);
    // ai: ANCH_REFIT's over the border and the anchors every term
    LF = LensFrame(0.5 * (lo + hi), 2.0 / max(max(hi.x - lo.x, hi.y - lo.y), 1.0), 0.5 * (smin + smax),
      2.0 / max(smax - smin, 1e-6), ${predict ? `(wu >= ${LENS_P}u ? 3u : (wu > 8u ? wu - 8u : 0u))` : "3u"}, nw, nw > 0u ? 1u : 0u);
    wok = 0u;
${predict ? "  }" : `    wnb = wnba;
    uint nq = (count + 4u) / 4u;
    float px = 0.0;
    float md = 0.0;
    for (uint c = 0u; c < 4u; c++) {
      uint a = (c * (nq - 1u)) % count;
      uint b = (((c + 1u) % 4u) * (nq - 1u)) % count;
      px += distance(pp[a].xy, pp[b].xy);
      md += distance(nodes[base + a].xy, nodes[base + b].xy);
    }
    wtau = 0.5 * px / max(md, 1.0);
  }`}
  barrier();
  ${predict ? `{` : "for (uint rd = 0u; rd < 4u; rd++) {"}
    if (li < 16u) {
      float acc[77] = ${zeros(77)};
      for (uint i = ${predict ? "count + li" : "li"}; i < npts; i += 16u) {
        float w = ${predict ? "pw[i]" : `(i < count && pd[min(i, ${NODES_MAX - 1}u)] == 1u) ? 0.0 : pw[i]`};
        if (!(w > 0.0)) { continue; }
        vec2 p = pp[i].xy;
        vec2 r = pp[i].zw;
        ${lensNode}
        ${lensAccumulate}
      }
      LS[li] = acc;
    }
    barrier();
    for (uint e = li; e < 77u; e += 64u) {
      float a = 0.0;
      for (uint s = 0u; s < 16u; s++) { a += LS[s][e]; }
      LP[e] = a;
    }
    barrier();
    if (li == 0u) {
      for (uint k = 0u; k < 4u; k++) { LK[k] = 0u; }
${lensNormalSolve}    }
    barrier();
    {
      vec4 rs = vec4(0.0);
      uint nu = 0u;
      for (uint i = ${predict ? "count + li" : "li"}; i < npts; i += 64u) {
        float w = ${predict ? "pw[i]" : `(i < count && pd[min(i, ${NODES_MAX - 1}u)] == 1u) ? 0.0 : pw[i]`};
        if (!(w > 0.0)) { continue; }
        nu += 1u;
        vec2 p = pp[i].xy;
        vec2 r = pp[i].zw;
        ${lensNode}
        ${[0, 1, 2, 3].map((K) => `if (LK[${K}] != 0u) { vec2 e = r - ${lensModel(K)}; rs[${K}] += w * dot(e, e); }`).join("\n        ")}
      }
      LR[li] = rs;
      LQ[li] = vec4(float(nu), 0.0, 0.0, 0.0);
    }
    barrier();
    if (li < 8u) {
      vec4 rss = LR[8u * li];
      float nu = LQ[8u * li].x;
      for (uint s = 8u * li + 1u; s < 8u * li + 8u; s++) { rss += LR[s]; nu += LQ[s].x; }
      LR8[li] = rss;
      LQ8[li] = vec4(nu, 0.0, 0.0, 0.0);
    }
    barrier();
    if (li == 0u) {
      vec4 rss = vec4(0.0);
      uint nu = 0u;
      for (uint s = 0u; s < 8u; s++) { rss += LR8[s]; nu += f2u(LQ8[s].x); }
      uint kb = 4u;
      float best = 3.0e38;
      float nobs = 2.0 * float(nu);
      for (uint k = 0u; k < 4u; k++) {
        float pk = 8.0 + float(k);
        if (LK[k] == 0u || nobs - pk - 1.0 <= 0.0) { continue; }
        float score = nobs * log(max(rss[k], 1e-30) / nobs) + 2.0 * pk + 2.0 * pk * (pk + 1.0) / (nobs - pk - 1.0);
        if (score < best) { best = score; kb = k; }
      }
      float tk[${LENS_P}] = ${zeros(LENS_P)};
      ${[0, 1, 2, 3].map((K) => `if (kb == ${K}u) { ${Array.from({ length: LENS_P }, (_, a) => `tk[${a}] = LT[${K}][${a}];`).join(" ")} }`).join("\n      ")}
      if (kb < 4u) {
        LZ = Lens(vec4(LF.ctr, LF.sig, LF.smid), vec4(LA, lir, LF.ish), vec4(tk[0], tk[1], tk[2], tk[3]),
          vec4(tk[4], tk[5], tk[6], tk[7]), vec4(tk[8], tk[9], tk[10], 0.0));
        wok = 1u;
      }
      wu = kb < 4u ? 1u : 0u;
    }
${predict ? `  }
` : `    { ${uniformLoad("uint wu1", "wu")} if (wu1 == 0u || rd == 3u) { break; } }
    Lens lz = LZ;
    for (uint n = li; n < count; n += 64u) { float dn = length(pp[n].zw - lensAt(lz, pp[n].xy)); pr[n] = pw[n] > 0.0 ? dn : 0.0; }
    if (li == 0u) { wmed = 0.0; wchg = 0u; }
    barrier();
    uint nb = wnb;
    for (uint n = li; n < count; n += 64u) {
      if (!(pw[n] > 0.0)) { continue; }
      float v = pr[n];
      uint rank = 0u;
      for (uint j = 0u; j < count; j++) { float u = pr[j]; rank += (pw[j] > 0.0 && (u < v || (u == v && j < n))) ? 1u : 0u; }
      if (rank == nb / 2u) { wmed = v; }
    }
    barrier();
    float tau = max(wtau, ${TAU_SPREAD} * wmed);
    for (uint n = li; n < count; n += 64u) {
      uint d = (pw[n] > 0.0 && pr[n] > tau) ? 1u : 0u;
      if (d != pd[n]) { atomicAdd(wchg, 1u); }
      pd[n] = d;
    }
    barrier();
    if (li == 0u) { wu = wchg; }
    { ${uniformLoad("uint wu2", "wu")} if (wu2 == 0u) { break; } }
  }
`}  { ${uniformLoad("uint wok0", "wok")} if (wok0 == 0u) { return; } }
  Lens lz = LZ;
${predict ? `  for (uint n = li; n < count; n += 64u) { meas[f * ${NODES_MAX}u + n] = vec4(pp[n].xy + lensAt(lz, pp[n].xy), 0.0, 2.0); }
  return;
` : ""}  if (li == 0u) {
    uint b = f * ${RESID}u + ${NODES_MAX}u;
    resid[b] = lz.c;
    resid[b + 1u] = lz.a;
    resid[b + 2u] = lz.t0;
    resid[b + 3u] = lz.t1;
    resid[b + 4u] = lz.t2;
  }
  for (uint n = li; n < count; n += 64u) {
    vec4 vk = vec4(pp[n].zw - lensAt(lz, pp[n].xy), pw[n], 0.0);
    vec4 vd = vec4(0.0, 0.0, pw[n], 0.0);
    vec4 v = pd[n] == 1u ? vd : vk;
    resid[f * ${RESID}u + n] = v;
    rw[n] = v;
  }
  ${STORAGE_BARRIER}
  barrier();
` + SIDE_FITS + /* glsl */ `
  ${STORAGE_BARRIER}
  barrier();
  if (li < NK && ak[li] == 1u) {
    vec2 X = vec2(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    vec4 e = anch[NK * f + li];
    vec4 q = pp[count + li];
    anch[NK * f + li] = vec4(q.zw - lensAt(lz, q.xy) - coons(f, first, nl, X), e.zw);
  }
}
`;
export const ANCH_REFIT = anchRefitSource(), ANCH_PREDICT = anchRefitSource({ predict: true });
