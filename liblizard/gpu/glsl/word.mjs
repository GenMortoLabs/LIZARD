// F8, the format word, in GLSL: the band read through the final map, each word cell against its track pairs' grey;
// the word, static or animated, by Gauss-Jordan over GF(256) from its most reliable bytes (../wordcode.mjs); the
// side-aware gate, held to the C's false-word rate; sel written (a static word's version, else HELD's, below) with
// the result: r[5] the version, or 256 + the position of an animated word, r[6] the bytes after it
// (fps + 256 ext0 + 65536 ext1, or the symbol's three bytes alike), 0 and 0 where no word is taken. One workgroup of
// 256 a frame.
// ai: The Word, Lattice and Held uniforms are std140, every member an array of uvec4 or vec4; the uniform's layout and
// ai: tables come from ../plan/word.mjs. No private array is indexed at run
// ai: time (the S26's Adreno returned zeros from one): the code's rows, GF(256)'s tables and the generator's columns are
// ai: read from the uniform, the lattice too.
import { VERSION, SAMPLERLESS, DIMS, CONVERT, MAP, uniformLoad } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { SLOTS } from "../plan/common.mjs";
import { LATTICE_MAX } from "../plan/back_transform.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { VERSION_MIN, VERSION_MAX, TAU, SIDE_KEEP, MAGIC, MAGIC_ANIM, POS_MAX } from "../wordcode.mjs";
import { CON_MIN, WMAX, ROWS, MIDS_MAX, VECS, UNKNOWN, SWAP, PER, SETS, TOP, BYTES_MAX, GATE, LAYOUT } from "../plan/word.mjs";

const TOPS = Math.ceil(SLOTS / 4);
const WORDS = ["x", "y", "z", "w"];
// ai: the Word uniform's members (std140, ../plan/word.mjs's layout), WORD's and HELD's
const WORD_UNI = `uvec4 code[${LAYOUT.range}]; uvec4 range[${RING_COUNT}]; uvec4 plan[${RING_COUNT}]; uvec4 tops[${TOPS}]; uvec4 gate[${(RING_COUNT * GATE) / 4}]; uvec4 gf[${LAYOUT.cols - LAYOUT.gf}]; uvec4 cols[${LAYOUT.size - LAYOUT.cols}];`;
// ai: the picture slot a version decodes in (the Word uniform's tops)
const SLOT_OF = `uint slotOf(uint v) {
  return 0u${Array.from({ length: SLOTS - 1 }, (_, k) => ` + (v > W.tops[${k >> 2}].${WORDS[k & 3]} ? 1u : 0u)`).join("")};
}`;

// ai: Gauss-Jordan over GF(256) (2026-11-15; ordered statistics decoding over the code's bit rows before): RS(n, 5) is
// ai: MDS, so the magic (byte 0) and any four other bytes give the word, for each magic in turn (the static word's and
// ai: the animated one's). Each byte's hard value (its 8 cells MSB first, dark 1), its reliability (the least |qs| of
// ai: its cells) and its least reliable bit; bytes 1 to n - 1 ranked by reliability (the lower index first on a tie),
// ai: the first TOP kept. A magic's set 0 solves the UNKNOWN most reliable; sets 1 to UNKNOWN SWAP swap one of them for
// ai: one of the next SWAP; the last UNKNOWN flip one's least reliable bit. A lane a set solves its UNKNOWN x UNKNOWN
// ai: system by Gauss-Jordan in its own rows of workgroup memory (the magic's column moved to the right side; the
// ai: Adreno's pivoted eliminations on private arrays returned zeros, workgroup memory is not private); a static solve
// ai: whose version is outside VERSION_MIN to VERSION_MAX, an animated one past POS_MAX, refused. A lane a (set, u32 of
// ai: the word) builds that word of the candidate from the code's rows and scores it against qs; a lane a set sums them;
// ai: lane 0 keeps the best, the first set on a tie (the static magic's first).
const SEARCH = /* glsl */ `
  uint at0 = rg * ${ROWS}u;
  uint nw = nb / 32u;
  uint nby = nb / 8u;
  for (uint bit = li; bit < nb; bit += 256u) {
    float x = ((keep >> (bit / vr.x)) & 1u) != 0u ? q[bit] : 0.0;
    qs[bit >> 2u][bit & 3u] = x;
  }
  barrier();
  if (li < nby) {
    uint c = (li & 3u) * vr.x + 8u * (li >> 2u);
    uint h = 0u;
    float lo = 3.0e38;
    uint wi = 0u;
    for (uint t = 0u; t < 8u; t++) {
      float x = qs[(c + t) >> 2u][(c + t) & 3u];
      h = (h << 1u) | (x > 0.0 ? 1u : 0u);
      if (abs(x) < lo) { lo = abs(x); wi = t; }
    }
    hb[li] = h;
    rl[li] = lo;
    wk[li] = wi;
  }
  barrier();
  if (li >= 1u && li < nby) {
    float r = rl[li];
    uint rank = 0u;
    for (uint i = 1u; i < nby; i++) { float o = rl[i]; if (o > r || (o == r && i < li)) { rank += 1u; } }
    if (rank < ${TOP}u) { tb[rank] = li; }
  }
  barrier();
  if (li < ${SETS}u) {
    uint kind = li / ${PER}u;
    uint t = li % ${PER}u;
    uint magic = kind == 0u ? ${MAGIC}u : ${MAGIC_ANIM}u;
    uint p0 = tb[0];
    uint p1 = tb[1];
    uint p2 = tb[2];
    uint p3 = tb[3];
    if (t >= 1u && t <= ${UNKNOWN * SWAP}u) {
      uint a = (t - 1u) / ${SWAP}u;
      uint b = tb[${UNKNOWN}u + (t - 1u) % ${SWAP}u];
      if (a == 0u) { p0 = b; } else if (a == 1u) { p1 = b; } else if (a == 2u) { p2 = b; } else { p3 = b; }
    }
    uint fl = t > ${UNKNOWN * SWAP}u ? t - ${UNKNOWN * SWAP + 1}u : ${UNKNOWN}u;
    uint m0 = li * 20u;
    setRow(m0, rg, p0, hb[p0] ^ (fl == 0u ? 0x80u >> wk[p0] : 0u), magic);
    setRow(m0 + 5u, rg, p1, hb[p1] ^ (fl == 1u ? 0x80u >> wk[p1] : 0u), magic);
    setRow(m0 + 10u, rg, p2, hb[p2] ^ (fl == 2u ? 0x80u >> wk[p2] : 0u), magic);
    setRow(m0 + 15u, rg, p3, hb[p3] ^ (fl == 3u ? 0x80u >> wk[p3] : 0u), magic);
    bool ok = true;
    for (uint c = 0u; c < 4u; c++) {
      uint pr = 4u;
      for (uint r = c; r < 4u; r++) { if (pr == 4u && mat[m0 + 5u * r + c] != 0u) { pr = r; } }
      if (pr == 4u) { ok = false; break; }
      if (pr != c) {
        for (uint j = 0u; j < 5u; j++) { uint x = mat[m0 + 5u * c + j]; mat[m0 + 5u * c + j] = mat[m0 + 5u * pr + j]; mat[m0 + 5u * pr + j] = x; }
      }
      uint iv = gexp(255u - glog(mat[m0 + 5u * c + c]));
      for (uint j = 0u; j < 5u; j++) { mat[m0 + 5u * c + j] = gmul(mat[m0 + 5u * c + j], iv); }
      for (uint r = 0u; r < 4u; r++) {
        uint fr = r == c ? 0u : mat[m0 + 5u * r + c];
        if (fr != 0u) { for (uint j = 0u; j < 5u; j++) { mat[m0 + 5u * r + j] ^= gmul(fr, mat[m0 + 5u * c + j]); } }
      }
    }
    uint x0 = mat[m0 + 4u];
    uint data = x0 | (mat[m0 + 9u] << 8u) | (mat[m0 + 14u] << 16u) | (mat[m0 + 19u] << 24u);
    bool inRange = kind == 0u ? (x0 >= ${VERSION_MIN}u && x0 <= ${VERSION_MAX}u) : x0 <= ${POS_MAX}u;
    sv[li] = uvec4(data, kind, ok && inRange ? 1u : 0u, 0u);
  }
  barrier();
  for (uint t = li; t < ${SETS * 16}u; t += 256u) {
    uint w = t & 15u;
    uvec4 c = sv[t >> 4u];
    float sc = 0.0;
    if (w < nw && c.z != 0u) {
      uint cw = W.code[${VECS}u * (at0 + c.y) + (w >> 2u)][w & 3u];
      for (uint b = 0u; b < 32u; b++) {
        if (((c.x >> b) & 1u) != 0u) { cw ^= W.code[${VECS}u * (at0 + 2u + b) + (w >> 2u)][w & 3u]; }
      }
      for (uint b = 0u; b < 8u; b++) {
        uint m = cw >> (4u * b);
        sc += dot(qs[8u * w + b], mix(vec4(-1.0), vec4(1.0), notEqual(uvec4(m) & uvec4(1u, 2u, 4u, 8u), uvec4(0u))));
      }
    }
    part[t] = sc;
  }
  barrier();
  if (li < ${SETS}u) {
    float t = 0.0;
    for (uint k = 0u; k < 16u; k++) { t += part[16u * li + k]; }
    tot[li] = t;
  }
  barrier();
  if (li == 0u) {
    float best = -1e9;
    uint bs = ${SETS}u;
    for (uint s = 0u; s < ${SETS}u; s++) { if (sv[s].z != 0u && tot[s] > best) { best = tot[s]; bs = s; } }
    uvec4 c = sv[min(bs, ${SETS - 1}u)];
    wu = uvec4(c.x, c.y, bs < ${SETS}u ? 1u : 0u, floatBitsToUint(best));
  }`;

export const WORD = VERSION + SAMPLERLESS + DIMS + CONVERT + MAP + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
struct Result { float r[32]; };
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Level1 { float lv1[]; };
layout(set = 0, binding = 2, std430) readonly buffer Level2 { float lv2[]; };
layout(set = 0, binding = 3, std430) readonly buffer Level3 { float lv3[]; };
layout(set = 0, binding = 4, std430) readonly buffer Level4 { float lv4[]; };
layout(set = 0, binding = 5, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 6, std140) uniform LevelDims { uvec4 d[5]; } LD;
layout(set = 0, binding = 7, std430) readonly buffer Maps { float maps[][16]; };
layout(set = 0, binding = 8, std430) buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 9, std430) readonly buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 10, std430) readonly buffer Pts { vec4 pts[]; };
layout(set = 0, binding = 11, std430) buffer Results { Result result[]; };
layout(set = 0, binding = 12, std140) uniform Word { ${WORD_UNI} } W;
layout(set = 0, binding = 13, std140) uniform Lattice { uvec4 at[${RING_COUNT}]; vec4 lat[${(RING_COUNT * LATTICE_MAX) / 4}]; } LT;
float latAt(uint i) { return LT.lat[i >> 2u][i & 3u]; }
float pix(uint l, uint f, int x, int y) {
  Frame F = frames[f];
  int w = int((F.w + (1u << l) - 1u) >> l);
  int h = int((F.h + (1u << l) - 1u) >> l);
  int xc = clamp(x, 0, w - 1);
  int yc = clamp(y, 0, h - 1);
  if (l == 0u) { return texelFetch(img, ivec3(xc, yc, int(f)), 0).r; }
  uvec4 s = LD.d[l];
  uint i = (f * s.y + uint(yc)) * s.x + uint(xc);
  switch (l) {
    case 1u: { return lv1[i]; }
    case 2u: { return lv2[i]; }
    case 3u: { return lv3[i]; }
    default: { return lv4[i]; }
  }
}
float look(uint l, uint f, vec2 p) {
  vec2 q = p / float(1u << l) - vec2(0.5);
  vec2 q0 = floor(q);
  vec2 a = q - q0;
  ivec2 i = f2i(q0);
  return mix(mix(pix(l, f, i.x, i.y), pix(l, f, i.x + 1, i.y), a.x), mix(pix(l, f, i.x, i.y + 1), pix(l, f, i.x + 1, i.y + 1), a.x), a.y);
}
` + COONS + /* glsl */ `
shared vec4 red[256];
shared float mids[${MIDS_MAX}];
shared float q[${4 * WMAX}];
shared uint go;
shared uvec4 wu;
shared vec4 qs[${WMAX}];
shared uint hb[${BYTES_MAX}];
shared float rl[${BYTES_MAX}];
shared uint wk[${BYTES_MAX}];
shared uint tb[${TOP}];
shared uvec4 sv[${SETS}];
shared uint mat[${SETS * 20}];
shared float part[${SETS * 16}];
shared float tot[${SETS}];
uint gbyte(uint i) { return (W.gf[i >> 4u][(i >> 2u) & 3u] >> (8u * (i & 3u))) & 255u; }
uint gexp(uint e) { return gbyte(e % 255u); }
uint glog(uint x) { return gbyte(256u + x); }
uint gmul(uint a, uint b) { return (a == 0u || b == 0u) ? 0u : gexp(glog(a) + glog(b)); }
// ai: a row of a set's system: position p's columns of the four unknown bytes, its value less the magic's part
void setRow(uint at, uint r, uint p, uint v, uint magic) {
  uint i = 2u * (${BYTES_MAX}u * r + p);
  uint g0 = W.cols[i >> 2u][i & 3u] & 255u;
  uint g = W.cols[(i + 1u) >> 2u][(i + 1u) & 3u];
  mat[at] = g & 255u;
  mat[at + 1u] = (g >> 8u) & 255u;
  mat[at + 2u] = (g >> 16u) & 255u;
  mat[at + 3u] = g >> 24u;
  mat[at + 4u] = v ^ gmul(g0, magic);
}
uint codeBit(uint at, uint kind, uint data, uint i) {
  uint k = i >> 5u;
  uint w = W.code[${VECS}u * (at + kind) + (k >> 2u)][k & 3u];
  for (uint b = 0u; b < 32u; b++) {
    if (((data >> b) & 1u) != 0u) { w ^= W.code[${VECS}u * (at + 2u + b) + (k >> 2u)][k & 3u]; }
  }
  return (w >> (i & 31u)) & 1u;
}
uint leastSide(vec4 s, uint skip) {
  uint b = 0u;
  float v = 3.0e38;
  if ((skip & 1u) == 0u && s.x < v) { v = s.x; b = 1u; }
  if ((skip & 2u) == 0u && s.y < v) { v = s.y; b = 2u; }
  if ((skip & 4u) == 0u && s.z < v) { v = s.z; b = 4u; }
  if ((skip & 8u) == 0u && s.w < v) { v = s.w; b = 8u; }
  return b;
}
uint gateAt(uint r, uint m) { uint k = r * ${GATE}u + (m >> 1u); return (W.gate[k >> 2u][k & 3u] >> (16u * (m & 1u))) & 0xffffu; }
${SLOT_OF}
vec2 imgAt(uint f, float M[16], uvec4 L, vec2 m) { vec2 p = toImage(M, m); return p + lensAt(lensOf(f), p) + coons(f, L.x, L.y, m); }
void reduce(uint li) {
  barrier();
  for (uint st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    barrier();
  }
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  if (li == 0u) { go = (frames[f].valid != 0u && sel[f].y != 0u) ? 1u : 0u; }
  { ${uniformLoad("uint go0", "go")} if (go0 == 0u) { return; } }
  uint rg = sel[f].x;
  uvec4 P = W.plan[rg];
  float M[16] = maps[f];
  uvec4 L = LT.at[rg];
  vec2 c0 = toImage(M, pts[P.x].xy);
  float u = distance(toImage(M, pts[P.x].xy + vec2(1.0, 0.0)), c0);
  uint l = f2u(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0));
  vec2 lg = vec2(0.0);
  for (uint s = li; s < P.y; s += 256u) {
    lg += vec2(look(l, f, imgAt(f, M, L, pts[P.x + 2u * s].xy)), look(l, f, imgAt(f, M, L, pts[P.x + 2u * s + 1u].xy)));
  }
  red[li] = vec4(lg, 0.0, 0.0);
  reduce(li);
  float con = (red[0].y - red[0].x) / float(max(P.y, 1u));
  uint mp = P.x + 2u * P.y;
  for (uint k = li; k < P.z; k += 256u) { mids[k] = 0.5 * (look(l, f, imgAt(f, M, L, pts[mp + 2u * k].xy)) + look(l, f, imgAt(f, M, L, pts[mp + 2u * k + 1u].xy))); }
  barrier();
  uint wp = mp + 2u * P.z;
  uvec4 vr = W.range[rg];
  uint nb = 4u * vr.x;
  for (uint bit = li; bit < nb; bit += 256u) {
    vec4 e = pts[wp + bit];
    uint lo = f2u(e.z) % 4096u;
    uint hi = f2u(e.z) / 4096u;
    float grey = 0.0;
    for (uint j = lo; j <= hi; j++) { grey += mids[j]; }
    q[bit] = clamp((grey / float(hi - lo + 1u) - look(l, f, imgAt(f, M, L, e.xy))) / max(con, 1e-3), -0.5, 0.5);
  }
  barrier();
  vec4 sa = vec4(0.0);
  for (uint bit = li; bit < nb; bit += 256u) { sa += mix(vec4(0.0), vec4(abs(q[bit])), equal(uvec4(0u, 1u, 2u, 3u), uvec4(bit / vr.x))); }
  red[li] = sa;
  reduce(li);
  vec4 sides = red[0];
  uint d1 = leastSide(sides, 0u);
  uint d2 = d1 | leastSide(sides, d1);
  float top = ${SIDE_KEEP} * max(max(sides.x, sides.y), max(sides.z, sides.w));
  uint low = (sides.x < top ? 1u : 0u) | (sides.y < top ? 2u : 0u) | (sides.z < top ? 4u : 0u) | (sides.w < top ? 8u : 0u);
  uint keep = 15u & ~(low & d2);
  barrier();${SEARCH}
  ${uniformLoad("uvec4 wn", "wu")}
  vec3 g = vec3(0.0);
  for (uint bit = li; bit < nb; bit += 256u) {
    if (abs(q[bit]) < ${TAU}) { continue; }
    uint s = 1u << (bit / vr.x);
    float one = 1024.0 + (((q[bit] > 0.0) != (codeBit(at0, wn.y, wn.x, bit) != 0u)) ? 1.0 : 0.0);
    g += mix(vec3(0.0), vec3(one), bvec3(true, (s & d1) == 0u, (s & d2) == 0u));
  }
  red[li] = vec4(g, 0.0);
  reduce(li);
  if (li == 0u) {
    vec4 G = red[0];
    uint m0 = f2u(G.x / 1024.0);
    uint m1 = f2u(G.y / 1024.0);
    uint m2 = f2u(G.z / 1024.0);
    uint e0 = gateAt(rg, m0);
    uint e1 = gateAt(rg, m1);
    uint e2 = gateAt(rg, m2);
    bool through = (e0 != 0xffffu && f2u(G.x) - 1024u * m0 <= e0) || (e1 != 0xffffu && f2u(G.y) - 1024u * m1 <= e1) || (e2 != 0xffffu && f2u(G.z) - 1024u * m2 <= e2);
    bool ok = con >= ${CON_MIN} && wn.z != 0u && through;
    uint b1 = wn.x & 255u;
    bool own = ok && wn.y == 0u;
    result[f].r[5] = ok ? float(wn.y == 0u ? b1 : 256u + b1) : 0.0;
    result[f].r[6] = ok ? float(wn.x >> 8u) : 0.0;
    result[f].r[7] = uintBitsToFloat(wn.w);
    // ai: its own static word's version, sel.w 2; else HELD's (sel.w 1, waiting for it)
    result[f].r[29] = own ? float(b1) : 0.0;
    result[f].r[30] = 0.0;
    sel[f] = own ? uvec4(rg, 1u + b1, slotOf(b1), 2u) : uvec4(rg, 0u, 0u, 1u);
  }
}
`;

// ai: F8's held version (2026-11-16), after WORD in its pass: one invocation walks the launch's frames in capture order;
// ai: a frame with its own static word (sel.w 2) makes its version the held one, a frame waiting (sel.w 1: found, no
// ai: static word of its own) decodes at the held one (sel.y 1 + it, sel.z its picture; none where 0), record words
// ai: 29 and 30 its version and whether it was held. The held version goes on from launch to launch in CARRY, one
// ai: word pair for every lane of the decoder (x the version, y the host's last post taken), so a frame's held version
// ai: is the last static word read before it, whatever the launch's size or how many are in flight (before, each
// ai: launch took the host's version at its submit: a word read earlier in the same launch, or in one still in flight,
// ai: was not seen). The host posts a version of its own (Held: version, post; a carousel's header, or 0 at a reset or
// ai: for a launch alone) by a new post number, taken once in CARRY's place; hold 0 keeps no version (a lab's arm).
// ai: Bindings: 0 sel, 1 the results, 2 CARRY, 3 the Word uniform, 4 Held (version, post, frames, hold).
export const HELD = VERSION + /* glsl */ `
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
struct Result { float r[32]; };
layout(set = 0, binding = 0, std430) buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 1, std430) buffer Results { Result result[]; };
layout(set = 0, binding = 2, std430) buffer Carry { uvec4 carry[]; };
layout(set = 0, binding = 3, std140) uniform Word { ${WORD_UNI} } W;
layout(set = 0, binding = 4, std140) uniform Held { uint version; uint post; uint frames; uint hold; } HB;
${SLOT_OF}
void main() {
  uvec4 c = carry[0];
  uint cur = HB.post != c.y ? HB.version : c.x;
  for (uint f = 0u; f < HB.frames; f++) {
    uvec4 s = sel[f];
    if (s.w == 2u) {
      if (HB.hold != 0u) { cur = s.y - 1u; }
      sel[f].w = 0u;
    } else if (s.w == 1u) {
      uint v = HB.hold != 0u ? cur : 0u;
      result[f].r[29] = float(v);
      result[f].r[30] = v > 0u ? 1.0 : 0.0;
      sel[f] = v > 0u ? uvec4(s.x, 1u + v, slotOf(v), 0u) : uvec4(s.x, 0u, 0u, 0u);
    }
  }
  carry[0] = uvec4(cur, HB.post, 0u, 0u);
}
`;
