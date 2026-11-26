// F8, the format word: the band read through the final map, each word cell against its track pairs' grey;
// the word by Gauss-Jordan over GF(256) from its most reliable bytes; the side-aware gate, held to
// the C's false-word rate; the word's index split into the configuration (its version from the Word uniform's
// before, its counts across the workgroup's lanes, each a 6/7 count's configurations) and the clock; sel written (the
// configuration; none for a frame whose word is not read, which is not decoded: nothing carries from frame to frame)
// with the result: r[5] the version, r[6] the word's last 32 bits as a float's (uintBitsToFloat; the index's low bits
// alone: src/fmt.h), r[30] the clock and the 1/2 count as a float's bits (fps | delay << 8 | the 1/2 count << 16); 0
// where no word is taken. One
// workgroup of 256 a frame.
// The Word and Lattice uniforms are std140, every member an array of uint4 or float4; the uniform's layout and
// tables come from core/dec/build.cpp WordTables. No private array is indexed at run
// time (the S26's Adreno returned zeros from one): the code's rows, GF(256)'s tables and the generator's columns are
// read from the uniform, the lattice too.
// Defines (core/dec/build.cpp WordPlan): the Word uniform's lengths W_CODE, W_TOPS, W_GATE, W_GF, W_COLS, W_BEFORE, RING_COUNT,
// LATTICE_VECS (core/dec/build.cpp WordPlan); MIDS_MAX, WMAX, BYTES_MAX, TOP, SETS, ROWS, VECS, GATE, UNKNOWN, SWAP, MAGIC,
// WORDS_HI and WORDS_LO (the words' count as two u32), CLOCKS, DELAY_MAX, VERSION_MIN, VERSION_MAX, VERSION_SLOTS,
// UNIT and UNIT0 to UNIT3, SLOT_COUNT (the picture slots, the Word uniform's tops one past the first each) (uint); TAU,
// SIDE_KEEP, CON_MIN (float); RESID (lib/lens.slang); lib/cfg.slang's.
#include "lib/sync.slang"
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/map.slang"
struct Result { float r[32]; };
[[vk::binding(0, 0)]] Texture2DArray<float> img;
[[vk::binding(1, 0)]] StructuredBuffer<float> lv1;
[[vk::binding(2, 0)]] StructuredBuffer<float> lv2;
[[vk::binding(3, 0)]] StructuredBuffer<float> lv3;
[[vk::binding(4, 0)]] StructuredBuffer<float> lv4;
[[vk::binding(5, 0)]] StructuredBuffer<Frame> frames;
struct LevelDimsBlock { uint4 d[5]; };
[[vk::binding(6, 0)]] ConstantBuffer<LevelDimsBlock> LD;
struct MapRec { float m[16]; };
[[vk::binding(7, 0)]] StructuredBuffer<MapRec> maps;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint4> sel;
[[vk::binding(9, 0)]] StructuredBuffer<float4> resid;
[[vk::binding(10, 0)]] StructuredBuffer<float4> pts;
[[vk::binding(11, 0)]] RWStructuredBuffer<Result> result;
struct WordBlock {
  uint4 code[W_CODE]; uint4 range[RING_COUNT]; uint4 plan[RING_COUNT]; uint4 tops[W_TOPS]; uint4 gate[W_GATE]; uint4 gf[W_GF];
  uint4 cols[W_COLS]; uint4 before[W_BEFORE];
};
[[vk::binding(12, 0)]] ConstantBuffer<WordBlock> W;
struct LatticeBlock { uint4 at[RING_COUNT]; float4 lat[LATTICE_VECS]; };
[[vk::binding(13, 0)]] ConstantBuffer<LatticeBlock> LT;
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
#include "lib/lens.slang"
groupshared float4 red[256];
groupshared float mids[MIDS_MAX];
groupshared float q[4u * WMAX];
groupshared uint go;
groupshared uint4 wu;
groupshared float4 qs[WMAX];
groupshared uint hb[BYTES_MAX];
groupshared float rl[BYTES_MAX];
groupshared uint wk[BYTES_MAX];
groupshared uint tb[TOP];
groupshared uint4 sv[SETS];
groupshared uint mat[SETS * 30u];
groupshared uint na[256];
groupshared uint4 wix;
groupshared float part[SETS * 16u];
groupshared float tot[SETS];
uint gbyte(uint i) { return (W.gf[i >> 4u][(i >> 2u) & 3u] >> (8u * (i & 3u))) & 255u; }
uint gexp(uint e) { return gbyte(e % 255u); }
uint glog(uint x) { return gbyte(256u + x); }
uint gmul(uint a, uint b) { return (a == 0u || b == 0u) ? 0u : gexp(glog(a) + glog(b)); }
#include "lib/cfg.slang"
// A candidate's five bytes after the magic, data lo (bytes 1 to 4, byte 1 low) and hi (byte 5), as the word's 40
// bits (src/fmt.c ob_fmt_pack: byte 1 the most significant): x.x the top 8 (byte 1), x.y the other 32, the word's
// index alone (no CRC since 2026-11-23); refused past the last word.
uint2 bitsOfWord(uint lo, uint hi) {
  return uint2(lo & 255u, (((lo >> 8u) & 255u) << 24u) | (((lo >> 16u) & 255u) << 16u) | ((lo >> 24u) << 8u) | (hi & 255u));
}
bool wordOk(uint lo, uint hi) {
  uint2 c = bitsOfWord(lo, hi);
  return c.x < WORDS_HI || (c.x == WORDS_HI && c.y < WORDS_LO);
}
// The index's parts (src/fmt.c accept): configuration x CLOCKS + k (k 0 a still, 1 + (fps - 1) x 30 + delay - 1
// the others), so q = c / CLOCKS the configuration's index and c % CLOCKS the clock, by long division in 16-bit
// steps (c under 2^40).
uint2 splitIndex(uint2 c) {
  uint a = (c.x << 16u) | (c.y >> 16u);
  uint q1 = a / CLOCKS;
  uint u = ((a % CLOCKS) << 16u) | (c.y & 0xffffu);
  return uint2((q1 << 16u) + u / CLOCKS, u % CLOCKS);
}
// The configurations in index order (src/fmt.c ob_fmt_index), in units of UNIT slots (640; the blocks UNIT0 to UNIT3,
// 7, 8, 9, 12): a version's room; the (t, c) that fit r units (src/fmt_tables.h FMT_P2, by its closed form,
// src/fmt_tables.h's closed form: the sum over c = 3k + j of floor((r - 3j) / 9) - 4k - j + 1); the (b, t, c) that fit (FMT_P3).
uint roomOf(uint v) { return v * VERSION_SLOTS / UNIT; }
uint p2Of(uint r) {
  int cm = int(r / 12u);
  int n = 0;
  for (int j = 0; j < 3; j++) {
    if (cm < j) { continue; }
    int k = (cm - j) / 3;
    int fl = (int(r) - 3 * j) / 9;
    n += (k + 1) * (fl - j + 1 - 2 * k);
  }
  return uint(n);
}
uint p3Of(uint r) {
  uint n = 0u;
  for (uint b = 0u; UNIT1 * b <= r; b++) { n += p2Of(r - UNIT1 * b); }
  return n;
}
uint beforeAt(uint k) { return W.before[k >> 2u][k & 3u]; }
// a row of a set's system: position p's columns of the five unknown bytes (bytes 1 to 4, then 5), its value less
// the magic's part
void setRow(uint at, uint r, uint p, uint v, uint magic) {
  uint i = 2u * (BYTES_MAX * r + p);
  uint g05 = W.cols[i >> 2u][i & 3u];
  uint g = W.cols[(i + 1u) >> 2u][(i + 1u) & 3u];
  mat[at] = g & 255u;
  mat[at + 1u] = (g >> 8u) & 255u;
  mat[at + 2u] = (g >> 16u) & 255u;
  mat[at + 3u] = g >> 24u;
  mat[at + 4u] = (g05 >> 8u) & 255u;
  mat[at + 5u] = v ^ gmul(g05 & 255u, magic);
}
uint codeBit(uint at, uint lo, uint hi, uint i) {
  uint k = i >> 5u;
  uint w = W.code[VECS * at + (k >> 2u)][k & 3u];
  for (uint b = 0u; b < 32u; b++) {
    if (((lo >> b) & 1u) != 0u) { w ^= W.code[VECS * (at + 1u + b) + (k >> 2u)][k & 3u]; }
  }
  for (uint b = 0u; b < 8u; b++) {
    if (((hi >> b) & 1u) != 0u) { w ^= W.code[VECS * (at + 33u + b) + (k >> 2u)][k & 3u]; }
  }
  return (w >> (i & 31u)) & 1u;
}
uint leastSide(float4 s, uint skip) {
  uint b = 0u;
  float v = 3.0e38;
  if ((skip & 1u) == 0u && s.x < v) { v = s.x; b = 1u; }
  if ((skip & 2u) == 0u && s.y < v) { v = s.y; b = 2u; }
  if ((skip & 4u) == 0u && s.z < v) { v = s.z; b = 4u; }
  if ((skip & 8u) == 0u && s.w < v) { v = s.w; b = 8u; }
  return b;
}
uint gateAt(uint r, uint m) { uint k = r * GATE + (m >> 1u); return (W.gate[k >> 2u][k & 3u] >> (16u * (m & 1u))) & 0xffffu; }
// the picture slot a version decodes in (the Word uniform's tops)
uint slotOf(uint v) {
  uint s = 0u;
  [ForceUnroll] for (int k = 0; k < int(SLOT_COUNT) - 1; k++) { const uint4 t = W.tops[k >> 2]; s = s + (v > t[k & 3] ? 1u : 0u); }
  return s;
}
float2 imgAt(uint f, float M[16], float2 m) { float2 p = toImage(M, m); return p + lensAt(lensOf(f), p); }
void reduce(uint li) {
  GroupMemoryBarrierWithGroupSync();
  for (uint st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    GroupMemoryBarrierWithGroupSync();
  }
}
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x;
  if (li == 0u) { go = (frames[f].valid != 0u && sel[f].y != 0u) ? 1u : 0u; }
  { UNIFORM_LOAD(uint go0, go) if (go0 == 0u) { return; } }
  uint rg = sel[f].x;
  uint4 P = W.plan[rg];
  float M[16] = maps[f].m;
  float2 c0 = toImage(M, pts[P.x].xy);
  float u = distance(toImage(M, pts[P.x].xy + float2(1.0, 0.0)), c0);
  uint l = f2u(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0));
  float2 lg = float2(0.0);
  for (uint s = li; s < P.y; s += 256u) {
    lg += float2(look(l, f, imgAt(f, M, pts[P.x + 2u * s].xy)), look(l, f, imgAt(f, M, pts[P.x + 2u * s + 1u].xy)));
  }
  red[li] = float4(lg, 0.0, 0.0);
  reduce(li);
  float con = (red[0].y - red[0].x) / float(max(P.y, 1u));
  uint mp = P.x + 2u * P.y;
  for (uint k = li; k < P.z; k += 256u) { mids[k] = 0.5 * (look(l, f, imgAt(f, M, pts[mp + 2u * k].xy)) + look(l, f, imgAt(f, M, pts[mp + 2u * k + 1u].xy))); }
  GroupMemoryBarrierWithGroupSync();
  uint wp = mp + 2u * P.z;
  uint4 vr = W.range[rg];
  uint nb = 4u * vr.x;
  for (uint bit = li; bit < nb; bit += 256u) {
    float4 e = pts[wp + bit];
    uint lo = f2u(e.z) % 4096u;
    uint hi = f2u(e.z) / 4096u;
    float grey = 0.0;
    for (uint j = lo; j <= hi; j++) { grey += mids[j]; }
    q[bit] = clamp((grey / float(hi - lo + 1u) - look(l, f, imgAt(f, M, e.xy))) / max(con, 1e-3), -0.5, 0.5);
  }
  GroupMemoryBarrierWithGroupSync();
  float4 sa = float4(0.0);
  for (uint bit = li; bit < nb; bit += 256u) { sa += select(uint4(0u, 1u, 2u, 3u) == uint4(bit / vr.x), float4(abs(q[bit])), float4(0.0)); }
  red[li] = sa;
  reduce(li);
  float4 sides = red[0];
  uint d1 = leastSide(sides, 0u);
  uint d2 = d1 | leastSide(sides, d1);
  float top = SIDE_KEEP * max(max(sides.x, sides.y), max(sides.z, sides.w));
  uint low = (sides.x < top ? 1u : 0u) | (sides.y < top ? 2u : 0u) | (sides.z < top ? 4u : 0u) | (sides.w < top ? 8u : 0u);
  uint keep = 15u & ~(low & d2);
  // Gauss-Jordan over GF(256) (2026-11-15; ordered statistics decoding over the code's bit rows before): RS(n, 6) is
  // MDS, so the magic (byte 0) and any five other bytes give the word. Each byte's hard value (its 8 cells MSB
  // first, dark 1), its reliability (the least |qs| of its cells) and its least reliable bit; bytes 1 to n - 1 ranked
  // by reliability (the lower index first on a tie), the first TOP kept. Set 0 solves the UNKNOWN most reliable;
  // sets 1 to UNKNOWN SWAP swap one of them for one of the next SWAP; the last UNKNOWN flip one's least reliable bit.
  // A lane a set solves its UNKNOWN x UNKNOWN system by Gauss-Jordan in its own rows of workgroup memory (the
  // magic's column moved to the right side; the Adreno's pivoted eliminations on private arrays returned zeros,
  // workgroup memory is not private); a solve whose index is past the last word refused (no CRC since 2026-11-23)
  // (wordOk; src/fmt.c accept). A lane a (set, u32 of the word) builds that word of the
  // candidate from the code's rows and scores it against qs; a lane a set sums them; lane 0 keeps the best, the
  // first set on a tie.
  GroupMemoryBarrierWithGroupSync();
  uint at0 = rg * ROWS;
  uint nw = nb / 32u;
  uint nby = nb / 8u;
  for (uint bit = li; bit < nb; bit += 256u) {
    float x = ((keep >> (bit / vr.x)) & 1u) != 0u ? q[bit] : 0.0;
    qs[bit >> 2u][bit & 3u] = x;
  }
  GroupMemoryBarrierWithGroupSync();
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
  GroupMemoryBarrierWithGroupSync();
  if (li >= 1u && li < nby) {
    float r = rl[li];
    uint rank = 0u;
    for (uint i = 1u; i < nby; i++) { float o = rl[i]; if (o > r || (o == r && i < li)) { rank += 1u; } }
    if (rank < TOP) { tb[rank] = li; }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < SETS) {
    uint t = li;
    uint magic = MAGIC;
    uint p0 = tb[0];
    uint p1 = tb[1];
    uint p2 = tb[2];
    uint p3 = tb[3];
    uint p4 = tb[4];
    if (t >= 1u && t <= UNKNOWN * SWAP) {
      uint a = (t - 1u) / SWAP;
      uint b = tb[UNKNOWN + (t - 1u) % SWAP];
      if (a == 0u) { p0 = b; } else if (a == 1u) { p1 = b; } else if (a == 2u) { p2 = b; } else if (a == 3u) { p3 = b; } else { p4 = b; }
    }
    uint fl = t > UNKNOWN * SWAP ? t - (UNKNOWN * SWAP + 1u) : UNKNOWN;
    uint m0 = li * 30u;
    setRow(m0, rg, p0, hb[p0] ^ (fl == 0u ? 0x80u >> wk[p0] : 0u), magic);
    setRow(m0 + 6u, rg, p1, hb[p1] ^ (fl == 1u ? 0x80u >> wk[p1] : 0u), magic);
    setRow(m0 + 12u, rg, p2, hb[p2] ^ (fl == 2u ? 0x80u >> wk[p2] : 0u), magic);
    setRow(m0 + 18u, rg, p3, hb[p3] ^ (fl == 3u ? 0x80u >> wk[p3] : 0u), magic);
    setRow(m0 + 24u, rg, p4, hb[p4] ^ (fl == 4u ? 0x80u >> wk[p4] : 0u), magic);
    bool ok = true;
    for (uint c = 0u; c < 5u; c++) {
      uint pr = 5u;
      for (uint r = c; r < 5u; r++) { if (pr == 5u && mat[m0 + 6u * r + c] != 0u) { pr = r; } }
      if (pr == 5u) { ok = false; break; }
      if (pr != c) {
        for (uint j = 0u; j < 6u; j++) { uint x = mat[m0 + 6u * c + j]; mat[m0 + 6u * c + j] = mat[m0 + 6u * pr + j]; mat[m0 + 6u * pr + j] = x; }
      }
      uint iv = gexp(255u - glog(mat[m0 + 6u * c + c]));
      for (uint j = 0u; j < 6u; j++) { mat[m0 + 6u * c + j] = gmul(mat[m0 + 6u * c + j], iv); }
      for (uint r = 0u; r < 5u; r++) {
        uint fr = r == c ? 0u : mat[m0 + 6u * r + c];
        if (fr != 0u) { for (uint j = 0u; j < 6u; j++) { mat[m0 + 6u * r + j] ^= gmul(fr, mat[m0 + 6u * c + j]); } }
      }
    }
    uint lo = mat[m0 + 5u] | (mat[m0 + 11u] << 8u) | (mat[m0 + 17u] << 16u) | (mat[m0 + 23u] << 24u);
    uint hi = mat[m0 + 29u];
    sv[li] = uint4(lo, hi, ok && wordOk(lo, hi) ? 1u : 0u, 0u);
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint t = li; t < SETS * 16u; t += 256u) {
    uint w = t & 15u;
    uint4 c = sv[t >> 4u];
    float sc = 0.0;
    if (w < nw && c.z != 0u) {
      uint cw = W.code[VECS * at0 + (w >> 2u)][w & 3u];
      for (uint b = 0u; b < 32u; b++) {
        if (((c.x >> b) & 1u) != 0u) { cw ^= W.code[VECS * (at0 + 1u + b) + (w >> 2u)][w & 3u]; }
      }
      for (uint b = 0u; b < 8u; b++) {
        if (((c.y >> b) & 1u) != 0u) { cw ^= W.code[VECS * (at0 + 33u + b) + (w >> 2u)][w & 3u]; }
      }
      for (uint b = 0u; b < 8u; b++) {
        uint m = cw >> (4u * b);
        sc += dot(qs[8u * w + b], select((uint4(m) & uint4(1u, 2u, 4u, 8u)) != uint4(0u), float4(1.0), float4(-1.0)));
      }
    }
    part[t] = sc;
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < SETS) {
    float t = 0.0;
    for (uint k = 0u; k < 16u; k++) { t += part[16u * li + k]; }
    tot[li] = t;
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    float best = -1e9;
    uint bs = SETS;
    for (uint s = 0u; s < SETS; s++) { if (sv[s].z != 0u && tot[s] > best) { best = tot[s]; bs = s; } }
    uint4 c = sv[min(bs, SETS - 1u)];
    wu = uint4(c.x, c.y, bs < SETS ? 1u : 0u, asuint(best));
  }
  UNIFORM_LOAD(uint4 wn, wu)
  float3 g = float3(0.0);
  for (uint bit = li; bit < nb; bit += 256u) {
    if (abs(q[bit]) < TAU) { continue; }
    uint s = 1u << (bit / vr.x);
    float one = 1024.0 + (((q[bit] > 0.0) != (codeBit(at0, wn.x, wn.y, bit) != 0u)) ? 1.0 : 0.0);
    g += select(bool3(true, (s & d1) == 0u, (s & d2) == 0u), float3(one), float3(0.0));
  }
  red[li] = float4(g, 0.0);
  reduce(li);
  if (li == 0u) {
    float4 G = red[0];
    uint m0 = f2u(G.x / 1024.0);
    uint m1 = f2u(G.y / 1024.0);
    uint m2 = f2u(G.z / 1024.0);
    uint e0 = gateAt(rg, m0);
    uint e1 = gateAt(rg, m1);
    uint e2 = gateAt(rg, m2);
    bool through = (e0 != 0xffffu && f2u(G.x) - 1024u * m0 <= e0) || (e1 != 0xffffu && f2u(G.y) - 1024u * m1 <= e1) || (e2 != 0xffffu && f2u(G.z) - 1024u * m2 <= e2);
    bool ok = con >= CON_MIN && wn.z != 0u && through;
    uint2 x = bitsOfWord(wn.x, wn.y);
    uint2 qc = splitIndex(x);
    uint v = VERSION_MIN;
    for (uint k = 1u; k <= VERSION_MAX - VERSION_MIN; k++) { if (beforeAt(k) <= qc.x) { v = VERSION_MIN + k; } }
    wix = uint4(ok ? 1u : 0u, v, qc.x - beforeAt(v - VERSION_MIN), qc.y);
  }
  // the configuration from its index (src/fmt.c ob_fmt_config): the version from the Word uniform's before
  // (lane 0, above), each lane a 6/7 count's layouts (FMT_P3 at the room past it, none where it passes the room),
  // then lane 0 walks the 6/7 count, the 3/4 count (FMT_P2), the 2/3 count, and the 1/2 count is what remains
  UNIFORM_LOAD(uint4 wx, wix)
  uint room = roomOf(wx.y);
  na[li] = wx.x != 0u && UNIT0 * li <= room ? p3Of(room - UNIT0 * li) : 0u;
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    bool ok = wx.x != 0u;
    uint v = wx.y;
    uint rem = wx.z;
    uint a = 0u;
    while (a < 255u && rem >= na[a] && na[a] != 0u) { rem -= na[a]; a++; }
    uint r = room - min(room, UNIT0 * a);
    uint b = 0u;
    for (; UNIT1 * b <= r; b++) { uint k = p2Of(r - UNIT1 * b); if (rem < k) { break; } rem -= k; }
    r = r - min(r, UNIT1 * b);
    uint t = 0u;
    for (; UNIT2 * t <= r; t++) { uint k = (r - UNIT2 * t) / UNIT3 + 1u; if (rem < k) { break; } rem -= k; }
    uint2 g = uint2((v << 24u) | (min(a, 255u) << 16u) | (min(b, 255u) << 8u) | min(t, 255u), min(rem, 255u));
    ok = ok && cfgOk(g);
    g = ok ? g : uint2(0u);
    uint fps = wx.w == 0u ? 0u : 1u + (wx.w - 1u) / DELAY_MAX;
    uint delay = wx.w == 0u ? 1u : 1u + (wx.w - 1u) % DELAY_MAX;
    result[f].r[5] = ok ? float(v) : 0.0;
    result[f].r[6] = ok ? asfloat(bitsOfWord(wn.x, wn.y).y) : 0.0;
    result[f].r[30] = ok ? asfloat(fps | (delay << 8u) | (g.y << 16u)) : 0.0;
    result[f].r[29] = asfloat(g.x);
    sel[f] = ok ? uint4(rg, 1u + g.x, slotOf(v), g.y) : uint4(rg, 0u, 0u, 0u);
  }
}
