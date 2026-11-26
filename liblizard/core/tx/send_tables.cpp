#include "send_tables.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

#include "../dec/codes.h"
#include "send_plan.h"

namespace lizard {

SendConsts::SendConsts() : pw(codes::crcPowers(plan::PAYLOAD)) {
  const auto cs = codes::read();
  if (static_cast<int>(cs.size()) != plan::NC || send::SLOTS < 1 + plan::NC) throw std::runtime_error("the paint's uniform holds " + std::to_string(send::SLOTS) + " sizes, too few for the codes' tiers");
  int o = 16 + 16 * static_cast<int>(cs.size());
  std::vector<int> rows, lay, pm;
  for (const auto& c : cs) { rows.push_back(o); o += c.mb + 1; lay.push_back(o); o += c.slots; pm.push_back(o); o += c.nt / 2; }
  const int white = o;
  o += codes::WHITE_SLOTS / 32 + 1;
  tab.assign(static_cast<size_t>(o), 0);
  tab[0] = static_cast<uint32_t>(white);
  for (size_t q = 0; q < cs.size(); q++) {
    const auto& c = cs[q];
    // ai: a block's slots its code's sent bits alone (the pilots apart from the blocks: no tail to paint)
    if (c.nt % 2 || c.blockSlots != c.nt || c.n / 32 > send::CW_WORDS)
      throw std::runtime_error(std::string("the ") + codes::NAMES[q] + " code's n " + std::to_string(c.n) + " (" + std::to_string(c.nt) + " sent, its block " + std::to_string(c.blockSlots) + " slots) does not fit the paint");
    codeRate.push_back(OB_TIER_RATE[q]);
    codeSlots.push_back(c.blockSlots);
    const uint32_t head[10] = {static_cast<uint32_t>(c.n), static_cast<uint32_t>(c.k), static_cast<uint32_t>(c.m), static_cast<uint32_t>(c.z), static_cast<uint32_t>(c.mb),
                               static_cast<uint32_t>(rows[q]), static_cast<uint32_t>(lay[q]), static_cast<uint32_t>(pm[q]), static_cast<uint32_t>(c.blockSlots), static_cast<uint32_t>(c.nt)};
    std::copy(head, head + 10, tab.begin() + 16 + 16 * static_cast<long>(q));
    for (int r = 0; r <= c.mb; r++) tab[static_cast<size_t>(rows[q] + r)] = static_cast<uint32_t>(c.lay[static_cast<size_t>(r)]);
    for (int e = 0; e < c.slots; e++)
      tab[static_cast<size_t>(lay[q] + e)] = static_cast<uint32_t>(c.lay[static_cast<size_t>(c.mb + 1 + 2 * e)]) | static_cast<uint32_t>(c.lay[static_cast<size_t>(c.mb + 2 + 2 * e)]) << 16;
    const auto p = codes::perm(c.nt);
    for (int i = 0; i < c.nt; i++) tab[static_cast<size_t>(pm[q] + (i >> 1))] |= static_cast<uint32_t>(codes::sentBit(c, p[static_cast<size_t>(i)])) << (16 * (i & 1));
  }
  const auto w = codes::whiten(codes::WHITE_SLOTS);
  for (size_t i = 0; i < w.size(); i++) if (w[i]) tab[static_cast<size_t>(white) + (i >> 5)] |= 1u << (i & 31);
}

namespace {
// ai: the coefficient at pos p as (u, v), the C's half-plane block layout (src/lizard_phy.c init)
void uvOf(int p, int n, int bw, int& u, int& v) {
  const int nb = n * bw;
  v = (p / nb) * bw + ((p % nb) % bw);
  const int uw = (p % nb) / bw;
  u = uw < n / 2 ? uw : uw - n;
}
uint32_t bitsOf(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
}  // namespace

// ai: the border: the C's paint of the symbol with every block zero (only the square depends on them, and not the
// ai: anchors' squares in it: lizard_encode paints them over the picture, the same whatever the blocks)
void sendBorder(const lizard_t& f, int W, std::vector<uint8_t>& border) {
  std::vector<uint8_t> zero(static_cast<size_t>(f.blocks) * f.block_bytes, 0), rgba(static_cast<size_t>(W) * W * 4);
  std::vector<float> drive(static_cast<size_t>(f.px) * f.px);
  lizard_encode(&f, zero.data(), drive.data());
  lizard_paint_rgba(&f, drive.data(), rgba.data());
  border.assign(4 * ((static_cast<size_t>(W) * W + 3) / 4), 0);
  for (size_t i = 0; i < static_cast<size_t>(W) * W; i++) border[i] = rgba[4 * i];
}

std::string sendTables(const lizard_t& f, const SendConsts& k, int frames, SendTables& t, int codes, int gap) {
  t = SendTables{};
  t.n = f.n; t.subch = f.subch; t.blocks = f.blocks; t.blockBytes = f.block_bytes; t.npos = f.subch * LIZARD_SUB;
  t.sdWords = f.subch * (LIZARD_SUB - 2) / 16;
  const int n = f.n, bw = f.bw;
  // ai: the spectrum transposed (the web encoder's too, through the wasm), so the inverse's second pass runs along
  // ai: the picture's rows (slang/src/send_ipic.slang writes it row-major): (u, v) at (v, u) where u > 0 or u = 0, else at its
  // ai: Hermitian partner's (-v, -u) conjugated (UV's bit 31); row v holds u = ustart .. umaxP then -umaxN .. -1
  // ai: (ustart 1 on v = 0), packed at off[v]
  const auto tOf = [&](int p, int& u, int& v) {
    int a, b;
    uvOf(p, n, bw, a, b);
    if (a > 0 || (a == 0 && b > 0)) { u = b; v = a; return 0u; }
    u = -b; v = -a;
    return 1u;
  };
  std::map<int, std::vector<int>> byV;
  for (int i = 0; i < t.npos; i++) { int u, v; tOf(f.pos[i], u, v); byV[v].push_back(u); }
  if (byV.empty()) return "no coefficients";
  t.Vr = byV.rbegin()->first + 1;
  std::vector<uint32_t> off(t.Vr), np(t.Vr), un(t.Vr);
  t.rows.assign(2048, 0);
  if (4 * t.Vr > 2048) return "the disc has " + std::to_string(t.Vr) + " rows, past the uniform's 512";
  uint32_t count = 0;
  for (int v = 0; v < t.Vr; v++) {
    const std::vector<int> us = byV.count(v) ? byV[v] : std::vector<int>{};
    const int ustart = v == 0 ? 1 : 0;
    int maxP = ustart - 1, maxN = 0;
    for (int u : us) { if (u >= 0) maxP = std::max(maxP, u); else maxN = std::max(maxN, -u); }
    const std::set<int> have(us.begin(), us.end());
    bool ok = static_cast<int>(us.size()) == maxP + 1 - ustart + maxN && have.size() == us.size();
    for (int u = ustart; u <= maxP && ok; u++) ok = have.count(u) != 0;
    for (int u = -maxN; u < 0 && ok; u++) ok = have.count(u) != 0;
    if (!ok) return "disc row v = " + std::to_string(v) + " is not contiguous";
    off[v] = count; np[v] = static_cast<uint32_t>(maxP + 1 - ustart); un[v] = static_cast<uint32_t>(maxN);
    t.rows[4 * v] = count; t.rows[4 * v + 1] = static_cast<uint32_t>(us.size()); t.rows[4 * v + 2] = np[v]; t.rows[4 * v + 3] = un[v];
    count += static_cast<uint32_t>(us.size());
  }
  if (static_cast<int>(count) != t.npos) return "the disc holds " + std::to_string(count) + " entries for " + std::to_string(t.npos);
  t.uv.resize(t.npos);
  for (int i = 0; i < t.npos; i++) {
    int u, v; const uint32_t cj = tOf(f.pos[i], u, v);
    const int ustart = v == 0 ? 1 : 0;
    t.uv[i] = (off[v] + (u >= 0 ? static_cast<uint32_t>(u - ustart) : np[v] + un[v] + static_cast<uint32_t>(u))) | cj << 31;
  }
  // ai: the twiddles as the web computes them, in double, kept as f32
  t.tw.resize(2 * n);
  for (int m = 0; m < n; m++) { const double th = -2 * M_PI * m / n; t.tw[2 * m] = static_cast<float>(std::cos(th)); t.tw[2 * m + 1] = static_cast<float>(std::sin(th)); }
  // ai: src/lizard_phy.c's level at the format's clip ratio (LIZARD_CLIP, SPEC 6.7):
  // ai: fround(fround(2 c) x fround(sqrt(fround(0.5 subch 320)))); its reciprocal halved in double
  const float lim = 2.0f * LIZARD_CLIP * std::sqrt(static_cast<float>(0.5 * f.subch * 320));
  t.su = {static_cast<uint32_t>(n), static_cast<uint32_t>(t.Vr), static_cast<uint32_t>(t.Vr * n), static_cast<uint32_t>(t.npos),
          static_cast<uint32_t>(n * n / 4), 0, 0, 0, bitsOf(lim), bitsOf(static_cast<float>(0.5 / lim)), 0, 0};
  // ai: the resampler (the C's taps): q pixels from the square's first, six taps each from i0, wrapped mod n
  int geo[4];
  const int* i0 = nullptr;
  const float* w = nullptr;
  if (lizard_resample_geom(&f, geo, &i0, &w)) return "the codec has no resampler tables";
  t.q = geo[0];
  if (geo[2] != n) return "the resampler's n " + std::to_string(geo[2]) + " is not the codec's " + std::to_string(n);
  t.taps.resize(12 * static_cast<size_t>(t.q));
  t.copy = true;
  for (int x = 0; x < t.q; x++) for (int s = 0; s < 6; s++) {
    t.taps[6 * x + s] = static_cast<uint32_t>((((i0[x] + s) % n) + n) % n);
    const float wt = w[6 * x + s];
    t.taps[6 * t.q + 6 * x + s] = bitsOf(wt);
    if (wt != (s == 2 ? 1.f : 0.f)) t.copy = false;
  }
  // ai: rsh's strip holds a code's samples for 4 LANES pixels at n at most 2 q (send_plan.h RSH_STRIP)
  if (!t.copy && n > 2 * t.q) return "the picture's " + std::to_string(n) + " samples are over twice its square's " + std::to_string(t.q) + " pixels: rsh's strip does not hold them";
  t.W = f.px + 2 * LIZARD_QUIET * f.pxm;
  t.sq = geo[1] + LIZARD_QUIET * f.pxm;
  if (codes < 1 || codes > 2) return "codes are 1 or 2, not " + std::to_string(codes);
  if (gap < 0 || gap > 64) return "the gap is 0 to 64 modules, not " + std::to_string(gap);
  t.codes = codes;
  t.gap = codes > 1 ? gap * f.pxm : 0;
  t.FW = codes * t.W + (codes - 1) * t.gap; t.RW = (t.FW + 3) / 4;
  t.FS = 64 * ((t.RW * t.W + 63) / 64);
  t.g = {static_cast<uint32_t>(n), static_cast<uint32_t>(t.q), static_cast<uint32_t>(t.sq), static_cast<uint32_t>(t.W), static_cast<uint32_t>(codes),
         static_cast<uint32_t>(t.FW), static_cast<uint32_t>(t.RW), static_cast<uint32_t>(t.FS), 1, 0, static_cast<uint32_t>(t.gap), 0};
  // ai: the paint's uniform: block count, the slots the blocks take and the coefficients the frame holds at slot 0 (the
  // ai: last block's workgroup paints the padding between); each tier at slot 1 + its code's place in TAB (first block,
  // ai: first slot, blocks; src/lizard_phy.c init packs the tiers by slots from the lowest frequencies in order); dims
  // ai: (blocks, npos, symbols an encode, parity | codes << 2: set an encode)
  t.pu.assign(send::P_WORDS, 0);
  int used = 0;
  for (int q = 0, b = 0; q < f.tiers; q++) {
    int c = -1;
    for (size_t i = 0; i < k.codeRate.size(); i++) if (k.codeRate[i] == f.tier[q].rate && k.codeSlots[i] == f.tier[q].slots) c = static_cast<int>(i);
    if (c < 0) return "a tier at rate " + std::to_string(f.tier[q].rate) + " of " + std::to_string(f.tier[q].slots) + " slots a block, which the paint has no code for";
    uint32_t* at = t.pu.data() + send::P_SIZES + 4 * (1 + c);
    if (at[2]) return "two tiers of one code";
    at[0] = static_cast<uint32_t>(b); at[1] = static_cast<uint32_t>(used); at[2] = static_cast<uint32_t>(f.tier[q].blocks);
    b += f.tier[q].blocks; used += f.tier[q].blocks * f.tier[q].slots;
  }
  if (used != f.used) return "the tiers take " + std::to_string(used) + " slots, the codec's blocks " + std::to_string(f.used);
  t.pu[send::P_SIZES] = static_cast<uint32_t>(f.blocks); t.pu[send::P_SIZES + 2] = static_cast<uint32_t>(used); t.pu[send::P_SIZES + 3] = static_cast<uint32_t>(t.npos);
  t.pu[send::P_DIMS] = static_cast<uint32_t>(f.blocks); t.pu[send::P_DIMS + 1] = static_cast<uint32_t>(t.sdWords); t.pu[send::P_DIMS + 2] = static_cast<uint32_t>(frames * codes);
  std::copy(k.pw.begin(), k.pw.end(), t.pu.begin() + send::P_PW);
  sendBorder(f, t.W, t.border);
  // ai: the anchors' runs in a symbol's pixels (src/lizard_phy.c lizard_encode: square i from module m0(i) of the drive, a
  // ai: modules a side, the drive LIZARD_QUIET modules into the symbol), d and e of rsh's G (gpu/slang/src/send_rsh.slang)
  int geo8[8];
  lizard_anchor_geom(&f, geo8);
  t.anch.assign(8, 0);
  for (int i = 0; i < 7; i++) t.anch[i] = static_cast<uint32_t>((geo8[1 + i] + LIZARD_QUIET) * f.pxm);
  t.anch[7] = static_cast<uint32_t>(geo8[0] * f.pxm);
  return "";
}

}  // namespace lizard
