// ai: The format's codes and bit-level tables as the GPU's hosts take them from the C codec, one definition for the
// ai: decoder's (dec/build.cpp: the rule stage's TAB, the LDPC's uniforms) and the sender's (tx/send_tables.cpp: the
// ai: paint's TAB and uniform): each code's shape and block-row tables (src/ldpc.h ldpc_tables), LINEAR's bit map and
// ai: the whitening (src/lizard_phy.c perm_fill, sent_bit, whiten_fill), the CRC-64/LIZARD split's powers (src/layout.c
// ai: ob_crc64's generator). Header-only, so a build of either host alone (the sender's under wasm too) takes it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "fmt.h"
#include "ldpc.h"
#include "lizard_phy.h"
}

namespace lizard::codes {

// ai: the codes' names in OB_TIER_RATE's order
inline const char* const NAMES[OB_TIERS] = {"6/7", "3/4", "2/3", "1/2"};

// ai: A code (ldpc_tables: lay is lay_ptr [mb + 1], then (col, shift) a slot), from a format of one block of each, in
// ai: OB_TIER_RATE's order; nt its sent bits (n - np), blockSlots its block's slots (lizard_block_slots)
struct Code {
  int n = 0, k = 0, m = 0, z = 0, zp = 0, mb = 0, kb = 0, norm = 0, slotsMax = 0, slotsTotal = 0, slots = 0, np = 0, nt = 0, blockSlots = 0;
  std::vector<int32_t> lay;
};
inline std::vector<Code> read() {
  lizard_tier_t tiers[LIZARD_TIERS];
  for (int t = 0; t < LIZARD_TIERS; t++) tiers[t] = {OB_TIER_RATE[t], 1, 0};
  lizard_t f;
  if (lizard_init_tiers(&f, LIZARD_PICTURES[0], tiers, LIZARD_TIERS, 0) != 0) throw std::runtime_error("the codec made no format of one block of each code");
  std::vector<Code> out;
  for (int t = 0; t < LIZARD_TIERS; t++) {
    int32_t d[12] = {};
    ldpc_tables(&f.code[t], d, nullptr);
    Code c;
    c.n = d[0]; c.k = d[1]; c.m = d[2]; c.z = d[3]; c.zp = d[4]; c.mb = d[5]; c.kb = d[6]; c.norm = d[7]; c.slotsMax = d[8]; c.slotsTotal = d[9]; c.slots = d[10]; c.np = d[11];
    c.nt = c.n - c.np;
    c.blockSlots = lizard_block_slots(OB_TIER_RATE[t]);
    if (c.nt > c.blockSlots || (c.blockSlots - c.nt) % 2) {
      lizard_free(&f);
      throw std::runtime_error(std::string("the ") + NAMES[t] + " code sends " + std::to_string(c.nt) + " bits, past its block's " + std::to_string(c.blockSlots) + " slots");
    }
    c.lay.resize(static_cast<size_t>(c.mb + 1 + 2 * c.slots));
    ldpc_tables(&f.code[t], d, c.lay.data());
    out.push_back(std::move(c));
  }
  lizard_free(&f);
  return out;
}

// ai: LINEAR's slot i takes sent bit i P mod n, P the integer nearest n over the golden ratio squared stepped up until
// ai: prime to n; sent bit j is codeword bit sentBit(j): every bit but the code's last np data bits (the 6/7 code's
// ai: zeros past the block)
inline std::vector<int> perm(int n) {
  int P = static_cast<int>(std::floor(0.3819660113 * n + 0.5));
  while (std::gcd(P, n) != 1) P++;
  std::vector<int> p(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) p[static_cast<size_t>(i)] = static_cast<int>((static_cast<int64_t>(i) * P) % n);
  return p;
}
inline int sentBit(const Code& c, int j) { return j < c.k - c.np ? j : j + c.np; }

// ai: the PRBS-23 (x^23 + x^18 + 1 from all ones), a bit a slot of the frame
inline std::vector<uint8_t> whiten(int count) {
  std::vector<uint8_t> w(static_cast<size_t>(count));
  uint32_t s = 0x7fffff;
  for (size_t i = 0; i < w.size(); i++) { const uint32_t b = ((s >> 22) ^ (s >> 17)) & 1; s = ((s << 1) | b) & 0x7fffff; w[i] = static_cast<uint8_t>(b); }
  return w;
}
// ai: the whitening's slots over the ladder's largest frame (every version's at the top)
constexpr int WHITE_SLOTS = LIZARD_STEP * OB_FMT_VERSION_MAX * 2 * LIZARD_SUB;

// ai: The CRC-64/LIZARD split's powers: the CRC of len bytes is the XOR over 4-byte chunks of (the chunk from a zero
// ai: state) times x^(8 x the bytes after it), plus the initial state pushed through the message, then the final
// ai: inversion (both halves linear in the message). Chunk w's power as (high, low), then the initial state's term with
// ai: the inversion folded in, to a whole vec4u; reflected, x^0 at bit 63
inline std::vector<uint32_t> crcPowers(int len) {
  constexpr uint64_t P = 0xc02c38b47d32d307ull, ONES = ~0ull;
  auto mulx = [](uint64_t c) { return (c >> 1) ^ (c & 1 ? P : 0); };
  auto xpow = [&](int e) { uint64_t c = 1ull << 63; for (int i = 0; i < e; i++) c = mulx(c); return c; };
  auto mulPoly = [&](uint64_t c, uint64_t p) { uint64_t r = 0, t = c; for (int j = 0; j < 64; j++) { if ((p >> (63 - j)) & 1) r ^= t; t = mulx(t); } return r; };
  const int words = (len + 3) / 4;
  std::vector<uint32_t> pw(4 * static_cast<size_t>((2 * words + 2 + 3) / 4), 0);
  auto put = [&](int i, uint64_t v) { pw[2 * static_cast<size_t>(i)] = static_cast<uint32_t>(v >> 32); pw[2 * static_cast<size_t>(i) + 1] = static_cast<uint32_t>(v); };
  for (int w = 0; w < words; w++) put(w, xpow(8 * (len - std::min(4 * w + 4, len))));
  put(words, mulPoly(ONES, xpow(8 * len)) ^ ONES);
  return pw;
}

}  // namespace lizard::codes
