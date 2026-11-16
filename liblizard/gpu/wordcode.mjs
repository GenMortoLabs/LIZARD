// The format word as a linear code, for the GPU reader (glsl/word.mjs). The word is its magic and four bytes, static
// ai: [0x4C, version, fps, ext0, ext1] or animated [0xB3, position, three bytes of a carousel symbol] (since 2026-11-15;
// ai: [0x4C, version, fps] before), and bytes - 5 Reed-Solomon parity bytes, one code a ring over every word cell:
// ai: RS(16,5), (32,5), (64,5) at W = 32, 64, 128 word cells a side, the 64, 128 and 256 rings (src/fmt.c, src/rs.c:
// ai: GF(256), polynomial 0x11D, roots alpha^0 upward, data first).
// Reed-Solomon is linear over GF(2), so every valid word is its magic's BASE xor the rows of its four bytes' set bits:
// ai: (VERSION_MAX - VERSION_MIN + 1) x 2^24 static words and (POS_MAX + 1) x 2^24 animated ones a ring (any ring may
// ai: carry any version since 2026-09-27): the reader solves the word from its most reliable bytes by Gauss-Jordan over
// ai: GF(256) and scores the candidates' bits over this code (glsl/word.mjs), where the reference decodes hard bytes
// ai: with erasures.
//
// Bit order is the reader's: q index side * W + cell, cell 0..W - 1 along the side; byte k lives on side k % 4 at
// cells 8 (k / 4) .. + 7, most significant bit first (src/fmt.c ob_fmt_bit). Cells past the codeword are light.
import { N_FOR, subchOf } from "../sim/lizard_pick.mjs";

const EXP = new Uint8Array(512), LOG = new Uint8Array(256);
{ let x = 1; for (let i = 0; i < 255; i++) { EXP[i] = x; LOG[x] = i; x <<= 1; if (x & 256) x ^= 0x11d; } for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255]; }
const mul = (a, b) => (a && b ? EXP[LOG[a] + LOG[b]] : 0);

export const BYTES_MAX = 64;
// ai: The word's first byte (src/fmt.h OB_FMT_MAGIC, OB_FMT_MAGIC_ANIM), its data bytes (OB_FMT_DATA) and the last
// ai: carousel position (OB_FMT_POS_MAX)
export const MAGIC = 0x4c, MAGIC_ANIM = 0xb3, DATA = 5, POS_MAX = 254;
// ai: A static word's data bytes and an animated one's
export const staticWord = (version, fps, ext = [0, 0]) => [MAGIC, version, fps, ext[0], ext[1]];
export const animWord = (pos, sym) => [MAGIC_ANIM, pos, sym[0], sym[1], sym[2]];
// ai: The versions a word states (src/fmt.h OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX): LIZARD-64's 4 to LIZARD-1024's
// ai: 64, sub-channels / 16 (sim/lizard_pick.mjs STEP, since 2026-10-29); a version byte of 0 to 3 or 65 to 255 is refused.
export const VERSION_MIN = 4;
export const VERSION_MAX = 64;
// The codeword's length for W word cells a side (src/fmt.c ob_fmt_bytes; no word under the 64 ring's 32 cells,
// ai: src/fmt.h OB_FMT_SIDE_CELLS).
export const bytesFor = (W) => (W < 32 ? 0 : Math.min(4 * Math.floor(W / 8), BYTES_MAX));

// src/rs.c rs_encode, for the word's 5 data bytes (`data`, its magic first) and bytes - 5 roots.
export function wordBytes(data, bytes) {
  const k = DATA, nroots = bytes - k, cw = new Uint8Array(bytes), g = new Uint8Array(nroots + 1);
  cw.set(data.slice(0, k));
  g[0] = 1;
  for (let i = 0; i < nroots; i++) { g[i + 1] = 0; for (let j = i + 1; j > 0; j--) g[j] ^= mul(g[j - 1], EXP[i]); }
  const par = new Uint8Array(nroots);
  for (let i = 0; i < k; i++) {
    const fb = cw[i] ^ par[0];
    par.copyWithin(0, 1); par[nroots - 1] = 0;
    if (fb) for (let j = 0; j < nroots; j++) par[j] ^= mul(fb, g[j + 1]);
  }
  cw.set(par, k);
  return cw;
}

// The word's 4W bits in the reader's order, as u32 (q index 0..31 in the first, and so on).
export function wordBits(cw, W) {
  const out = new Uint32Array(Math.ceil((4 * W) / 32));
  for (let side = 0; side < 4; side++) for (let c = 0; c < W; c++) {
    const i = side * W + c, k = 4 * (c >> 3) + side;
    if (k < cw.length && (cw[k] >> (7 - (c & 7))) & 1) out[i >> 5] |= 1 << (i & 31);
  }
  return out;
}

// The two magics' BASEs (static, animated) and the 32 rows (data byte 1's bits 0..7, then byte 2's, 3's and 4's) for W
// word cells a side.
export function wordCode(W) {
  const bytes = bytesFor(W), bits = (d) => wordBits(wordBytes(d, bytes), W);
  const base = [bits([MAGIC, 0, 0, 0, 0]), bits([MAGIC_ANIM, 0, 0, 0, 0])], rows = [];
  for (let j = 1; j < DATA; j++) for (let b = 0; b < 8; b++) { const d = [0, 0, 0, 0, 0]; d[j] = 1 << b; rows.push(bits(d)); }
  return { W, bytes, base, rows };
}

// ai: The word's gate (2026-10-25, research STATUS "The ring's word: the C's RS reader against the GPU's search"): the
// ai: best word is taken if it disagrees with few enough of the cells that count, the cells at |q| >= TAU of the sides
// ai: counted. The sides counted are tried three ways (GATE_CHOICES): all four, all but the one with the least sum of
// ai: |q|, all but the two least; the word is taken if any of the three passes. A glared side reads its signs as a coin
// ai: at |q| about 0.33 where a clean one reads 0.5 (scripts/exp/fmt_word.mjs DAMAGE), so no threshold on a cell finds
// ai: it, and one fixed share of the best side's sum was too thin a cut (scripts/gpu/word_check.mjs: a glared side at
// ai: 0.758 of the best kept, a clean one at 0.74 dropped). E[m] is the most disagreements on m cells that keep the
// ai: union bound (every word of the ring within E[m] of a random hard word on those m cells) at BUDGET / GATE_CHOICES,
// ai: so the three together stay at BUDGET, the C's own false-word rate in that ring (SPEC 5.4, three roots left: its
// ai: steps' pass rate on garbage times a magic's 1/256 and its range's share of the byte after it, cFalseWord below);
// ai: -1 where no count does. The bound counts every word a word may be, WORDS: (VERSION_MAX - VERSION_MIN + 1) x 2^24
// ai: static and (POS_MAX + 1) x 2^24 animated. Which cells count depends on the magnitudes
// ai: alone, which a garbage frame's signs do not, so the bound holds for each choice as it stands. SIDE_KEEP is the
// ai: search's: F8 zeroes a side under SIDE_KEEP of the best side's sum, at most the two least, before it scores
// ai: (0.75, the best of 0.85, 0.75 and 0.70 on the lab's damage sweep).
export const TAU = 0.1, SIDE_KEEP = 0.75, GATE_CHOICES = 3;
// ai: The C's false-word rate on garbage (soft values uniform in -0.5 to 0.5) in a ring of W word cells a side, exact
// ai: as test/fmt_test.c p_pass has it: RS(bytes, 5) passing errors only, then each erasure step (2, 4, ... nroots - 3
// ai: least confident bytes, src/fmt.c ob_fmt_decode), a false codeword's data bytes uniform, so 1/256 for each magic
// ai: and the versions' share of the version byte, or the positions' of the position byte.
export const WORDS = (VERSION_MAX - VERSION_MIN + 1 + POS_MAX + 1) * 2 ** 24;
export function cFalseWord(W) {
  const bytes = bytesFor(W), nroots = bytes - DATA;
  const binom = (n, k) => { let r = 1; for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1); return r; };
  const pass = (f) => { const t = Math.floor((nroots - f) / 2); let v = 0; for (let i = 0; i <= t; i++) v += binom(bytes - f, i) * 255 ** i; return v / 256 ** (nroots - f); };
  let steps = pass(0);
  for (let f = 2; f <= nroots - 3; f += 2) steps += pass(f);
  return (steps * (VERSION_MAX - VERSION_MIN + 1 + POS_MAX + 1)) / 65536;
}
export const BUDGET = Object.fromEntries([32, 64, 128].map((W) => [W, cFalseWord(W)]));
export function gateTable(W) {
  if (!(W in BUDGET)) throw new Error(`a ring of ${W} word cells a side: the gate holds ${Object.keys(BUDGET).join(", ")}`);
  const nb = 4 * W, words = WORDS, E = new Int16Array(nb + 1).fill(-1), budget = BUDGET[W] / GATE_CHOICES;
  for (let m = 0; m <= nb; m++) {
    let s = 0, c = 1;
    for (let d = 0; d <= m; d++) { s += c; if ((words * s) / 2 ** m > budget) break; E[m] = d; c = (c * (m - d)) / (d + 1); }
  }
  return E;
}
// ai: The sides as F8 orders them: the bit of the least sum of |q| among the sides not in `skip` (ties to the lower side).
export function leastSide(sums, skip) {
  let b = 0, v = Infinity;
  for (let s = 0; s < 4; s++) if (!((skip >> s) & 1) && sums[s] < v) { v = sums[s]; b = 1 << s; }
  return b;
}
// ai: The three ways the gate counts sides (bit s for side s), and the sides the search keeps.
export function gateChoices(sums) {
  const d1 = leastSide(sums, 0), d2 = d1 | leastSide(sums, d1);
  return [15, 15 & ~d1, 15 & ~d2];
}
export function searchKeep(sums) {
  const top = SIDE_KEEP * Math.max(...sums), d2 = gateChoices(sums)[2] ^ 15;
  let low = 0;
  for (let s = 0; s < 4; s++) if (sums[s] < top) low |= 1 << s;
  return 15 & ~(low & d2);
}

// ai: The versions whose picture is n (sim/lizard_pick.mjs N_FOR), first and last: the word names the version, and the
// ai: version the picture (gpu/tables.mjs PICTURES).
export function versionRange(n) {
  let lo = 0, hi = -1;
  for (let v = VERSION_MIN; v <= VERSION_MAX; v++) if (N_FOR(subchOf(v)) === n) { if (!lo) lo = v; hi = v; }
  return [lo, hi];
}
