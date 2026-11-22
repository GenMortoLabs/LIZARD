// The format word as a linear code, for the GPU reader (glsl/word.mjs). The word is its magic and five bytes
// ai: [0x4C, then 40 bits, the first the most significant: the word's index (35 bits: (configuration x 241 + fps) x 30
// ai: + delay - 1, the configuration's own index configIndex's) and its CRC-5 (src/fmt.c crc5: x^5 + x^2 + 1)] (since
// ai: 2026-11-22; the version less 4, the 6/7, 2/3 and 1/2 counts and a CRC-4 in 32 bits from 2026-11-18; another magic
// ai: refused: src/fmt.h), and bytes - 6 Reed-Solomon parity bytes, one code a ring over every word cell:
// ai: RS(16,6), (32,6), (64,6) at W = 32, 64, 128 word cells a side, the 64, 128 and 256 rings (src/fmt.c, src/rs.c:
// ai: GF(256), polynomial 0x11D, roots alpha^0 upward, data first).
// Reed-Solomon is linear over GF(2), so every valid word is the magic's BASE xor the rows of its five bytes' set bits:
// ai: WORDS words a ring (any ring may carry any version since 2026-09-27): the reader solves the word from its most
// ai: reliable bytes by Gauss-Jordan over
// ai: GF(256) and scores the candidates' bits over this code (glsl/word.mjs), where the reference decodes hard bytes
// ai: with erasures.
//
// Bit order is the reader's: q index side * W + cell, cell 0..W - 1 along the side; byte k lives on side k % 4 at
// cells 8 (k / 4) .. + 7, most significant bit first (src/fmt.c ob_fmt_bit). Cells past the codeword are light.
import { N_FOR, subchOf, layoutOf } from "../sim/lizard_pick.mjs";

const EXP = new Uint8Array(512), LOG = new Uint8Array(256);
{ let x = 1; for (let i = 0; i < 255; i++) { EXP[i] = x; LOG[x] = i; x <<= 1; if (x & 256) x ^= 0x11d; } for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255]; }
const mul = (a, b) => (a && b ? EXP[LOG[a] + LOG[b]] : 0);

export const BYTES_MAX = 64;
// ai: The word's first byte (src/fmt.h OB_FMT_MAGIC) and its data bytes (OB_FMT_DATA)
export const MAGIC = 0x4c, DATA = 6;
// ai: The versions a word states (src/fmt.h OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX): 4 to 67, sub-channels / 16 (sim/lizard_pick.mjs
// ai: STEP), 64 to 1072 sub-channels (since 2026-11-18; 4 to 64 from 2026-10-29), its six bits' every value.
export const VERSION_MIN = 4;
export const VERSION_MAX = 67;
// ai: The clock (src/fmt.h OB_FMT_FPS_MAX, OB_FMT_DELAY_MAX, OB_FMT_CLOCKS): fps 0 to 240 (0 a still), delay 1 to 30.
export const FPS_MAX = 240, DELAY_MAX = 30, CLOCKS = (FPS_MAX + 1) * DELAY_MAX;
// ai: The configurations in index order (src/fmt.c ob_fmt_index): a version's, then by the 6/7 count a, the 2/3 count t
// ai: and the 1/2 count c, each from 0. A block's slots (src/fmt.c ob_block_slots: its code's sent bits) and a version's
// ai: data slots (OB_FMT_SLOTS); room the slots past a, t and the one 3/4 block a frame needs; cCount the 1/2 counts there.
export const SLOTS67 = 4480, SLOTS34 = 5120, SLOTS23 = 5760, SLOTS12 = 7680, VERSION_SLOTS = 10176;
const room = (v, a, t) => v * VERSION_SLOTS - SLOTS34 - a * SLOTS67 - t * SLOTS23;
const cCount = (r) => (r < 0 ? 0 : Math.min(127, Math.floor(r / SLOTS12)) + 1);
const tCount = (v, a) => { let n = 0; for (let t = 0; t <= 127; t++) { const k = cCount(room(v, a, t)); if (!k) break; n += k; } return n; };
// ai: BEFORE[v - VERSION_MIN]: the configurations of every version before v (src/fmt.c OB_FMT_BEFORE), the last CONFIGS
export const BEFORE = (() => {
  const b = [0];
  for (let v = VERSION_MIN; v <= VERSION_MAX; v++) { let n = 0; for (let a = 0; a <= 255; a++) n += tCount(v, a); b.push(b.at(-1) + n); }
  return b;
})();
export const CONFIGS = BEFORE.at(-1), WORDS = CONFIGS * CLOCKS;
export function configIndex(version, count) {
  if (version < VERSION_MIN || version > VERSION_MAX || !layoutOf(subchOf(version), count).length) return -1;
  let i = BEFORE[version - VERSION_MIN];
  for (let a = 0; a < count[0]; a++) i += tCount(version, a);
  for (let t = 0; t < count[1]; t++) i += cCount(room(version, count[0], t));
  return i + count[2];
}
export function configOf(index) {
  if (!(index >= 0 && index < CONFIGS)) return null;
  let v = VERSION_MIN;
  while (BEFORE[v + 1 - VERSION_MIN] <= index) v++;
  let r = index - BEFORE[v - VERSION_MIN], a = 0, t = 0;
  for (let k; r >= (k = tCount(v, a)); a++) r -= k;
  for (let k; r >= (k = cCount(room(v, a, t))); t++) r -= k;
  return { version: v, count: [a, t, r] };
}
// ai: src/fmt.c crc5: x^5 + x^2 + 1 over the 35-bit index, most significant first, from 0
export function crc5(c) {
  let r = BigInt(c) << 5n;
  for (let i = 39n; i >= 5n; i--) if ((r >> i) & 1n) r ^= 0x25n << (i - 5n);
  return Number(r & 31n);
}
// ai: The word's six data bytes (src/fmt.c ob_fmt_pack): the magic, then the index and its CRC-5 in 40 bits
export function word(version, count, fps = 60, delay = 1) {
  const i = configIndex(version, count);
  if (i < 0 || !(fps >= 0 && fps <= FPS_MAX) || !(delay >= 1 && delay <= DELAY_MAX)) throw new Error(`no word: version ${version}, counts ${count}, clock ${fps}/${delay}`);
  const c = (i * (FPS_MAX + 1) + fps) * DELAY_MAX + delay - 1, x = BigInt(c) * 32n + BigInt(crc5(c));
  return [MAGIC, ...[32n, 24n, 16n, 8n, 0n].map((s) => Number((x >> s) & 255n))];
}
// ai: A word's fields from its five bytes after the magic, or null where its CRC-5 or index refuses it
export function wordOf(bytes) {
  let x = 0n;
  for (const b of bytes) x = (x << 8n) | BigInt(b);
  const c = Number(x >> 5n);
  if (Number(x & 31n) !== crc5(c) || c >= WORDS) return null;
  const k = c % CLOCKS, cfg = configOf(Math.floor(c / CLOCKS));
  return { ...cfg, fps: Math.floor(k / DELAY_MAX), delay: 1 + (k % DELAY_MAX) };
}
// The codeword's length for W word cells a side (src/fmt.c ob_fmt_bytes; no word under the 64 ring's 32 cells,
// ai: src/fmt.h OB_FMT_SIDE_CELLS).
export const bytesFor = (W) => (W < 32 ? 0 : Math.min(4 * Math.floor(W / 8), BYTES_MAX));

// src/rs.c rs_encode, for the word's 6 data bytes (`data`, its magic first) and bytes - 6 roots.
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

// The magic's BASE and the 40 rows (data byte 1's bits 0..7, then byte 2's to 5's) for W word cells a side.
export function wordCode(W) {
  const bytes = bytesFor(W), bits = (d) => wordBits(wordBytes(d, bytes), W);
  const base = [bits([MAGIC, ...Array(DATA - 1).fill(0)])], rows = [];
  for (let j = 1; j < DATA; j++) for (let b = 0; b < 8; b++) { const d = Array(DATA).fill(0); d[j] = 1 << b; rows.push(bits(d)); }
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
// ai: steps' pass rate on garbage times a magic's 1/256 and its check's share of the 40 bits after it, cFalseWord below);
// ai: -1 where no count does. The bound counts every word a word may be, WORDS. Which cells count
// ai: depends on the magnitudes
// ai: alone, which a garbage frame's signs do not, so the bound holds for each choice as it stands. SIDE_KEEP is the
// ai: search's: F8 zeroes a side under SIDE_KEEP of the best side's sum, at most the two least, before it scores
// ai: (0.75, the best of 0.85, 0.75 and 0.70 on the lab's damage sweep).
export const TAU = 0.1, SIDE_KEEP = 0.75, GATE_CHOICES = 3;
// ai: The C's false-word rate on garbage (soft values uniform in -0.5 to 0.5) in a ring of W word cells a side, exact
// ai: as test/fmt_test.c p_pass has it: RS(bytes, 6) passing errors only, then each erasure step (1, 3, ... nroots - 3
// ai: least confident bytes on an even nroots, src/fmt.c ob_fmt_decode), a false codeword's data bytes uniform, so 1/256
// ai: for the magic and the words' share of the 40 bits after it (WORDS / 2^40: every configuration with every clock,
// ai: each with its one CRC-5): 1.7e-8 a read since 2026-11-22 (5.3e-10 with the 32-bit word before).
export function cFalseWord(W) {
  const bytes = bytesFor(W), nroots = bytes - DATA, first = 2 - ((nroots - 3) & 1);
  const binom = (n, k) => { let r = 1; for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1); return r; };
  const pass = (f) => { const t = Math.floor((nroots - f) / 2); let v = 0; for (let i = 0; i <= t; i++) v += binom(bytes - f, i) * 255 ** i; return v / 256 ** (nroots - f); };
  let steps = pass(0);
  for (let f = first; f <= nroots - 3; f += 2) steps += pass(f);
  return (steps * (WORDS / 2 ** 40)) / 256;
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
