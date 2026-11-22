// The LDPC's learned stop for a code and the CRC-64 split's powers, shared by the rate profile's stage (tiers.mjs,
// glsl/back_tiers.mjs) and the sender's paint (../encoder.mjs, wgsl/send.mjs).
import { loadStop, ruleUrlFor } from "./stop/rule.mjs";

export const PAYLOAD = 472;                   // bytes a block, the id first (src/lizard_phy.h LIZARD_BLOCK)

// ai: A code's learned stop (DESIGN section 6), what glsl/back_tiers.mjs builds the code's kernel from: after
// ai: iterations 1 to 29 the MLP of stop/ on (bad, low, first, prev, t, grow) gives up when its score is under its
// ai: threshold, each code its own file by its m checks (stop/rule.mjs ruleUrlFor, loadStop, once a file). It replaced
// ai: the funnel and the C's stall test on 2026-09-28 (scripts/gpu/back/stop/README.md).
export async function stopFor(m) {
  const net = await loadStop(ruleUrlFor(m));
  if (net.m !== m) throw new Error(`stop rule: the file for ${m} checks was trained on a code of ${net.m}`);
  return net;
}
// ---- the CRC-64 split (DESIGN section 7) ----
//
// The block's CRC is the BCH-derived CRC-64 (src/layout.c ob_crc64: 0xE0CB4CBE2D1C3403, reflected 0xC02C38B47D32D307,
// init and final xor all ones; SPEC 7.1). The reflected bitwise step is multiplication by x in GF(2)[x] / P with x^0 at bit 63. Both halves of the
// algorithm are linear in the message, so the CRC of 472 bytes is the XOR over 4-byte chunks of (chunk from a zero
// state) times x^(8 * bytes after the chunk), plus the initial state pushed through the whole message, then the final
// inversion. The shader gives each lane one chunk, its power and a 64-step multiply, a 64-bit value as two u32 (high,
// low).
const P = 0xc02c38b47d32d307n, ONES = (1n << 64n) - 1n;   // ai: the block CRC's generator, reflected
const mulx = (c) => (c >> 1n) ^ (c & 1n ? P : 0n);
function xpow(e) { let c = 1n << 63n; for (let i = 0; i < e; i++) c = mulx(c); return c; }
// c times the polynomial p (both in the reflected representation).
function mulPoly(c, p) { let r = 0n, t = c; for (let j = 0n; j < 64n; j++) { if ((p >> (63n - j)) & 1n) r ^= t; t = mulx(t); } return r; }
// pw[2 w], pw[2 w + 1] (high, low) for chunk w (bytes 4 w .. 4 w + 3, the last one shorter where len is no multiple of 4);
// at 2 words the init term with the final inversion folded in; zeros to a whole vec4. What the shader's uniform carries.
export function crcPowers(len = PAYLOAD) {
  const words = Math.ceil(len / 4), pw = new Uint32Array(4 * Math.ceil((2 * words + 2) / 4));
  const put = (i, v) => { pw[2 * i] = Number(v >> 32n); pw[2 * i + 1] = Number(v & 0xffffffffn); };
  for (let w = 0; w < words; w++) { const end = Math.min(4 * w + 4, len); put(w, xpow(8 * (len - end))); }
  put(words, mulPoly(ONES, xpow(8 * len)) ^ ONES);
  return pw;
}
// The split's arithmetic on a block's bytes, the shader's: the CRC as a BigInt (held to ob_crc64 by its check).
export function crcSplit(bytes, pw = crcPowers(bytes.length)) {
  const words = Math.ceil(bytes.length / 4);
  let acc = 0n;
  for (let w = 0; w < words; w++) {
    let c = 0n;
    for (let t = 0; t < 4 && 4 * w + t < bytes.length; t++) { c ^= BigInt(bytes[4 * w + t]); for (let q = 0; q < 8; q++) c = mulx(c); }
    acc ^= mulPoly(c, (BigInt(pw[2 * w]) << 32n) | BigInt(pw[2 * w + 1]));
  }
  return acc ^ ((BigInt(pw[2 * words]) << 32n) | BigInt(pw[2 * words + 1]));
}
