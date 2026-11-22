// The sender's paint uniform layout (send.mjs) and the split CRC-64, shared by the rate profile's LDPC kernel
// (back_tiers.mjs) and the sender's paint.
// ai: The one-rate LDPC kernel these came with (ldpcSource: layered min-sum on the 3/4 code, the stall rule and the
// ai: learned stop) went 2026-10-22; the profile's kernel is back_tiers.mjs tldpcSource.
// The CRC-64 is split over lanes (ref_ldpc.mjs crcPowers): lane w runs the reflected bitwise CRC over its 4-byte chunk
// from a zero state, multiplies by x^(8 * bytes after it) mod P from a table, and XORs into two words (high, low); lane
// 0 adds the initial state's term with the final inversion folded in. 118 lanes, 32 steps and a 64-step multiply each,
// a 64-bit value as two u32, no table in workgroup memory.
import { SLOTS } from "../plan/common.mjs";
import { PW_WORDS } from "../plan/back_tiers.mjs";
import { PAYLOAD } from "../back/ref_ldpc.mjs";
// ai: the payload is whole words (472 B, 118), its CRC-64 the two words after it
if (PAYLOAD % 4) throw new Error(`the block's ${PAYLOAD} B are no whole words`);
export const CRC_CHUNKS = PAYLOAD / 4;

// ai: The sender's paint uniform P in u32 words (wgsl/send.mjs paintSource): sizes, a vec4u a picture slot; dims; pw,
// ai: PW_VECS vec4u (the CRC-64 powers, crcPowers(PAYLOAD): each chunk's two words, high then low, then the init term's).
// ai: PARAMS_STRUCT is its WGSL.
export const PW_VECS = Math.ceil((2 * CRC_CHUNKS + 2) / 4);
if (4 * PW_VECS !== PW_WORDS) throw new Error(`the CRC powers take ${4 * PW_VECS} words, the LDPC's uniform ${PW_WORDS}`);
export const PARAMS_AT = { sizes: 0, dims: 4 * SLOTS, pw: 4 * SLOTS + 4, words: 4 * SLOTS + 4 + 4 * PW_VECS };
export const PARAMS_STRUCT = `struct Params { sizes: array<vec4<u32>, ${SLOTS}>, dims: vec4<u32>, pw: array<vec4<u32>, ${PW_VECS}> }`;

// ai: The split CRC-64 as WGSL, shared with the sender's paint (wgsl/send.mjs): crcOf(i) is called by every lane of the
// ai: workgroup with its lane index (it holds two barriers) once the 472 payload bytes sit in the workgroup array `packw`
// ai: (120 u32, four bytes a word in memory order, words 0 to 117; words 118 and 119 the CRC's) and returns the CRC to
// ai: every lane as (high, low). `crcw` names the two workgroup atomics it folds into (array<atomic<u32>, 2>), `crcStep`
// ai: the name the step function is declared under, `pw` the powers table expression (crcPowers(PAYLOAD), as the LDPC's
// ai: uniform holds it: chunk w's power at vec4 w / 2, its .xy where w is even, .zw where odd). The names are parameters
// ai: so a shader with its own arrays binds them without a copy of the arithmetic.
export function CRC_SPLIT({ packw = "packw", crcw = "crcw", crcStep = "crcStep", pw = "P.pw" } = {}) {
  return /* wgsl */ `
const CRC_CHUNKS: u32 = ${CRC_CHUNKS}u;
const CRC_POLY_HI: u32 = 0xc02c38b4u;
const CRC_POLY_LO: u32 = 0x7d32d307u;
fn ${crcStep}(c: vec2<u32>) -> vec2<u32> {
  let m = 0u - (c.y & 1u);
  return vec2<u32>((c.x >> 1u) ^ (CRC_POLY_HI & m), ((c.y >> 1u) | (c.x << 31u)) ^ (CRC_POLY_LO & m));
}
fn crcPow(w: u32) -> vec2<u32> {
  let v = ${pw}[w >> 1u];
  return select(v.xy, v.zw, (w & 1u) != 0u);
}
fn crcOf(i: u32) -> vec2<u32> {
  if (i == 0u) { atomicStore(&${crcw}[0], 0u); atomicStore(&${crcw}[1], 0u); }
  workgroupBarrier();
  if (i < CRC_CHUNKS) {
    let w = ${packw}[i];
    var c = vec2<u32>(0u, 0u);
    for (var t: u32 = 0u; t < 4u; t = t + 1u) {
      c.y = c.y ^ ((w >> (8u * t)) & 0xffu);
      for (var q: u32 = 0u; q < 8u; q = q + 1u) { c = ${crcStep}(c); }
    }
    let p = crcPow(i);
    var acc = vec2<u32>(0u, 0u);
    var tt = c;
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
      if (((p.x >> (31u - j)) & 1u) != 0u) { acc = acc ^ tt; }
      tt = ${crcStep}(tt);
    }
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
      if (((p.y >> (31u - j)) & 1u) != 0u) { acc = acc ^ tt; }
      tt = ${crcStep}(tt);
    }
    if (i == 0u) { acc = acc ^ crcPow(CRC_CHUNKS); }
    atomicXor(&${crcw}[0], acc.x);
    atomicXor(&${crcw}[1], acc.y);
  }
  workgroupBarrier();
  return vec2<u32>(atomicLoad(&${crcw}[0]), atomicLoad(&${crcw}[1]));
}
`;
}
