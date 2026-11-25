// The back half's rule stage (src/tnoise.slang, talign.slang, tsoft.slang, tblocks.slang, tldpc.slang; TAB's readers
// src/lib/tab.slang): tnoise (each sub-channel's moments once: its estimate, its soft values' scale and its noise), talign
// (the grid's shift fitted from the pilots, each weighted by 1 / N of its sub-channel, a workgroup a frame), tsoft (a
// sub-channel's soft values at tnoise's scale, whitened by slot, in data-slot order, and its two pilots' values),
// tblocks (a block's estimate and its decline, and the frame's two pilot readings) and tldpc (a code's layered min-sum
// in int16 with its learned stop and the CRC-64, in forms differing in workgroup memory alone). TAB's layout, the codes'
// shapes and the forms: ../plan/back_tiers.mjs (tabLayout, tierShape, LDPC_FORMS); the arithmetic: ../back/DESIGN.md 4
// and 6.
import { CFG_DEFS } from "./common.mjs";
import { slang } from "./load.mjs";
import { TIER_BLOCKS, CHUNKS, KNOWN, NC, PW_WORDS, PB_REF, DCOEF, SUB_SLOTS, PILOT_READ, CODES, tierUniformWords } from "../plan/back_tiers.mjs";
import { STEP } from "../../sim/lizard_pick.mjs";
import { SLOTS } from "../plan/common.mjs";
import { CRC_CHUNKS } from "../plan/send.mjs";

const TWO_PI = (2 * Math.PI).toFixed(9);
// ai: the frames' configurations in the gate's lists (lib/lists.slang)
const LISTS = { SLOT_COUNT: `${SLOTS}u` };
// ai: TAB's readers' defines (lib/tab.slang): layout L (tabLayout), codes [{ n, k, slots, bar }] (n the sent bits, slots a
// ai: block's: CODES', the same since the pilots left the blocks)
function tabDefs(L, codes) {
  if (codes.length !== 4 || NC !== 4) throw new Error(`lib/tab.slang reads four codes, not ${codes.length}`);
  codes.forEach((x, c) => { if (x.slots !== CODES[c].slots) throw new Error(`code ${c}: ${x.slots} slots, CODES' ${CODES[c].slots}`); });
  const barBits = (x) => { const f = new Float32Array([x]); return new Uint32Array(f.buffer)[0]; };
  const d = { TAB_WHITE_AT: `${L.WHITE}u`, TAB_PB_REF: `${PB_REF}u`, TAB_SUB_SLOTS: `${SUB_SLOTS}u`, TAB_NCODES: `${NC}u`, ...CFG_DEFS };
  codes.forEach((x, c) => { d[`CODE${c}_N`] = `${x.n}u`; d[`CODE${c}_BAR`] = `${barBits(x.bar)}u`; });
  return d;
}

export const tnoiseSource = ({ f16 = false }) => ({ slang: slang("tnoise.slang", { F16: f16 ? 1 : 0, CHUNKS: `${CHUNKS}u`, ...LISTS, ...CFG_DEFS }) });

export const talignSource = ({ f16 = false, layout, codes }) => ({ slang: slang("talign.slang", {
  F16: f16 ? 1 : 0, TIER_BLOCKS: `${TIER_BLOCKS}u`, STEP: `${STEP}u`, PILOT_READ: `${PILOT_READ}u`, TWO_PI, ...LISTS, ...tabDefs(layout, codes),
}) });

// ai: tsoft's pairs (src/tsoft.slang), unrolled: a lane's PAIRS pairs m = l + 32 k of data coefficients (2 m, 2 m + 1),
// ai: the last lane's last pair the sub-channel's two pilots
const PAIRS = 5;
export const tsoftSource = ({ f16 = false, layout, codes }) => {
  const ks = [...Array(PAIRS).keys()];
  const last = PAIRS - 1;
  let body = "";
  for (const k of ks) {
    const m = `(l + ${32 * k}u)`;
    if (k < last) body += `  uint a${k} = fcoefIn(2u * ${m});\n  uint b${k} = fcoefIn(2u * ${m} + 1u);\n`;
    else body += `  uint a${k} = pil ? 0u : fcoefIn(2u * ${m});\n  uint b${k} = pil ? PB_REF : fcoefIn(2u * ${m} + 1u);\n`;
  }
  for (const k of ks) body += `  float2 y${2 * k} = float2(S[sBase + UV[ub + a${k}]]);\n  float2 y${2 * k + 1} = float2(S[sBase + UV[ub + b${k}]]);\n`;
  body += `  if (turn) {\n`;
  for (const k of ks) for (const h of [0, 1]) {
    const c = 2 * k + h, w = `w${c}`;
    body += `    uint ${w} = UV[cb + ${h ? "b" : "a"}${k}];\n    y${c} = turned(y${c}, kk * (float(int(${w}) >> 16u) * sh.x + float(${w} & 0xffffu) * sh.y));\n`;
  }
  body += `  }\n`;
  for (const k of ks) for (const h of [0, 1]) {
    const c = 2 * k + h;
    body += `  { uint n = whiteOf(j, ${h ? "b" : "a"}${k}); y${c} = float2((n & 1u) != 0u ? -y${c}.x : y${c}.x, (n & 2u) != 0u ? -y${c}.y : y${c}.y); }\n`;
  }
  body += `  GroupMemoryBarrierWithGroupSync();\n  if (pil) { EP[f * P.subchMax + j].yzw = float3(y${2 * last}.x + y${2 * last}.y, y${2 * last + 1}.x + y${2 * last + 1}.y, kq); }\n`;
  let store = "";
  for (const k of ks) {
    store += `  ${k === last ? "if (!pil) " : ""}{ int2 r0 = int2(roundEven(clamp(kq * y${2 * k}, float2(-10.0), float2(10.0)) * 8.0));\n    int2 r1 = int2(roundEven(clamp(kq * y${2 * k + 1}, float2(-10.0), float2(10.0)) * 8.0));\n`;
    store += `    L[lBase${k ? ` + ${32 * k}u` : ""}] = (uint(r0.x) & 0xffu) | ((uint(r0.y) & 0xffu) << 8u) | ((uint(r1.x) & 0xffu) << 16u) | ((uint(r1.y) & 0xffu) << 24u); }\n`;
  }
  return { slang: slang("tsoft.slang", {
    F16: f16 ? 1 : 0, CHUNKS: `${CHUNKS}u`, PILOT_READ: `${PILOT_READ}u`, DCOEF: `${DCOEF}u`, TWO_PI, ...LISTS, ...tabDefs(layout, codes),
  }, { pairs: body, store }) };
};

export const tblocksSource = ({ layout, codes }) => ({ slang: slang("tblocks.slang", {
  TIER_BLOCKS: `${TIER_BLOCKS}u`, STEP: `${STEP}u`, PILOT_READ: `${PILOT_READ}u`, ...LISTS, ...tabDefs(layout, codes),
}) });

const packWords = (n) => Math.max(120, Math.ceil(n / 32) + 1);

// ai: The LDPC kernel (src/tldpc.slang): sh from ../plan/back_tiers.mjs tierShape (its form), c the code's index in CODES,
// ai: inv its INV section in TAB, net { hidden, at } its learned stop.
export function tldpcSource(sh, { c, inv, layout, codes, net }) {
  const { n: N, k: K, z: Z, mb: MB, ns: NS, threads: TH, one } = sh;
  if (K < KNOWN || NS > 32 || (one && NS > 12)) throw new Error(`tier code ${JSON.stringify(sh)}`);
  if (!(net?.hidden > 0)) throw new Error(`the code ${c}'s kernel needs its learned stop`);
  const H = net.hidden;
  // ai: the posteriors' store (src/tldpc.slang): "pair" two int16 halves a uint, "i32", or "i16" (sh.narrow)
  const lw = sh.wide ? "i32" : sh.narrow ? "i16" : "pair";
  return { slang: slang("tldpc.slang", {
    LW_PAIR: lw === "pair" ? 1 : 0, LW_I32: lw === "i32" ? 1 : 0, LW_I16: lw === "i16" ? 1 : 0, ONE: one ? 1 : 0,
    TH: `${TH}u`, LDPC_CODE: `${c}u`, LDPC_INV_AT: `${inv}u`, LDPC_N: `${N}u`, LDPC_K: `${K}u`, LDPC_Z: `${Z}u`, LDPC_MB: `${MB}u`,
    LDPC_NS: `${NS}u`, LDPC_KNOWN: `${KNOWN}u`, LDPC_NET_H: `${H}u`, LDPC_NET_AT: `${net.at}u`, CRC64_CHUNKS: `${CRC_CHUNKS}u`,
    PACK_WORDS: packWords(N), ROWV: sh.rowv, PAIRV: sh.pairv, PW_VECS: PW_WORDS / 4, ...LISTS, ...tabDefs(layout, codes),
  }) };
}
// ai: the uniform's words (one layout, one buffer: tierUniform in ../back/tiers.mjs writes it)
export { tierUniformWords };
