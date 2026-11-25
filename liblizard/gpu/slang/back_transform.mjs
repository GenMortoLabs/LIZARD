// The back half's transform (src/gate.slang, pass1.slang, reduce.slang, pass2.slang): the gate (a batch's frames listed by
// picture slot, every later dispatch sized), the fused pass 1 (each frame sampled through its map on a node grid with the
// anchors' mesh, the squares filled with the shading, the gain applied, then the column FFT of four columns), reduce
// (the five surface coefficients) and pass 2 (the row FFT and the detrend). Built a batch size and a picture size each;
// the FFTs' stages, unrolled for each size, generated here. The LISTS and ARGS layouts: ../plan/back_transform.mjs; the
// arithmetic: ../back/DESIGN.md 1 to 3 and 10.
import { CFG_DEFS } from "./common.mjs";
import { slang } from "./load.mjs";
import { SLOTS } from "../plan/common.mjs";
import { ARGS_WORDS, CPW, LATTICE_MAX, ANCH_SIDE, ANCH_REC, ANCH_SURF, ANCH_GAIN, ANCH_ERASE, listsHead, listsRows, listsRing, argsLdpc } from "../plan/back_transform.mjs";
import { NC } from "../plan/back_tiers.mjs";
import { stagesOf } from "../back/disc.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { RESID } from "../plan/register.mjs";

if (CPW !== 4) throw new Error(`pass 1 (src/pass1.slang) is written for one quad of columns a workgroup, not ${CPW} columns`);

export function gateSource(B, { blkStride, counts, chunks } = {}) {
  if (!(counts > 1)) throw new Error(`gate: rows table of ${counts} versions`);
  if (!(chunks > 0)) throw new Error(`gate: ${chunks} chunks a version`);
  return { slang: slang("gate.slang", {
    BATCH: `${B}u`, SLOT_COUNT: `${SLOTS}u`, GATE_ROWS: Math.ceil((counts * SLOTS) / 4), LISTS_HEAD: `${listsHead(B)}u`,
    ARGS_LDPC: `${argsLdpc(0)}u`, ARGS_WORDS: `${ARGS_WORDS}u`, NC: `${NC}u`, CHUNKS: `${chunks}u`, BLK_STRIDE: `${blkStride}u`, ...CFG_DEFS,
  }) };
}


// ai: The exchange buffer's element type and the conversions in and out of it: f16 is half2, computed in f32 (float2).
const store = (prec) => ({
  v2: "float2", bar: "GroupMemoryBarrierWithGroupSync();",
  ty: prec === "f16" ? "half2" : "float2",
  ld: (name, i) => (prec === "f16" ? `float2(${name}[${i}])` : `${name}[${i}]`),
  st: (name, i, x) => (prec === "f16" ? `${name}[${i}] = half2(${x});` : `${name}[${i}] = ${x};`),
});

// ai: The stages after the first (the WGSL's laterStages, lanes the same). Since 2026-11-08 every picture is transformed
// ai: at its own size (384 to 1536: 8 x 2^a 3^b 5^c), so an odd stage may come before the last (1152's two 3s: oddStage)
// ai: or be a 5 (1280's: oddLastStage), the sender's inverse runs them too (../slang/src/send_irows.slang, send_ipic.slang); a last 3 stays
// ai: radix3Stage.
export function laterStages(n, prec, lanes = [["v", "buf"]]) {
  const { ld, st, v2, bar } = store(prec);
  const T = n / 8, stages = stagesOf(n);
  let Ns = stages[0], out = "";
  for (let s = 1; s < stages.length; s++) {
    if (stages[s] === 3 || stages[s] === 5) {
      const R = stages[s];
      out += s < stages.length - 1 ? oddStage(n, R, Ns, prec, lanes) : R === 3 ? radix3Stage(n, prec, lanes) : oddLastStage(n, R, prec, lanes);
      Ns *= R; continue;
    }
    const R = stages[s], per = 8 / R, span = n / R, step = n / (Ns * R);
    out += `  // stage ${s + 1}: radix ${R}, Ns ${Ns}\n`;
    for (let q = 0; q < per; q++) {
      const j = per === 1 ? "tid" : `(tid + ${q * T}u)`;
      for (const [v, buf] of lanes) for (let r = 0; r < R; r++) out += `  ${v}[${q * R + r}] = ${ld(buf, `${j} + ${r * span}u`)};\n`;
    }
    out += `  ${bar}\n`;
    for (let q = 0; q < per; q++) {
      const j = per === 1 ? "tid" : `(tid + ${q * T}u)`;
      out += `  {\n    uint j = ${j};\n    uint k = j & ${Ns - 1}u;\n`;
      out += `    ${v2} w1 = tw[k * ${step}u];\n`;
      if (R >= 4) out += `    ${v2} w2 = cmul(w1, w1);\n    ${v2} w3 = cmul(w2, w1);\n`;
      if (R === 8) out += `    ${v2} w4 = cmul(w2, w2);\n    ${v2} w5 = cmul(w4, w1);\n    ${v2} w6 = cmul(w3, w3);\n    ${v2} w7 = cmul(w4, w3);\n`;
      out += `    uint d = (j - k) * ${R}u + k;\n`;
      for (const [v, buf] of lanes) {
        for (let r = 1; r < R; r++) out += `    ${v}[${q * R + r}] = cmul(${v}[${q * R + r}], w${r});\n`;
        out += R === 8 ? `    dft8(${v});\n` : R === 4 ? `    dft4(${v}, ${q * R}u);\n` : `    dft2(${v}, ${q * R}u);\n`;
        for (let r = 0; r < R; r++) out += `    ${st(buf, `d + ${r * Ns}u`, `${v}[${q * R + r}]`)}\n`;
      }
      out += `  }\n`;
    }
    out += `  ${bar}\n`;
    Ns *= R;
  }
  return out;
}

// ai: A radix-3 or radix-5 butterfly (the WGSL's oddButterfly): p(r) the r-th input (twiddled), put(r, e) its r-th
// ai: output's store.
function oddButterfly(R, p, put, ind, v2) {
  let out = `${ind}{\n`;
  for (let r = 0; r < R; r++) out += `${ind}  ${v2} p${r} = ${p(r)};\n`;
  if (R === 3) {
    out += `${ind}  ${v2} t1 = p1 + p2;\n${ind}  ${v2} t2 = mneg_i(p1 - p2) * S3;\n${ind}  ${v2} m = p0 - 0.5 * t1;\n`;
    out += `${ind}  ${put(0, "p0 + t1")}\n${ind}  ${put(1, "m + t2")}\n${ind}  ${put(2, "m - t2")}\n`;
  } else {
    out += `${ind}  ${v2} t1 = p1 + p4;\n${ind}  ${v2} t2 = p2 + p3;\n${ind}  ${v2} t3 = p1 - p4;\n${ind}  ${v2} t4 = p2 - p3;\n`;
    out += `${ind}  ${v2} m1 = p0 + 0.30901699437494742 * t1 - 0.80901699437494742 * t2;\n${ind}  ${v2} m2 = p0 - 0.80901699437494742 * t1 + 0.30901699437494742 * t2;\n`;
    out += `${ind}  ${v2} n1 = mneg_i(0.95105651629515357 * t3 + 0.58778525229247313 * t4);\n${ind}  ${v2} n2 = mneg_i(0.58778525229247313 * t3 - 0.95105651629515357 * t4);\n`;
    out += `${ind}  ${put(0, "p0 + t1 + t2")}\n${ind}  ${put(1, "m1 + n1")}\n${ind}  ${put(4, "m1 - n1")}\n${ind}  ${put(2, "m2 + n2")}\n${ind}  ${put(3, "m2 - n2")}\n`;
  }
  return out + `${ind}}\n`;
}

// ai: An odd stage before the last (radix R, 3 or 5, at Ns; the WGSL's oddStage): Stockham as the even stages (read
// ai: buf[j + r n / R], twiddle by exp(-2 pi i r k / (Ns R)), k = j mod Ns, write buf[(j - k) R + k + r Ns]), its n / R
// ai: butterflies over T = n / 8 threads, Q = ceil(n / R / T) a thread, their points held in registers across the
// ai: barrier; each twiddle read whole from the table, tw[r k n / (Ns R)]. Ns is not a power of two after another odd
// ai: stage (k by a modulo there).
function oddStage(n, R, Ns, prec, lanes) {
  const { ld, st, v2, bar } = store(prec);
  const T = n / 8, M = n / R, Q = Math.ceil(M / T), step = n / (Ns * R), pow2 = (Ns & (Ns - 1)) === 0;
  const jOf = (q) => (q === 0 ? "tid" : `(tid + ${q * T}u)`), open = (q) => ((q + 1) * T > M ? `    if (${jOf(q)} < ${M}u) {\n` : `    {\n`);
  let out = `  // stage: radix ${R}, Ns ${Ns}\n  {\n`;
  for (const [v] of lanes) out += `    ${v2} ${v}g[${Q * R}];\n`;
  for (let q = 0; q < Q; q++) {
    out += open(q);
    for (const [v, buf] of lanes) for (let r = 0; r < R; r++) out += `      ${v}g[${q * R + r}] = ${ld(buf, `${jOf(q)} + ${r * M}u`)};\n`;
    out += `    }\n`;
  }
  out += `    ${bar}\n`;
  for (let q = 0; q < Q; q++) {
    out += open(q) + `      uint j = ${jOf(q)};\n      uint k = ${pow2 ? `j & ${Ns - 1}u` : `j % ${Ns}u`};\n      uint d = (j - k) * ${R}u + k;\n`;
    for (let r = 1; r < R; r++) out += `      ${v2} w${r} = tw[k * ${r * step}u];\n`;
    for (const [v, buf] of lanes) out += oddButterfly(R, (r) => (r === 0 ? `${v}g[${q * R}]` : `cmul(${v}g[${q * R + r}], w${r})`), (r, e) => st(buf, `d + ${r * Ns}u`, e), "      ", v2);
    out += `    }\n`;
  }
  return out + `    ${bar}\n  }\n`;
}

// ai: The last stage when it is odd and not radix 3 (the WGSL's oddLastStage, radix3Stage's in-place form for any R):
// ai: Ns = n / R, a butterfly reads and writes the same R points, buf[j + r n / R], j = tid, tid + T, ... below n / R,
// ai: its twiddles tw[r j].
function oddLastStage(n, R, prec, lanes) {
  const { ld, st, v2, bar } = store(prec);
  const T = n / 8, Ns = n / R;
  let out = `  // stage: radix ${R}, Ns ${Ns}, in place\n  for (uint j = tid; j < ${Ns}u; j += ${T}u) {\n`;
  for (let r = 1; r < R; r++) out += `    ${v2} w${r} = tw[${r === 1 ? "" : `${r}u * `}j];\n`;
  for (const [, buf] of lanes) out += oddButterfly(R, (r) => (r === 0 ? ld(buf, "j") : `cmul(${ld(buf, `j + ${r * Ns}u`)}, w${r})`), (r, e) => st(buf, r === 0 ? "j" : `j + ${r * Ns}u`, e), "    ", v2);
  return out + `  }\n  ${bar}\n`;
}

// ai: The last stage when n has a factor of 3, in place (the WGSL's radix3Stage).
function radix3Stage(n, prec, lanes) {
  const { ld, st, v2, bar } = store(prec);
  const T = n / 8, Ns = n / 3;
  let out = `  // stage: radix 3, Ns ${Ns}, in place\n  for (uint j = tid; j < ${Ns}u; j += ${T}u) {\n`;
  out += `    ${v2} w1 = tw[j];\n    ${v2} w2 = tw[2u * j];\n`;
  for (const [, buf] of lanes) {
    out += `    {\n      ${v2} a = ${ld(buf, "j")};\n      ${v2} b = cmul(${ld(buf, `j + ${Ns}u`)}, w1);\n      ${v2} c = cmul(${ld(buf, `j + ${2 * Ns}u`)}, w2);\n`;
    out += `      ${v2} t1 = b + c;\n      ${v2} t2 = mneg_i(b - c) * S3;\n      ${v2} m = a - 0.5 * t1;\n`;
    out += `      ${st(buf, "j", "a + t1")}\n      ${st(buf, `j + ${Ns}u`, "m + t2")}\n      ${st(buf, `j + ${2 * Ns}u`, "m - t2")}\n    }\n`;
  }
  return out + `  }\n  ${bar}\n`;
}

// ai: The first stage (Ns = 1, no twiddles), the registers loaded by the caller.
export function firstStage(prec, lanes = [["v", "buf"]]) {
  const { st, bar } = store(prec);
  let out = `  // stage 1: radix 8, Ns 1\n`;
  for (const [v, buf] of lanes) {
    out += `  dft8(${v});\n`;
    for (let r = 0; r < 8; r++) out += `  ${st(buf, `tid * 8u + ${r}u`, `${v}[${r}]`)}\n`;
  }
  out += `  ${bar}\n`;
  return out;
}

const ANCH_ROWS = ANCH_SIDE + 4;

// ai: Pass 1 (src/pass1.slang): n the picture's size, prec, B; its column FFT's stages generated here (two lanes).
export function pass1Source({ n, prec, B }) {
  const lanes = [["v0", "buf0"], ["v1", "buf1"]];
  return { slang: slang("pass1.slang", {
    F16: prec === "f16" ? 1 : 0, LANES: n / 8, PICTURE_N: `${n}u`, BATCH: `${B}u`, LISTS_HEAD: `${listsHead(B)}u`,
    LISTS_ROWS: `${listsRows(B)}u`, LISTS_RING: `${listsRing(B)}u`, ANCH_SIDE: `${ANCH_SIDE}u`, ANCH_ROWS: `${ANCH_ROWS}u`,
    ANCH_REC: `${ANCH_REC}u`, ANCH_SURF: `${ANCH_SURF}u`, ANCH_GAIN: `${ANCH_GAIN}u`, ANCH_ERASE, RING_COUNT,
    LATTICE_VECS: (RING_COUNT * LATTICE_MAX) / 4, RESID: `${RESID}u`, LUMH_BINDING: 11,
  }, { fft: firstStage(prec, lanes) + laterStages(n, prec, lanes) }) };
}

// ai: Reduce (src/reduce.slang): the five surface coefficients from the column partials.
export function reduceSource({ n, B }) {
  return { slang: slang("reduce.slang", { PICTURE_N: `${n}u`, BATCH: `${B}u` }) };
}

// ai: Pass 2 (src/pass2.slang): row v of one frame along x; its row FFT's stages generated here.
export function pass2Source({ n, prec, B }) {
  return { slang: slang("pass2.slang", { F16: prec === "f16" ? 1 : 0, LANES: n / 8, PICTURE_N: `${n}u`, BATCH: `${B}u`, LISTS_ROWS: `${listsRows(B)}u` },
    { fft: firstStage(prec) + laterStages(n, prec) }) };
}
