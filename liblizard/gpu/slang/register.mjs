// F7, registration (src/nodes.slang, refit.slang; the lens src/lib/lens.slang): NODES (nodesSource: each border node found
// by normalised cross-correlation of the border as painted, over offsets about the finder's map on one lattice of
// samples and offsets), REFIT (the homography over the nodes by Gauss-Newton, weighted by each node's correlation; the
// lens record cleared), and the lens fit's generated pieces the anchors' ANCH_REFIT takes (lensAccumulate, lensModel,
// lensNormalSolve, gaussJordan). The map past F7 and the anchors is the C's one model: REFIT's homography, the lens
// ANCH_REFIT fits over the border and the anchors, and ANCH_FIT's mesh. The constants: ../plan/register.mjs.
// ai: Every solve is straight-line with literal indices, generated here (the S26's Adreno returned zeros from a private
// ai: array indexed at run time).
import { zeros, zeroDefs } from "./common.mjs";
import { slang } from "./load.mjs";
import {
  WINDOW, SUB, NODES_MAX, LIGHT_MAX, LENS_SLOTS, RESID, ROUND, latticeSide, WMIN, DELTA, ITERS, LENS_P, lensIdx,
} from "../plan/register.mjs";

const TMPL_MAX = WINDOW * WINDOW * SUB * SUB;
const ITEMS = 512;
const unroll = (n, f) => Array.from({ length: n }, (_, i) => f(i)).join("\n");

// ai: NODES and its centred form, a kernel a round (src/nodes.slang). Every round's template samples and offsets lie on
// ai: one lattice (latticeSide), whose values are read once a node and gathered by each offset.
export const nodesSource = ({ centred = false, round = ROUND, chunk = TMPL_MAX / 2, items = ITEMS } = {}) => {
  const grid = latticeSide(round);
  if (!(grid > 0)) throw new Error(`round ${round}: its samples and offsets share no lattice`);
  return { slang: slang("nodes.slang", {
    NODES_CENTRED: centred ? 1 : 0,
    NODES_CHUNK: `${chunk}u`,
    NODES_ITEMS: `${items}u`,
    NODES_GRID: `${grid}u`,
    NODES_MAX: `${NODES_MAX}u`,
    LIGHT_VECS: LIGHT_MAX / 4,
    NODES_SPACING: round[1] === SUB * SUB ? "1.0" : `1.0 / ${SUB}.0`,
    NODES_RATIO: `${Math.round((round[1] === SUB * SUB ? 1 : 1 / SUB) / round[2])}u`,
    HALF_WINDOW: `${WINDOW / 2}.0`,
    SUB_F: `${SUB}.0`,
  }) };
};
export const NODES = nodesSource();

// ai: Gauss-Jordan with partial pivoting as straight-line code (gaussJordan): the pivot chosen and
// ai: swapped in by selects, every index a literal.
export function gaussJordan(N, rows, cols, eps, ok) {
  const s = [];
  for (let col = 0; col < rows; col++) {
    s.push(`{ int piv = ${col}; float pv = abs(${N}[${col}][${col}]);`);
    for (let r = col + 1; r < rows; r++) s.push(`{ float a = abs(${N}[${r}][${col}]); piv = a > pv ? ${r} : piv; pv = max(pv, a); }`);
    for (let r = col + 1; r < rows; r++) {
      s.push(`{ bool sw = piv == ${r};`);
      for (let j = col; j < cols; j++) s.push(`{ float a = ${N}[${col}][${j}]; float b = ${N}[${r}][${j}]; ${N}[${col}][${j}] = sw ? b : a; ${N}[${r}][${j}] = sw ? a : b; }`);
      s.push("}");
    }
    s.push(`${ok} = ${ok} && abs(${N}[${col}][${col}]) >= ${eps};`);
    for (let r = 0; r < rows; r++) {
      if (r === col) continue;
      s.push(`{ float fc = ${N}[${r}][${col}] / ${N}[${col}][${col}];`);
      for (let j = col; j < cols; j++) s.push(`${N}[${r}][${j}] -= fc * ${N}[${col}][${j}];`);
      s.push("}");
    }
    s.push("}");
  }
  return s.join("\n");
}
// ai: parameter k's two perturbed copies named (an array is no argument's constructor in Slang)
const jacobian = unroll(8, (k) => {
  const arr = (sign) => `{ ${Array.from({ length: 8 }, (_, j) => (j === k ? `t[${j}] ${sign} hs${k}` : `t[${j}]`)).join(", ")} }`;
  return `float hs${k} = 1e-3;\nfloat tp${k}[8] = ${arr("+")};\nfloat tm${k}[8] = ${arr("-")};\nfloat2 J${k} = (fwd(tp${k}, m, c, S) - fwd(tm${k}, m, c, S)) / (2.0 * hs${k});`;
});
const accumulate = (() => {
  const s = [];
  let q = 0;
  for (let a = 0; a < 8; a++) {
    s.push(`jr[${a}] += w * dot(J${a}, r);`);
    for (let b = a; b < 8; b++) s.push(`jj[${q++}] += w * dot(J${a}, J${b});`);
  }
  return s.join("\n");
})();
const normal = (() => {
  const s = [];
  let q = 0;
  for (let a = 0; a < 8; a++) for (let b = a; b < 8; b++) { s.push(`N[${a}][${b}] = A[${q}]; N[${b}][${a}] = A[${q}];`); q++; }
  return s.join("\n");
})();
const lensRows = () => {
  const rx = ["d.x", "d.y", null, null, "1.0", null, "(-d.x * d.x)", "(-d.y * d.x)", "(t * q.x)", "(t2 * q.x)", "(t3 * q.x)"];
  const ry = [null, null, "d.x", "d.y", null, "1.0", "(-d.x * d.y)", "(-d.y * d.y)", "(t * q.y)", "(t2 * q.y)", "(t3 * q.y)"];
  return { rx, ry };
};
export const lensAccumulate = (() => {
  const { rx, ry } = lensRows(), s = [];
  for (let a = 0; a < LENS_P; a++) for (let b = a; b < LENS_P; b++) {
    const t = [];
    if (rx[a] && rx[b]) t.push(`${rx[a]} * ${rx[b]}`);
    if (ry[a] && ry[b]) t.push(`${ry[a]} * ${ry[b]}`);
    if (t.length) s.push(`acc[${lensIdx(a, b)}] += w * (${t.join(" + ")});`);
  }
  for (let a = 0; a < LENS_P; a++) {
    const t = [];
    if (rx[a]) t.push(`${rx[a]} * r.x`);
    if (ry[a]) t.push(`${ry[a]} * r.y`);
    s.push(`acc[${66 + a}] += w * (${t.join(" + ")});`);
  }
  return s.join("\n        ");
})();
export const lensModel = (K) => {
  const { rx, ry } = lensRows(), x = [], y = [];
  for (let a = 0; a < 8 + K; a++) {
    if (rx[a]) x.push(`LT[${K}][${a}] * ${rx[a]}`);
    if (ry[a]) y.push(`LT[${K}][${a}] * ${ry[a]}`);
  }
  return `float2(${x.join(" + ")}, ${y.join(" + ")})`;
};
const lensSolveTo = (kTop) => {
  const s = [];
  for (let j = 0; j < LENS_P; j++) {
    let sum = `Ns[${j}][${j}]`;
    for (let k = 0; k < j; k++) sum += ` - Ln[${j}][${k}] * Ln[${j}][${k}]`;
    s.push(`{ float v = ${sum}; uint om = min(okD, ${j}u); okD = !(v > 1e-7) ? om : okD; Ln[${j}][${j}] = sqrt(max(v, 1e-7)); }`);
    for (let i = j + 1; i < LENS_P; i++) {
      let e = `Ns[${i}][${j}]`;
      for (let k = 0; k < j; k++) e += ` - Ln[${i}][${k}] * Ln[${j}][${k}]`;
      s.push(`Ln[${i}][${j}] = (${e}) / Ln[${j}][${j}];`);
    }
  }
  for (let i = 0; i < LENS_P; i++) {
    let e = `bs[${i}]`;
    for (let k = 0; k < i; k++) e += ` - Ln[${i}][${k}] * z[${k}]`;
    s.push(`z[${i}] = (${e}) / Ln[${i}][${i}];`);
  }
  for (let K = 0; K <= kTop; K++) {
    const D = 8 + K;
    s.push(`if (${D}u <= okD && ${K}u <= LF.kmax) {`, `  float y[${D}] = ${zeros(D)};`);
    for (let i = D - 1; i >= 0; i--) {
      let e = `z[${i}]`;
      for (let k = i + 1; k < D; k++) e += ` - Ln[${k}][${i}] * y[${k}]`;
      s.push(`  y[${i}] = (${e}) / Ln[${i}][${i}];`);
    }
    for (let a = 0; a < LENS_P; a++) s.push(`  LT[${K}][${a}] = ${a < D ? `Dg[${a}] * y[${a}]` : "0.0"};`);
    s.push(`  LK[${K}] = 1u;`, "}");
  }
  return s.join("\n      ");
};
const lensNormalSolveTo = (kTop) => `      float Ns[${LENS_P}][${LENS_P}] = ${zeros(LENS_P, LENS_P)};
      float bs[${LENS_P}] = ${zeros(LENS_P)};
      float Dg[${LENS_P}] = ${zeros(LENS_P)};
      float Ln[${LENS_P}][${LENS_P}] = ${zeros(LENS_P, LENS_P)};
      float z[${LENS_P}] = ${zeros(LENS_P)};
      float Ap[77] = ${zeros(77)};
      ${Array.from({ length: 77 }, (_, i) => `Ap[${i}] = LP[${i}];`).join(" ")}
      ${(() => { const s = []; for (let a = 0; a < LENS_P; a++) for (let b = a; b < LENS_P; b++) s.push(`Ns[${a}][${b}] = Ap[${lensIdx(a, b)}]; Ns[${b}][${a}] = Ap[${lensIdx(a, b)}];`); return s.join("\n      "); })()}
      ${Array.from({ length: LENS_P }, (_, a) => `Dg[${a}] = 1.0 / sqrt(max(Ns[${a}][${a}], 1e-30));`).join("\n      ")}
      ${Array.from({ length: LENS_P }, (_, a) => `${Array.from({ length: LENS_P }, (_, b) => `Ns[${a}][${b}] *= Dg[${a}] * Dg[${b}];`).join(" ")}\n      Ns[${a}][${a}] = Ns[${a}][${a}] * 1.0001 + 1e-9; bs[${a}] = Ap[${66 + a}] * Dg[${a}];`).join("\n      ")}
      uint okD = ${LENS_P}u;
      ${lensSolveTo(kTop)}
`;
export const lensNormalSolve = lensNormalSolveTo(3);
// ai: REFIT: a workgroup of 64 a frame (src/refit.slang), its solves straight-line here.
export const REFIT = { slang: slang("refit.slang", { REFIT_ITERS: ITERS, NODES_MAX: `${NODES_MAX}u`, LENS_SLOTS: `${LENS_SLOTS}u`, RESID: `${RESID}u`, WMIN, DELTA, ...zeroDefs(8, 9, 16, 36, [8, 9]) }, {
  start: unroll(3, (j) => `{ float s = ${j < 2 ? "Mod" : "1.0"}; h[${j}] = (m0[${j}] - c.x * m0[${6 + j}]) / S * s; h[${3 + j}] = (m0[${3 + j}] - c.y * m0[${6 + j}]) / S * s; h[${6 + j}] = m0[${6 + j}] * s; }`),
  theta: unroll(8, (j) => `th[${j}] = h[${j}] / h[8];`),
  jacobian,
  accumulate,
  sums: unroll(36, (q) => `A[${q}] = Ag[${q}];`),
  rhs: unroll(8, (a) => `g[${a}] = Ag[${36 + a}];`),
  normal,
  scale: unroll(8, (a) => `D[${a}] = 1.0 / sqrt(max(N[${a}][${a}], 1e-30));`),
  scaled: unroll(8, (a) => `${unroll(8, (b) => `N[${a}][${b}] *= D[${a}] * D[${b}];`)}\nN[${a}][${a}] = N[${a}][${a}] * 1.001 + 1e-9; N[${a}][8] = g[${a}] * D[${a}];`),
  solve: gaussJordan("N", 8, 9, "1e-20", "okSolve"),
  step: unroll(8, (a) => `th[${a}] += D[${a}] * N[${a}][8] / N[${a}][${a}];`),
  map: unroll(3, (j) => `{ float s = ${j < 2 ? "1.0 / Mod" : "1.0"}; mo[${j}] = (S * t[${j}] + c.x * t[${6 + j}]) * s; mo[${3 + j}] = (S * t[${3 + j}] + c.y * t[${6 + j}]) * s; mo[${6 + j}] = t[${6 + j}] * s; }`),
}) };
