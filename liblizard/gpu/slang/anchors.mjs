// The anchors (SPEC 6.13), src/anch_*.slang: ANCH_FIND (a workgroup of 128 an (anchor, frame): the ring's squares found by
// normalised correlation from integral images, coarse then fine, through REFIT's homography and the sides' curves),
// ANCH_SUMS (the squares' fill's sums over the picture's points outside them, a slice of the points a workgroup, so
// the fit's ANCH_FILL^2 reads run side by side; 2026-11-20), ANCH_FIT (the mesh's knots, a missing one the mean of its
// found neighbours; the squares' fill, the least-squares shading surface outside them, from ANCH_SUMS' slices; the
// picture's gain from the signal's swing, SPEC 6.10; the record pass 1 reads), and
// anchRefitSource: ANCH_REFIT (the lens about the axis over the border and the anchors, robust over the border, the
// sides' curves of what it leaves) and ANCH_PREDICT (the fit over the anchors alone, the border's places for its
// second search). Their constants and the uniform's layout: ../plan/anchors.mjs.
// ai: Workgroup memory is not zeroed but where a kernel reads it before writing it: ANCH_REFIT's three counters (wcnt,
// ai: wnba, wchg), whose atomics start from zero, are zeroed by lane 0 before a barrier.
import { lit, zeroDefs } from "./common.mjs";
import { slang } from "./load.mjs";
import { LATTICE_MAX, ANCH_SIDE, ANCH_K, ANCH_REC, ANCH_SURF, ANCH_GAIN } from "../plan/back_transform.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { NODES_MAX, RESID, WMIN, LENS_P } from "../plan/register.mjs";
import { ANCH_TS, ANCH_PEAK, ANCH_KNOT_ROUNDS, ANCH_FILL, ANCH_BAND2, ANCH_GS, SW_T, SW_R, SW_LAG, SW_MIN, SW_BINS } from "../plan/anchors.mjs";
import { lensAccumulate, lensModel, lensNormalSolve, gaussJordan } from "./register.mjs";

const RR = ANCH_TS, TC = ANCH_TS / 2, RC = RR / 2, WC = TC + 2 * RC, W1C = WC + 1, OC = 2 * RC + 1;
const FD = 3, WF = ANCH_TS + 2 * FD, W1F = WF + 1, OF = 2 * FD + 1;
const pattern = (ts) => {
  const dist = (t) => Math.abs((t + 0.5) / (ts / 2) - 1);
  const span1 = (lim) => { const q = [...Array(ts).keys()].filter((t) => dist(t) < lim); return [q[0], q.at(-1)]; };
  const [i25, j25] = span1(0.25), [i6, j6] = span1(0.6);
  const dark = (j6 - i6 + 1) ** 2 - (j25 - i25 + 1) ** 2, light = ts * ts - dark;
  const tm = (light - dark) / (ts * ts);
  return { ts, i25, j25, i6, j6, tm, tn: Math.fround(Math.sqrt(light * (1 - tm) ** 2 + dark * (1 + tm) ** 2)) };
};

// ai: what every anchors' stage binds first (lib/anch_head.slang)
const HEAD = { RING_COUNT, LATTICE_VECS: (RING_COUNT * LATTICE_MAX) / 4, RESID: `${RESID}u` };
// ai: the fill's points (lib/anch_fill.slang)
const FILL = { ANCH_FILL, ANCH_BAND2, ANCH_GS: `${ANCH_GS}u` };
// ai: a window's square pattern as ANCH_FIND's SCORE takes it, prefix P
const patternDefs = (P, { ts, i25, j25, i6, j6, tm, tn }) => ({
  [`${P}_TS`]: `${ts}u`, [`${P}_AREA`]: lit(ts * ts), [`${P}_I25`]: `${i25}u`, [`${P}_N25`]: `${j25 - i25 + 1}u`,
  [`${P}_I6`]: `${i6}u`, [`${P}_N6`]: `${j6 - i6 + 1}u`, [`${P}_TM1`]: lit(1 - tm), [`${P}_TN`]: lit(tn),
});

export const ANCH_FIND = { slang: slang("anch_find.slang", {
  ...HEAD, ANCH_TS: `${ANCH_TS}u`, ANCH_K: `${ANCH_K}u`, ANCH_SIDE: `${ANCH_SIDE}u`, ANCH_PEAK: lit(ANCH_PEAK),
  WC: `${WC}u`, W1C: `${W1C}u`, OC: `${OC}u`, RC, WF: `${WF}u`, W1F: `${W1F}u`, OF: `${OF}u`,
  ...patternDefs("PC", pattern(TC)), ...patternDefs("PF", pattern(ANCH_TS)),
}) };

// ai: ANCH_SUMS: each lane FILL_PER of the fill's points in its workgroup's slice (src/anch_sums.slang)
const FILL_PER = (ANCH_FILL * ANCH_FILL) / (64 * ANCH_GS);
if (!Number.isInteger(FILL_PER) || FILL_PER < 1) throw new Error(`ANCH_FILL ${ANCH_FILL}^2 points in ${ANCH_GS} slices of 64 lanes`);
export const ANCH_SUMS = { slang: slang("anch_sums.slang", { ...HEAD, ...FILL, FILL_PER: `${FILL_PER}u` }) };

export const ANCH_FIT = { slang: slang("anch_fit.slang", {
  ...HEAD, ...FILL, ANCH_K: `${ANCH_K}u`, ANCH_SIDE: `${ANCH_SIDE}u`, ANCH_REC: `${ANCH_REC}u`, ANCH_SURF: `${ANCH_SURF}u`,
  ANCH_GAIN: `${ANCH_GAIN}u`, ANCH_KNOT_ROUNDS: `${ANCH_KNOT_ROUNDS}u`, SW_T: `${SW_T}u`, SW_R: `${SW_R}u`, SW_LAG: `${SW_LAG}u`,
  SW_MIN: `${SW_MIN}u`, SW_BINS: `${SW_BINS}u`, SPREADS3: lit(3 * 1.4826),
}, { solve: gaussJordan("N", 6, 7, "1e-12", "ok") }) };

// ai: ANCH_REFIT and ANCH_PREDICT (src/anch_refit.slang; its header says what each does).
const TAU_SPREAD = String(Math.fround(3.0 * 1.4826));   // ai: three robust spreads, 3.0 * 1.4826 folded in double precision then rounded to f32
export const anchRefitSource = ({ predict = false } = {}) => ({ slang: slang("anch_refit.slang", {
  PREDICT: predict ? 1 : 0, ANCH_K: `${ANCH_K}u`, ANCH_SIDE: `${ANCH_SIDE}u`, NODES_MAX: `${NODES_MAX}u`, LENS_P: `${LENS_P}u`,
  RESID: `${RESID}u`, WMIN, TAU_SPREAD, RING_COUNT, LATTICE_VECS: (RING_COUNT * LATTICE_MAX) / 4, ...zeroDefs(77, LENS_P),
}, {
  accumulate: lensAccumulate,
  solve: lensNormalSolve,
  residuals: [0, 1, 2, 3].map((K) => `if (LK[${K}] != 0u) { float2 e = r - ${lensModel(K)}; rs[${K}] += w * dot(e, e); }`).join("\n        "),
  pick: [0, 1, 2, 3].map((K) => `if (kb == ${K}u) { ${Array.from({ length: LENS_P }, (_, a) => `tk[${a}] = LT[${K}][${a}];`).join(" ")} }`).join("\n      "),
}) });
export const ANCH_REFIT = anchRefitSource(), ANCH_PREDICT = anchRefitSource({ predict: true });
