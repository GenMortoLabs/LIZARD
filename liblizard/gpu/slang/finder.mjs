// F5 to F6 and the second round's REDO (src/vote.slang, peaks, gather, complete, trace, score, pick, redo; the
// homography and the bend lib/homography.slang, lib/bend.slang): VOTE (each classified reading votes along its inward diagonal for the
// symbol's centre, a ray a ring), PEAKS (the accumulators' local maxima, the frame's list), GATHER (the strongest
// centres' quads from the readings that support them), COMPLETE (a quad of two or three corners, on a frame with none
// of four, completed by one fitted homography), TRACE (each quad's bend, traced along its ring), SCORE (every quad,
// orientation, ring and bend read against the known timing track near the corners), PICK (the argmax, the ring and
// the map) and REDO (the frames the first round missed, sent round again from level 1). The constants:
// ../plan/finder.mjs.
import { lit, zeroDefs } from "./common.mjs";
import { slang } from "./load.mjs";
import {
  CENTRE, K, CELL, ORIENT, RING_COUNT, HYP, SMIN, VOTE_READS, RAY_LO, RAY_HI, RAY_SPREAD, NEAR, SUPPRESS, PEAK_LIST, STATIONS, R2_N,
} from "../plan/finder.mjs";

// ai: COMPLETE's side reads: pass 0 reads the stretch ALONG0 or ALONG1 (below) at a reach of REACH0 modules either way
// ai: of its guessed line; the later PASSES, [extent, reach], each about the line the pass before fitted, its extent in
// ai: modules of the corner's own module size from the first station (Infinity the whole run), three times the last.
const PROFILE = 13, NCC_MIN = 0.6, REACH0 = 3.0, PASSES = [[36, 1.5], [108, 1.5], [Infinity, 1.5]], GROUP = 256 / K;
const REACH_MAX = Math.max(REACH0, ...PASSES.map((p) => p[1]));
// ai: a pass's extent along the side, from t0: m modules of u0, at most the run
const ext = (m) => (m === Infinity ? "(t1 - t0)" : `min(${lit(m)} * u0, t1 - t0)`);

const ALONG0 = [4, 15], ALONG1 = [13, 25];
const RULER_LO = 0.6, RULER_HI = 1.6, RULER_CELLS0 = 64, RULER_NP = 1024, RULER_DMAX = 2.0;
const RULER_STEP0 = 0.5 / (16 + 2 * RULER_CELLS0);
// ai: stage 0's scale step, RULER_COARSE times RULER_STEP0: stage 1 searches 3 of RULER_STEP0 either side of its best
const RULER_COARSE = 3, RULER_STEP0C = RULER_COARSE * RULER_STEP0;
const RULER_N0 = Math.ceil(Math.log(RULER_HI / RULER_LO) / RULER_STEP0C);
const RULER_ND = Math.round((2 * RULER_DMAX) / 0.5) + 1;
const RULER_STEP1 = 0.0005, RULER_N1 = Math.ceil((6 * RULER_STEP0) / RULER_STEP1) + 1;
const RULER_DEPTHS = [2.5, 6.5], RULER_NDEP = 9;
// ai: the fit's weights, one over each measurement's spread squared in modules: a ring line's end (its stations' fit,
// ai: counted by half its stations), a found mark's centre (the classifier's), a ruled cell (by the ruler's correlation,
// ai: at most 32 a side, each weighted for the 4 cells of the cell lattice's quarter it stands for)
const FIT_LINE = 1 / (0.15 * 0.15), FIT_MARK = 1, FIT_RULER = 1 / (0.3 * 0.3), FIT_PIVOT = 1e-6, FIT_ROWS = 16 + 4 * STATIONS;
if (10 * FIT_ROWS > 2 * STATIONS * (2 * Math.round(4 * REACH_MAX) + 2 * PROFILE + 1)) throw new Error("COMPLETE's fit rows past prof");
if (STATIONS > 32) throw new Error(`COMPLETE keeps a side's stations in one word: ${STATIONS} past 32`);
const TR_MAX = 24, TR_STEP = 12, TR_L = 32;

// ai: Every stage's constants as defines: an integer a uint (`<n>u`), any other number its JS text; COMPLETE's reach,
// ai: band depth and mark weight floats (their values whole).
const C = {
  CELL, K, ORIENT, RING_COUNT, HYP, SMIN, VOTE_READS, RAY_LO, RAY_HI, RAY_SPREAD, NEAR, SUPPRESS, PEAK_LIST, STATIONS, R2_N,
  GROUP, PROFILE, NCC_MIN, RULER_CELLS0, RULER_HI, RULER_N0, RULER_N1, RULER_ND, RULER_NDEP, RULER_NP, RULER_STEP0,
  RULER_STEP0C, RULER_STEP1, FIT_LINE, FIT_PIVOT, FIT_ROWS, TR_MAX, TR_STEP, TR_L,
  ALONG0_LO: ALONG0[0], ALONG0_HI: ALONG0[1], ALONG1_LO: ALONG1[0], ALONG1_HI: ALONG1[1],
};
const defs = (...names) => Object.fromEntries(names.map((n) => [n, Number.isInteger(C[n]) ? `${C[n]}u` : C[n]]));
const CENTRES = { CENTRE0: lit(CENTRE[0]), CENTRE1: lit(CENTRE[1]) };

// ai: F5a VOTE, dispatched (ceil(cap / VOTE_READS), frames); F5b PEAKS, a lane a cell, dispatched (ceil(gx gy / 256),
// ai: frames); F5b's selection and F5c's quads, GATHER, a workgroup a frame.
export const VOTE = { slang: slang("vote.slang", { ...defs("CELL", "RAY_HI", "RAY_LO", "RAY_SPREAD", "RING_COUNT", "SMIN", "VOTE_READS"), ...CENTRES }) };
export const PEAKS = { slang: slang("peaks.slang", defs("CELL", "PEAK_LIST", "RING_COUNT", "SUPPRESS")) };
export const GATHER = { slang: slang("gather.slang", { ...defs("CELL", "GROUP", "K", "PEAK_LIST", "RAY_HI", "RAY_LO", "RAY_SPREAD", "RING_COUNT", "SMIN"), ...CENTRES }) };

// ai: F5d COMPLETE (src/complete.slang's header says what it does); its later passes PASSES, each PASS_AFTER's
export const COMPLETE = { slang: slang("complete.slang", {
  ...defs("FIT_LINE", "FIT_PIVOT", "FIT_ROWS", "K", "NCC_MIN", "PROFILE", "RULER_CELLS0", "RULER_HI", "RULER_N0", "RULER_N1", "RULER_ND",
    "RULER_NDEP", "RULER_NP", "RULER_STEP0", "RULER_STEP0C", "RULER_STEP1", "STATIONS", "ALONG0_LO", "ALONG0_HI", "ALONG1_LO", "ALONG1_HI"),
  LANES: 2 * STATIONS, REACH0: lit(REACH0), RULER_DMAX: lit(RULER_DMAX), FIT_MARK: lit(FIT_MARK), PROF_LEN: 2 * Math.round(4 * REACH_MAX) + 2 * PROFILE + 1,
  RULER_LOG_LO: lit(Math.log(RULER_LO)), FIT_RULER_CELL: lit(FIT_RULER / 4), RULER_DEPTH0: lit(RULER_DEPTHS[0]),
  RULER_DEPTH_STEP: lit((RULER_DEPTHS[1] - RULER_DEPTHS[0]) / (RULER_NDEP - 1)),
}, { passes: PASSES.map(([m, r]) => `PASS_AFTER(${ext(m)}, ${lit(r)})`).join("\n") }) };

// ai: TRACE (src/trace.slang): each quad's bend, a workgroup a (frame, quad); SCORE: every quad, orientation, ring and
// ai: bend read against the timing track; PICK: the argmax, the ring and the map; REDO: the frames round 1 missed
export const TRACE = { slang: slang("trace.slang", defs("K", "TR_L", "TR_MAX", "TR_STEP")) };
export const SCORE = { slang: slang("score.slang", { ...defs("HYP", "K", "NEAR", "ORIENT", "RING_COUNT"), ...zeroDefs(9) }) };
export const PICK = { slang: slang("pick.slang", { ...defs("HYP", "K", "ORIENT", "RING_COUNT"), ...zeroDefs(9, 16, 32) }) };
export const REDO = { slang: slang("redo.slang", defs("K", "R2_N", "RING_COUNT")) };
