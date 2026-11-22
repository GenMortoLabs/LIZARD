// ai: F8's shape and its tables (glsl/word.mjs WORD): the search's constants, the Word uniform's layout and the packers
// ai: of its code, GF(256)'s tables, the generator's columns, plan points and lattice (the host's build;
// ai: scripts/gpu/word_check.mjs's JS mirror).
import { SLOTS } from "./common.mjs";
import { LATTICE_MAX } from "./back_transform.mjs";
import { RING_COUNT } from "./finder.mjs";
import { wordCode, wordBytes, bytesFor, VERSION_MAX, DATA, gateTable, BEFORE } from "../wordcode.mjs";

export const CON_MIN = 0.08;
// ai: WMAX: word cells a side at most (the 256 ring; the 128's 64 until 2026-10-10), so a word's 4 WMAX bits fill VECS
// ai: vec4u (a row of the code); ROWS: the magic's BASE and the 40 rows a ring (wordcode.mjs wordCode); MIDS_MAX:
// ai: track pairs a frame (the 256 ring's 512), each invocation taking every 256th.
export const WMAX = 128, ROWS = 1 + 8 * (DATA - 1), MIDS_MAX = 512;
export const VECS = (4 * WMAX) / 128;
// ai: vec4u the pictures' last versions take, four a vec4u.
const TOPS = Math.ceil(SLOTS / 4);
// ai: The search's shape (Gauss-Jordan over GF(256) since 2026-11-15, ordered statistics decoding over the code's bit
// ai: rows before): the word's bytes by reliability (a byte's least |q| of its 8 cells); the magic known, the five
// ai: other data bytes solved from five bytes (RS is MDS, any six bytes give the word): the UNKNOWN most reliable,
// ai: each swapped for one of the next SWAP, each with its least reliable bit flipped: SETS sets, each candidate
// ai: scored by its correlation with q (two magics, 2 PER sets, until 2026-11-19). NW the u32 of a word at most.
export const UNKNOWN = DATA - 1, SWAP = 6, PER = 1 + UNKNOWN * SWAP + UNKNOWN, SETS = PER, TOP = UNKNOWN + SWAP;
export const NW = (4 * WMAX) / 32, BYTES_MAX = (4 * WMAX) / 8;
if (SETS > 256) throw new Error(`F8 solves ${SETS} sets a lane each: more than 256 lanes`);
// ai: The gate's table a ring: E[m] for m = 0 to 4 W as u16, two a u32 (m even low), 0xffff where no count passes;
// ai: GATE u32 a ring holds 4 WMAX + 1 of them, padded to a vec4u.
export const GATE = 4 * Math.ceil((4 * WMAX + 2) / 8);
// ai: The Word uniform's layout in vec4u: code (every ring's ROWS rows, a row VECS vec4u: ring r's row k at (r ROWS +
// ai: k) VECS), then range (W, 0, 0, 0) a ring, then plan a ring, then tops (the last version of pictures 0 to
// ai: SLOTS - 1), then gate (GATE u32 a ring), then gf (GF(256)'s exp at bytes 0 to 254 and log at 256 to 511, four a
// ai: u32: src/rs.c's field, 0x11d), then cols (a ring's BYTES_MAX columns, two u32 a column: byte p of the word is the
// ai: GF(256) sum of the data bytes 0 to 5 times the column's bytes, byte 0's in the first u32's low byte and byte 5's
// ai: in its second, bytes 1 to 4's in the second u32, low first: wordcode.mjs wordBytes of the unit words), then before
// ai: (the configurations before each version, wordcode.mjs BEFORE, 65 u32: the index's version).
export const BEFORE_VECS = Math.ceil(BEFORE.length / 4);
export const LAYOUT = (() => {
  const range = RING_COUNT * ROWS * VECS, plan = range + RING_COUNT, tops = plan + RING_COUNT, gate = tops + TOPS;
  const gf = gate + (RING_COUNT * GATE) / 4, cols = gf + 512 / 16, before = cols + (RING_COUNT * BYTES_MAX * 2) / 4;
  return { code: 0, range, plan, tops, gate, gf, cols, before, size: before + BEFORE_VECS };
})();
// ai: GF(256) as src/rs.c builds it (alpha 2, 0x11d): exp[i] for i 0 to 254, log[x] for x 1 to 255
export const GF_EXP = new Uint8Array(256), GF_LOG = new Uint8Array(256);
for (let i = 0, x = 1; i < 255; i++) { GF_EXP[i] = x; GF_LOG[x] = i; x <<= 1; if (x & 256) x ^= 0x11d; }

// ai: The Word uniform for the rings, from gpu/tables.mjs formatTables() rings (word.cells), the plan wordPoints lays
// ai: out, and the pictures (their last versions, which pick a version's slot), as u32 in LAYOUT's order.
export function wordUniform(rings, plans, pictures) {
  if (rings.length !== RING_COUNT) throw new Error(`${rings.length} rings: F8 is built for ${RING_COUNT}`);
  if (pictures.length !== SLOTS || pictures.at(-1).hi !== VERSION_MAX) throw new Error(`F8 names a picture of ${SLOTS} by its last version, the last VERSION_MAX`);
  const u = new Uint32Array(4 * LAYOUT.size);
  rings.forEach(({ word }, r) => {
    if (word.cells > WMAX) throw new Error(`ring ${r}: ${word.cells} word cells a side, F8 holds ${WMAX}`);
    const { base, rows } = wordCode(word.cells);
    [...base, ...rows].forEach((v, k) => u.set(v, 4 * VECS * (r * ROWS + k)));
    gateTable(word.cells).forEach((e, m) => { u[4 * LAYOUT.gate + GATE * r + (m >> 1)] |= (e < 0 ? 0xffff : e) << (16 * (m & 1)); });
    u.set([word.cells, 0, 0, 0], 4 * (LAYOUT.range + r));
    u.set(plans[r], 4 * (LAYOUT.plan + r));
    const n = bytesFor(word.cells), e = Array.from({ length: DATA }, (_, j) => wordBytes(Array.from({ length: DATA }, (_, i) => +(i === j)), n));
    if (n > BYTES_MAX) throw new Error(`ring ${r}: a word of ${n} bytes, F8 holds ${BYTES_MAX}`);
    for (let p = 0; p < n; p++) {
      const at = 4 * LAYOUT.cols + 2 * (BYTES_MAX * r + p);
      u[at] = e[0][p] | (e[5][p] << 8);
      u[at + 1] = (e[1][p] | (e[2][p] << 8) | (e[3][p] << 16) | (e[4][p] << 24)) >>> 0;
    }
  });
  for (let i = 0; i < 256; i++) { u[4 * LAYOUT.gf + (i >> 2)] |= GF_EXP[i] << (8 * (i & 3)); u[4 * LAYOUT.gf + 64 + (i >> 2)] |= GF_LOG[i] << (8 * (i & 3)); }
  u.set(pictures.map((p) => p.hi), 4 * LAYOUT.tops);
  u.set(BEFORE, 4 * LAYOUT.before);
  return u;
}

// ai: The plan's points as the kernel reads them, every ring after the one before: per ring the references (2 a
// ai: step, the line then the gap), the track pairs (2 a pair), then the word cells in bit order, as vec4f (x, y,
// ai: lo + 4096 hi, 0) in module coordinates, lo and hi the pair range a cell's grey is the mean of. plans[r] =
// ai: (first point, reference steps, pairs, word cells). Throws where the plan leaves the kernel's shape.
export function wordPoints(rings) {
  const pts = [], plans = [];
  for (const { cells: ring, word: w } of rings) {
    if (w.reps !== 1 || w.words !== 4 * w.cells || w.cells > WMAX) throw new Error(`ring ${ring}: ${w.words} word points of ${w.cells} cells a side, reps ${w.reps}: F8 reads one point a cell, W <= ${WMAX}`);
    if (w.mids > MIDS_MAX) throw new Error(`ring ${ring}: ${w.mids} track pairs, F8 holds ${MIDS_MAX}`);
    plans.push([pts.length / 4, w.refSteps, w.mids, w.words]);
    const np = 2 * w.refSteps + 2 * w.mids;
    for (let k = 0; k < np; k++) pts.push(w.pts[2 * k], w.pts[2 * k + 1], 0, 0);
    for (let k = 0; k < w.words; k++) {
      const [lo, hi, bit] = w.widx.subarray(3 * k, 3 * k + 3);
      if (bit !== k || lo < 0 || hi < lo || hi >= w.mids) throw new Error(`ring ${ring}: word point ${k} is bit ${bit}, pairs ${lo} to ${hi}`);
      pts.push(w.pts[2 * (np + k)], w.pts[2 * (np + k) + 1], lo + 4096 * hi, 0);
    }
  }
  return { pts: new Float32Array(pts), plans };
}

// ai: The LT uniform: at[r] = (first coordinate, nodes a side, 0, 0), then every ring's lattice coordinates,
// ai: LATTICE_MAX a ring, four a vec4f (the fused pass 1's PIC holds them the same way).
export function wordLattice(rings) {
  const ab = new ArrayBuffer(16 * RING_COUNT + 4 * RING_COUNT * LATTICE_MAX), u = new Uint32Array(ab), f = new Float32Array(ab);
  rings.forEach(({ cells, nodeX }, r) => {
    if (nodeX.length > LATTICE_MAX || nodeX.length < 2) throw new Error(`ring ${cells}: ${nodeX.length} lattice nodes a side, LT holds 2 to ${LATTICE_MAX}`);
    u.set([LATTICE_MAX * r, nodeX.length, 0, 0], 4 * r);
    f.set(nodeX, 4 * RING_COUNT + LATTICE_MAX * r);
  });
  return new Uint8Array(ab);
}
