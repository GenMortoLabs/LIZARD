// F8, the format word (src/word.slang): the band read through the final map, each word cell against its track pairs'
// grey; the word by Gauss-Jordan over GF(256) from its most reliable bytes (../wordcode.mjs); the side-aware gate, held
// to the C's false-word rate; the word's index split into the configuration and the clock; sel and the result written.
// ai: The Word and Lattice uniforms' layout and tables come from ../plan/word.mjs.
import { CFG_DEFS } from "./common.mjs";
import { slang } from "./load.mjs";
import { SLOTS } from "../plan/common.mjs";
import { LATTICE_MAX } from "../plan/back_transform.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { RESID } from "../plan/register.mjs";
import { VERSION_MIN, VERSION_MAX, TAU, SIDE_KEEP, MAGIC, DELAY_MAX, CLOCKS, WORDS as WORD_COUNT, VERSION_SLOTS, UNIT, UNITS } from "../wordcode.mjs";
import { CON_MIN, WMAX, ROWS, MIDS_MAX, VECS, UNKNOWN, SWAP, SETS, TOP, BYTES_MAX, GATE, LAYOUT, BEFORE_VECS } from "../plan/word.mjs";

if (UNKNOWN !== 5) throw new Error(`F8's solve is written for five unknown bytes, the word has ${UNKNOWN}`);
// ai: the words' count as two u32 (high, low): an index at or past it is no word
const WORDS_HI = Math.floor(WORD_COUNT / 2 ** 32), WORDS_LO = WORD_COUNT % 2 ** 32;

const TOPS = Math.ceil(SLOTS / 4);
const WORDS = ["x", "y", "z", "w"];

export const WORD = { slang: slang("word.slang", {
  W_CODE: LAYOUT.range, RING_COUNT, W_TOPS: TOPS, W_GATE: (RING_COUNT * GATE) / 4, W_GF: LAYOUT.cols - LAYOUT.gf,
  W_COLS: LAYOUT.before - LAYOUT.cols, W_BEFORE: BEFORE_VECS, LATTICE_VECS: (RING_COUNT * LATTICE_MAX) / 4,
  MIDS_MAX, WMAX: `${WMAX}u`, BYTES_MAX: `${BYTES_MAX}u`, TOP: `${TOP}u`, SETS: `${SETS}u`, ROWS: `${ROWS}u`, VECS: `${VECS}u`,
  GATE: `${GATE}u`, UNKNOWN: `${UNKNOWN}u`, SWAP: `${SWAP}u`, MAGIC: `${MAGIC}u`, WORDS_HI: `${WORDS_HI}u`, WORDS_LO: `${WORDS_LO}u`,
  CLOCKS: `${CLOCKS}u`, DELAY_MAX: `${DELAY_MAX}u`, VERSION_MIN: `${VERSION_MIN}u`, VERSION_MAX: `${VERSION_MAX}u`,
  VERSION_SLOTS: `${VERSION_SLOTS}u`, UNIT: `${UNIT}u`, UNIT0: `${UNITS[0]}u`, UNIT1: `${UNITS[1]}u`, UNIT2: `${UNITS[2]}u`,
  UNIT3: `${UNITS[3]}u`, TAU, SIDE_KEEP, CON_MIN, RESID: `${RESID}u`, ...CFG_DEFS,
}, {
  // ai: the picture slot a version decodes in: one step a picture past the first (the Word uniform's tops)
  slot_steps: Array.from({ length: SLOTS - 1 }, (_, k) => ` + (v > W.tops[${k >> 2}].${WORDS[k & 3]} ? 1u : 0u)`).join(""),
}) };
