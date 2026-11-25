// The JS the decoder's Slang modules share: lib/cfg.slang's defines (CFG_DEFS) and the Slang text of float literals and
// zero arrays (lit, zeros, zeroDefs). The Slang itself is src/ (load.mjs).
import { CODES, VERSION_SLOTS } from "../plan/back_tiers.mjs";
import { VERSION_MIN, VERSION_MAX } from "../wordcode.mjs";

// ai: lib/cfg.slang's defines: the four codes' slots, the versions' range and a version's data slots
export const CFG_DEFS = {
  CODE0_SLOTS: `${CODES[0].slots}u`, CODE1_SLOTS: `${CODES[1].slots}u`, CODE2_SLOTS: `${CODES[2].slots}u`, CODE3_SLOTS: `${CODES[3].slots}u`,
  CFG_VERSION_MIN: `${VERSION_MIN}u`, CFG_VERSION_MAX: `${VERSION_MAX}u`, CFG_VERSION_SLOTS: `${VERSION_SLOTS}u`,
};
// ai: lit(x) a JS number as a float literal; zeros(n) or zeros(n, m) a float array's zero as an initialiser list
// ai: (a function variable's zero, written out; a declaration's initialiser or a return's value)
export const lit = (x) => (Number.isInteger(x) ? `${x}.0` : String(x));
export const zeros = (n, m) => (m === undefined ? `{ ${Array(n).fill("0.0").join(", ")} }` : `{ ${Array(n).fill(zeros(m)).join(", ")} }`);
// ai: the defines ZEROS_<n> and ZEROS_<n>_<m> (zeros(n), zeros(n, m)) for a stage's zero float arrays: a shape a number or [n, m]
export const zeroDefs = (...shapes) => Object.fromEntries(shapes.map((s) => [`ZEROS_${[].concat(s).join("_")}`, zeros(...[].concat(s))]));
