// ai: The rate profile's stage's constants, TAB's layout and each code's shape (glsl/back_tiers.mjs talign, tsoft, tblocks,
// ai: tldpc; ../back/tiers.mjs builds it), which the sender's paint (../wgsl/send.mjs, ../encoder.mjs) shares: CODES, KNOWN, NC.
import { VERSION_MAX } from "../wordcode.mjs";
import { STEP, SUB_SLOTS } from "../../sim/lizard_pick.mjs";

// ai: The pilots apart from the blocks (2026-11-21, src/lizard_phy.c): in every sub-channel coefficient 0 is the count
// ai: pilot (both axes the whitening XOR bit j & 1 of the painted count, j the sub-channel) and coefficient PB_REF the
// ai: reference pilot (the whitening alone); the blocks pack over the other DCOEF coefficients in order (fcoefOf: data
// ai: coefficient dc's coefficient in the frame, src/lizard_phy.c pb_fcoef), SUB_SLOTS data slots a sub-channel. The
// ai: whitening stays by the frame's slot (WHITE_SLOTS), a data coefficient taking its frame coefficient's.
// ai: The grid's shift by the pilots (2026-10-01; the format paints the pilots whatever the phone, so a decoder may
// ai: steer by them; src/lizard_phy.c pilot_align, the same fit). A grid (dx, dy) samples off turns coefficient (u, v) by
// ai: 2 pi (u dx + v dy) / n. `talign` (one workgroup a frame, TIER_BLOCKS lanes over its sub-channels, before the soft
// ai: values in their pass and on their bind groups) reads every sub-channel's two pilots (each its sign cancels: every
// ai: term the fit sums is a product of two of its parts), fits zi = k zr (u dx + v dy) by least squares twice (the
// ai: second about the first's answer), and keeps the shift wherever the fit stands (8 pilots or more, its matrix
// ai: invertible; the significance test ALIGN_CHI 13.8 went 2026-11-07, as the C's), else 0, 0; each pilot unweighted,
// ai: where the C weights it by 1 / N of its sub-channel (the moments before the fit, a pass of its own, not built). It
// ai: goes to PILOT's end: PILOT[PILOT_READ B + 2 f] and the next; before it, PILOT_READ floats a frame, its readings
// ai: (tblocks: r, its standard error, r2, its standard error; src/lizard_phy.c lizard_pilot).
// ai: The soft stage turns every coefficient back by it as it gathers (a turn leaves |y|, so the moments and the
// ai: estimate read as before). UV's second half (from P.ucOff) is each coefficient's (u, v), u in the high 16 bits
// ai: (signed), v in the low.

// ai: blocks a frame at most (talign's and tblocks' lanes): the most a word may state, 149 at version 67 (127 6/7 blocks
// ai: and 22 3/4 ones; sim/lizard_pick.mjs blocksMaxFor), rounded up to whole 32-lane sets; 128 before 2026-11-18
export const TIER_BLOCKS = 160;
// ai: the versions a word names (../wordcode.mjs VERSION_MAX)
export const VERSIONS = VERSION_MAX;
// ai: the 8-sub-channel chunks a version takes (16 sub-channels, sim/lizard_pick.mjs STEP): the soft stage's workgroups
// ai: are a chunk each, and the gate's rows table (gpu/back/transform.mjs rowsOfVersion) counts chunks.
export const CHUNKS = STEP / 8;
export const PB_REF = 160;
export const DCOEF = SUB_SLOTS / 2;   // ai: a sub-channel's data coefficients (318)
export { SUB_SLOTS };
export const VERSION_SLOTS = STEP * SUB_SLOTS;   // ai: a version's data slots (10,176; src/fmt.c ob_fmt_layout)
export const PILOT_READ = 4;
export const KNOWN = 3840;        // ai: (472 + 8) x 8: the bits a block's payload and CRC fill, the k of the 3/4, 2/3 and 1/2 codes; past them
                                  // ai: the 6/7 code's data bits are zeros
// ai: the whitening a slot over the ladder's largest frame (16 VERSION_MAX sub-channels of 640 frame slots, the pilots'
// ai: four among them: 1072 since 2026-11-18, 1024 before)
export const WHITE_SLOTS = STEP * VERSION_MAX * 640;
// ai: The LDPC kernel's forms (2026-10-08, the decoder's speed; research STATUS "The LDPC's speed"), all counting the
// ai: violated checks after each iteration from a bitmap of the posteriors' signs, 32 checks of a row at a time
// ai: (violatedChunks), and keeping each check's record in its lane's own array (lane i is check i of every row, so no
// ai: other lane reads it): "reg" two posteriors a word in workgroup memory (atomic stores); "reg+i32" a posterior a
// ai: word with plain loads and stores where the code's workgroup memory still fits 32 KB; "reg+i16" (2026-10-31) a
// ai: posterior an int16_t with plain loads and stores, "reg"'s bytes without its atomics, its module needing the
// ai: device's shaderInt16 (its pipelines' `needs`: the host leaves it out where that is missing). All are exact (the
// ai: same decisions and iterations) and differ by workgroup memory alone, so the one a device runs faster is the
// ai: faster on any batch: a generated tree carries every form and the native host times them on its first batch and
// ai: keeps the faster ("reg+i16" on the iGPU and the S26, at one and two frames a launch and batched: research STATUS
// ai: "The LDPC's posteriors as int16_t"; "reg" and "reg+i32" before). The pair form (a check on two lanes, the 1/2 code's rows side by side) went 2026-10-31: in GLSL with
// ai: the slot loops unrolled "reg+i32" beat it on the S26's iterating captures at every launch, and its lower cost on
// ai: clean batches at one and two frames a launch turned the first-batch pick wrong (research STATUS "GLSL as the GPU
// ai: decoder's native source"). The lab's other forms (records in workgroup memory, a parity bit array; LIZ_TLDPC)
// ai: went 2026-10-22; the check's own values in 16-bit registers ("reg+i16+r16") was slower on both devices
// ai: (2026-10-31).
export const LDPC_FORMS = ["reg", "reg+i32", "reg+i16"];
// ai: the codes, in order: src/lizard_phy.c lizard_tiers_for's four rates (ldpc.h index, slots a block: the code's
// ai: sent bits, src/lizard_phy.c lizard_block_slots; since 2026-11-21 the pilots apart from the blocks, two
// ai: coefficients a sub-channel, so a block has no tail: 4496, 5152, 5792 and 7712 with it). Every block's slots are a
// ai: multiple of 16, so every block's first data slot is too (the blocks are packed from the frame's first data slot,
// ai: the remainder padded).
export const CODES = [{ rate: 6, slots: 4480, name: "6/7" }, { rate: 4, slots: 5120, name: "3/4" }, { rate: 3, slots: 5760, name: "2/3" }, { rate: 2, slots: 7680, name: "1/2" }];
// ai: 2/3 among the codes since 2026-10-07, in the format's profile since 2026-10-08 (four rates, 6/7, 3/4, 2/3, 1/2).
// ai: A frame's blocks of each are its word's (../glsl/common.mjs CFG; TAB's TT table, a row a version, until 2026-11-18).
export const NC = CODES.length;

// ai: TAB's sections' word offsets, from the codes' n and the nets' words (all codes' together).
export function tabLayout(codes, netWords) {
  let o = 0;
  const inv = codes.map((c) => { const at = o; o += Math.ceil(c.n / 2); return at; });
  const WHITE = o;
  o += WHITE_SLOTS / 32 + 1;   // ai: one word past the last slot: whiteWord reads the word after a slot's
  const NET = o;
  o += netWords;
  return { inv, WHITE, NET, words: o };
}

// ai: A code's module constants: its shape (n, k, z, block rows mb, the most slots a row, staircase included), the
// ai: workgroup's lanes (z rounded up to 32, at least 128), whether a check's record fits one word, and the kernel's
// ai: form (LDPC_FORMS; wide: i32, only where it fits 32 KB; narrow: i16).
export function tierShape(code, form = LDPC_FORMS[0]) {
  const { n, k, z, mb, lay } = code;
  let nd = 0;
  for (let r = 0; r < mb; r++) nd = Math.max(nd, lay[r + 1] - lay[r]);
  const ns = nd + 2, slots = lay[mb];
  const parts = form.split("+");
  if (!parts.includes("reg") || parts.some((p) => !["reg", "i32", "i16"].includes(p)) || (parts.includes("i32") && parts.includes("i16")))
    throw new Error(`LDPC form ${form}: one of ${LDPC_FORMS.join(", ")}`);
  return { n, k, z, mb, ns, slots, threads: Math.max(128, 32 * Math.ceil(z / 32)), one: ns <= 12, rowv: Math.ceil((mb + 1) / 4), pairv: Math.ceil(slots / 4), form,
    wide: parts.includes("i32") && 4 * n + 4 * packWords(n) + 12 + 4 * (6 + 2 * 64) <= 32768, narrow: parts.includes("i16") };
}
// ai: packw's words: the 120 of a block's data (118 the payload's, 2 its CRC-64's) for the CRC, and between iterations the posteriors' signs (N bits and a
// ai: word past them, which get32 may read).
const packWords = (n) => Math.max(120, Math.ceil(n / 32) + 1);
// ai: Workgroup bytes a code's kernel declares (32 KB is the floor every phone we know offers).
// ai: ("reg+i16"'s int16_t array, 2 ceil(n / 2) of them, is "reg"'s bytes)
export const tierWgBytes = (sh, net) => (sh.wide ? 4 * sh.n : 4 * Math.ceil(sh.n / 2)) + 4 * packWords(sh.n) + 20 + 4 * (6 + 2 * net.hidden);

// ai: the CRC-64's split powers in a code's uniform: 118 chunks' and the init term's, two words each, to whole vec4s
// ai: (../back/ref_ldpc.mjs crcPowers; ../wgsl/back_ldpc.mjs PW_VECS, held to it)
export const PW_WORDS = 240;
export const tierUniformWords = (sh) => 4 + 4 * sh.rowv + 4 * sh.pairv + PW_WORDS;
