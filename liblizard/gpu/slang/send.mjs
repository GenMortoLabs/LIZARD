// The sender's kernels in Slang (src/send_paint.slang, send_rsv.slang, send_rsh.slang, send_irows.slang,
// send_ipic.slang, send_rq_blocks.slang; ../encoder.mjs runs them in the browser through the WGSL slangc writes from them, the
// native painter the SPIR-V, both through ../../gen/sender.mjs): PAINT, a block's codeword onto the disc (every code of the format's rate
// profile; the test stream's payloads from BLAKE3 of their ids), the inverse transforms IROWS and IPIC (a picture size
// each, on the spectrum transposed: PIC, n x n f32 a symbol in 0..1, clipped, row-major; or, the copy form, the frame's
// bytes), RSV and RSH (and its border form), the C's arithmetic
// (src/lizard_phy.c resample, lizard_paint_rgba): the border, the margin and the anchors' squares the C's own bytes,
// only the resampled square computed, within a grey level of the C's; and the native senders' RaptorQ blocks. Their
// constants and uniforms' layouts: ../plan/send.mjs.
import { slang } from "./load.mjs";
import { firstStage, laterStages } from "./back_transform.mjs";
import { KNOWN, NC, PB_REF, DCOEF } from "../plan/back_tiers.mjs";
import { SLOTS } from "../plan/common.mjs";
import { CRC_CHUNKS, PW_VECS, CW_WORDS, PAINT_THREADS, RSV_THREADS, RSH_THREADS, B3_PERM, TAB_DEG, TAB_PLANS, TAB_PLAN, RQ_OUT_WORDS } from "../plan/send.mjs";

if (PAINT_THREADS !== 128) throw new Error(`src/send_paint.slang is written for 128 lanes, not ${PAINT_THREADS}`);

// ai: BLAKE3's compression over v0..v15, the message m0 (the id's low word) and m1 (its fifth byte), the words that are
// ai: always zero left out: seven rounds of the eight G's, the message permuted between rounds
function b3Steps() {
  const add = (a, b) => (b === null ? a : `(${a} + ${b})`), rot = (x, n) => `rotr(${x}, ${n}u)`;
  let pos = [...Array(16).keys()];
  const out = [];
  const g = (a, b, c, d, x, y) => {
    out.push(`v${a} = ${add(add(`v${a}`, `v${b}`), x)};`, `v${d} = ${rot(`v${d} ^ v${a}`, 16)};`, `v${c} = ${add(`v${c}`, `v${d}`)};`,
      `v${b} = ${rot(`v${b} ^ v${c}`, 12)};`, `v${a} = ${add(add(`v${a}`, `v${b}`), y)};`, `v${d} = ${rot(`v${d} ^ v${a}`, 8)};`,
      `v${c} = ${add(`v${c}`, `v${d}`)};`, `v${b} = ${rot(`v${b} ^ v${c}`, 7)};`);
  };
  for (let r = 0; r < 7; r++) {
    const w = (k) => (pos[k] === 0 ? "m0" : pos[k] === 1 ? "m1" : null);
    g(0, 4, 8, 12, w(0), w(1)); g(1, 5, 9, 13, w(2), w(3)); g(2, 6, 10, 14, w(4), w(5)); g(3, 7, 11, 15, w(6), w(7));
    g(0, 5, 10, 15, w(8), w(9)); g(1, 6, 11, 12, w(10), w(11)); g(2, 7, 8, 13, w(12), w(13)); g(3, 4, 9, 14, w(14), w(15));
    pos = pos.map((_, i) => pos[B3_PERM[i]]);
  }
  return "  " + out.join("\n  ");
}

export const paintSource = () => slang("send_paint.slang", {
  SLOT_COUNT: SLOTS, PW_VECS, PAINT_KNOWN: `${KNOWN}u`, CW_WORDS: `${CW_WORDS}u`, NCODES: `${NC}u`, DCOEF: `${DCOEF}u`, PB_REF: `${PB_REF}u`,
  CRC64_CHUNKS: `${CRC_CHUNKS}u`,
}, { b3: b3Steps() });
export const rsvSource = () => slang("send_rsv.slang", { LANES: RSV_THREADS });
export const rshSource = ({ border = false } = {}) => slang("send_rsh.slang", { BORDER_ONLY: border ? 1 : 0, LANES: RSH_THREADS });
export const irowsSource = ({ n }) => slang("send_irows.slang", { LANES: n / 8, PICTURE_N: `${n}u` }, { fft: firstStage("f32") + laterStages(n, "f32") });
// ai: frame: the copy form's, the frame's bytes written from workgroup memory (FRAME_OUT; RSH's border form the rest)
export const ipicSource = ({ n, frame = false }) => {
  const lanes = [["v0", "buf0"], ["v1", "buf1"]];
  return slang("send_ipic.slang", { LANES: n / 8, PICTURE_N: `${n}u`, FRAME_OUT: frame ? 1 : 0 }, { fft: firstStage("f32", lanes) + laterStages(n, "f32", lanes) });
};
export const rqBlocksSource = ({ outWords = RQ_OUT_WORDS } = {}) => slang("send_rq_blocks.slang", {
  TAB_DEG: `${TAB_DEG}u`, TAB_PLANS: `${TAB_PLANS}u`, TAB_PLAN: `${TAB_PLAN}u`, OUT_WORDS: `${outWords}u`,
});
// ai: the web's present (a render pipeline: a triangle over the canvas, the frame shown in the code's colours), its two
// ai: stages
export const presentVsSource = () => slang("send_present_vs.slang");
export const presentFsSource = () => slang("send_present_fs.slang");
