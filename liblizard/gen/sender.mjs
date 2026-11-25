// ai: The app's GPU sender (2026-10-02; the CPU already held 60 a second at LIZARD-1024): the sender's kernels
// ai: (gpu/slang/send.mjs: paint, irows and ipic (and its frame form) a picture size, rsv, rsh and its border form, RaptorQ's blocks) as
// ai: Slang, for gen.mjs to compile with slangc (SPIR-V for the native painter, WGSL for the web's encoder, from the same source), and the tables no format changes,
// ai: which the web host computes from the wasm and core/tx/gpu_painter.cpp takes from here: the paint's TAB (every
// ai: code of the format's rate profile, src/lizard_phy.h lizard_tiers_for, and the whitening: gpu/encoder.mjs sendTab, since
// ai: 2026-10-07; one code's bit map a block, PERMW, before) and the CRC's powers. What a format does change (the disc's
// ai: rows, the resampler's taps, the border, its tiers in the paint's uniform) the painter takes from the C's codec on
// ai: the phone (core/tx/send_tables.cpp), held to the web's by gen/send_tables_ref.mjs and `tx_check gputables`.
import { createHash } from "node:crypto";

const LIB = new URL("../", import.meta.url);
const imp = (p) => import(new URL(p, LIB).href);

// ai: { modules: { name: Slang }, manifest (setup/send.json), tab: TAB's bytes (blobs/<hash>.bin) }
export async function senderSetup() {
  const { paintSource, rsvSource, rshSource, irowsSource, ipicSource, rqBlocksSource, presentVsSource, presentFsSource } = await imp("gpu/slang/send.mjs");
  const { RSV_THREADS, RSH_THREADS, GA_BYTES, PARAMS_AT, RQ_OUT_WORDS } = await imp("gpu/plan/send.mjs");
  const { SLOTS, SIZES } = await imp("gpu/plan/common.mjs");
  const { CODES } = await imp("gpu/plan/back_tiers.mjs");
  const { crcPowers, PAYLOAD } = await imp("gpu/back/ref_ldpc.mjs");
  const { ruleTables } = await imp("gpu/back/codes.mjs");
  const { GAP_MODULES, sendTab } = await imp("gpu/encoder.mjs");
  const modules = { paint: paintSource(), rsv: rsvSource(), rsh: rshSource(), rshBorder: rshSource({ border: true }) };
  // ai: RaptorQ's blocks (2026-11-16, gpu/slang/src/send_rq_blocks.slang): each frame's blocks made from their ids, written into the
  // ai: painter's BLOCKS (core/tx/gpu_stream.h)
  Object.assign(modules, { rqBlocks: rqBlocksSource({ outWords: RQ_OUT_WORDS }) });
  if (RQ_OUT_WORDS * 4 !== 480) throw new Error(`the RQ blocks kernel writes ${RQ_OUT_WORDS * 4} B a block, the paint reads 480`);
  for (const n of SIZES) {
    modules[`irows${n}`] = irowsSource({ n });
    modules[`ipic${n}`] = ipicSource({ n });
    modules[`ipicFrame${n}`] = ipicSource({ n, frame: true });
  }
  // ai: the codes from the codec (the wasm's, as the web's encoder reads them), in CODES' order: the paint's uniform
  // ai: names a tier by its place there
  const rt = await ruleTables(), tab = sendTab(rt);
  if (SLOTS < 1 + CODES.length) throw new Error(`the paint's uniform holds ${SLOTS} sizes, the codes' tiers need ${1 + CODES.length}`);
  const tabBytes = new Uint8Array(tab.buffer, tab.byteOffset, tab.byteLength);
  const tabHash = createHash("sha256").update(tabBytes).digest("hex").slice(0, 16);
  const manifest = {
    modules: Object.keys(modules).map((k) => `send_${k}`), sizes: SIZES, slots: SLOTS, paramsAt: PARAMS_AT,
    rsvThreads: RSV_THREADS, rshThreads: RSH_THREADS, blockBytes: 480, payload: PAYLOAD, gapModules: GAP_MODULES,
    codes: CODES.map(({ rate, slots }) => ({ rate, slots })), pw: Array.from(crcPowers(PAYLOAD)), tab: { blob: `blobs/${tabHash}.bin`, words: tab.length },
    gWords: GA_BYTES / 4,
    // ai: paint makes the test stream's payloads from their ids where Params' dims.w has bit 31 (BLAKE3 of the id, SPEC
    // ai: 9.3; 2026-11-15): a host that sets it refuses a setup without this (core/tx/gpu_painter.cpp)
    stream: "blake3",
    // ai: a block's id 12 bytes (the bare id and the 48-bit TOI, src/xfer.h), the payload from word 3: paint's test
    // ai: stream and rq_enc's BLOCKS write that layout, and a host refuses a setup without it
    ids: 12,   // ai: the block id's bytes the kernels write
    // ai: rq_enc's PAR: the blocks, the controls' first symbol, the source blocks, the controls, the file's TOI's high 24
    // ai: bits and the session's (the session header's); a host refuses a setup without it (core/tx/gpu_painter.cpp;
    // ai: 5 words, the TSI the fifth, before the 48-bit TOI)
    rqPar: 6,
    // ai: the disc is the spectrum transposed (gpu/back/disc.mjs sendDiscLayout, core/tx/send_tables.cpp: UV's bit 31
    // ai: conjugates) and ipic writes PIC row-major, no tpose (2026-11-24): a host refuses a setup without it
    disc: "transposed",
  };
  // ai: what the web's encoder runs (gpu/encoder.mjs, through gen.mjs webWgsl): every kernel but RaptorQ's blocks, a
  // ai: stage each, and the present's two stages
  const web = { ...Object.fromEntries(Object.entries(modules).filter(([k]) => k !== "rqBlocks").map(([k, code]) => [k, { code, stage: "compute" }])),
    presentVs: { code: presentVsSource(), stage: "vertex" }, presentFs: { code: presentFsSource(), stage: "fragment" } };
  return { modules, manifest, tab: { hash: tabHash, bytes: tabBytes }, web };
}
