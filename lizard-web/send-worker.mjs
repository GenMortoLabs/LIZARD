// The sender's encoder, off the page's thread (lizard-web/send.mjs), for a LIZARD symbol. It builds each frame's
// blocks (the test stream, or the fountain for a file), encodes them and writes the finished RGBA pixels in wasm
// (src/lizard_phy.c lizard_paint_rgba), a few frames ahead of the page, so the page only puts pixels on its canvas. At
// LIZARD-1024 that was 13 ms here and a few on the page, where the page used to spend 45 ms a frame on both (2026-09-23,
// n = 2048 then).
//
// Messages run one at a time in arrival order (the chain below), so a frame asked for after a re-pick is encoded with
// the new code and one asked for before it with the old: the page's block ids stay right across the change.
//
// A file goes as a chunked transfer (sim/xfer.mjs): its header and manifest in the light, a fountain per chunk, and the
// block ids chosen here by the transfer's schedule, so the page's baseId only counts blocks. It is made when the page's
// file is picked ("prepare"), and a start names it by its pick; a start with bytes makes one there, as before.
// ai: codes: symbols a frame, 1 or 2 (the page's "codes"). Two are painted side by side in one buffer, a gap of
// ai: gapModules modules between them (the page's slider since 2026-10-03, sent with every start and respec;
// ai: 2026-09-30: widened by about 2 modules, then 2 more: 4; then 12, as the 8
// ai: modules of white between the rims could not be seen; packed to their quiet zones from 2026-09-28, since with two
// ai: codes the whole performance rests on packing), the first with the frame's first blocksPerFrame ids and the second
// ai: with the next, one format and one word for both.
// ai: Under the page's GPU encoder (gpu/encoder.mjs, since 2026-09-29) frames are asked for a batch at a time
// ai: ("frames": count frames from seq) and come back as their blocks alone, one buffer, frame after frame, each codes x
// ai: blocksPerFrame x 476 bytes; the page encodes them. The test stream's come as their ids alone (test, 2026-11-15):
// ai: the encoder makes each payload from its id on the device (BLAKE3, SPEC 9.3), so nothing here hashes.
// ai: Rate refresh (refresh.mjs, the page's "refresh", sent with every start too): a block of a held rate repeats the
// ai: id at its place in the frame before, so only the new places take ids (the transfer's schedule, or the test
// ai: stream's from baseId); the frame before is forgotten at start, an accepted respec and stop.
import { makePhy } from "../liblizard/sim/phy.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initFountain, Encoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { packing } from "../liblizard/sim/pack.mjs";
import { isControlId, ID_BYTES, idPut } from "../liblizard/sim/xfer.mjs";
import { makeXfer } from "./xfer-pool.mjs";
import { Refresh, batchIds } from "./refresh.mjs";

let phy = null, xs = null, fps = 0, codes = 1, chain = Promise.resolve();
// ai: The transfer made at a pick's first Start ("prepare", sent just before "start"; at the pick from 2026-10-19 to
// ai: 2026-11-01): the file packed where that pays (sim/pack.mjs, the default client's compression) and its root hashed. made: { pick, xs } while no run sends it; a run takes
// ai: it (xsPick its pick) and a stop gives it back, so the next Start sends the same transfer on, with nothing made again.
let made = null, xsPick = null;
const refresh = new Refresh();
// ai: the test stream's n new ids from the page's base (a frame's own range: baseId to baseId + its blocks)
const run = (base, n) => Float64Array.from({ length: n }, (_, i) => base + i);   // ai: ids are 40 bits (sim/xfer.mjs)
// ai: a frame's new data blocks: its new ids but the transfer's header and manifest (every test stream block is data)
const dataOf = (ids, fresh) => { let n = 0; for (let t = 0; t < ids.length; t++) if (fresh[t] && (!xs || !isControlId(ids[t]))) n++; return n; };
function dropRun() {
  if (xs) { if (xsPick !== null && !made) made = { pick: xsPick, xs }; else xs.free(); }
  xs = null; xsPick = null;
}
// ai: a frame's blocks held by the pool's workers (xfer-pool.mjs), fetched before it is painted; the rest are xs's own
let fetched = null;
const source = () => (id, out) => { const b = fetched?.get(id); if (b) out.set(b); else xs.block(id, out); };
async function fetchFor(ids) { fetched = ids && xs?.remote ? await xs.remote.fetch(ids) : null; }
// ai: a pool's transfer made whole behind its first frames (xfer-pool.mjs, streaming): the bytes it carries and its lap
// ai: then said to the page ("made", by pick), or the failure; nothing for one let go first
function whenMade(x, said) {
  x.remote?.made?.then(
    () => postMessage({ type: "made", ...said, sent: x.length, lap: x.sched?.lap ?? 0 }),
    (e) => { if (x.remote) postMessage({ type: "made", ...said, error: String(e?.message ?? e) }); });
}
// ai: The file's transfer: packed where that pays (sim/pack.mjs; the bytes the transfer carries are `sent`), made by
// ai: the pool (xfer-pool.mjs). Between the pack's frames and the pool's chunks the worker takes its other messages
// ai: (pause) and asks `stopped`: a preparation the page cancels ends there, a frame (4 MiB) or a chunk's hash on
// ai: (2026-11-16; the page ended the whole worker before, which Chrome holds back about 2 s while it runs a script).
const CANCELLED = new Error("cancelled");
const pauseCh = new MessageChannel(), pauseQ = [];
pauseCh.port1.onmessage = () => pauseQ.shift()();
const pause = () => new Promise((r) => { pauseQ.push(r); pauseCh.port2.postMessage(0); });   // ai: a task, no timer's clamp
async function transferOf(bytes, xfer, stopped = () => false) {
  await Promise.all([initFountain(), initZstd()]);
  const g = packing(bytes, xfer.name, xfer.type, Z);
  let r;
  while (!(r = g.next()).done) { await pause(); if (stopped()) throw CANCELLED; }
  const p = r.value;
  return makeXfer(p.bytes, { ...xfer, name: p.name, type: p.type }, { M: await initOb(), Encoder, stopped });
}
// ai: the newest preparation the page let go ("cancel", taken as it comes, outside the chain; its number the page's)
let cancelledUpTo = 0;
// ai: the gap between two codes, in modules (the page's slider, start and respec's gap), and in the paint's pixels for
// ai: this codec (phy.cell samples a module)
let gapModules = 0;
const gapOf = (p) => gapModules * (p.cell || 0);

async function handle(m) {
  if (m.type === "prepare") {
    // ai: its own reply ("prepared", by pick and the page's number for it, outside the runs' gen), a failure in it or
    // ai: its cancel too
    made?.xs.free(); made = null;
    const stopped = () => m.prep <= cancelledUpTo;
    try {
      const x = await transferOf(new Uint8Array(m.bytes), m.xfer, stopped);
      if (stopped()) { x.free(); postMessage({ type: "prepared", pick: m.pick, prep: m.prep, cancelled: true }); return; }
      made = { pick: m.pick, xs: x };
      postMessage({ type: "prepared", pick: m.pick, prep: m.prep, sent: x.length });
      whenMade(x, { pick: m.pick });
    } catch (e) {
      postMessage({ type: "prepared", pick: m.pick, prep: m.prep, ...(stopped() ? { cancelled: true } : { error: String(e?.message ?? e) }) });
    }
  } else if (m.type === "start") {
    phy?.free(); phy = null; dropRun();
    phy = await makePhy(m.spec);
    if (m.pick != null) {
      if (made?.pick !== m.pick) throw new Error("the file's transfer was not made: choose it again");
      xs = made.xs; xsPick = m.pick; made = null;
      phy.setSource(source());
    } else if (m.bytes) {
      xs = await transferOf(new Uint8Array(m.bytes), m.xfer);
      whenMade(xs, { pick: null });
      phy.setSource(source());
    }
    // ai: Developer Tools' Seed (2026-11-14): the schedule drawn from it before the first frame (a transfer made at the pick
    // ai: drew a random one); none given, the random one stands
    if (xs && m.seed != null) xs.reseed(m.seed);
    fps = m.fps; phy.setFps?.(fps); codes = m.codes ?? 1; gapModules = m.gap;
    if (m.every) refresh.setEvery(m.every);
    refresh.setFormat(m.spec.subch, phy.blocksPerFrame);
    postMessage({ type: "ready", gen: m.gen, label: phy.label, usefulBytes: phy.usefulBytes, blocksPerFrame: phy.blocksPerFrame, xfer: xs?.info() });
  } else if (m.type === "respec") {
    // A re-pick (the room changed): a new code for the same transfer. Refused if the block size would move, which
    // would break the fountain's stream, exactly as the page used to refuse it.
    const next = await makePhy(m.spec);
    if (next.usefulBytes !== phy.usefulBytes) { next.free(); postMessage({ type: "respec", gen: m.gen, ok: false }); return; }
    phy.free(); phy = next; phy.setFps?.(fps); codes = m.codes ?? codes; gapModules = m.gap;
    if (xs) phy.setSource(source());
    refresh.setFormat(m.spec.subch, phy.blocksPerFrame);
    postMessage({ type: "respec", gen: m.gen, ok: true, label: phy.label, blocksPerFrame: phy.blocksPerFrame });
  } else if (m.type === "fps") {
    fps = m.fps; phy?.setFps?.(fps);
  } else if (m.type === "refresh") {
    refresh.setEvery(m.every);
  } else if (m.type === "stop") {
    phy?.free(); phy = null; dropRun(); refresh.reset();
  } else if (m.type === "frame") {
    if (!phy) return;
    const t = performance.now(), V = phy.blocksPerFrame;
    const { ids, fresh } = refresh.next(m.seq, V * codes, (n) => (xs ? xs.frameIds(n) : run(m.baseId, n)));
    await fetchFor(xs ? ids : null);
    // ai: Each symbol's pixels are a view into the wasm heap, good until the next paint, so each is copied row by row
    // ai: into its place (side k of codes) before the next is painted. The page hands its last buffer back to be filled.
    // ai: Two codes sit a gap of gapModules apart (since 2026-09-30; packed to their quiet zones before), the gap the
    // ai: colour of the margin beside it; the frame is
    // ai: codes w + (codes - 1) gap px wide and the message says the gap.
    let out = null, w = 0, gap = 0, fw = 0;
    for (let k = 0; k < codes; k++) {
      const part = phy.frameRGBA(m.seq, ids.subarray(k * V, (k + 1) * V)), row = 4 * part.w;
      if (!out) {
        w = part.w; gap = codes > 1 ? gapOf(phy) : 0; fw = codes * w + (codes - 1) * gap;
        const bytes = 4 * fw * w;
        out = m.buf && m.buf.byteLength === bytes ? new Uint8Array(m.buf) : new Uint8Array(bytes);
      }
      for (let y = 0; y < w; y++) {
        const at = 4 * (y * fw + k * (w + gap));
        out.set(part.rgba.subarray(y * row, (y + 1) * row), at);
        if (k + 1 < codes) for (let x = 0; x < gap; x++) out.set(part.rgba.subarray((y + 1) * row - 4, (y + 1) * row), at + row + 4 * x);
      }
    }
    // ai: data: the frame's new data blocks, its header and manifest blocks and its repeats not counted; the page
    // ai: counts laps with it (xfer.lap data blocks a lap). Every new block of the test stream is data.
    const data = dataOf(ids, fresh);
    postMessage({ type: "frame", gen: m.gen, seq: m.seq, w, codes, gap, buf: out.buffer, ms: performance.now() - t, data }, [out.buffer]);
  } else if (m.type === "frames") {
    if (!phy) return;
    const t = performance.now(), V = phy.blocksPerFrame, B = phy.usefulBytes + ID_BYTES, per = codes * V * B, bytes = m.count * per;
    const out = m.buf && m.buf.byteLength === bytes ? new Uint8Array(m.buf) : new Uint8Array(bytes), T = V * codes;
    // ai: the batch's frames in order, each holding against the one before (the batch before's last for the first)
    const all = Array.from({ length: m.count }, (_, f) => refresh.next(m.seq + f, T, (n) => (xs ? xs.frameIds(n) : run(m.baseId + f * T, n))));
    if (xs) await fetchFor(batchIds(all));
    const data = all.map(({ ids, fresh }) => dataOf(ids, fresh)), test = !xs;
    for (let f = 0; f < m.count; f++) {
      if (test) { for (let t = 0; t < T; t++) idPut(out, f * per + t * B, all[f].ids[t]); continue; }
      for (let k = 0; k < codes; k++) out.set(phy.frameBlocks(m.seq + f, all[f].ids.subarray(k * V, (k + 1) * V)), f * per + k * V * B);
    }
    postMessage({ type: "frames", gen: m.gen, seq: m.seq, count: m.count, codes, test, buf: out.buffer, ms: performance.now() - t, data }, [out.buffer]);
  }
}

onmessage = ({ data: m }) => {
  if (m.type === "cancel") { cancelledUpTo = Math.max(cancelledUpTo, m.prep); return; }
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", gen: m.gen, message: String(e?.message ?? e) }));
};
