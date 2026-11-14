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
// ai: blocksPerFrame x 476 bytes; the page encodes them.
// ai: Rate refresh (refresh.mjs, the page's "refresh", sent with every start too): a block of a held rate repeats the
// ai: id at its place in the frame before, so only the new places take ids (the transfer's schedule, or the test
// ai: stream's from baseId); the frame before is forgotten at start, an accepted respec and stop.
import { makePhy } from "../liblizard/sim/phy.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initFountain, Encoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { pack } from "../liblizard/sim/pack.mjs";
import { isControlId } from "../liblizard/sim/xfer.mjs";
import { makeXfer } from "./xfer-pool.mjs";
import { Refresh, batchIds } from "./refresh.mjs";

let phy = null, xs = null, fps = 0, codes = 1, chain = Promise.resolve();
// ai: The transfer made at a pick's first Start ("prepare", sent just before "start"; at the pick from 2026-10-19 to
// ai: 2026-11-01): the file packed where that pays (sim/pack.mjs, the default client's compression) and its root hashed. made: { pick, xs } while no run sends it; a run takes
// ai: it (xsPick its pick) and a stop gives it back, so the next Start sends the same transfer on, with nothing made again.
let made = null, xsPick = null;
const refresh = new Refresh();
// ai: the test stream's n new ids from the page's base (a frame's own range: baseId to baseId + its blocks)
const run = (base, n) => Uint32Array.from({ length: n }, (_, i) => base + i);
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
// ai: the pool (xfer-pool.mjs)
async function transferOf(bytes, xfer) {
  await Promise.all([initFountain(), initZstd()]);
  const p = pack(bytes, xfer.name, xfer.type, Z);
  return makeXfer(p.bytes, { ...xfer, name: p.name, type: p.type }, { M: await initOb(), Encoder });
}
// ai: the gap between two codes, in modules (the page's slider, start and respec's gap), and in the paint's pixels for
// ai: this codec (phy.cell samples a module)
let gapModules = 0;
const gapOf = (p) => gapModules * (p.cell || 0);

async function handle(m) {
  if (m.type === "prepare") {
    // ai: its own reply ("prepared", by pick, outside the runs' gen), a failure in it too
    made?.xs.free(); made = null;
    try {
      const x = await transferOf(new Uint8Array(m.bytes), m.xfer);
      made = { pick: m.pick, xs: x };
      postMessage({ type: "prepared", pick: m.pick, sent: x.length });
      whenMade(x, { pick: m.pick });
    } catch (e) { postMessage({ type: "prepared", pick: m.pick, error: String(e?.message ?? e) }); }
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
    const t = performance.now(), V = phy.blocksPerFrame, B = phy.usefulBytes + 4, per = codes * V * B, bytes = m.count * per;
    const out = m.buf && m.buf.byteLength === bytes ? new Uint8Array(m.buf) : new Uint8Array(bytes), T = V * codes;
    // ai: the batch's frames in order, each holding against the one before (the batch before's last for the first)
    const all = Array.from({ length: m.count }, (_, f) => refresh.next(m.seq + f, T, (n) => (xs ? xs.frameIds(n) : run(m.baseId + f * T, n))));
    if (xs) await fetchFor(batchIds(all));
    const data = all.map(({ ids, fresh }) => dataOf(ids, fresh));
    for (let f = 0; f < m.count; f++) {
      for (let k = 0; k < codes; k++) out.set(phy.frameBlocks(m.seq + f, all[f].ids.subarray(k * V, (k + 1) * V)), f * per + k * V * B);
    }
    postMessage({ type: "frames", gen: m.gen, seq: m.seq, count: m.count, codes, buf: out.buffer, ms: performance.now() - t, data }, [out.buffer]);
  }
}

onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", gen: m.gen, message: String(e?.message ?? e) }));
};
