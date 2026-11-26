// The sender's encoder, off the page's thread (lizard-web/send.mjs), for a LIZARD symbol. It builds each frame's
// blocks (the test stream, or the fountain for a file), encodes them and writes the finished RGBA pixels in wasm
// (src/lizard_phy.c lizard_paint_rgba), a few frames ahead of the page, so the page only puts pixels on its canvas. At
// LIZARD-1024 that was 13 ms here and a few on the page, where the page used to spend 45 ms a frame on both (2026-09-23,
// n = 2048 then).
//
// Messages run one at a time in arrival order (the chain below), so a frame asked for after a re-pick is encoded with
// the new code and one asked for before it with the old: the page's block ids stay right across the change.
//
// A file goes as a chunked transfer (lizard.h liz_tx, the C's sending end through the threaded module,
// liblizard/bindings/wasm/dist/lizard-mt.mjs, since 2026-11-25: its source blocks made on the module's threads behind the
// first frames): its header in the light, a fountain per source block, and the block ids chosen by the transfer's
// schedule, so the page's baseId only counts blocks. It is made when the page's file is picked ("prepare"), and a start
// names it by its pick; a start with bytes makes one there, as before.
// ai: codes: symbols a frame, 1 or 2 (the page's "codes"). Two are painted side by side in one buffer, a gap of
// ai: gapModules modules between them (the page's slider since 2026-10-03, sent with every start and respec;
// ai: 2026-09-30: widened by about 2 modules, then 2 more: 4; then 12, as the 8
// ai: modules of white between the rims could not be seen; packed to their quiet zones from 2026-09-28, since with two
// ai: codes the whole performance rests on packing), the first with the frame's first blocksPerFrame ids and the second
// ai: with the next, one format and one word for both.
// ai: Under the page's GPU encoder (gpu/encoder.mjs, since 2026-09-29) frames are asked for a batch at a time
// ai: ("frames": count frames from seq) and come back as their blocks alone, one buffer, frame after frame, each codes x
// ai: blocksPerFrame x 472 bytes; the page encodes them. The test stream's come as their ids alone (test, 2026-11-15):
// ai: the encoder makes each payload from its id on the device (BLAKE3, liblizard/src/xfer.h stream_fill), so nothing here hashes.
// ai: Refresh (the C's rule: ob.mjs Refresh over core/tx/refresh.h; the page's "refresh", sent with every start too): a held block repeats the
// ai: id at its place in the frame before, so only the new places take ids (the transfer's schedule, or the test
// ai: stream's from baseId); the frame before is forgotten at start, an accepted respec and stop.
import createMt from "../liblizard/bindings/wasm/dist/lizard-mt.mjs";
import { init as initOb, Lizard, frameBlocks, isControlId, ID_BYTES, BLOCK, idPut, idGet, toiGet, hex, Refresh } from "../liblizard/sim/ob.mjs";

// ai: fc: the paint's codec (the C's paint on the plain module, sim/ob.mjs Lizard), made at each start and respec; buf
// ai: its frame's blocks
let fc = null, buf = null, xs = null, codes = 1, chain = Promise.resolve();
// ai: the code's colours the paint takes (the page's Colours, 2026-11-16: "colours", and start's), 0xRRGGBB, kept for each
// ai: codec made (a start, a respec)
let colours = [0x000000, 0xffffff];
// ai: the sender's clock the word states, [fps, the largest Refresh delay] (src/fmt.h): the page's, at Start and as it moves
let clock = [60, 1];
// ai: The transfer made at a pick's first Start ("prepare", sent just before "start"; at the pick from 2026-10-19 to
// ai: 2026-11-01): the file packed where that pays (core/tx/pack.h, the default client's compression) and its root hashed. made: { pick, xs } while no run sends it; a run takes
// ai: it (xsPick its pick) and a stop gives it back, so the next Start sends the same transfer on, with nothing made again.
let made = null, xsPick = null;
// ai: the test stream's TOI, drawn at each start and kept over a re-pick: the format has no test stream (TOI 0 is none),
// ai: so it goes under a TOI as any send's (src/xfer.h; the native senders' rule)
let testToi = 1;
// ai: 48 bits (the TOI's), 0 none stepped over
const drawToi = () => { const r = crypto.getRandomValues(new Uint32Array(2)); return ((r[0] & 0xffffff) * 2 ** 24 + (r[1] & 0xffffff)) || 1; };
const refresh = new Refresh();
// ai: the test stream's n new ids from the page's base (a frame's own range: baseId to baseId + its blocks)
const run = (base, n) => Float64Array.from({ length: n }, (_, i) => base + i);   // ai: ids are 40 bits (src/xfer.h)
// ai: a frame's new data blocks: its new ids but the transfer's header (every test stream block is data)
const dataOf = (ids, fresh) => { let n = 0; for (let t = 0; t < ids.length; t++) if (fresh[t] && (!xs || !isControlId(ids[t]))) n++; return n; };
function dropRun() {
  if (xs) { if (xsPick !== null && !made) made = { pick: xsPick, xs }; else xs.free(); }
  xs = null; xsPick = null;
}
// ai: The threaded module, made at the first file (one heap for the transfer; the file never in it: read where it lies,
// ai: a source block a thread), told the cores its sending end takes a worker each (LIZ_CORES, core/tx/xfer_tx.cpp
// ai: countCores: a page is told only its hardware threads, two a core on x86, one on ARM); its OPFS mounted where the
// ai: origin keeps files (the packed stream and the spill there), else memory; its heap's views made again after a
// ai: growth on another thread
let MT = null, STORE = "";
const cores = () => { const n = navigator.hardwareConcurrency || 1; return /arm|aarch64|Android|iPhone|iPad/i.test(`${navigator.userAgentData?.platform ?? ""} ${navigator.userAgent}`) ? n : Math.max(1, n >> 1); };
const mt = async (n = cores()) => {
  if (MT) return MT;
  MT = await createMt({ preRun: [(m) => { m.ENV.LIZ_CORES = String(n); }] });
  STORE = MT._lizw_opfs() === 0 ? "/opfs" : "";
  return MT;
};
const cstr = (s) => { const n = MT.lengthBytesUTF8(s) + 1, p = MT._malloc(n); MT.stringToUTF8(s, p, n); return p; };
function heap() {
  if (MT.HEAPU8.byteLength !== MT.wasmMemory.buffer.byteLength) { const b = MT.wasmMemory.buffer; MT.HEAPU8 = new Uint8Array(b); MT.HEAPU32 = new Uint32Array(b); MT.HEAP32 = new Int32Array(b); }
  return MT.HEAPU8;
}
// ai: A picked file's transfer, the C's (liblizard/bindings/wasm/send.cpp lizw_prepare over lizard.h liz_tx_new_read), made
// ai: on a thread of the module's: the file read where it lies, a slice at a time, by its reader (blob-reader.mjs, a
// ai: worker holding the file, as the module's threads ask); packed where that pays (core/tx/pack.h, the default client's: the stream in a folder of its own under
// ai: /opfs/lizard-send); its chaining values hashed and its header made, then its source blocks made by the module's
// ai: threads, a worker a core, their symbols spilled to the same folder (no encoder kept); until every one is made a
// ai: frame carries the header alone. prepare() polls it, a cancel (stopped) stopping the pack at its next round of
// ai: frames and the hash at its next source block. frameIds(n) takes the next n blocks of its schedule and gives their
// ai: ids; block and toiFor answer from the blocks taken (kept while a frame of the run may still name them: keep). made
// ai: settles once every source block is (the bytes it carries and its cycle then said to the page). The reader lives
// ai: as long as the transfer (the spill's segments are made again from the file).
class Tx {
  static async prepare(blob, { name, sbLog2, cores: n }, stopped = () => false) {
    const M = await mt(n || undefined), t = new Tx();
    const ps = [name, STORE].map(cstr);
    const prep = M._lizw_prepare(blob.size, ps[0], sbLog2 ?? -1, 1, ps[1]);
    ps.forEach((p) => M._free(p));
    const reader = new Worker(new URL("./blob-reader.mjs", import.meta.url), { type: "module" });
    reader.postMessage({ memory: M.wasmMemory, slot: M._lizw_prep_slot(prep), blob });
    for (let i = 0; !M._lizw_prep_state(prep); i++) { if (stopped()) M._lizw_prep_cancel(prep); await new Promise((r) => setTimeout(r, Math.min(20, 1 + i))); }
    if (M._lizw_prep_state(prep) < 0) {
      const why = M.UTF8ToString(M._lizw_prep_why(prep));
      M._lizw_prep_free(prep); reader.terminate();
      throw why === "cancelled" ? CANCELLED : new Error(why);
    }
    t.prep = prep; t.reader = reader; t.h = M._lizw_prep_tx(prep);
    t.length = M._lizw_prep_sent(prep); t.name = M.UTF8ToString(M._lizw_prep_name(prep)); t.compression = M._lizw_prep_compression(prep);
    t.blocks = new Map(); t.buf = 0; t.cap = 0;
    t.made = (async () => { for (let i = 0; !t.info().cycle; i++) { if (!t.h) throw new Error("let go"); await new Promise((r) => setTimeout(r, Math.min(50, 2 + i))); } return t.info(); })();
    t.made.catch(() => {});
    return t;
  }
  // ai: liz_tx_info (64 bytes): the bytes it carries, its source blocks and their size, the cycle (0 until every
  // ai: source block is made), the root, the seed
  info() {
    const p = MT._malloc(64);
    MT._liz_tx_info_get(this.h, p);
    const H = heap(), dv = new DataView(H.buffer, p, 64);
    const i = { length: Number(dv.getBigUint64(8, true)), sourceBlocks: dv.getUint32(16, true), cycle: dv.getUint32(20, true), root: hex(H.slice(p + 24, p + 56)), seed: dv.getUint32(56, true), sbLog2: dv.getUint32(60, true) };
    MT._free(p);
    return i;
  }
  // ai: the schedule drawn from seed (liz_tx_seed: before the first frame)
  reseed(seed) { if (MT._liz_tx_seed(this.h, seed >>> 0) < 0) throw new Error(MT.UTF8ToString(MT._liz_last_error())); }
  frameIds(n) {
    if (n > this.cap) { if (this.buf) MT._free(this.buf); this.buf = MT._malloc(n * BLOCK); this.cap = n; }
    if (MT._liz_tx_next(this.h, n, this.buf) < 0) throw new Error(MT.UTF8ToString(MT._liz_last_error()));
    const b = heap().slice(this.buf, this.buf + n * BLOCK), ids = new Float64Array(n);
    for (let k = 0; k < n; k++) { ids[k] = idGet(b, k * BLOCK); this.blocks.set(ids[k], b.subarray(k * BLOCK, (k + 1) * BLOCK)); }
    return ids;
  }
  block(id, out) { out.set(this.blocks.get(id).subarray(ID_BYTES)); return out; }
  toiFor(id) { return toiGet(this.blocks.get(id), 0); }
  // ai: the blocks taken let go but those of ids (the last frame's, which the next may hold)
  keep(ids) { const want = new Set(ids); for (const id of this.blocks.keys()) if (!want.has(id)) this.blocks.delete(id); }
  // ai: the transfer let go (its threads joined, the reader serving them meanwhile), then its reader
  free() { if (!this.h) return; MT._lizw_prep_free(this.prep); this.reader.terminate(); if (this.buf) MT._free(this.buf); this.h = 0; this.blocks.clear(); }
}
// ai: a transfer made whole behind its first frames: the bytes it carries and its cycle then said to the page ("made",
// ai: by pick), or the failure; nothing for one let go first
function whenMade(x, said) {
  x.made.then((i) => postMessage({ type: "made", ...said, sent: x.length, cycle: i.cycle, heap: heap().length }), (e) => { if (x.h) postMessage({ type: "made", ...said, error: String(e?.message ?? e) }); });
}
const CANCELLED = new Error("cancelled");
// ai: the newest preparation the page let go ("cancel", taken as it comes, outside the chain; its number the page's)
let cancelledUpTo = 0;
// ai: the gap between two codes, in modules (the page's slider, start and respec's gap), and in the paint's pixels for
// ai: this codec (fc.cell samples a module)
let gapModules = 0;
const gapOf = (p) => gapModules * (p.cell || 0);
// ai: The codec for the page's spec (send.mjs spec(): n, subch, span, bitmap); taken as the paint's with its blocks'
// ai: buffer (use); the label #tx shows
async function codecOf(s) {
  await initOb();
  return new Lizard(s.n, s.subch, 1, { span: s.span, bitmap: s.bitmap });
}
const use = (c) => { fc = c; buf = new Uint8Array(c.blocks * BLOCK); fc.setClock(...clock); };
const labelOf = (s) => `lizard ${s.n} ${s.subch}sub${s.span ? " span" + s.span : ""}`;
// ai: A frame's blocks for its ids (sim/ob.mjs frameBlocks): a file's from its transfer, each under its own TOI (Tx
// ai: toiFor), else the test stream's under the drawn TOI; a view reused by the next call
const blocksOf = (ids) => frameBlocks(ids, fc.blocks, xs ? { toi: (id) => xs.toiFor(id), payload: (id, out) => xs.block(id, out) } : { toi: testToi }, buf);
// ai: A frame as the pixels the page paints (src/wasm.c lizard_tx_rgba, in the code's colours), its pilots' count the
// ai: painted count mod 4 (SPEC 7.3): { w, rgba }, a view into the wasm heap until the next paint
function paint(seq, ids) {
  fc.setParity(seq & 3);
  return fc.encodeRGBA(blocksOf(ids), colours[0] & 0xffffff, colours[1] & 0xffffff);
}

async function handle(m) {
  if (m.type === "prepare") {
    // ai: its own reply ("prepared", by pick and the page's number for it, outside the runs' gen), a failure in it or
    // ai: its cancel too
    made?.xs.free(); made = null;
    const stopped = () => m.prep <= cancelledUpTo;
    try {
      const x = await Tx.prepare(m.blob, m.xfer, stopped);
      if (stopped()) { x.free(); postMessage({ type: "prepared", pick: m.pick, prep: m.prep, cancelled: true }); return; }
      made = { pick: m.pick, xs: x };
      postMessage({ type: "prepared", pick: m.pick, prep: m.prep, sent: x.length });
      whenMade(x, { pick: m.pick });
    } catch (e) {
      postMessage({ type: "prepared", pick: m.pick, prep: m.prep, ...(stopped() ? { cancelled: true } : { error: String(e?.message ?? e) }) });
    }
  } else if (m.type === "start") {
    fc?.free(); fc = null; dropRun();
    testToi = drawToi();
    if (m.colours) colours = m.colours;
    if (m.clock) clock = m.clock;
    use(await codecOf(m.spec));
    if (m.pick != null) {
      if (made?.pick !== m.pick) throw new Error("the file's transfer was not made: choose it again");
      xs = made.xs; xsPick = m.pick; made = null;
    }
    // ai: Developer Tools' Seed (2026-11-14): the schedule drawn from it before the first frame (a transfer made at the pick
    // ai: drew a random one); none given, the random one stands
    if (xs && m.seed != null) xs.reseed(m.seed);
    codes = m.codes ?? 1; gapModules = m.gap;
    if (m.every) refresh.setEvery(m.every);
    refresh.setFormat(m.spec.subch, fc.blocks);
    postMessage({ type: "ready", gen: m.gen, label: labelOf(m.spec), usefulBytes: fc.blockBytes - ID_BYTES, blocksPerFrame: fc.blocks, xfer: xs?.info() });
  } else if (m.type === "respec") {
    // A re-pick (the room changed): a new code for the same transfer. Refused if the block size would move, which
    // would break the fountain's stream, exactly as the page used to refuse it.
    const next = await codecOf(m.spec);
    if (next.blockBytes !== fc.blockBytes) { next.free(); postMessage({ type: "respec", gen: m.gen, ok: false }); return; }
    fc.free(); use(next); codes = m.codes ?? codes; gapModules = m.gap;
    refresh.setFormat(m.spec.subch, fc.blocks);
    postMessage({ type: "respec", gen: m.gen, ok: true, label: labelOf(m.spec), blocksPerFrame: fc.blocks });
  } else if (m.type === "colours") {
    colours = [m.ink, m.paper];
  } else if (m.type === "refresh") {
    refresh.setEvery(m.every);
  } else if (m.type === "clock") {
    clock = m.clock; fc?.setClock(...clock);
  } else if (m.type === "stop") {
    fc?.free(); fc = null; dropRun(); refresh.reset();
  } else if (m.type === "frame") {
    if (!fc) return;
    const t = performance.now(), V = fc.blocks;
    const { ids, fresh } = refresh.next(m.seq, V * codes, (n) => (xs ? xs.frameIds(n) : run(m.baseId, n)));
    // ai: Each symbol's pixels are a view into the wasm heap, good until the next paint, so each is copied row by row
    // ai: into its place (side k of codes) before the next is painted. The page hands its last buffer back to be filled.
    // ai: Two codes sit a gap of gapModules apart (since 2026-09-30; packed to their quiet zones before), the gap the
    // ai: colour of the margin beside it; the frame is
    // ai: codes w + (codes - 1) gap px wide and the message says the gap.
    let out = null, w = 0, gap = 0, fw = 0;
    for (let k = 0; k < codes; k++) {
      const part = paint(m.seq, ids.subarray(k * V, (k + 1) * V)), row = 4 * part.w;
      if (!out) {
        w = part.w; gap = codes > 1 ? gapOf(fc) : 0; fw = codes * w + (codes - 1) * gap;
        const bytes = 4 * fw * w;
        out = m.buf && m.buf.byteLength === bytes ? new Uint8Array(m.buf) : new Uint8Array(bytes);
      }
      for (let y = 0; y < w; y++) {
        const at = 4 * (y * fw + k * (w + gap));
        out.set(part.rgba.subarray(y * row, (y + 1) * row), at);
        if (k + 1 < codes) for (let x = 0; x < gap; x++) out.set(part.rgba.subarray((y + 1) * row - 4, (y + 1) * row), at + row + 4 * x);
      }
    }
    // ai: data: the frame's new data blocks, its header and its repeats not counted; the page
    // ai: counts cycles with it (xfer.cycle data blocks a cycle). Every new block of the test stream is data.
    const data = dataOf(ids, fresh);
    xs?.keep(ids);
    postMessage({ type: "frame", gen: m.gen, seq: m.seq, w, codes, gap, buf: out.buffer, ms: performance.now() - t, data }, [out.buffer]);
  } else if (m.type === "frames") {
    if (!fc) return;
    const t = performance.now(), V = fc.blocks, B = fc.blockBytes, per = codes * V * B, bytes = m.count * per;
    const out = m.buf && m.buf.byteLength === bytes ? new Uint8Array(m.buf) : new Uint8Array(bytes), T = V * codes;
    // ai: the batch's frames in order, each holding against the one before (the batch before's last for the first)
    const all = Array.from({ length: m.count }, (_, f) => refresh.next(m.seq + f, T, (n) => (xs ? xs.frameIds(n) : run(m.baseId + f * T, n))));
    const data = all.map(({ ids, fresh }) => dataOf(ids, fresh)), test = !xs;
    for (let f = 0; f < m.count; f++) {
      if (test) { for (let t = 0; t < T; t++) idPut(out, f * per + t * B, all[f].ids[t], testToi); continue; }
      for (let k = 0; k < codes; k++) out.set(blocksOf(all[f].ids.subarray(k * V, (k + 1) * V)), f * per + k * V * B);
    }
    if (xs && m.count) xs.keep(all[m.count - 1].ids);
    postMessage({ type: "frames", gen: m.gen, seq: m.seq, count: m.count, codes, test, buf: out.buffer, ms: performance.now() - t, data }, [out.buffer]);
  }
}

onmessage = ({ data: m }) => {
  if (m.type === "cancel") { cancelledUpTo = Math.max(cancelledUpTo, m.prep); return; }
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", gen: m.gen, message: String(e?.message ?? e) }));
};
