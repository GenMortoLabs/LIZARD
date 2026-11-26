// ai: The receiver's engine (2026-11-25): one worker holding the threaded module (liblizard/bindings/wasm/dist/lizard-mt.mjs,
// ai: its engine liblizard/bindings/wasm/engine.cpp), in place of a pool of workers each a module and heap of its own and
// ai: a fountain worker beside them. The decoders are the module's threads over its one shared heap (core/cpu/pool.h:
// ai: one at the start, one more while frames are lost to busy ones, one parked after ten slack seconds); each frame's
// ai: verified blocks go from its decode thread into the transfer (core/rx/xfer_rx.h: the judge, the dedupe and the
// ai: fountain on threads of their own), by pointer; the store core/rx/store.h's files under /opfs (WasmFS's OPFS
// ai: backend: lizard-xfer/<id>/, the layout the fountain worker kept, so a transfer it parked resumes here), or memory
// ai: where the origin keeps no files (a private window) or the page asks (?store=memory). A finished file is filed
// ai: into the received files by the C (lizard-files/<id>/data and meta.json, a packed one unpacked: core/tx/pack.h).
// ai: The page keeps its frames' path (a VideoFrame transferred here, its crop copied into the shared heap by copyTo,
// ai: vframes.mjs lumaOf), its tracker and its stats row; this worker answers each frame.
// ai:   page to engine: { type: "config", store, workers, ceiling } (once); { type: "frame", tag, frame, x, y, w, h, axis,
// ai:     verify } (a VideoFrame and its crop in display pixels), { type: "frame", tag, rgba | luma, w, h, axis, verify }
// ai:     (a canvas grab's RGBA, a file's luma; transferred); { type: "lost", repeatShare } (a frame lost to busy
// ai:     decoders); { type: "tick", busySkips } (a second); { type: "release" } (the camera stopped for a file in:
// ai:     the finished transfers' TOIs let go, the same file at a later Start received again)
// ai:   engine to page: { type: "ready", size, ceiling, store, storeWhy, frames }; { type: "result", tag, found, seen,
// ai:     bad, test, judged, fresh, ms, quad, fmt, blocks, size, heap, levels, probe, vf } a frame decoded; { type: "busy",
// ai:     tag } a frame no decoder took (lost) or { type: "busy", tag, noFrames } one whose VideoFrame could not be read;
// ai:     { type: "xfer", ...the transfer's progress (engine.cpp lizw_rx_progress), mem }; { type: "file", name, root,
// ai:     length, sent, compression, lib | bytes } once a file is in; { type: "error", message }
// ai: No top-level await (a module worker drops messages meanwhile).
import createModule from "../liblizard/bindings/wasm/dist/lizard-mt.mjs";
import { lumaOf, cornerProbe, greyLevels } from "./vframes.mjs";

let M = null, rx = 0, RES = 0, resBuf = 0, store = "", storeWhy = "", seq = 0, levelTick = 0, lastXfer = 0, chain = Promise.resolve();
const frames = new Map(), filed = new Set(), LIB = "/opfs/lizard-files";
// ai: the heap's views made again where another thread grew it (a shared memory's buffer keeps the length it had when
// ai: its views were made)
function heap() {
  if (M.HEAPU8.byteLength !== M.wasmMemory.buffer.byteLength) {
    const b = M.wasmMemory.buffer;
    M.HEAPU8 = new Uint8Array(b); M.HEAP32 = new Int32Array(b); M.HEAPU32 = new Uint32Array(b); M.HEAPF32 = new Float32Array(b); M.HEAPF64 = new Float64Array(b);
  }
  return M.HEAPU8;
}
const cstr = (s) => { const n = M.lengthBytesUTF8(s) + 1, p = M._malloc(n); M.stringToUTF8(s, p, n); return p; };
const withStr = (strs, f) => { const ps = strs.map(cstr); try { return f(...ps); } finally { ps.forEach((p) => M._free(p)); } };

async function config(c) {
  M = await createModule();
  RES = M._lizw_rx_res(); resBuf = M._malloc(64 * RES * 8);
  if (c.store === "memory") storeWhy = "asked for (?store=memory)";
  else if (M._lizw_opfs() === 0) store = "/opfs";
  else storeWhy = "no OPFS (a private window, or a browser without sync access handles)";
  rx = withStr([store], (p) => M._lizw_rx_new(p, c.workers ?? 0, c.ceiling ?? 0));
  if (!rx) throw new Error("the receiver's engine was not made");
  globalThis.lizwWake = drain;
  postMessage({ type: "ready", size: M._lizw_rx_size(rx), ceiling: M._lizw_rx_ceiling(rx), store: store ? "opfs" : "memory", storeWhy, frames: typeof VideoFrame === "function" });
}

// ai: A frame into the shared heap and offered to the decoders. The crop's luma: a VideoFrame's copied by copyTo straight
// ai: into the engine's buffer (vframes.mjs lumaOf), a grab's RGBA turned to luma by the C (the codec's formula), a
// ai: file's luma copied in. The page sends a frame only where a decoder is idle; one taken by none comes back "busy".
async function frame(m) {
  const id = ++seq, { w, h } = m;
  let off = 0, stride = w, vf = "", buf = null;
  const into = (n) => { const p = M._lizw_rx_luma(rx, n); return new Uint8Array(heap().buffer, p, n); };
  if (m.frame) {
    try { ({ off, stride, format: vf, buf } = await lumaOf(m.frame, m.x, m.y, w, h, into)); }
    catch (e) { return postMessage({ type: "busy", tag: m.tag, noFrames: String(e?.message ?? e) }); }
  } else if (m.rgba) {
    // ai: the RGBA after the luma in the engine's buffer, turned to luma in place before it
    buf = into(5 * w * h);
    buf.set(new Uint8Array(m.rgba), w * h);
    M._lizw_luma_of_rgba(buf.byteOffset + w * h, w * h, buf.byteOffset);
  } else {
    buf = into(w * h);
    buf.set(new Uint8Array(m.luma));
  }
  // ai: the grey range every 32nd frame (the page's Washed out), the crop's corners under ?verify (check_rates)
  const levels = (levelTick++ & 31) === 0 ? greyLevels(buf, off, stride, w, h) : null;
  const probe = m.verify ? cornerProbe(buf, off, stride, w, h) : null;
  const [ax, ay] = m.axis ?? [NaN, NaN];
  if (!M._lizw_rx_offer(rx, buf.byteOffset + off, stride, 0, 0, w, h, id, ax, ay)) return postMessage({ type: "busy", tag: m.tag });
  frames.set(id, { tag: m.tag, levels, probe, vf });
}

// ai: The decoders' answers, as the module wakes this thread (engine.cpp lizwWake): one result a frame; the transfer's
// ai: progress at most every 250 ms and with every file in
function drain() {
  if (!rx) return;
  for (;;) {
    const n = M._lizw_rx_results(rx, resBuf, 64);
    if (!n) break;
    const F = heap() && M.HEAPF64, size = M._lizw_rx_size(rx);
    for (let i = 0; i < n; i++) {
      const r = F.subarray((resBuf >> 3) + i * RES, (resBuf >> 3) + (i + 1) * RES), id = r[0], f = frames.get(id);
      frames.delete(id);
      if (!f) continue;
      const out = { type: "result", tag: f.tag, ms: r[1], found: r[2], quad: r[2] ? Array.from(r.subarray(13, 21)) : null, seen: r[22], bad: r[23], judged: r[24], fresh: r[25], test: r[26], blocks: r[27], size, heap: M.HEAPU8.length };
      // ai: the word the frame read (src/fmt.h): its version, counts and the sender's clock, and the ring it was read in
      if (r[5]) out.fmt = { version: r[6], count: [r[7], r[8], r[9], r[10]], fps: r[11], delay: r[12], ring: r[3] };
      if (f.levels) out.levels = f.levels;
      if (f.probe) out.probe = f.probe;
      if (f.vf) out.vf = f.vf;
      postMessage(out);
    }
  }
  xfer(false);
}

// ai: The transfer's progress for the page (its state line, its meter, the stats row's xfer), each source block's share
// ai: for a file of 256 or fewer; then every file in not yet offered, filed and offered
function xfer(force) {
  const now = performance.now();
  if (!force && now - lastXfer < 250) return;
  lastXfer = now;
  const p = JSON.parse(M.UTF8ToString(M._lizw_rx_progress(rx, 1)));
  if (p.per?.length > 256) delete p.per;
  postMessage({ ...p, type: "xfer", storeWhy, mem: { codecHeap: heap().length, storeBytes: 0 } });
  for (const f of p.files) {
    const key = `${f.toi}.${f.root}`;
    if (!f.done || filed.has(key)) continue;
    filed.add(key);
    offer(f);
  }
}
setInterval(() => { if (rx) xfer(false); }, 500);

// ai: A finished file offered to the page: on OPFS filed by the C into the received files (lizard-files/<id>/, Home's
// ai: list told), else its bytes from memory, a packed one unpacked first (the C's, engine.cpp lizw_unpack)
function offer(f) {
  const base = { type: "file", ok: true, name: f.name, root: f.root, length: f.length, sent: f.length, compression: f.compression };
  if (store) {
    const r = JSON.parse(M.UTF8ToString(withStr([f.path, f.name, f.root, LIB], (pp, np, rp, lp) => M._lizw_file_away(pp, np, rp, f.compression, lp, Date.now()))));
    if (r.error) return postMessage({ type: "error", message: r.error });
    if (typeof BroadcastChannel === "function") { const bc = new BroadcastChannel("lizard-files"); bc.postMessage("changed"); bc.close(); }
    return postMessage({ ...base, name: r.name, length: r.size, lib: { ...r, root: f.root } });
  }
  const lp = M._malloc(8);
  try {
    let p = M._lizw_rx_data(rx, lp), n = (heap(), M.HEAPF64[lp >> 3]), name = f.name, own = false;
    if (!p) return postMessage({ type: "error", message: `${f.name}: the file's bytes are gone` });
    if (f.compression === 1) {
      const q = M._lizw_unpack(p, n, lp);
      if (q) { p = q; n = (heap(), M.HEAPF64[lp >> 3]); own = true; if (name.endsWith(".zst")) name = name.slice(0, -4); }
    }
    const bytes = heap().slice(p, p + n);
    if (own) M._free(p);
    postMessage({ ...base, name, length: bytes.length, bytes: bytes.buffer }, [bytes.buffer]);
  } finally { M._free(lp); }
}

async function handle(m) {
  if (m.type === "config") return config(m);
  if (!rx) return m.frame?.close();
  if (m.type === "frame") return frame(m);
}
onmessage = ({ data: m }) => {
  // ai: what does not wait on a frame's copy: the pool's policy and the transfer's own calls
  if (rx && m.type === "lost") return M._lizw_rx_lost(rx, m.repeatShare);
  if (rx && m.type === "tick") return M._lizw_rx_tick(rx, m.busySkips);
  if (rx && m.type === "release") { M._lizw_rx_release(rx); return xfer(true); }
  chain = chain.then(() => handle(m)).catch((e) => { m.frame?.close(); postMessage({ type: "error", tag: m.tag, message: String(e?.message ?? e) }); });
};
// ai: Home's Delete of an unfinished transfer this engine holds open (library.mjs forgetUnfinished): forgotten, its folder
// ai: removed, then Home told
if (typeof BroadcastChannel === "function") new BroadcastChannel("lizard-xfer").onmessage = ({ data }) => {
  if (!rx || typeof data?.forget !== "string") return;
  withStr([data.forget], (p) => M._lizw_rx_forget(rx, p));
  const bc = new BroadcastChannel("lizard-files"); bc.postMessage("changed"); bc.close();
  xfer(true);
};
