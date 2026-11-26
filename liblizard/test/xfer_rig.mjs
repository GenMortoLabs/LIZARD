// The rig's chunked transfer without a camera, through the C (lizard.h liz_tx and liz_rx over the plain wasm module:
// core/tx/xfer_tx.cpp and core/rx/xfer_rx_core.cpp as emscripten builds them; the JS transfer, sim/xfer.mjs, until
// 2026-11-25): the sender's carousel, its blocks dropped the ways a camera drops them, and the receiver putting the file
// back from what is left, in memory and in a folder store (core/rx/store.h on the module's file system), held to the
// file and to its BLAKE3. Files of no bytes, one block, one source block, several, 41 small source blocks and 10 MB;
// loss at random, by ring, in bursts, a late join, a corrupted block that passes its CRC (the file refused whole at
// its root, or a source block of one refused at its solve); then a folder's transfer resumed by the next receiver, a
// sender started again (its seed the same, or another), two senders of one file pooled, and a finished file's
// release. Laps is the airtime from the join over one cycle of the file's blocks. About 10 s.
//   node test/xfer_rig.mjs
import { init as initOb, Sender, blake3, hex, PAYLOAD, BLOCK, ID_BYTES, isControlId, idSbn, idEsi, idGet } from "../sim/ob.mjs";

const M = await initOb();
let fails = 0;
const say = (ok, line) => { if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} ${line}`); };
const rand = (n, seed) => { const a = new Uint8Array(n); let s = seed >>> 0 || 1; for (let i = 0; i < n; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a[i] = s >>> 16; } return a; };
const cstr = (s) => { const n = M.lengthBytesUTF8(s) + 1, p = M._malloc(n); M.stringToUTF8(s, p, n); return p; };

// ai: a receiver: dir null in memory, else a folder store under dir (liz_store); frame(blocks) one frame's 472-byte
// ai: blocks; progress (liz_progress, 80 bytes); data the file (memory) once done
let folders = 0;
function receiver(dir = null) {
  const sp = M._malloc(16);
  M.HEAPU32.fill(0, sp >> 2, (sp >> 2) + 4);
  if (dir) M.HEAPU32[sp >> 2] = cstr(dir);
  const h = M._liz_rx_new(sp), buf = M._malloc(64 * BLOCK), vp = M._malloc(20), pp = M._malloc(80);
  if (!h) throw new Error(M.UTF8ToString(M._liz_last_error()));
  return {
    frame(blocks) { M.HEAPU8.set(blocks, buf); if (M._liz_rx_frame(h, buf, blocks.length / BLOCK, vp) < 0) throw new Error(M.UTF8ToString(M._liz_last_error())); },
    get progress() { M._liz_rx_progress(h, pp); const dv = new DataView(M.HEAPU8.buffer, pp, 80); return { header: dv.getInt32(0, true), done: !!dv.getInt32(4, true), verified: dv.getUint32(36, true), rejected: dv.getUint32(40, true), over: dv.getUint32(76, true), fraction: dv.getFloat64(24, true) }; },
    get root() { return M.UTF8ToString(M._liz_rx_meta(h, 1)); },
    get name() { return M.UTF8ToString(M._liz_rx_meta(h, 0)); },
    get data() { const dp = M._malloc(8); try { if (M._liz_rx_data(h, dp, dp + 4)) return null; const a = M.HEAPU32[dp >> 2], n = M.HEAPU32[(dp + 4) >> 2]; return M.HEAPU8.slice(a, a + n); } finally { M._free(dp); } },
    clear() { M._liz_rx_clear(h); },
    free() { M._liz_rx_free(h); M._free(buf); M._free(vp); M._free(pp); M._free(sp); },
  };
}
const same = (a, b) => !!a && a.length === b.length && Buffer.compare(Buffer.from(a), Buffer.from(b)) === 0;

// ai: ring: the outer share of a frame's slots never read. burst: frames lost, 48 of every 240 (2 s of 10 at 24 fps).
// ai: corrupt: the source block whose first symbol sent goes with a flipped bit (once, as symbols do not repeat). Every
// ai: case's carousel from a seed of its own, so a run repeats.
function run(label, { length, log2 = 22, B = 12, loss = 0, ring = 0, burst = 0, join = 0, corrupt = -1, maxCycles = 6, wantRejected = 0 }) {
  const file = rand(length, length + 7), root = hex(blake3(file));
  const xs = new Sender(file, { name: "f.bin", sbLog2: log2, seed: (Math.imul(length + 1, 2654435761) ^ log2) >>> 0 }), info = xs.info;
  const mem = receiver(), dir = receiver(`/rig/${++folders}`);
  let s = 12345;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const K = Array.from({ length: info.sourceBlocks }, (_, c) => xs.K(c)).reduce((a, b) => a + b, 0), cycle = Math.max(1, K), limit = join + Math.ceil((maxCycles * cycle) / B) + 50;
  let f = 0, sent = 0, csym = -1;
  for (; f < limit && !(mem.progress.done && dir.progress.done); f++) {
    const ids = xs.frameIds(B);
    if (f < join) continue;
    if (burst && f % 240 < 48) { sent += B; continue; }
    const kept = [];
    for (let k = 0; k < B; k++) {
      sent++;
      if (rnd() < loss || k >= B * (1 - ring)) continue;
      const b = new Uint8Array(BLOCK);
      b.set(xs.blocks.get(ids[k]));
      if (!isControlId(ids[k]) && idSbn(ids[k]) === corrupt) { if (csym < 0) csym = idEsi(ids[k]); if (idEsi(ids[k]) === csym) b[ID_BYTES + 5] ^= 1; }
      kept.push(b);
    }
    const all = new Uint8Array(kept.length * BLOCK);
    kept.forEach((b, i) => all.set(b, i * BLOCK));
    mem.frame(all); dir.frame(all);
  }
  const pm = mem.progress, pd = dir.progress;
  const ok = pm.done && pd.done && same(mem.data, file) && mem.root === root && dir.root === root && info.root === root && pm.rejected === wantRejected && pd.rejected === wantRejected;
  say(ok, `${label}: ${length} B, ${info.sourceBlocks} source block${info.sourceBlocks === 1 ? "" : "s"} of 2^${log2}; ${(sent / cycle).toFixed(3)} cycles, memory and folder ${pm.done && pd.done ? "whole" : "NOT whole"}, ${pm.rejected}/${pd.rejected} rejected, ${pm.over} solve${pm.over === 1 ? "" : "s"} past K`);
  xs.free(); mem.free(); dir.free();
}
run("empty", { length: 0 });
run("one block, unfountained", { length: PAYLOAD });
run("two blocks", { length: PAYLOAD + 1 });
run("small", { length: 50000 });
run("small, 30% lost", { length: 50000, loss: 0.3 });
run("3 source blocks", { length: 3 * 2 ** 20 - 12345, log2: 20 });
run("3 source blocks, the outer 25% of rings lost", { length: 3 * 2 ** 20 - 12345, log2: 20, ring: 0.25 });
run("3 source blocks, joined 200 frames late", { length: 3 * 2 ** 20 - 12345, log2: 20, join: 200 });
run("3 source blocks, 2 s lost in every 10", { length: 3 * 2 ** 20 - 12345, log2: 20, burst: 1 });
// ai: the wrong source block caught at the file's root (no manifest): the file refused whole, its 3 source blocks collected again
run("3 source blocks, source block 1 sent wrong in cycle 1, refused whole at the root", { length: 3 * 2 ** 20 - 12345, log2: 20, corrupt: 1, wantRejected: 3 });
// ai: a file of one source block sent wrong: refused at its solve, its symbols dropped and collected again
run("1 source block sent wrong in cycle 1, refused at once and collected again", { length: 2 ** 20 - 12345, log2: 20, corrupt: 0, wantRejected: 1 });
run("41 source blocks of 1 KiB", { length: 40 * 1024 + 100, log2: 10 });
run("12 source blocks, as many as a frame's blocks, outer 25% lost", { length: 12 * 2 ** 16 - 5, log2: 16, ring: 0.25 });
run("10 MB", { length: 10e6 });
run("10 MB, 40% lost", { length: 10e6, loss: 0.4 });
for (let k = 0; k < 10; k++) run(`200 KB, 20% lost, seed ${k}`, { length: 200000 + k, log2: 16, loss: 0.2 });
// ai: joined after the first cycle, so repair blocks only: RaptorQ then wants a block or two past K now and again
run("41 source blocks of 1 KiB, repair blocks only", { length: 40 * 1024 + 100, log2: 10, join: 100 });
for (let k = 0; k < 5; k++) run(`400 KB in 16 KiB source blocks, repair blocks only, seed ${k}`, { length: 400000 + 37 * k, log2: 14, join: 80 });

// ai: a sender's frames into receivers, until each is done or n frames
function feed(xs, rxs, n, B = 12, keep = () => true) {
  for (let f = 0; f < n && !rxs.every((r) => r.progress.done); f++) {
    const ids = xs.frameIds(B), kept = ids.filter((id, k) => keep(f, k)), all = new Uint8Array(kept.length * BLOCK);
    kept.forEach((id, i) => all.set(xs.blocks.get(id), i * BLOCK));
    for (const r of rxs) r.frame(all);
  }
}
{
  // ai: transfers kept across receivers (a folder store whose folders outlive its receiver, as OPFS outlives a page):
  // ai: part of a file into one receiver, which is let go, the rest into the next on the same folder; the rest alone
  // ai: cannot carry the file (each source block wants a cycle's share of its blocks)
  const file = rand(3 * 2 ** 20 - 999, 5), root = hex(blake3(file)), xs = new Sender(file, { name: "keep.bin", sbLog2: 20, seed: 11 }), cyc = xs.info.cycle;
  const a = receiver("/rig/keep");
  feed(xs, [a], Math.ceil((0.6 * cyc) / 12));
  const was = a.progress;
  a.free();
  const b = receiver("/rig/keep"), lone = receiver();
  feed(xs, [b, lone], Math.ceil((0.5 * cyc) / 12));
  say(!was.done && b.progress.done && b.root === root && !lone.progress.done, `a folder's transfer resumed by the next receiver: ${(100 * was.fraction).toFixed(0)}% in the first, whole in the second (${(100 * lone.progress.fraction).toFixed(0)}% from the rest alone)`);
  b.free(); lone.free(); xs.free();
}
{
  // ai: a sender started again (the app closed and sent the file anew), its seed the same (its ids repeat) or another
  // ai: (fresh ids): the file's TOI its content's either way, so one transfer, resumed
  for (const again of ["same", "other"]) {
    const file = rand(2 * 2 ** 20 - 7, 9), root = hex(blake3(file)), first = new Sender(file, { name: "again.bin", sbLog2: 20, seed: 21 }), cyc = first.info.cycle;
    const r = receiver();
    feed(first, [r], Math.ceil((0.55 * cyc) / 12));
    first.free();
    const second = new Sender(file, { name: "again.bin", sbLog2: 20, seed: again === "same" ? 21 : 22 });
    feed(second, [r], Math.ceil((1.5 * cyc) / 12));
    say(r.progress.done && r.root === root, `a sender started again, its seed ${again === "same" ? "the same" : "another"}: one transfer, whole`);
    second.free(); r.free();
  }
}
{
  // ai: one file from two senders (another name and seed) is one transfer: half a cycle from each, pooled
  const file = rand(3 * 2 ** 20 - 77, 13), root = hex(blake3(file)), x1 = new Sender(file, { name: "File A.bin", sbLog2: 20, seed: 31 }), x2 = new Sender(file, { name: "File A copy.bin", sbLog2: 20, seed: 32 }), cyc = x1.info.cycle;
  const r = receiver();
  feed(x1, [r], Math.ceil((0.6 * cyc) / 12));
  feed(x2, [r], Math.ceil((0.6 * cyc) / 12));
  say(r.progress.done && r.root === root && r.name === "File A.bin", `two senders of one file pooled: whole from both's blocks in 1.2 cycles, under the first description's name (${r.name})`);
  x1.free(); x2.free(); r.free();
}
{
  // ai: a finished transfer's TOIs let go (the receive's release when the camera stops for a file in, lizard.h
  // ai: liz_rx_clear here: every transfer and file let go): the same file still in the light is received again
  const file = rand(60000, 17), xs = new Sender(file, { name: "twice.bin", seed: 41 }), r = receiver();
  feed(xs, [r], 400);
  const first = r.progress.done;
  r.clear();
  feed(xs, [r], 400);
  say(first && r.progress.done && same(r.data, file), "a finished file let go, the same file in the light received again");
  xs.free(); r.free();
}
console.log(fails ? `${fails} FAILED` : "all ok");
process.exit(fails ? 1 : 0);
