// ai: The sender's transfer made by a pool of workers (2026-11-06; the native sender's prepare across cores,
// ai: liblizard/core/tx/xfer_tx.cpp). No source block reads another, so each worker (source-block-worker.mjs) takes the next source block as it
// ai: comes free and makes it whole (its fountain, its chaining value); the encoders' solves run in parallel too,
// ai: as each worker keeps the fountains it made (a wasm heap's state cannot move to another), and so the fountains
// ai: spread over the workers' heaps (one heap holds 2 GiB at most). A frame's blocks are fetched from the workers that
// ai: hold them before it is painted (send-worker.mjs). The ids and bytes are the serial sender's (sim/xfer.mjs).
// ai: Streaming (2026-11-06, the native sender's): the workers hash every source block first (the chaining values and the root,
// ai: all the header needs) and makeXfer returns; the source blocks are made behind it while frames go, each handed to the
// ai: sender as it comes (XferSender.add), remote.made settling with the last.
import { XferSender, PAYLOAD, CONTROL, idSbn } from "../liblizard/sim/xfer.mjs";

// ai: a worker's answers taken in the order they were asked (it handles one message at a time); end() refuses what is
// ai: still asked of a worker let go
function link(w) {
  const waiting = [];
  const fail = (why) => { for (const p of waiting.splice(0)) p.reject(new Error(why)); };
  w.onmessage = ({ data }) => { const p = waiting.shift(); if (data?.error) p.reject(new Error(data.error)); else p.resolve(data); };
  w.onerror = (e) => { e.preventDefault(); fail(e.message || "a chunk worker failed"); };
  const call = (msg, transfer = []) => new Promise((resolve, reject) => { waiting.push({ resolve, reject }); w.postMessage(msg, transfer); });
  call.end = () => fail("let go");
  return call;
}

// ai: The workers: one a core, not a hardware thread (the native sender's rule, liblizard/core/tx/xfer_tx.cpp workers()),
// ai: found by timing (2026-11-06). A page is told only hardwareConcurrency, its hardware threads, and nothing says
// ai: whether two of them share a core; so the file's first MiB is made as a source block by half the threads' workers at once,
// ai: then by all of them at once, and all are kept only where doubling them made at least 1.5 times as much: a
// ai: second thread on a core gives about 1 to 1.3 (on a 16-core 32-thread x86 in Chrome, 32 workers made 256 MiB no
// ai: faster than 16, at twice the memory, about 30 MiB a worker), a core of its own about 2. A round first warms every
// ai: worker (its codec loaded, its heap grown). A MiB, not a whole source block: still past a core's cache, at a quarter of
// ai: the cost (whole source blocks took 1.16 s for 256 MiB that then took 0.57 s). Found once a page (the send worker's
// ai: module lives as long), and only where it decides anything: more source blocks than half the threads. found: the
// ai: workers kept, null until timed.
let found = null;
async function timeCores(calls, sample, sbLog2) {
  const n = calls.length, half = n >> 1;
  const round = async (k) => {
    const t = performance.now();
    await Promise.all(calls.slice(0, k).map((call) => call({ type: "trial", buf: sample, sbLog2 })));
    return performance.now() - t;
  };
  await round(n);
  const a = half / await round(half), b = n / await round(n);
  return b >= 1.5 * a ? n : half;
}

// ai: The file's transfer (XferSender's options in opts; the source block size src/xfer.h xfer_sb_log2_for's unless given),
// ai: made by up to `workers` workers (the cores found unless given), one a source block at most; a file of one source block, or one
// ai: worker, made here as before. `stopped` (2026-11-16) is asked before each source block's hash: true ends the pool and
// ai: throws "cancelled" (the source blocks made behind the return are the transfer's, never stopped so).
export async function makeXfer(bytes, opts, { M, Encoder, workers = null, stopped = () => false }) {
  const sbLog2 = opts.sbLog2 ?? M._xfer_sb_log2_of(bytes.length), size = 2 ** sbLog2, C = Math.ceil(bytes.length / size);
  if (sbLog2 < 0) throw new Error(`${bytes.length} B is more than the transfer carries (65,535 chunks of 16 MiB)`);
  opts = { ...opts, sbLog2 };
  const n = navigator.hardwareConcurrency || 1, timed = workers == null && found == null && n >= 4 && C > n >> 1;
  let P = Math.min(C, workers ?? found ?? n);
  if (P < 2) return new XferSender(bytes, { ...opts, M, Encoder });
  const ws = Array.from({ length: timed ? n : P }, () => new Worker(new URL("./source-block-worker.mjs", import.meta.url), { type: "module" }));
  const calls = ws.map(link), owner = new Int32Array(C);
  const free = () => { ws.forEach((w) => w.terminate()); calls.forEach((call) => call.end()); };
  try {
    const part = (c) => bytes.slice(c * size, Math.min(bytes.length, (c + 1) * size));
    // ai: each worker the next source block as it comes free (the work of a pass), the pass done when every one is
    const pass = (job, stop = () => false) => {
      let next = 0;
      return Promise.all(calls.map(async (call, i) => { for (let c; (c = next++) < C;) { if (stop()) throw new Error("cancelled"); await job(call, i, c); } }));
    };
    const cvs = new Uint8Array(32 * C);
    await pass(async (call, i, c) => { const plain = part(c); cvs.set((await call({ type: "cv", c, sbLog2, buf: plain.buffer }, [plain.buffer])).cv, 32 * c); }, stopped);
    let xs = null;
    const remote = {
      // ai: the blocks of the ids whose source blocks the workers hold (every source block's fountain), each worker asked once for its own
      async fetch(ids) {
        const per = calls.map(() => []), got = new Map();
        for (const id of ids) { const c = idSbn(id); if (c !== CONTROL) per[owner[c]].push(id); }   // ai: the header the sender's own
        await Promise.all(per.map(async (list, i) => {
          if (!list.length) return;
          const b = new Uint8Array((await calls[i]({ type: "blocks", ids: Float64Array.from(list) })).buf);
          list.forEach((id, k) => got.set(id, b.subarray(k * PAYLOAD, (k + 1) * PAYLOAD)));
        }));
        return got;
      },
      free,
      made: null,   // ai: settles once every source block is in xs (or a worker failed)
    };
    xs = new XferSender(null, { ...opts, M, Encoder, cvs, length: bytes.length, remote });
    // ai: the cores timed after the header (every thread's worker hashed: a short pass), the extra workers let go before
    // ai: any source block is made, so the first frame waits on the hash alone
    const cores = async () => {
      if (!timed) return;
      found = await timeCores(calls, bytes.slice(0, Math.min(size, 2 ** 20)), sbLog2);
      P = Math.min(C, found);
      ws.splice(P).forEach((w) => w.terminate());
      calls.splice(P);
    };
    remote.made = cores().then(() => pass(async (call, i, c) => {
      const plain = part(c), k = await call({ type: "make", c, sbLog2, buf: plain.buffer }, [plain.buffer]);
      owner[c] = i;
      xs.add(c, k);
    }));
    remote.made.catch(() => {});   // ai: the caller's to read; a transfer let go before it is whole ends it so
    return xs;
  } catch (e) { free(); throw e; }
}
