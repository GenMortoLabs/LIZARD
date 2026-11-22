// The rig's chunked transfer (sim/xfer.mjs) without a camera: the sender's carousel, its blocks dropped the ways a
// camera drops them, and the receiver putting the file back from what is left, held to the file and to its BLAKE3
// (the vendored C through the wasm). Files of no bytes, one block, one source block, several, 41 small source blocks and 10 MB,
// random and prose (packed by the default client, sim/pack.mjs, sent as the packed stream and unpacked after); loss at
// random, by ring, in bursts, a late join, a corrupted block that passes its CRC; the pack's own rules.
// Laps is the airtime from the join over one cycle of the file's blocks. About 8 s.
// ai: Since 2026-10-22, transfers kept across receivers (store keeps): a store whose folders outlive its receiver, as
// ai: OPFS outlives a page; part of a file into one receiver, which is dropped, the rest into the next; a file parked by
// ai: another's header and resumed by its own; a parked transfer forgotten.
//   node test/xfer_rig.mjs
import { init as initOb } from "../sim/ob.mjs";
import { init as initWh, Encoder, Decoder } from "../sim/fountain.mjs";
import { init as initZstd, Z } from "../sim/zstd.mjs";
import { XferSender, XferReceiver, MemoryStore, ID_HEADER, blake3, hex, isControlId, folderOf, PAYLOAD, ID_BYTES, makeSourceBlock, sbCv, sameFile, idSbn, idEsi, idPut, COMP_ZSTD } from "../sim/xfer.mjs";
import { pack, unpack, PACK_FRAME } from "../sim/pack.mjs";

const M = await initOb();
await initWh(); await initZstd();
let fails = 0, over = 0;
const rand = (n, seed) => { const a = new Uint8Array(n); let s = seed >>> 0 || 1; for (let i = 0; i < n; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a[i] = s >>> 16; } return a; };
// Text-like bytes from a small vocabulary: the pack shrinks them to about a sixth.
const prose = (n, seed) => { const w = ["lizard ", "frame ", "block ", "ring ", "pilot ", "camera ", "screen ", "the ", "a ", "of ", "and ", "reads ", "paints ", "\n"]; const a = new Uint8Array(n); let s = seed >>> 0 || 1, k = 0; while (k < n) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; const t = w[(s >>> 16) % w.length]; for (let i = 0; i < t.length && k < n; i++) a[k++] = t.charCodeAt(i); } return a; };

// ring: the outer share of a frame's slots never read. burst: frames lost, 48 of every 240 (2 s of 10 at 24 fps).
// corrupt: source block whose first symbol sent (a send's first symbols are its seed's) goes with a flipped bit (once, as
// symbols do not repeat). Every case's carousel from a seed of its own (XferSender seed), so a run repeats.
// stream: a streaming sender (2026-11-06): every chaining value first, then a source block handed it every `stream` frames, in
// a scrambled order, the frames from the first; warm counts the frames before the
// last source block. text: prose packed (sim/pack.mjs), the packed stream sent, the received one unpacked and held to the prose.
async function run(label, { length, log2 = 22, B = 12, loss = 0, ring = 0, burst = 0, join = 0, corrupt = -1, text = false, maxCycles = 6, wantRejected = 0, stream = 0 }) {
  const plain = text ? prose(length, length + 7) : rand(length, length + 7), pk = text ? pack(plain, "f.txt", Z) : null;
  const file = pk ? pk.bytes : plain;
  length = file.length;
  const size = 2 ** log2, C = Math.ceil(length / size);
  const opts = { name: pk?.name ?? "f.bin", compression: pk?.compression ?? 0, sbLog2: log2, M, Encoder, seed: (Math.imul(length + 1, 2654435761) ^ log2) >>> 0 };
  let xs, ks = null, order = [];
  if (stream) {
    const cvs = new Uint8Array(32 * C), part = (c) => file.subarray(c * size, Math.min(length, (c + 1) * size));
    for (let c = 0; c < C; c++) cvs.set(sbCv(M, part(c), c, log2), 32 * c);
    ks = Array.from({ length: C }, (_, c) => makeSourceBlock(part(c), c, log2, false, { M, Encoder }));
    order = Array.from({ length: C }, (_, c) => (c * 7 + 3) % C);
    if (new Set(order).size !== C) order = Array.from({ length: C }, (_, c) => C - 1 - c);
    xs = new XferSender(null, { ...opts, cvs, length });
  } else xs = new XferSender(file, opts);
  const rx = new XferReceiver({ M, Decoder, store: new MemoryStore() });
  let s = 12345;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const buf = new Uint8Array(PAYLOAD), cycle = Math.max(1, (ks ? ks.map((k) => k.K) : xs.K).reduce((a, b) => a + b, 0)), limit = join + C * stream + Math.ceil((maxCycles * cycle) / B) + 50;
  let f = 0, sent = 0, warm = 0, csym = -1;
  for (; f < limit && !rx.done; f++) {
    if (stream && order.length && f % stream === stream - 1) { const c = order.shift(); xs.add(c, ks[c]); }
    if (!xs.carousel) warm++;
    const ids = xs.frameIds(B);
    if (f < join) continue;
    if (burst && f % 240 < 48) { sent += B; continue; }
    for (let k = 0; k < B; k++) {
      sent++;
      if (rnd() < loss || k >= B * (1 - ring)) continue;
      const b = xs.block(ids[k], buf).slice();
      if (!isControlId(ids[k]) && idSbn(ids[k]) === corrupt) { if (csym < 0) csym = idEsi(ids[k]); if (idEsi(ids[k]) === csym) b[5] ^= 1; }
      await rx.add(ids[k], xs.toiFor(ids[k]), b);
    }
  }
  const r = rx.result, p = rx.progress();
  let ok = !!r && r.bytes.length === length && r.bytes.every((v, i) => v === file[i]) && r.root === hex(blake3(M, file)) && r.root === hex(xs.root) && p.rejected === wantRejected;
  if (pk) {
    const back = ok && r?.compression === COMP_ZSTD ? await unpack(r.bytes, Z) : null;
    ok &&= pk.packed && r.name === "f.txt.zst" && !!back && back.length === plain.length && back.every((v, i) => v === plain[i]);
  }
  over += p.over ?? 0;
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${pk ? `${plain.length} B packed to ` : ""}${length} B, ${xs.sourceBlocks} source block${xs.sourceBlocks === 1 ? "" : "s"} of 2^${log2}; ${(sent / cycle).toFixed(3)} cycles${stream ? ` (${warm} frames before the last source block)` : ""}, ${p.rejected ?? 0} rejected, ${p.over ?? 0} solve${p.over === 1 ? "" : "s"} past K`);
  xs.free();
}
await run("empty", { length: 0 });
await run("one block, unfountained", { length: PAYLOAD });
await run("two blocks", { length: PAYLOAD + 1 });
await run("small", { length: 50000 });
await run("small, 30% lost", { length: 50000, loss: 0.3 });
await run("3 source blocks", { length: 3 * 2 ** 20 - 12345, log2: 20 });
await run("3 source blocks, the outer 25% of rings lost", { length: 3 * 2 ** 20 - 12345, log2: 20, ring: 0.25 });
await run("3 source blocks, joined 200 frames late", { length: 3 * 2 ** 20 - 12345, log2: 20, join: 200 });
await run("3 source blocks, 2 s lost in every 10", { length: 3 * 2 ** 20 - 12345, log2: 20, burst: 1 });
// ai: the wrong source block caught at the file's root (no manifest since 2026-11-21): the file refused whole, its 3
// ai: source blocks collected again
await run("3 source blocks, source block 1 sent wrong in cycle 1, refused whole at the root", { length: 3 * 2 ** 20 - 12345, log2: 20, corrupt: 1, wantRejected: 3 });
// ai: a file of one source block sent wrong: refused at its solve, its symbols dropped (no list of refused symbols kept,
// ai: 2026-11-21) and collected again from the next cycles
await run("1 source block sent wrong in cycle 1, refused at once and collected again", { length: 2 ** 20 - 12345, log2: 20, corrupt: 0, wantRejected: 1 });
await run("41 source blocks of 1 KiB", { length: 40 * 1024 + 100, log2: 10 });
await run("12 source blocks, as many as a frame's blocks, outer 25% lost", { length: 12 * 2 ** 16 - 5, log2: 16, ring: 0.25 });
await run("10 MB", { length: 10e6 });
await run("10 MB, 40% lost", { length: 10e6, loss: 0.4 });
// Prose: packed by the default client (sim/pack.mjs), the stream sent, unpacked after.
await run("small, prose, packed", { length: 50000, text: true });
await run("1200 B of prose, one block once packed", { length: 1200, text: true });
await run("2 source blocks of a packed stream, 30% lost", { length: 9 * 2 ** 20 - 12345, log2: 20, loss: 0.3, text: true });
await run("10 MB of prose packed, 40% lost", { length: 10e6, loss: 0.4, text: true });
for (let k = 0; k < 20; k++) await run(`200 KB, 20% lost, seed ${k}`, { length: 200000 + k, log2: 16, loss: 0.2 });
// Joined after the first cycle, so repair blocks only: RaptorQ then wants a block or two past K now and again, and the
// source block takes the rest as they come.
await run("41 source blocks of 1 KiB, repair blocks only", { length: 40 * 1024 + 100, log2: 10, join: 100 });
await run("3 source blocks, streamed, a source block every 40 frames", { length: 3 * 2 ** 20 - 12345, log2: 20, stream: 40 });
await run("41 source blocks of 1 KiB, streamed out of turn, 30% lost", { length: 40 * 1024 + 100, log2: 10, stream: 3, loss: 0.3 });
await run("2 source blocks of a packed stream, streamed, outer 25% of rings lost", { length: 9 * 2 ** 20 - 12345, log2: 20, stream: 40, text: true, ring: 0.25 });
await run("10 MB, streamed, joined in the warm frames, 40% lost", { length: 10e6, stream: 60, join: 70, loss: 0.4 });
for (let k = 0; k < 10; k++) await run(`400 KB in 16 KiB source blocks, repair blocks only, seed ${k}`, { length: 400000 + 37 * k, log2: 14, join: 80 });

// ai: a receiver's file by name: done, and its result (XferReceiver holds several transfers since 2026-11-19)
const doneAs = (rx, name) => rx.files().some((t) => t.done && t.hdr.name === name);
const resultAs = (rx, name) => rx.files().find((t) => t.done && t.hdr.name === name)?.result ?? null;

// ai: The fountain worker's OPFS store in memory (lizard-web/fountain-worker.mjs OpfsStore): a folder a transfer by its
// ai: id, in a map that outlives the receiver (the origin's OPFS outliving a page).
class KeepStore {
  constructor(folders) { this.kind = "keep"; this.keeps = true; this.folders = folders; this.cur = null; this.id = ""; }
  // ai: another transfer's store on the same folders (XferReceiver takes one a transfer)
  make() { return new KeepStore(this.folders); }
  async open(id, header, length) {
    const f = this.folders.get(id);
    this.id = id;
    if (f && sameFile(M, f.header, header)) {
      this.cur = f;
      return { found: true, verified: [...f.verified], cvs: new Map(f.cvs), sourceBlocks: [...f.recs.keys()] };
    }
    this.cur = { header: header.slice(), out: new Uint8Array(length), recs: new Map(), verified: [], cvs: new Map(), meta: null };
    this.folders.set(id, this.cur);
    return { found: false };
  }
  async append(c, esi, bytes) {
    const r = this.cur.recs.get(c) ?? [];
    const rec = new Uint8Array(ID_BYTES + PAYLOAD);
    idPut(rec, 0, esi);
    rec.set(bytes.subarray(0, PAYLOAD), ID_BYTES);
    r.push(rec); this.cur.recs.set(c, r);
  }
  async read(c) { const r = this.cur.recs.get(c) ?? [], b = new Uint8Array(r.length * (ID_BYTES + PAYLOAD)); r.forEach((x, i) => b.set(x, i * x.length)); return b; }
  async drop(c) { this.cur?.recs.delete(c); }
  async writeOut(off, bytes) { this.cur.out.set(bytes, off); }
  async noteVerified(c, cv) { this.cur.verified.push(c); if (cv) this.cur.cvs.set(c, cv.slice()); }
  async forgetVerified() { this.cur.verified = []; this.cur.cvs = new Map(); }
  async noteMeta(j) { this.cur.meta = JSON.parse(j); }
  async finish() { const out = this.cur.out; this.folders.delete(this.id); return { bytes: out }; }
  async content(length) { return this.cur.out.slice(0, length); }
  async park() { this.cur = null; this.id = ""; }
  async clear() { if (this.id) this.folders.delete(this.id); this.cur = null; this.id = ""; }
  // ai: a folder by its id (22 hex digits), or every folder of a root (store.h forget)
  async forget(id) {
    for (const k of [...this.folders.keys()]) if (id.length === 22 ? k === id : k.startsWith(id.slice(0, 16))) { if (k === this.id) await this.clear(); else this.folders.delete(k); }
  }
  held() { return 0; }
}
// ai: share of a cycle of x's blocks into one receiver, then (other) a second file whole into the same receiver (a transfer
// ai: of its own beside x's since 2026-11-19; it parked x's before), then x into a new receiver on the same folders (or
// ai: the same one, other) until it is done
// ai: restart: the rest from a sender made again (the app closed and sent the file anew), "same" its seed (its ids
// ai: repeat those it sent before), "other" another (fresh ids): the file's TOI its content's either way, so the same
// ai: transfer, resumed (2026-11-21; another seed another transfer, the parked one left, from 2026-11-20)
async function resume(label, { length, log2 = 20, B = 12, loss = 0, join = 0, share = 0.5, other = false, forget = false, restart = "" }) {
  const mk = (n, seed) => rand(n, seed), seed = (Math.imul(length + 3, 2654435761) ^ log2) >>> 0;
  const file = mk(length, length + 11), xs = new XferSender(file, { name: "x.bin", sbLog2: log2, M, Encoder, seed });
  xs.name = "x.bin";
  const folders = new Map(), id = folderOf(hex(xs.root), xs.toi), buf = new Uint8Array(PAYLOAD);
  let s = 4242;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const feed = async (rx, tx, frames) => {
    let f = 0;
    for (; f < frames && !doneAs(rx, tx.name); f++) { const ids = tx.frameIds(B); for (let k = 0; k < B; k++) if (rnd() >= loss) await rx.add(ids[k], tx.toiFor(ids[k]), tx.block(ids[k], buf).slice()); }
    return f;
  };
  for (let f = 0; f < join; f++) xs.frameIds(B);
  const cycle = Math.max(1, xs.K.reduce((a, b) => a + b, 0)), first = Math.ceil((share * cycle) / B);
  let rx = new XferReceiver({ M, Decoder, store: new KeepStore(folders) });
  await feed(rx, xs, first);
  const part = rx.progress(), parked = () => folders.get(id);
  let ok = !rx.done && part.verified < part.header.sourceBlocks, extra = "";
  if (other) {
    const y = mk(length + 333, 99), ys = new XferSender(y, { name: "y.bin", sbLog2: log2, M, Encoder });
    ys.name = "y.bin";
    await feed(rx, ys, 50 * Math.ceil(ys.K.reduce((a, b) => a + b, 0) / B));
    const ry = resultAs(rx, "y.bin");
    ok &&= !!ry && ry.bytes.every((v, i) => v === y[i]) && !doneAs(rx, "x.bin");
    extra = `, y.bin whole between`;
    ys.free();
  } else rx = new XferReceiver({ M, Decoder, store: new KeepStore(folders) });   // ai: the page closed: the receiver dropped as it was
  ok &&= !!parked()?.meta && parked().meta.name === "x.bin";
  if (forget) {
    await rx.forget(hex(xs.root));
    ok &&= !folders.has(id);
    console.log(`${ok ? "ok  " : "FAIL"} ${label}: parked at ${Math.round(100 * part.verified / part.header.sourceBlocks)}% of source blocks verified, then forgotten`);
    if (!ok) fails++;
    xs.free();
    return;
  }
  const tx = restart ? new XferSender(file, { name: "x.bin", sbLog2: log2, M, Encoder, seed: restart === "same" ? seed : (seed ^ 0x5a5a5a5a) >>> 0 }) : xs;
  tx.name = "x.bin";
  const more = await feed(rx, tx, 50 * Math.ceil(cycle / B));
  if (tx !== xs) tx.free();
  const r = resultAs(rx, "x.bin");
  ok &&= !!r && r.bytes.length === length && r.bytes.every((v, i) => v === file[i]) && r.root === hex(blake3(M, file)) && !folders.has(id);
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${xs.sourceBlocks} source blocks, ${part.verified} verified and ${Math.round(100 * (first * B) / cycle)}% of a cycle in before${extra}; the rest in ${(more * B / cycle).toFixed(3)} cycles`);
  xs.free();
}
await resume("resumed by the next receiver", { length: 6 * 2 ** 20 - 99, log2: 20 });
await resume("resumed by the next receiver, 30% lost", { length: 6 * 2 ** 20 - 99, log2: 20, loss: 0.3 });
await resume("resumed by the next receiver, repair blocks only", { length: 400000, log2: 14, join: 80 });
await resume("another file's header beside it: both held at once, each whole", { length: 3 * 2 ** 20 - 12345, log2: 20, other: true });
await resume("a parked transfer forgotten", { length: 3 * 2 ** 20 - 12345, log2: 20, forget: true });
await resume("resumed from a sender made again with the same seed, 30% lost", { length: 6 * 2 ** 20 - 99, log2: 20, loss: 0.3, share: 0.8, restart: "same" });
await resume("resumed from a sender made again with another seed, 30% lost: the same file's TOI", { length: 6 * 2 ** 20 - 99, log2: 20, loss: 0.3, share: 0.8, restart: "other" });

// ai: One file from two senders is one transfer (2026-11-21: a file's TOI its content's; two transfers from 2026-11-20):
// ai: half a cycle of File A from one sender, then the other's under another name and seed into one receiver
// ai: (park: dropped between, the second on the same folders); the file whole from both senders' blocks pooled, under
// ai: the first description's name (park: the second's, which the second receiver read), in less than the cycle one
// ai: sender's alone takes (or its folder resumed and done).
async function twoSenders(label, { length, log2 = 20, B = 12, loss = 0.2, park = false }) {
  const file = rand(length, length + 21), a = new XferSender(file, { name: "a.bin", sbLog2: log2, M, Encoder, seed: 11 });
  const b = new XferSender(file, { name: "b.bin", sbLog2: log2, M, Encoder, seed: 977 });
  const folders = new Map(), buf = new Uint8Array(PAYLOAD), cycle = a.K.reduce((x, y) => x + y, 0);
  let s = 31337;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const feed = async (rx, tx, frames) => { let f = 0; for (; f < frames && !rx.done; f++) { const ids = tx.frameIds(B); for (let k = 0; k < B; k++) if (rnd() >= loss) await rx.add(ids[k], tx.toiFor(ids[k]), tx.block(ids[k], buf).slice()); } return f; };
  let rx = new XferReceiver({ M, Decoder, store: park ? new KeepStore(folders) : new MemoryStore() });
  const first = await feed(rx, a, Math.ceil((0.5 * cycle) / B)), held = rx.progress();
  if (park) rx = new XferReceiver({ M, Decoder, store: new KeepStore(folders) });
  // ai: the name its receiver's description gives: the first sender's, or (park) the second's, the only one the second
  // ai: receiver read
  const more = await feed(rx, b, 50 * Math.ceil(cycle / B)), r = resultAs(rx, park ? "b.bin" : "a.bin"), p = rx.progress();
  const ok = first > 0 && a.toi === b.toi && !!r && r.bytes.every((v, i) => v === file[i]) && r.root === hex(a.root) && p.rejected === 0 && more * B / cycle < 1 &&
    rx.files().length === 1 && (!park || !folders.has(folderOf(hex(a.root), a.toi)));
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${a.sourceBlocks} source blocks, ${held.verified} verified and half a cycle in from a.bin; the file whole after ${(more * B / cycle).toFixed(3)} cycles of b.bin's blocks${park ? ", the first's folder resumed" : ""}`);
  a.free(); b.free();
}
await twoSenders("File A from two senders, another name and seed each, one receiver: one transfer, both senders' blocks pooled", { length: 6 * 2 ** 20 - 77 });
await twoSenders("File A from two senders, the receiver dropped between: the second resumes the first's folder", { length: 6 * 2 ** 20 - 77, park: true });

// ai: The transfer Wirehair stalled (2026-11-14, STATUS "Wirehair's long tail is precode twins"): a 4 MiB file (one
// ai: source block, 8,887 blocks) from carousel seed 305886, whose source block starts at symbol 246,656, with symbols 256,520 to
// ai: 256,524 lost, which Wirehair took past K + 1,024 and then refused; RaptorQ has it back within K + 2.
{
  const file = rand(2 ** 22, 4242), tx = new XferSender(file, { name: "r.bin", M, Encoder, seed: 305886 });
  const lost = new Set([256520, 256521, 256522, 256523, 256524]), buf = new Uint8Array(PAYLOAD), B = 60, K = tx.K[0];
  const rx = new XferReceiver({ M, Decoder, store: new MemoryStore() });
  let data = 0;
  for (let f = 0; f < 2000 && !rx.done; f++) {
    const ids = tx.frameIds(B);
    for (let k = 0; k < B && !rx.done; k++) {
      if (!isControlId(ids[k]) && lost.has(idEsi(ids[k]))) continue;
      if (!isControlId(ids[k])) data++;
      await rx.add(ids[k], tx.toiFor(ids[k]), tx.block(ids[k], buf).slice());
    }
  }
  const r = rx.result, ok = !!r && r.bytes.length === file.length && r.bytes.every((v, i) => v === file[i]) && r.root === hex(tx.root) && data - K <= 2;
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} the transfer Wirehair stalled: ${r ? `back at K + ${data - K} data blocks` : "NOT back"}`);
  tx.free();
}

// ai: The pack's rules (sim/pack.mjs): the name kept and compression 1, what does not pay, and a damaged stream.
{
  const sure = (ok, what) => { if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} pack: ${what}`); };
  const r0 = pack(rand(5e6, 3), "r.bin", Z), e0 = pack(new Uint8Array(0), "e", Z);
  sure(!r0.packed && r0.compression === 0 && !e0.packed && e0.compression === 0, "random and empty files go as they are");
  const big = prose(3 * PACK_FRAME + 12345, 5), p = pack(big, "big.txt", Z), back = await unpack(p.bytes, Z);
  sure(p.packed && p.compression === 1 && p.name === "big.txt.zst" && back.length === big.length && back.every((v, i) => v === big[i]),
    `${big.length} B of prose in 4 frames, ${p.bytes.length} B, compression 1, named big.txt.zst`);
  const cut = p.bytes.slice(0, p.bytes.length - 9);
  let threw = false;
  try { await unpack(cut, Z); } catch { threw = true; }
  sure(threw, "a stream cut short refused");
}
// ai: The session header's rules (src/xfer.h): the description is the file's own map; a name cut to keep it in the
// ai: header; a description with an unknown critical key refused. The keys' rule RFC 9581's: an unsigned key is
// ai: critical, a negative one elective.
{
  const buf = new Uint8Array(PAYLOAD);
  // ai: a name of 300 bytes and the zstd compression: its stem cut, its extension kept, the description in the header
  {
    const name = "n".repeat(296) + ".log", y = rand(5000, 79);
    const ys = new XferSender(y, { name, compression: COMP_ZSTD, M, Encoder, seed: 84 }), r2 = new XferReceiver({ M, Decoder, store: new MemoryStore() });
    const h = ys.block(ID_HEADER, buf).slice(), sc = M._malloc(8), ph = M._malloc(PAYLOAD);
    M.HEAPU8.set(h, ph); M._xfer_session_read(ph, sc);
    const base = new Int32Array(M.HEAPU8.buffer, sc, 2)[1];
    [sc, ph].forEach((q) => M._free(q));
    for (let f = 0; f < 50 && !r2.done; f++) for (const id of ys.frameIds(30)) await r2.add(id, ys.toiFor(id), ys.block(id, buf).slice());
    const got = r2.results()[0];
    const ok = base <= 460 && r2.done && got?.name === "n".repeat(251) + ".log" && got.compression === COMP_ZSTD && got.bytes.every((v, i) => v === y[i]);
    if (!ok) fails++;
    console.log(`${ok ? "ok  " : "FAIL"} a file of a 300-byte name, zstd: its stem cut to ${got?.name.length} bytes, .log kept, its description ${base} B of the header's 460`);
    ys.free();
  }
  // ai: a header whose description carries an unknown critical key (2): not taken, no file opened; an elective one (-5)
  // ai: skipped
  {
    const zs = new XferSender(rand(20000, 85), { name: "z.bin", M, Encoder, seed: 86 }), r2 = new XferReceiver({ M, Decoder, store: new MemoryStore() });
    const hdr = zs.block(ID_HEADER, buf).slice(), sc = M._malloc(8), ph = M._malloc(PAYLOAD);
    M.HEAPU8.set(hdr, ph); M._xfer_session_read(ph, sc);
    const dlen = new Int32Array(M.HEAPU8.buffer, sc, 2)[1];
    [sc, ph].forEach((q) => M._free(q));
    const ext = new Uint8Array(PAYLOAD); ext.set(hdr.subarray(0, dlen)); ext[0] = hdr[0] + 1; ext.set([0x02, 0xf5], dlen);   // ai: {3: .., 4: .., 5: .., 6: .., 2: true}
    for (let f = 0; f < 40; f++) for (const id of zs.frameIds(30)) await r2.add(id, zs.toiFor(id), isControlId(id) ? ext : zs.block(id, buf).slice());
    const ok = !r2.files().length && r2.progress().descsRefused > 0;
    if (!ok) fails++;
    console.log(`${ok ? "ok  " : "FAIL"} a session header with an unknown critical key (2): refused (${r2.progress().descsRefused}), no file opened`);
    const el = ext.slice(); el.set([0x24, 0xf5], dlen);   // ai: {..., -5: true}
    const r3 = new XferReceiver({ M, Decoder, store: new MemoryStore() }), ys2 = new XferSender(rand(20000, 85), { name: "z.bin", M, Encoder, seed: 86 });
    for (let f = 0; f < 200 && !r3.done; f++) for (const id of ys2.frameIds(30)) await r3.add(id, ys2.toiFor(id), isControlId(id) ? el : ys2.block(id, buf).slice());
    const ok3 = r3.done;
    if (!ok3) fails++;
    console.log(`${ok3 ? "ok  " : "FAIL"} the same with an unknown elective key (-5): skipped, the file whole`);
    ys2.free();
    zs.free();
  }
}
// ai: A finished transfer's TOIs let go (releaseDone: the session's and the page's when the camera stops for a file
// ai: in): the same file still in the light is received again, a download clicked twice (2026-11-20); before the
// ai: release its blocks go nowhere.
{
  const file = rand(2 * 2 ** 20 + 4321, 77), xs = new XferSender(file, { name: "again.bin", sbLog2: 20, M, Encoder, seed: 5 });
  const rx = new XferReceiver({ M, Decoder, store: new MemoryStore() }), buf = new Uint8Array(PAYLOAD), B = 30;
  const feed = async (want, n) => {
    let f = 0;
    for (; f < n && rx.results().length < want; f++) { const ids = xs.frameIds(B); for (const id of ids) await rx.add(id, xs.toiFor(id), xs.block(id, buf).slice()); }
    return f;
  };
  const f1 = await feed(1, 5000), f0 = await feed(2, 200);
  const held = rx.results().length;
  rx.releaseDone();
  const f2 = await feed(2, 5000), rs = rx.results();
  const ok = held === 1 && f0 === 200 && rs.length === 2 && rs.every((r) => r.bytes.every((v, i) => v === file[i])) && rx.files().length === 2 && rx.has(xs.toi);
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} a finished transfer's TOIs let go: the same file again (${f1} frames, then nothing in 200 held, then ${f2} frames after the release; ${rs.length} received)`);
  xs.free();
}

console.log(`${over} source block solves needed more than K blocks and took them as they came`);
console.log(fails ? `${fails} FAILED` : "all ok");
process.exit(fails ? 1 : 0);
