// The rig's chunked transfer (sim/xfer.mjs) without a camera: the sender's schedule, its blocks dropped the ways a
// camera drops them, and the receiver putting the file back from what is left, held to the file and to its BLAKE3
// (the vendored C through the wasm). Files of no bytes, one block, one chunk, several, 41 small chunks and 10 MB,
// random and prose (packed by the default client, sim/pack.mjs, sent as the packed stream and unpacked after); loss at
// random, by ring, in bursts, a late join, a corrupted block that passes its CRC; the pack's own rules.
// Laps is the airtime from the join over one lap of the file's blocks. About 8 s.
// ai: Since 2026-10-22, transfers kept across receivers (store keeps): a store whose folders outlive its receiver, as
// ai: OPFS outlives a page; part of a file into one receiver, which is dropped, the rest into the next; a file parked by
// ai: another's header and resumed by its own; a parked transfer forgotten.
//   node test/xfer_rig.mjs
import { init as initOb } from "../sim/ob.mjs";
import { init as initWh, Encoder, Decoder } from "../sim/fountain.mjs";
import { init as initZstd, Z } from "../sim/zstd.mjs";
import { XferSender, XferReceiver, MemoryStore, blake3, hex, isControlId, PAYLOAD, makeChunk, chunkCv } from "../sim/xfer.mjs";
import { pack, unpack, packType, packedType, unpackedName, PACK_FRAME } from "../sim/pack.mjs";

const M = await initOb();
await initWh(); await initZstd();
let fails = 0, over = 0;
const rand = (n, seed) => { const a = new Uint8Array(n); let s = seed >>> 0 || 1; for (let i = 0; i < n; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a[i] = s >>> 16; } return a; };
// Text-like bytes from a small vocabulary: the pack shrinks them to about a sixth.
const prose = (n, seed) => { const w = ["lizard ", "frame ", "block ", "ring ", "pilot ", "camera ", "screen ", "the ", "a ", "of ", "and ", "reads ", "paints ", "\n"]; const a = new Uint8Array(n); let s = seed >>> 0 || 1, k = 0; while (k < n) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; const t = w[(s >>> 16) % w.length]; for (let i = 0; i < t.length && k < n; i++) a[k++] = t.charCodeAt(i); } return a; };

// ring: the outer share of a frame's slots never read. burst: frames lost, 48 of every 240 (2 s of 10 at 24 fps).
// corrupt: chunk whose first symbol sent (a send's first symbols are its seed's) goes with a flipped bit (once, as
// symbols do not repeat). Every case's schedule from a seed of its own (XferSender seed), so a run repeats.
// stream: a streaming sender (2026-11-06): every chaining value first, then a chunk handed it every `stream` frames, in
// a scrambled order (manifest groups whole out of turn), the frames from the first; warm counts the frames before the
// last chunk. text: prose packed (sim/pack.mjs), the packed stream sent, the received one unpacked and held to the prose.
async function run(label, { length, log2 = 22, B = 12, loss = 0, ring = 0, burst = 0, join = 0, corrupt = -1, text = false, maxLaps = 6, wantRejected = 0, stream = 0 }) {
  const plain = text ? prose(length, length + 7) : rand(length, length + 7), pk = text ? pack(plain, "f.txt", "text/plain", Z) : null;
  const file = pk ? pk.bytes : plain;
  length = file.length;
  const size = 2 ** log2, C = Math.ceil(length / size);
  const opts = { name: pk?.name ?? "f.bin", type: pk?.type ?? "application/octet-stream", chunkLog2: log2, M, Encoder, seed: (Math.imul(length + 1, 2654435761) ^ log2) >>> 0 };
  let xs, ks = null, order = [];
  if (stream) {
    const cvs = new Uint8Array(32 * C), part = (c) => file.subarray(c * size, Math.min(length, (c + 1) * size));
    for (let c = 0; c < C; c++) cvs.set(chunkCv(M, part(c), c, log2), 32 * c);
    ks = Array.from({ length: C }, (_, c) => makeChunk(part(c), c, log2, false, { M, Encoder }));
    order = Array.from({ length: C }, (_, c) => (c * 7 + 3) % C);
    if (new Set(order).size !== C) order = Array.from({ length: C }, (_, c) => C - 1 - c);
    xs = new XferSender(null, { ...opts, cvs, length });
  } else xs = new XferSender(file, opts);
  const rx = new XferReceiver({ M, Decoder, store: new MemoryStore() });
  let s = 12345;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const buf = new Uint8Array(PAYLOAD), lap = Math.max(1, (ks ? ks.map((k) => k.K) : xs.K).reduce((a, b) => a + b, 0)), limit = join + C * stream + Math.ceil((maxLaps * lap) / B) + 50;
  let f = 0, sent = 0, warm = 0, csym = -1, bitsMax = 0;
  for (; f < limit && !rx.done; f++) {
    if (stream && order.length && f % stream === stream - 1) { const c = order.shift(); xs.add(c, ks[c]); }
    if (!xs.sched) warm++;
    const ids = xs.frameIds(B);
    if (f < join) continue;
    if (burst && f % 240 < 48) { sent += B; continue; }
    for (let k = 0; k < B; k++) {
      sent++;
      if (rnd() < loss || k >= B * (1 - ring)) continue;
      const b = xs.block(ids[k], buf).slice();
      if (!isControlId(ids[k]) && ids[k] >>> 18 === corrupt) { if (csym < 0) csym = ids[k] & 0x3ffff; if ((ids[k] & 0x3ffff) === csym) b[5] ^= 1; }
      await rx.add(ids[k], b);
    }
    // ai: the most bytes a chunk's symbol sets hold (sim/xfer.mjs Bits): its symbols from a random start in runs
    for (const c of rx.ch?.values?.() ?? []) bitsMax = Math.max(bitsMax, [c.have, c.ban].reduce((a, bs) => a + (bs?.runs ?? []).reduce((x, r) => x + r.a.length, 0), 0));
  }
  const r = rx.result, p = rx.progress();
  let ok = !!r && r.bytes.length === length && r.bytes.every((v, i) => v === file[i]) && r.root === hex(blake3(M, file)) && r.root === hex(xs.root) && p.rejected === wantRejected && bitsMax <= 8192;
  if (pk) {
    const original = packedType(r?.mediaType), back = ok && original === "text/plain" ? await unpack(r.bytes, Z) : null;
    ok &&= pk.packed && r.name === "f.txt.zst" && unpackedName(r.name) === "f.txt" && !!back && back.length === plain.length && back.every((v, i) => v === plain[i]);
  }
  over += p.over ?? 0;
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${pk ? `${plain.length} B packed to ` : ""}${length} B, ${xs.chunks} chunk${xs.chunks === 1 ? "" : "s"} of 2^${log2}, ${xs.mcount} manifest; ${(sent / lap).toFixed(3)} laps${stream ? ` (${warm} frames before the last chunk)` : ""}, ${p.rejected ?? 0} rejected, ${p.over ?? 0} solve${p.over === 1 ? "" : "s"} past K`);
  xs.free();
}
await run("empty", { length: 0 });
await run("one block, unfountained", { length: PAYLOAD });
await run("two blocks", { length: PAYLOAD + 1 });
await run("small", { length: 50000 });
await run("small, 30% lost", { length: 50000, loss: 0.3 });
await run("3 chunks", { length: 3 * 2 ** 20 - 12345, log2: 20 });
await run("3 chunks, the outer 25% of rings lost", { length: 3 * 2 ** 20 - 12345, log2: 20, ring: 0.25 });
await run("3 chunks, joined 200 frames late", { length: 3 * 2 ** 20 - 12345, log2: 20, join: 200 });
await run("3 chunks, 2 s lost in every 10", { length: 3 * 2 ** 20 - 12345, log2: 20, burst: 1 });
await run("3 chunks, chunk 1 sent wrong in lap 1", { length: 3 * 2 ** 20 - 12345, log2: 20, corrupt: 1, wantRejected: 1 });
await run("41 chunks of 1 KiB, 3 manifest blocks", { length: 40 * 1024 + 100, log2: 10 });
await run("12 chunks, as many as a frame's blocks, outer 25% lost", { length: 12 * 2 ** 16 - 5, log2: 16, ring: 0.25 });
await run("10 MB", { length: 10e6 });
await run("10 MB, 40% lost", { length: 10e6, loss: 0.4 });
// Prose: packed by the default client (sim/pack.mjs), the stream sent, unpacked after.
await run("small, prose, packed", { length: 50000, text: true });
await run("1200 B of prose, one block once packed", { length: 1200, text: true });
await run("2 chunks of a packed stream, 30% lost", { length: 9 * 2 ** 20 - 12345, log2: 20, loss: 0.3, text: true });
await run("10 MB of prose packed, 40% lost", { length: 10e6, loss: 0.4, text: true });
for (let k = 0; k < 20; k++) await run(`200 KB, 20% lost, seed ${k}`, { length: 200000 + k, log2: 16, loss: 0.2 });
// Joined after the first lap, so repair blocks only: Wirehair then wants a block past K now and again, and the chunk
// takes the rest as they come.
await run("41 chunks of 1 KiB, repair blocks only", { length: 40 * 1024 + 100, log2: 10, join: 100 });
await run("3 chunks, streamed, a chunk every 40 frames", { length: 3 * 2 ** 20 - 12345, log2: 20, stream: 40 });
await run("41 chunks of 1 KiB, streamed out of turn, 30% lost", { length: 40 * 1024 + 100, log2: 10, stream: 3, loss: 0.3 });
await run("2 chunks of a packed stream, streamed, outer 25% of rings lost", { length: 9 * 2 ** 20 - 12345, log2: 20, stream: 40, text: true, ring: 0.25 });
await run("10 MB, streamed, joined in the warm frames, 40% lost", { length: 10e6, stream: 60, join: 70, loss: 0.4 });
for (let k = 0; k < 10; k++) await run(`400 KB in 16 KiB chunks, repair blocks only, seed ${k}`, { length: 400000 + 37 * k, log2: 14, join: 80 });

// ai: The fountain worker's OPFS store in memory (lizard-web/fountain-worker.mjs OpfsStore): a folder a transfer by its
// ai: id, in a map that outlives the receiver (the origin's OPFS outliving a page).
class KeepStore {
  constructor(folders) { this.kind = "keep"; this.keeps = true; this.folders = folders; this.cur = null; this.id = ""; }
  async open(id, header, length) {
    const f = this.folders.get(id);
    this.id = id;
    if (f && f.header.length === header.length && f.header.every((v, i) => v === header[i])) {
      this.cur = f;
      return { found: true, manifest: f.manifest.map(([m, b]) => [m, b.slice()]), verified: [...f.verified], chunks: [...f.recs.keys()] };
    }
    this.cur = { header: header.slice(), out: new Uint8Array(length), recs: new Map(), manifest: [], verified: [], meta: null };
    this.folders.set(id, this.cur);
    return { found: false };
  }
  async append(c, sym, bytes) {
    const r = this.cur.recs.get(c) ?? [];
    const rec = new Uint8Array(4 + PAYLOAD);
    rec[0] = sym & 255; rec[1] = (sym >>> 8) & 255; rec[2] = (sym >>> 16) & 255; rec[3] = sym >>> 24;
    rec.set(bytes.subarray(0, PAYLOAD), 4);
    r.push(rec); this.cur.recs.set(c, r);
  }
  async read(c) { const r = this.cur.recs.get(c) ?? [], b = new Uint8Array(r.length * (4 + PAYLOAD)); r.forEach((x, i) => b.set(x, i * x.length)); return b; }
  async drop(c) { this.cur?.recs.delete(c); }
  async writeOut(off, bytes) { this.cur.out.set(bytes, off); }
  async noteManifest(m, b) { this.cur.manifest.push([m, b.slice()]); }
  async noteVerified(c) { this.cur.verified.push(c); }
  async noteMeta(j) { this.cur.meta = JSON.parse(j); }
  async truncate(c, n) { const r = this.cur.recs.get(c); if (r) r.length = Math.min(r.length, n); }
  async finish() { const out = this.cur.out; this.folders.delete(this.id); return { bytes: out }; }
  async park() { this.cur = null; this.id = ""; }
  async clear() { if (this.id) this.folders.delete(this.id); this.cur = null; this.id = ""; }
  async forget(id) { if (id === this.id) return this.clear(); this.folders.delete(id); }
  held() { return 0; }
}
// ai: share of a lap of x's blocks into one receiver, then (other) a second file whole into the same receiver, then x
// ai: into a new receiver on the same folders (or the same one, other) until it is done
// ai: restart: the rest from a sender made again (the app closed and sent the file anew), "same" its seed, "other" another
async function resume(label, { length, log2 = 20, B = 12, loss = 0, join = 0, share = 0.5, other = false, forget = false, restart = "" }) {
  const mk = (n, seed) => rand(n, seed), seed = (Math.imul(length + 3, 2654435761) ^ log2) >>> 0;
  const file = mk(length, length + 11), xs = new XferSender(file, { name: "x.bin", chunkLog2: log2, M, Encoder, seed });
  xs.name = "x.bin";
  const folders = new Map(), id = hex(xs.root).slice(0, 16), buf = new Uint8Array(PAYLOAD);
  let s = 4242;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const feed = async (rx, tx, frames) => {
    let f = 0;
    for (; f < frames && !(rx.done && rx.result?.name === tx.name); f++) { const ids = tx.frameIds(B); for (let k = 0; k < B; k++) if (rnd() >= loss) await rx.add(ids[k], tx.block(ids[k], buf).slice()); }
    return f;
  };
  for (let f = 0; f < join; f++) xs.frameIds(B);
  const lap = Math.max(1, xs.K.reduce((a, b) => a + b, 0)), first = Math.ceil((share * lap) / B);
  let rx = new XferReceiver({ M, Decoder, store: new KeepStore(folders) });
  await feed(rx, xs, first);
  const part = rx.progress(), parked = () => folders.get(id);
  let ok = !rx.done && part.verified < part.header.chunks, extra = "";
  if (other) {
    const y = mk(length + 333, 99), ys = new XferSender(y, { name: "y.bin", chunkLog2: log2, M, Encoder });
    ys.name = "y.bin";
    await feed(rx, ys, 50 * Math.ceil(ys.K.reduce((a, b) => a + b, 0) / B));
    ok &&= !!rx.result && rx.result.bytes.every((v, i) => v === y[i]) && rx.result.name === "y.bin";
    extra = `, y.bin whole between`;
    ys.free();
  } else rx = new XferReceiver({ M, Decoder, store: new KeepStore(folders) });   // ai: the page closed: the receiver dropped as it was
  ok &&= !!parked()?.meta && parked().meta.name === "x.bin";
  if (forget) {
    await rx.forget(hex(xs.root));
    ok &&= !folders.has(id);
    console.log(`${ok ? "ok  " : "FAIL"} ${label}: parked at ${Math.round(100 * part.verified / part.header.chunks)}% of chunks verified, then forgotten`);
    if (!ok) fails++;
    xs.free();
    return;
  }
  const tx = restart ? new XferSender(file, { name: "x.bin", chunkLog2: log2, M, Encoder, seed: restart === "same" ? seed : (seed ^ 0x5a5a5a5a) >>> 0 }) : xs;
  tx.name = "x.bin";
  const more = await feed(rx, tx, 50 * Math.ceil(lap / B));
  if (tx !== xs) tx.free();
  const r = rx.result;
  ok &&= !!r && r.bytes.length === length && r.bytes.every((v, i) => v === file[i]) && r.root === hex(blake3(M, file)) && !folders.has(id);
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${xs.chunks} chunks, ${part.verified} verified and ${Math.round(100 * (first * B) / lap)}% of a lap in before${extra}; the rest in ${(more * B / lap).toFixed(3)} laps`);
  xs.free();
}
await resume("resumed by the next receiver", { length: 6 * 2 ** 20 - 99, log2: 20 });
await resume("resumed by the next receiver, 30% lost", { length: 6 * 2 ** 20 - 99, log2: 20, loss: 0.3 });
await resume("resumed by the next receiver, repair blocks only", { length: 400000, log2: 14, join: 80 });
await resume("parked by another file's header, resumed by its own", { length: 3 * 2 ** 20 - 12345, log2: 20, other: true });
await resume("a parked transfer forgotten", { length: 3 * 2 ** 20 - 12345, log2: 20, forget: true });
await resume("resumed from a sender made again with the same seed, 30% lost", { length: 6 * 2 ** 20 - 99, log2: 20, loss: 0.3, share: 0.8, restart: "same" });
await resume("resumed from a sender made again with another seed, 30% lost", { length: 6 * 2 ** 20 - 99, log2: 20, loss: 0.3, share: 0.8, restart: "other" });

// ai: The pack's rules (sim/pack.mjs): its type and name both ways, what does not pay, and a damaged stream.
{
  const t = 'text/plain; charset="utf-8"', pt = packType(t), sure = (ok, what) => { if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} pack: ${what}`); };
  sure(pt === 'application/zstd; original="text/plain; charset=\\"utf-8\\""' && packedType(pt) === t, `a quoted type escaped and back (${pt})`);
  sure(packedType("application/zstd") === null && packedType('application/zstd; original="a"b"') === null && packedType('application/zstd; original=""') === "", "a type of no pack, a bad one, an empty original");
  sure(packType("x".repeat(130)) === "" && packType("x".repeat(129)).length === 158, "a type past the header's 158 bytes once wrapped");
  sure(unpackedName("a.txt.zst") === "a.txt" && unpackedName(".zst") === ".zst" && unpackedName("a.txt") === "a.txt", "names back");
  sure(!pack(rand(5e6, 3), "r.bin", "", Z).packed && !pack(new Uint8Array(0), "e", "", Z).packed, "random and empty files go as they are");
  const big = prose(3 * PACK_FRAME + 12345, 5), p = pack(big, "b".repeat(300), "text/plain", Z), back = await unpack(p.bytes, Z);
  sure(p.packed && new TextEncoder().encode(p.name).length === 255 && p.name.endsWith(".zst") && back.length === big.length && back.every((v, i) => v === big[i]),
    `${big.length} B of prose in 4 frames, ${p.bytes.length} B, a long name cut to 255 bytes with its .zst`);
  const cut = p.bytes.slice(0, p.bytes.length - 9);
  let threw = false;
  try { await unpack(cut, Z); } catch { threw = true; }
  sure(threw, "a stream cut short refused");
}
console.log(`${over} chunk solves needed more than K blocks and took them as they came`);
console.log(fails ? `${fails} FAILED` : "all ok");
process.exit(fails ? 1 : 0);
