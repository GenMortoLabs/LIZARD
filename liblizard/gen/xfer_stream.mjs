// ai: The native transfer's check stream (liblizard/core/tools/xfer_check.cpp): a random file of several source blocks sent
// ai: by the wasm build of the C (lizard.h liz_tx over the plain module: core/tx/xfer_tx.cpp as emscripten builds it)
// ai: through a lossy channel, written as flat 472-byte blocks (the id, 12 bytes: the bare id and the 48-bit TOI (the
// ai: file's, or its session header's), then a reserved byte; then 460 payload bytes), and the receiving chain of the
// ai: same C's wasm build run on the same stream as the reference (liz_rx in memory: core/rx/xfer_rx_core.h's judge, its
// ai: dedupe and the core, a frame of 40 blocks a call), so the native build is held to the wasm build (C against C;
// ai: the JS chain, sim/xfer.mjs and the pages' dedupe, until 2026-11-25).
//
//   node liblizard/gen/xfer_stream.mjs [outDir] [bytes]    outDir defaults to liblizard/build/xfer, bytes to 9,000,000
//
// ai: Writes stream.bin (the file's blocks as received), file.bin (the file), bad.bin (stream.bin with one block of source block 1
// ai: wrong), renamed.bin (below), test.bin (the BLAKE3 test stream, one block of frame 7 wrong), ref.json (what the wasm's chain made of each). The channel, seeded: the receiver joins 30% into
// ai: the first cycle; each block is lost with p 0.2; a frame is read twice with p 0.08 (a camera's repeat); a frame's
// ai: blocks come in shuffled, and neighbouring frames swap with p 0.1 (decode threads finishing out of order).
import { mkdirSync, writeFileSync } from "node:fs";
const LIB = new URL("../", import.meta.url);
const { init: initOb, streamBlock, hex, PAYLOAD, ID_BYTES, ID_HEADER, BLOCK, idGet, idPut, toiGet, idSbn } = await import(new URL("sim/ob.mjs", LIB).href);

const OUT = process.argv[2] ?? new URL("../build/xfer/", import.meta.url).pathname;
const LENGTH = Number(process.argv[3] ?? 9_000_000), B = 40, REC = BLOCK;
mkdirSync(OUT, { recursive: true });
const M = await initOb();
const fail = () => { throw new Error(M.UTF8ToString(M._liz_last_error())); };

let s = 0x2545f491;
const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 2 ** 32; };
const file = new Uint8Array(LENGTH);
for (let i = 0; i < LENGTH; i++) file[i] = (rnd() * 256) | 0;

// ai: a sender over the file (every source block made before the first frame: the plain module has no threads), its
// ai: seed given; its info (liz_tx_info, 64 bytes)
const fp = M._malloc(LENGTH);
M.HEAPU8.set(file, fp);
function sender(name, seed) {
  const nn = M.lengthBytesUTF8(name) + 1, np = M._malloc(nn);
  M.stringToUTF8(name, np, nn);
  // ai: source blocks of 4 MiB, given: a 9 MB file at the default 16 MiB would be one source block, and the check is the receiver on several
  const t = M._liz_tx_new_sized(fp, LENGTH, np, 0, 22);
  M._free(np);
  if (!t || M._liz_tx_seed(t, seed >>> 0) < 0) fail();
  return t;
}
function infoOf(t) {
  const p = M._malloc(64);
  M._liz_tx_info_get(t, p);
  const dv = new DataView(M.HEAPU8.buffer, p, 64), i = { cycle: dv.getUint32(20, true), root: hex(M.HEAPU8.slice(p + 24, p + 56)), seed: dv.getUint32(56, true) };
  M._free(p);
  return i;
}
const txSeed = Math.floor(rnd() * 2 ** 32) >>> 0, tx = sender("check-9MB.bin", txSeed), info = infoOf(tx);
const cycle = info.cycle, frames = Math.ceil((1.8 * cycle) / (B - 1)), join = Math.floor((0.3 * cycle) / (B - 1));
const fb = M._malloc(B * REC), sent = [];
let txToi = 0;
for (let f = 0; f < frames; f++) {
  if (M._liz_tx_next(tx, B, fb) < 0) fail();
  const all = M.HEAPU8.slice(fb, fb + B * REC);
  for (let k = 0; !txToi && k < B; k++) if (idGet(all, k * REC) !== ID_HEADER) txToi = toiGet(all, k * REC);   // ai: the file's TOI
  if (f < join) continue;
  sent.push(Array.from({ length: B }, (_, k) => all.slice(k * REC, (k + 1) * REC)));
}
M._liz_tx_free(tx);
for (let f = 0; f + 1 < sent.length; f++) if (rnd() < 0.1) [sent[f], sent[f + 1]] = [sent[f + 1], sent[f]];
const got = [];
for (const blocks of sent) {
  for (let r = rnd() < 0.08 ? 2 : 1; r > 0; r--) {
    const kept = blocks.filter(() => rnd() >= 0.2);
    for (let i = kept.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [kept[i], kept[j]] = [kept[j], kept[i]]; }
    got.push(...kept);
  }
}
const stream = new Uint8Array(got.length * REC);
got.forEach((b, i) => stream.set(b, i * REC));
writeFileSync(`${OUT}/stream.bin`, stream);
writeFileSync(`${OUT}/file.bin`, file);

// ai: The test stream: ids from 0, frame seq's blocks a run from seq * B, each block its id's BLAKE3; frame 7's block 3 one
// ai: byte off, as a block that passed its CRC wrongly would be.
const TEST_FRAMES = 60, test = new Uint8Array(TEST_FRAMES * B * REC);
for (let i = 0; i < TEST_FRAMES * B; i++) {
  idPut(test, i * REC, i, 1);   // ai: under TOI 1 (0 is none: a receiver ignores it)
  test.set(streamBlock(i, PAYLOAD), i * REC + ID_BYTES);
}
test[(7 * B + 3) * REC + ID_BYTES + 100] ^= 0x40;
writeFileSync(`${OUT}/test.bin`, test);

// ai: The wasm's chain: liz_rx in memory, the stream in frames of 40 (as xfer_check cuts it), each frame's verdict summed;
// ai: the receiver's own counts (liz_progress, 80 bytes) at the end
const rb = M._malloc(B * REC), vp = M._malloc(20), pp = M._malloc(80);
function receive(bytes) {
  const rx = M._liz_rx_new(0);
  if (!rx) fail();
  const t = { frames: 0, seen: 0, bad: 0, test: 0, judged: 0, fresh: 0 }, n = bytes.length / REC;
  for (let f = 0; f * B < n; f++) {
    const k = Math.min(B, n - f * B);
    M.HEAPU8.set(bytes.subarray(f * B * REC, (f * B + k) * REC), rb);
    if (M._liz_rx_frame(rx, rb, k, vp) < 0) fail();
    const [seen, bad, judged, fresh, isTest] = M.HEAP32.subarray(vp >> 2, (vp >> 2) + 5);
    t.frames++; t.seen += seen; t.bad += bad; t.test += isTest; t.judged += judged; t.fresh += fresh;
  }
  M._liz_rx_progress(rx, pp);
  const dv = new DataView(M.HEAPU8.buffer, pp, 80), done = dv.getInt32(4, true) !== 0;
  let exact = false;
  if (done) {
    const dp = M._malloc(8);
    if (M._liz_rx_data(rx, dp, dp + 4) === 0) { const a = M.HEAPU32[dp >> 2], len = M.HEAPU32[(dp + 4) >> 2]; exact = len === file.length && Buffer.compare(Buffer.from(M.HEAPU8.subarray(a, a + len)), Buffer.from(file)) === 0; }
    M._free(dp);
  }
  const r = { ...t, forwarded: Number(dv.getBigUint64(56, true)), doneAt: done ? Number(dv.getBigUint64(64, true)) : null, done, verified: dv.getUint32(36, true), rejected: dv.getUint32(40, true),
    refused: dv.getUint32(72, true), over: dv.getUint32(76, true), fraction: dv.getFloat64(24, true), exact, root: M.UTF8ToString(M._liz_rx_meta(rx, 1)) };
  M._liz_rx_free(rx);
  return r;
}
// ai: The same stream with one data block of source block 1 wrong (a block that passed its CRC wrongly): the source block
// ai: decodes to other bytes, the file is refused at its root and collected again, and both chains must agree on
// ai: everything after.
const bad = stream.slice();
for (let i = 0; i < got.length; i++) if (toiGet(bad, i * REC) === txToi && idSbn(idGet(bad, i * REC)) === 1) { bad[i * REC + ID_BYTES + 200] ^= 0x01; break; }
writeFileSync(`${OUT}/bad.bin`, bad);
// ai: The same transfer under another name (src/xfer.h xfer_same_file: its sender's TOIs, the seed's, the
// ai: same): renamed.bin is stream.bin with every header block of its second half that sender's header; a receiver takes
// ai: it as the same transfer.
const other = sender("check-9MB-renamed.txt", txSeed);
let otherHeader = null;
for (let f = 0; !otherHeader && f < 10000; f++) {
  M._liz_tx_next(other, B, fb);
  for (let k = 0; k < B; k++) if (idGet(M.HEAPU8, fb + k * REC) === ID_HEADER) { otherHeader = M.HEAPU8.slice(fb + k * REC + ID_BYTES, fb + (k + 1) * REC); break; }
}
M._liz_tx_free(other);
const renamed = stream.slice();
// ai: from the frame xfer_check.cpp splits at (half the blocks, down to a frame of B), so its second receiver reads no
// ai: header of the first sender's
for (let i = Math.floor(got.length / 2 / B) * B; i < got.length; i++) if (idGet(renamed, i * REC) === ID_HEADER) renamed.set(otherHeader, i * REC + ID_BYTES);
writeFileSync(`${OUT}/renamed.bin`, renamed);
const ref = { blocks: got.length, cycle, join, sentFrames: sent.length, root: info.root, toi: txToi, file: receive(stream), bad: receive(bad), test: receive(test), renamed: receive(renamed) };
writeFileSync(`${OUT}/ref.json`, JSON.stringify(ref, null, 1));
console.log(JSON.stringify(ref));
