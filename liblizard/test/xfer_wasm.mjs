// The transfer's exports in the wasm (src/xfer.h, src/wasm.c): BLAKE3 against the official test
// vectors, then files of 41 chunks, 3, 1 and a block, random and prose, end to end through the exports and RaptorQ (in
// the same module, sim/fountain.mjs): the header and manifest blocks written and read back, each chunk's source
// symbols its own bytes, its decoder made from the header alone and fed repair symbols only, each recovered chunk
// verified, a corrupted
// chunk, a corrupted manifest, another transfer's manifest block and a header with a reserved byte set caught. Also the
// block size the layout assumes, and what hashing and the pack's zstd cost here. About 5 s.
//   node test/xfer_wasm.mjs [path/to/ob.mjs]      default build/ob.mjs
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { init as initRq, Encoder, Decoder } from "../sim/fountain.mjs";
import { init as initZstd, compress, decompress } from "../sim/zstd.mjs";

const M = await (await import(resolve(process.argv[2] ?? "build/ob.mjs"))).default();
await initRq(); await initZstd();
let checks = 0, fails = 0;
const check = (ok, what) => { checks++; if (!ok) { fails++; console.log("FAIL", what); } };
const hex = (a) => Buffer.from(a).toString("hex");
const put = (bytes) => { const p = M._malloc(Math.max(1, bytes.length)); M.HEAPU8.set(bytes, p); return p; };
const take = (p, n) => M.HEAPU8.slice(p, p + n);

const L = new Int32Array(13);
{ const p = M._malloc(4 * 13); M._xfer_layout(p); L.set(new Int32Array(M.HEAPU8.buffer, p, 13)); M._free(p); }
const [ID_BYTES, PAYLOAD, SYM_BITS, CHUNK_BITS, CONTROL, MAX_MANIFEST, PER_BLOCK, NAME_MAX, TYPE_MAX, , , MAX_CHUNKS, HASH] = L;
check(ID_BYTES === 5 && PAYLOAD === 471 && SYM_BITS === 24 && CHUNK_BITS === 16 && CONTROL === 0xffff && MAX_MANIFEST === 4682 && PER_BLOCK === 14 && TYPE_MAX === 158 && MAX_CHUNKS === 0xffff && HASH === 1, "layout");
check(M._lizard_setup(512, 128, 0) === ID_BYTES + PAYLOAD, "a LIZARD block is the id and the payload");
const ID_HEADER = 0xffff * 2 ** 24;
check(M._xfer_id_of(3, 7) === 3 * 2 ** 24 + 7 && M._xfer_id_of(65534, 2 ** 24 - 1) === 0xfffeffffff && M._xfer_kind_of(ID_HEADER) === 1 && M._xfer_kind_of(ID_HEADER + 5) === 2 &&
      M._xfer_kind_of(ID_HEADER + 4682) === 2 && M._xfer_kind_of(ID_HEADER + 4683) === -1 && M._xfer_kind_of(5) === 0, "ids");

const b3 = (bytes) => { const p = put(bytes), o = M._malloc(32); M._xfer_b3_hash(p, bytes.length, o); const h = take(o, 32); M._free(p); M._free(o); return h; };
const cvsOf = (file, log2) => {
  const C = 2 ** log2, n = Math.ceil(file.length / C), out = new Uint8Array(32 * n), o = M._malloc(32);
  for (let i = 0; i < n; i++) {
    const chunk = file.subarray(i * C, Math.min(file.length, (i + 1) * C)), p = put(chunk);
    check(M._xfer_chunk_cv(p, chunk.length, i, log2, o) === 0, `cv of chunk ${i}`);
    out.set(take(o, 32), 32 * i); M._free(p);
  }
  M._free(o);
  return out;
};
const rootOf = (cvs) => { const p = put(cvs), o = M._malloc(32); check(M._xfer_root(p, cvs.length / 32, o) === 0, "root"); const r = take(o, 32); M._free(p); M._free(o); return r; };

// The chunk size a sender picks (xfer_chunk_log2_for): 16 MiB, to a file of 65,535 chunks (about 1 TiB).
check(M._xfer_log2_for(0) === 24 && M._xfer_log2_for(65535 * 2 ** 22 + 1) === 24 && M._xfer_log2_for(65535 * 2 ** 24) === 24 && M._xfer_log2_for(65535 * 2 ** 24 + 1) === -1, "the chunk size by length");

// The official vectors: whole, as the root subtree, and split at 2^10 .. 2^17.
const vec = JSON.parse(readFileSync(new URL("../vendor/blake3/test_vectors/test_vectors.json", import.meta.url), "utf8"));
let splits = 0;
for (const { input_len: n, hash } of vec.cases) {
  const input = Uint8Array.from({ length: n }, (_, i) => i % 251), want = hash.slice(0, 64);
  check(hex(b3(input)) === want, `hash of ${n}`);
  const p = put(input), o = M._malloc(32);
  M._xfer_b3_sub(p, n, 0, 1, o);
  check(hex(take(o, 32)) === want, `root subtree of ${n}`);
  M._free(p); M._free(o);
  for (let log2 = 10; log2 <= 17; log2++) if (n > 2 ** log2) { check(hex(rootOf(cvsOf(input, log2))) === want, `${n} in chunks of 2^${log2}`); splits++; }
}
console.log(`official vectors: ${vec.cases.length} cases through the wasm, ${splits} of them split into chunks`);

let s = 99;
const rand = (n) => Uint8Array.from({ length: n }, () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16) & 255);
// Text-like bytes: words from a small vocabulary, no two chunks alike.
const prose = (n) => { const w = ["lizard ", "frame ", "block ", "ring ", "pilot ", "camera ", "screen ", "the ", "a ", "of ", "and ", "reads ", "paints ", "\n"]; const out = new Uint8Array(n); let k = 0; while (k < n) { const t = w[((s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16) % w.length]; for (let i = 0; i < t.length && k < n; i++) out[k++] = t.charCodeAt(i); } return out; };
const enc = new TextEncoder();

function transfer(label, file, log2, name, type) {
  const length = file.length, C = 2 ** log2, chunks = Math.ceil(length / C), lenOf = (i) => Math.min(C, length - i * C);
  // Sender: the chaining values and root; the header and manifest payloads.
  const cvs = cvsOf(file, log2), root = chunks > 1 ? rootOf(cvs) : b3(file);
  check(hex(root) === hex(b3(file)), `${label}: the root is the file's b3sum`);
  const nm = enc.encode(name), ty = enc.encode(type), pr = put(root), pn = put(nm), pt = put(ty), ph = M._malloc(PAYLOAD);
  check(M._xfer_hdr_write(ph, length, log2, pr, pn, nm.length, pt, ty.length) === 0, `${label}: header write`);
  const header = take(ph, PAYLOAD);
  const mcount = M._xfer_manifest_blocks(chunks), pc = put(cvs), manifest = [];
  for (let m = 0; m < mcount; m++) { check(M._xfer_manifest_write(pc, chunks, pr, m, ph) === 0, `${label}: manifest ${m}`); manifest.push(take(ph, PAYLOAD)); }

  // Receiver: nothing but the payloads.
  const sc = M._malloc(40), rr = M._malloc(32), rn = M._malloc(NAME_MAX), rt = M._malloc(TYPE_MAX), hp = M._malloc(PAYLOAD);
  M.HEAPU8.set(header, hp);
  check(M._xfer_hdr_parse(hp, sc, rr, rn, rt) === 0, `${label}: header parse`);
  const f = new Int32Array(M.HEAPU8.buffer, sc, 9).slice();
  const got = { hash: f[0], log2: f[1], length: (f[4] >>> 0) + (f[5] >>> 0) * 2 ** 32, chunks: f[6], mcount: f[7], name: new TextDecoder().decode(take(rn, f[2])), type: new TextDecoder().decode(take(rt, f[3])) };
  check(got.hash === HASH && got.log2 === log2 && got.length === length && got.chunks === chunks && got.mcount === mcount && got.name === name && got.type === type && hex(take(rr, 32)) === hex(root), `${label}: header fields`);
  // ai: the reserved bytes (461 to 470) zero, or the header is refused
  for (const at of [461, 466, 470]) {
    const odd = header.slice(); odd[at] = 1; M.HEAPU8.set(odd, ph);
    check(M._xfer_hdr_parse(ph, sc, 0, 0, 0) !== 0, `${label}: a header with reserved byte ${at} set refused`);
  }
  const list = M._malloc(32 * Math.max(1, chunks));
  for (let m = mcount - 1; m >= 0; m--) { M.HEAPU8.set(manifest[m], ph); check(M._xfer_mf_parse(ph, hp, m, list) === 0, `${label}: manifest ${m} parse`); }
  if (mcount) check(M._xfer_manifest_check(list, chunks, rr) === 0, `${label}: list against the root`);
  let past = 0;
  for (let i = 0; i < chunks; i++) {
    const bytes = lenOf(i), blocks = Math.ceil(bytes / PAYLOAD);
    const plain = file.subarray(i * C, i * C + bytes);
    // The source symbols are the chunk's own bytes, the last zero padded (RFC 6330's systematic property); the decoder
    // made from the header alone (the chunk's length), fed repair symbols only (ids from the source count up), so it
    // can only come back by solving; a chunk of one block too.
    const e = new Encoder(plain, PAYLOAD), d = new Decoder(bytes, PAYLOAD);
    const padded = new Uint8Array(blocks * PAYLOAD); padded.set(plain);
    let same = true;
    for (let k = 0; k < blocks && same; k++) same = hex(e.block(k)) === hex(padded.subarray(k * PAYLOAD, (k + 1) * PAYLOAD));
    check(same, `${label}: chunk ${i}'s source symbols are its bytes`);
    let id = blocks, done = false;
    while (!done && id < blocks * 2 + 20) done = d.add(id, e.block(id++));
    const out = done ? d.recover() : new Uint8Array(0);
    past = Math.max(past, id - 2 * blocks);
    e.free(); d.free();
    check(out.length === bytes, `${label}: chunk ${i} came back ${out.length} of ${bytes} bytes`);
    const pd = put(out);
    check(M._xfer_chunk_ok(hp, i, pd, out.length, mcount ? list : 0) === 0, `${label}: chunk ${i} verified`);
    if (i === Math.floor(chunks / 2)) {
      M.HEAPU8[pd + (out.length >> 1)] ^= 4;
      check(M._xfer_chunk_ok(hp, i, pd, out.length, mcount ? list : 0) !== 0, `${label}: corrupt chunk ${i} caught`);
    }
    M._free(pd);
  }
  if (mcount) {
    M.HEAPU8[list + 32 * (chunks - 1)] ^= 1;
    check(M._xfer_manifest_check(list, chunks, rr) !== 0, `${label}: corrupt manifest caught`);
    // A manifest block of another transfer (its tag, the root's first 8 bytes, another): refused.
    const bad = manifest[0].slice();
    bad[0] ^= 0xff;
    M.HEAPU8.set(bad, ph);
    check(M._xfer_mf_parse(ph, hp, 0, list) !== 0, `${label}: another transfer's manifest block refused`);
  }
  [pr, pn, pt, ph, pc, sc, rr, rn, rt, list, hp].forEach((p) => M._free(p));
  console.log(`${label}: ${length} B in ${chunks} chunk${chunks > 1 ? "s" : ""} of 2^${log2}, ${mcount} manifest block${mcount === 1 ? "" : "s"}, every chunk solved from repair symbols (the most past K: ${past})`);
}
transfer("3 chunks", rand(2 * 2 ** 20 + 304000), 20, "IMG_2041.jpg", "image/jpeg");
transfer("1 chunk", rand(700001), 20, "notes.txt", "text/plain");
transfer("41 chunks", rand(40 * 1024 + 100), 10, "", "");
transfer("3 chunks, prose", prose(2 * 2 ** 20 + 304000), 20, "notes.txt", "text/plain");
transfer("1 chunk, prose", prose(700001), 20, "notes.txt", "text/plain");
transfer("1 block", rand(400), 20, "tiny.bin", "");
transfer("1 chunk of 2 blocks", rand(PAYLOAD + 1), 20, "two.bin", "");
transfer("a last chunk of one block", rand(2 ** 16 + 100), 16, "end.bin", "");

// What a chunk's chaining value and the pack's zstd (a 4 MiB frame) cost here, for the pages to budget (a phone is slower).
{
  const n = 4 * 2 ** 20, p = put(rand(n)), o = M._malloc(32);
  const t = performance.now();
  for (let r = 0; r < 4; r++) M._xfer_chunk_cv(p, n, r, 22, o);
  const ms = (performance.now() - t) / 4;
  console.log(`chaining value of a 4 MiB chunk: ${ms.toFixed(1)} ms, ${(n / 1048576 / (ms / 1000)).toFixed(0)} MiB/s in this wasm on this machine`);
  M._free(p); M._free(o);
  const text = prose(n), t1 = performance.now(), z = compress(text), t2 = performance.now(), back = decompress(z, n), t3 = performance.now();
  check(back && back.every((v, i) => v === text[i]), "4 MiB of prose round trip");
  console.log(`zstd on 4 MiB of prose: ${(z.length / n * 100).toFixed(1)}% of the bytes, compress ${(t2 - t1).toFixed(0)} ms (${(n / 1048576 / ((t2 - t1) / 1000)).toFixed(0)} MiB/s), decompress ${(t3 - t2).toFixed(0)} ms`);
}
console.log(`${checks} checks, ${fails} failed`);
process.exit(fails ? 1 : 0);
