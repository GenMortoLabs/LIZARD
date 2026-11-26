// The transfer's exports in the wasm (src/xfer.h, src/wasm.c): BLAKE3 against the official test
// vectors, then files of 41 source blocks, 3, 1 and a block, random and prose, end to end through the exports and RaptorQ (in
// the same module, sim/fountain.mjs): the session header written and read back, each source block's source symbols its own
// bytes, its decoder made from the header alone and fed repair symbols only, each recovered source block's chaining value
// made and the values combined into the root (a one-source-block file checked against it at once), a corrupted source
// block, a wrong chaining value, another transfer's root and a header with a reserved byte set caught; the key rule
// (unsigned keys critical, negative ones elective). Also the block size the layout assumes, and what hashing and the
// pack's zstd cost here. About 5 s.
//   node test/xfer_wasm.mjs [path/to/ob.mjs]      default build/ob.mjs
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { init as initRq, Encoder, Decoder } from "../sim/fountain.mjs";
import { init as initZstd, compress, decompress } from "../sim/zstd.mjs";
// ai: two descriptions the same file (src/xfer.h xfer_same_file, through the wasm's _xfer_same_file_of)
const sameFile = (M, a, b) => {
  const pa = M._malloc(Math.max(1, a.length)), pb = M._malloc(Math.max(1, b.length));
  M.HEAPU8.set(a, pa); M.HEAPU8.set(b, pb);
  const r = M._xfer_same_file_of(pa, a.length, pb, b.length);
  M._free(pa); M._free(pb);
  return r === 1;
};

const M = await (await import(resolve(process.argv[2] ?? "build/ob.mjs"))).default();
await initRq(); await initZstd();
let checks = 0, fails = 0;
const check = (ok, what) => { checks++; if (!ok) { fails++; console.log("FAIL", what); } };
const hex = (a) => Buffer.from(a).toString("hex");
const put = (bytes) => { const p = M._malloc(Math.max(1, bytes.length)); M.HEAPU8.set(bytes, p); return p; };
const take = (p, n) => M.HEAPU8.slice(p, p + n);

const L = new Int32Array(14);
{ const p = M._malloc(4 * 14); M._xfer_layout(p); L.set(new Int32Array(M.HEAPU8.buffer, p, 14)); M._free(p); }
const [ID_BYTES, PAYLOAD, SYM_BITS, SBN_BITS, CONTROL, NAME_MAX, , , MAX_SOURCE_BLOCKS, HASH, TOI_BITS, ID_TOI, DESC_LONGEST, DESC_MAX] = L;
check(ID_BYTES === 12 && PAYLOAD === 460 && SYM_BITS === 24 && SBN_BITS === 16 && CONTROL === 0xffff && NAME_MAX === 255 &&
  MAX_SOURCE_BLOCKS === 0xffff && HASH === 1 && TOI_BITS === 48 && ID_TOI === 5 && DESC_LONGEST === 309 && DESC_MAX === 460, "layout");
// ai: a name past 255 bytes cut in its stem, its extension kept (src/xfer.h xfer_name_cut)
{ const nm = new TextEncoder().encode("n".repeat(296) + ".txt"), pn = put(nm), po = M._malloc(NAME_MAX), n = M._xfer_name_cut_of(pn, nm.length, po), got = new TextDecoder().decode(take(po, n));
  check(n === 255 && got === "n".repeat(251) + ".txt", "a 300-byte name cut in its stem, .txt kept"); [pn, po].forEach((q) => M._free(q)); }
// ai: a file's TOI its content's (src/xfer.h xfer_toi_of): the C test's vector
{ const r = Uint8Array.from({ length: 32 }, (_, i) => i), pr = M._malloc(32); M.HEAPU8.set(r, pr);
  check(M._xfer_toi_of_wasm(pr, 1000, 24) === 0x735b8ee29147, "the TOI's vector: root 00..1f, 1000 bytes, k 24 is 735b8ee29147"); M._free(pr); }
check(M._lizard_setup(512, 128, 0) === ID_BYTES + PAYLOAD, "a LIZARD block is the id and the payload");
const ID_HEADER = 0xffff * 2 ** 24;
check(M._xfer_id_of(3, 7) === 3 * 2 ** 24 + 7 && M._xfer_id_of(65534, 2 ** 24 - 1) === 0xfffeffffff && M._xfer_kind_of(ID_HEADER) === 1 && M._xfer_kind_of(ID_HEADER + 1) === -1 &&
      M._xfer_kind_of(ID_HEADER + 4683) === -1 && M._xfer_kind_of(5) === 0, "ids (every control id but the header's reserved)");

const b3 = (bytes) => { const p = put(bytes), o = M._malloc(32); M._xfer_b3_hash(p, bytes.length, o); const h = take(o, 32); M._free(p); M._free(o); return h; };
const cvsOf = (file, log2) => {
  const C = 2 ** log2, n = Math.ceil(file.length / C), out = new Uint8Array(32 * n), o = M._malloc(32);
  for (let i = 0; i < n; i++) {
    const sbn = file.subarray(i * C, Math.min(file.length, (i + 1) * C)), p = put(sbn);
    check(M._xfer_sb_cv(p, sbn.length, i, log2, o) === 0, `cv of source block ${i}`);
    out.set(take(o, 32), 32 * i); M._free(p);
  }
  M._free(o);
  return out;
};
const rootOf = (cvs) => { const p = put(cvs), o = M._malloc(32); check(M._xfer_root(p, cvs.length / 32, o) === 0, "root"); const r = take(o, 32); M._free(p); M._free(o); return r; };

// The source block size a sender picks (xfer_sb_log2_for): 16 MiB, to a file of 65,535 source blocks (about 1 TiB).
check(M._xfer_sb_log2_of(0) === 24 && M._xfer_sb_log2_of(65535 * 2 ** 22 + 1) === 24 && M._xfer_sb_log2_of(65535 * 2 ** 24) === 24 && M._xfer_sb_log2_of(65535 * 2 ** 24 + 1) === -1, "the source block size by length");

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
  for (let log2 = 10; log2 <= 17; log2++) if (n > 2 ** log2) { check(hex(rootOf(cvsOf(input, log2))) === want, `${n} in source blocks of 2^${log2}`); splits++; }
}
console.log(`official vectors: ${vec.cases.length} cases through the wasm, ${splits} of them split into source blocks`);

let s = 99;
const rand = (n) => Uint8Array.from({ length: n }, () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16) & 255);
// Text-like bytes: words from a small vocabulary, no two source blocks alike.
const prose = (n) => { const w = ["lizard ", "frame ", "block ", "ring ", "pilot ", "camera ", "screen ", "the ", "a ", "of ", "and ", "reads ", "paints ", "\n"]; const out = new Uint8Array(n); let k = 0; while (k < n) { const t = w[((s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16) % w.length]; for (let i = 0; i < t.length && k < n; i++) out[k++] = t.charCodeAt(i); } return out; };
const enc = new TextEncoder();

function transfer(label, file, log2, name) {
  const length = file.length, C = 2 ** log2, sourceBlocks = Math.ceil(length / C), lenOf = (i) => Math.min(C, length - i * C);
  // Sender: the chaining values and root (the BLAKE3 tree's), and the session header naming the file.
  const cvs = cvsOf(file, log2), root = sourceBlocks > 1 ? rootOf(cvs) : b3(file);
  check(hex(root) === hex(b3(file)), `${label}: the root is the file's b3sum`);
  const nm = enc.encode(name), pr = put(root), pn = put(nm), ph = M._malloc(PAYLOAD);
  const toi = M._xfer_toi_of_wasm(pr, length, log2), etoi = 0x7e57a07e57a0;   // ai: the file's TOI its content's; the session's drawn
  const zstd = sourceBlocks > 1 ? 1 : 0;
  // Sender: the file's description (its own CBOR map, src/xfer.h xfer_desc_write, 2026-11-22), then a session header
  // carrying it.
  const ep = M._malloc(DESC_LONGEST);
  const en = M._xfer_desc_write_of(ep, length, log2, pr, pn, nm.length, zstd), desc = take(ep, Math.max(0, en)), ext = desc;
  check(en > 0 && en <= DESC_LONGEST && desc[0] >> 5 === 5, `${label}: the description written, a CBOR map of ${en} bytes`);
  const px = put(ext);
  check(M._xfer_session_write_of(ph, px, ext.length) === 0, `${label}: session header write`);
  const header = take(ph, PAYLOAD);

  // Receiver: nothing but the payloads. The session header read, its description read under the session's TOI.
  const sc = M._malloc(44), rr = M._malloc(32), rn = M._malloc(NAME_MAX), hp = M._malloc(PAYLOAD), cp = M._malloc(4), tp = M._malloc(8);
  M.HEAPU8.set(header, hp);
  const s2 = (M._xfer_session_read(hp, sc), new Int32Array(M.HEAPU8.buffer, sc, 2).slice());
  check(s2[0] === 0 && s2[1] === ext.length, `${label}: the session header read: its description from 0, ${s2[1]} bytes`);
  const got_ext = header.subarray(s2[0], s2[0] + s2[1]), pe = put(got_ext);
  check(M._xfer_desc_parse(pe, got_ext.length, etoi, sc, rr, rn) === 0, `${label}: its description read under the session's TOI`);
  M._free(pe);
  const f = new Int32Array(M.HEAPU8.buffer, sc, 11).slice();
  const got = { hash: f[0], log2: f[1], length: (f[3] >>> 0) + (f[4] >>> 0) * 2 ** 32, sourceBlocks: f[5], name: new TextDecoder().decode(take(rn, f[2])), comp: f[7], toi: (f[9] >>> 0) + (f[10] >>> 0) * 2 ** 32 };
  check(got.hash === HASH && got.log2 === log2 && got.length === length && got.sourceBlocks === sourceBlocks && got.name === name && hex(take(rr, 32)) === hex(root) &&
    got.comp === zstd && got.toi === toi && f[8] === 0, `${label}: description fields, the TOI derived`);
  // ai: the bytes past the description ignored; an unknown critical key (unsigned, 2) flagged, an elective one
  // ai: (negative, -5) not (RFC 9581's rule); a description under its own file's TOI refused
  { const odd = header.slice(); odd[PAYLOAD - 1] = 1; M.HEAPU8.set(odd, hp);
    check(M._xfer_session_read(hp, sc) === 0 && new Int32Array(M.HEAPU8.buffer, sc, 2)[1] === ext.length, `${label}: the bytes past the description ignored`);
    const x = new Uint8Array(ext.length + 2); x.set(ext); x[0] = ext[0] + 1; x.set([0x02, 0xf5], ext.length);   // ai: {..., 2: true}
    const px2 = put(x);
    check(M._xfer_desc_parse(px2, x.length, etoi, sc, 0, 0) === 0 && new Int32Array(M.HEAPU8.buffer, sc, 11)[8] === 1, `${label}: an unknown critical key: flagged`);
    M._free(px2);
    const y = x.slice(); y.set([0x24, 0xf5], ext.length);   // ai: {..., -5: true}
    const py = put(y);
    check(M._xfer_desc_parse(py, y.length, etoi, sc, 0, 0) === 0 && new Int32Array(M.HEAPU8.buffer, sc, 11)[8] === 0, `${label}: an unknown elective key: skipped`);
    M._free(py); }
  { const pe2 = put(ext); check(M._xfer_desc_parse(pe2, ext.length, toi, sc, 0, 0) < 0, `${label}: a description under its own file's TOI refused`); M._free(pe2); }
  { const a = take(ep, en), b2 = a.slice(); check(sameFile(M, a, b2), `${label}: sameFile on itself`); }
  [px, cp, tp].forEach((q) => M._free(q));
  M.HEAPU8.set(desc, hp);
  // ai: each source block's chaining value as it decodes (two source blocks or more), the root checked at the last
  const list = M._malloc(32 * Math.max(1, sourceBlocks));
  let past = 0;
  for (let i = 0; i < sourceBlocks; i++) {
    const bytes = lenOf(i), blocks = Math.ceil(bytes / PAYLOAD);
    const plain = file.subarray(i * C, i * C + bytes);
    // The source symbols are the source block's own bytes, the last zero padded (RFC 6330's systematic property); the decoder
    // made from the header alone (the source block's length), fed repair symbols only (ids from the source count up), so it
    // can only come back by solving; a source block of one block too.
    const e = new Encoder(plain, PAYLOAD), d = new Decoder(bytes, PAYLOAD);
    const padded = new Uint8Array(blocks * PAYLOAD); padded.set(plain);
    let same = true;
    for (let k = 0; k < blocks && same; k++) same = hex(e.block(k)) === hex(padded.subarray(k * PAYLOAD, (k + 1) * PAYLOAD));
    check(same, `${label}: source block ${i}'s source symbols are its bytes`);
    let id = blocks, done = false;
    while (!done && id < blocks * 2 + 20) done = d.add(id, e.block(id++));
    const out = done ? d.recover() : new Uint8Array(0);
    past = Math.max(past, id - 2 * blocks);
    e.free(); d.free();
    check(out.length === bytes, `${label}: source block ${i} came back ${out.length} of ${bytes} bytes`);
    const pd = put(out);
    check(M._xfer_sb_ok(hp, desc.length, i, pd, out.length, sourceBlocks > 1 ? list + 32 * i : 0) === 0, `${label}: source block ${i} taken`);
    if (sourceBlocks === 1) {
      M.HEAPU8[pd + (out.length >> 1)] ^= 4;
      check(M._xfer_sb_ok(hp, desc.length, i, pd, out.length, 0) !== 0, `${label}: a corrupt source block of one caught at once`);
    }
    M._free(pd);
  }
  if (sourceBlocks > 1) {
    check(M._xfer_root_check(list, sourceBlocks, rr) === 0, `${label}: the chaining values make the root`);
    // ai: a source block decoded wrong: its chaining value another, the root not made (the file refused whole)
    const mid = Math.floor(sourceBlocks / 2);
    M.HEAPU8[list + 32 * mid] ^= 1;
    check(M._xfer_root_check(list, sourceBlocks, rr) !== 0, `${label}: a wrong source block caught at the root`);
    M.HEAPU8[list + 32 * mid] ^= 1;
    // ai: another transfer's root: refused
    const other = take(rr, 32); other[3] ^= 0x80; const po = put(other);
    check(M._xfer_root_check(list, sourceBlocks, po) !== 0, `${label}: the values against another root refused`);
    M._free(po);
  }
  [pr, pn, ph, sc, rr, rn, list, hp].forEach((p) => M._free(p));
  console.log(`${label}: ${length} B in ${sourceBlocks} source block${sourceBlocks > 1 ? "s" : ""} of 2^${log2}, every source block solved from repair symbols (the most past K: ${past})`);
}
transfer("3 source blocks", rand(2 * 2 ** 20 + 304000), 20, "IMG_2041.jpg");
transfer("1 source block", rand(700001), 20, "notes.txt");
transfer("41 source blocks", rand(40 * 1024 + 100), 10, "");
transfer("3 source blocks, prose", prose(2 * 2 ** 20 + 304000), 20, "notes.txt");
transfer("1 source block, prose", prose(700001), 20, "notes.txt");
transfer("1 block", rand(400), 20, "tiny.bin");
transfer("1 source block of 2 blocks", rand(PAYLOAD + 1), 20, "two.bin");
transfer("a last source block of one block", rand(2 ** 16 + 100), 16, "end.bin");

// What a source block's chaining value and the pack's zstd (a 4 MiB frame) cost here, for the pages to budget (a phone is slower).
{
  const n = 4 * 2 ** 20, p = put(rand(n)), o = M._malloc(32);
  const t = performance.now();
  for (let r = 0; r < 4; r++) M._xfer_sb_cv(p, n, r, 22, o);
  const ms = (performance.now() - t) / 4;
  console.log(`chaining value of a 4 MiB source block: ${ms.toFixed(1)} ms, ${(n / 1048576 / (ms / 1000)).toFixed(0)} MiB/s in this wasm on this machine`);
  M._free(p); M._free(o);
  const text = prose(n), t1 = performance.now(), z = compress(text), t2 = performance.now(), back = decompress(z, n), t3 = performance.now();
  check(back && back.every((v, i) => v === text[i]), "4 MiB of prose round trip");
  console.log(`zstd on 4 MiB of prose: ${(z.length / n * 100).toFixed(1)}% of the bytes, compress ${(t2 - t1).toFixed(0)} ms (${(n / 1048576 / ((t2 - t1) / 1000)).toFixed(0)} MiB/s), decompress ${(t3 - t2).toFixed(0)} ms`);
}
console.log(`${checks} checks, ${fails} failed`);
process.exit(fails ? 1 : 0);
