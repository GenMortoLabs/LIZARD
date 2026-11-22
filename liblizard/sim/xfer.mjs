// The transfer in the light for the rig (src/xfer.h holds the layouts and their reasons): the sender's carousel, and a
// receiver that puts a file back together from blocks alone, each source block checked against the file's BLAKE3 root.
//
// Pure JS. The codec's wasm (sim/ob.mjs init(), for BLAKE3 and the header) and RaptorQ (sim/fountain.mjs) are handed
// in, so sim/phy.mjs and the receiver page take the id test from here without loading either.
//
// THE BYTES (src/xfer.h): a source block's RaptorQ source block is the source block's own bytes; the
// transfer compresses nothing (a client compresses a file before it enters). Every source block's source symbols follow from
// the file's length. A receiver solves each source block once its K symbols are in, makes its chaining value, and at the
// last holds the values combined to the root (no manifest since 2026-11-21: the block's CRC-64 leaves nothing for a
// per-source-block check to find).
//
// THE SCHEDULE: every source block interleaved with every other, in proportion to its blocks, over the whole file (a cycle),
// and cycles repeated for as long as the sender runs, each with fresh repair ids. Not source blocks in turn, because a camera's
// loss is steady and bursty, not rare: LIZARD degrades instead of stopping, so at a distance the outer rings go and a
// frame brings a share of its blocks, and a hand that moves loses whole seconds. In turn, a source block gets a fixed budget
// a cycle (K plus a margin m, since nothing comes back to say it arrived); a receiver losing more than m / (1 + m) of
// its blocks finishes no source block in the first cycle, and the margin itself is airtime a receiver that loses nothing throws
// away. Interleaved, a file takes 1 / (1 - loss) cycles at any loss. Counted over 64 MB at 4 MiB, LIZARD-512, cycles of
// airtime from the join to the last source block (1.000 the least possible; STATUS.md has the table):
//                         interleaved   in turn m=0.1   m=0.25   m=0.5   ideal
//   nothing lost             1.001          1.099        1.246    1.492   1.000
//   joined 50% into a cycle    1.001          1.094        1.225    1.459   1.000
//   15% lost at random       1.184          2.183        1.249    1.495   1.176
//   outer 25% of rings       1.347          2.186        2.480    1.497   1.333
//   2 s lost every 10 s      1.271          2.126        3.743    2.977   1.250
// A late join costs neither carousel anything: in turn the source block being sent when the camera arrived comes round again
// at the end of its cycle, which is when the file would end anyway.
// What it costs: no source block completes before the end of the first cycle, so the receiver holds every source block's blocks at
// once. It keeps them on disk (the origin private file system, OPFS) where the browser has one, so its memory is a
// source block or two at a solve, not the file. The sender holds every source block's RaptorQ encoder (its intermediate symbols), about
// 1.02 times the file.
//
// Blocks are spread over a frame's slots by a shuffle drawn from the frame count: slot k of a LIZARD frame is ring k,
// the outer ones the first to go, and a plain round robin would pin source block c to the same slots whenever the source block count
// divides the frame's (every source block on a ring that a distant camera never reads). The control slots (the header) are
// shuffled with the data, so no one slot carries them: until 2026-10-10 they took slot 0, the lowest
// ring, thought the one read first and lost last, which under the rate profile carries the 6/7 code on the
// innermost sub-channel and for some contents never decodes, the header's among them.

// src/xfer.h, held against the codec's xfer_layout whenever a wasm is handed in. The receiver page has no codec and
// needs the id test anyway.
// A block's id is 96 bits: the bare id (16 bits of source block over 24 of symbol) in bytes 0 to 4, the 48-bit TOI in
// bytes 5 to 10, both little-endian, and a reserved byte (0). Here an id is the bare one, 40 bits: past JS's 32-bit
// operators, so a number taken apart by arithmetic (exact to 2^53), a list of ids a Float64Array; the TOI travels beside
// it (the sender's: its file's, its content's, or the session's for the session header, XferSender.toiFor), a number
// too (48 bits, exact), read and written apart (toiGet, idPut's TOI).
export const PAYLOAD = 460, ID_BYTES = 12, ESI_BITS = 24, SBN_BITS = 16, TOI_BITS = 48, CONTROL = 0xffff, MAX_SOURCE_BLOCKS = 0xffff;
const SYMBOLS = 2 ** ESI_BITS;
export const ID_HEADER = CONTROL * SYMBOLS, ESI_MASK = SYMBOLS - 1;
// ai: the TOI's first byte in the id (src/xfer.h XFER_ID_TOI); the session header's description one CBOR map from its
// ai: first byte, at most DESC_MAX bytes (its keys src/xfer.h's XFER_K_*); COMP_ZSTD the compression of the default
// ai: client's zstd stream (pack.mjs)
export const ID_TOI = 5, DESC_MAX = 460, NAME_MAX = 255, COMP_ZSTD = 1;
const TOI_SPAN = 2 ** 48;
const blocksOf = (len) => Math.ceil(len / PAYLOAD);
export const idSbn = (id) => Math.floor(id / SYMBOLS);
export const idEsi = (id) => id % SYMBOLS;
export const isControlId = (id) => idSbn(id) === CONTROL;
export const idOf = (sbn, symbol) => sbn * SYMBOLS + symbol;
// A block's bare id from its first five bytes and its TOI from the next six (little-endian); and both into them, the
// reserved byte 11 written 0.
export const idGet = (b, o = 0) => b[o] + b[o + 1] * 2 ** 8 + b[o + 2] * 2 ** 16 + b[o + 3] * 2 ** 24 + b[o + 4] * 2 ** 32;
export const toiGet = (b, o = 0) => {
  let t = 0;
  for (let i = 5; i >= 0; i--) t = t * 256 + b[o + ID_TOI + i];
  return t;
};
export const idPut = (b, o, id, toi = 0) => {
  for (let i = 0; i < 5; i++, id = Math.floor(id / 256)) b[o + i] = id % 256;
  for (let i = 0; i < 6; i++, toi = Math.floor(toi / 256)) b[o + ID_TOI + i] = toi % 256;
  b[o + 11] = 0;
};

// 16 MiB, the source block size a sender takes (src/xfer.h xfer_sb_log2_for, XFER_LOG2_DEFAULT): a file waits on its
// slowest source block, so the fewer and larger the source blocks the shorter its tail (src/xfer.h has the figures). The receiver holds
// one source block at a solve several times over (the blocks read back, the decoder of about 35 MB, the recovered bytes and
// their copy for the hash), about 100 MB at the peak of a solve where a 4 MiB source block took a quarter of it.
export const DEFAULT_LOG2 = 24;
// One control block (the session header) for this many data blocks (0.5% of the airtime), more often for a short file
// (eight a cycle at least), so a small file is not waiting on its header. 189 since 2026-11-21, the header's rate for a
// file of two source blocks or more before (a control every 63, the manifest's symbols on two of three): the manifest
// gone, its 1.1% of the airtime goes to data. A camera that joins meets the header within 3 frames at LIZARD-1024 (124
// blocks), 5 at LIZARD-512 (61), 19 at LIZARD-96 (11), the blocks before it held.
const CONTROL_EVERY = 189;
const REC = ID_BYTES + PAYLOAD;   // a stored block: its symbol, then its payload
const symAt = (b, o) => idGet(b, o);   // a stored block's symbol

function wasm(M) {
  const put = (bytes) => { const p = M._malloc(Math.max(1, bytes.length)); M.HEAPU8.set(bytes, p); return p; };
  const take = (p, n) => M.HEAPU8.slice(p, p + n);
  const p = M._malloc(4 * 14);
  M._xfer_layout(p);
  const L = new Int32Array(M.HEAPU8.buffer, p, 14).slice();
  M._free(p);
  if (L[0] !== ID_BYTES || L[1] !== PAYLOAD || L[2] !== ESI_BITS || L[3] !== SBN_BITS || L[4] !== CONTROL || L[5] !== NAME_MAX ||
      L[8] !== MAX_SOURCE_BLOCKS || L[9] !== 1 || L[10] !== TOI_BITS || L[11] !== ID_TOI || L[13] !== DESC_MAX)
    throw new Error(`src/xfer.h moved (layout ${L.join(",")}): sim/xfer.mjs is out of date`);
  const b3 = (bytes) => { const a = put(bytes), o = M._malloc(32); M._xfer_b3_hash(a, bytes.length, o); const h = take(o, 32); M._free(a); M._free(o); return h; };
  // ai: a file's TOI, its content's (src/xfer.h xfer_toi_of)
  const toi = (root, length, sbLog2) => { const a = put(root), t = M._xfer_toi_of_wasm(a, length, sbLog2); M._free(a); return t; };
  // ai: the name a description carries (src/xfer.h xfer_name_cut: one past 255 bytes cut in its stem, its extension kept)
  const nameCut = (name) => {
    const b = new TextEncoder().encode(name ?? ""), a = put(b), o = M._malloc(NAME_MAX), n = M._xfer_name_cut_of(a, b.length, o), r = take(o, n);
    M._free(a); M._free(o);
    return r;
  };
  return { put, take, b3, toi, nameCut, descLongest: L[12], logMin: L[6], logMax: L[7], log2For: (length) => M._xfer_sb_log2_of(length) };
}
// Whether two descriptions (their CBOR bytes) describe one transfer's blocks (src/xfer.h xfer_same_file: the form's
// source block size, the length and the root; not the TOI, the key's, nor the name or compression:
// one under the same TOI that differs in them alone is the same transfer).
export const sameFile = (M, a, b) => {
  const W = wasm(M), pa = W.put(a), pb = W.put(b), r = M._xfer_same_file_of(pa, a.length, pb, b.length);
  [pa, pb].forEach((p) => M._free(p));
  return r === 1;
};
export const hex = (a) => Array.from(a, (v) => v.toString(16).padStart(2, "0")).join("");
export const b64 = (a) => { let s = ""; for (const v of a) s += String.fromCharCode(v); return btoa(s); };
// BLAKE3 of a buffer through the vendored C, for a harness that has the codec's wasm to hand.
export const blake3 = (M, bytes) => wasm(M).b3(bytes);


// ai: The carousel's draws (core/tx/xfer_tx.cpp's): mix32, murmur3's finaliser, a word's bits mixed; xorshift32 on a
// ai: state that must not be 0; a random seed where a sender is given none.
const mix32 = (x) => {
  x = (x ^ (x >>> 16)) >>> 0; x = Math.imul(x, 0x85ebca6b) >>> 0; x = (x ^ (x >>> 13)) >>> 0; x = Math.imul(x, 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
};
const xorshift = (s) => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; return s >>> 0; };
const nonzero = (s) => (s ? s : 0x9e3779b9);
const randomSeed = () => globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
// ai: The session's TOI as the reference senders draw it (core/tx/xfer_tx.cpp sessionToi): from the send's seed,
// ai: mix32(seed ^ 0x3c6ef372) in its high 24 bits and mix32(seed ^ 0x165667b1) in its low 24, 0 taken as 1, and one past
// ai: the file's where it is that (a file's TOI is its content's, XferSender.toi).
export const sessionToi = (seed, avoid) => {
  let t = (mix32((seed ^ 0x3c6ef372) >>> 0) & 0xffffff) * 2 ** 24 + (mix32((seed ^ 0x165667b1) >>> 0) & 0xffffff);
  if (!t) t = 1;
  if (t === avoid) t = t % (TOI_SPAN - 1) + 1;
  return t;
};
// ai: A source block's symbols, pass by pass (2026-11-15; core/tx/xfer_tx.cpp esiOf, the same arithmetic): its v-th is the
// ai: (v mod 2^24)-th of pass floor(v / 2^24) under a keyed shuffle of the 2^24 ids (four Feistel rounds over two 12-bit
// ai: halves, the round function mix32), the pass's key drawn from the source block's (the seed's) and the pass: every id once
// ai: a pass, each pass in an order of its own, so no receiver meets the same ids at the same times twice (consecutive
// ai: ids from a start before: each pass repeated the last).
const passKey = (key, pass) => mix32((key ^ mix32((Math.imul(pass >>> 0, 0x85ebca6b) + 0x27d4eb2f) >>> 0)) >>> 0);
const shuffle24 = (key, k) => {
  let l = (k >>> 12) & 0xfff, r = k & 0xfff;
  for (let i = 0; i < 4; i++) { const t = r; r = (l ^ (mix32((key ^ Math.imul(i, 0x9e3779b9) ^ r) >>> 0) & 0xfff)) >>> 0; l = t; }
  return ((l << 12) | r) >>> 0;
};
export const esiOf = (key, v) => shuffle24(passKey(key, Math.floor(v / SYMBOLS)), v % SYMBOLS);

// Which source block the next data block is for, and its symbol. A round visits every source block of the most blocks once; a source block
// of fewer is visited in proportion to its blocks by an accumulator, plus 2 sqrt(K) visits a cycle and at least 64: a
// small source block's count of blocks received wanders more than a big one's, and the file waits on whichever source block is
// last. Counted (10 MB, 8.5 MB, 4.2 MB, 64 MB; 0, 15 and 30% lost) when only a last source block could be short, that costs a
// receiver that loses nothing at most 0.55% and is level with or ahead of plain proportion everywhere else (4.2 MB at
// 30% lost: 1.431 cycles against 1.472, the least being 1.429); 4 sqrt(K) + 64 cost 1.4% at 10 MB, which the rig
// measured. Only a last source block is short (no source block's blocks depend on its content); it takes the
// margin, 2 / sqrt(K) of its airtime at most (2.6% at 6,000 blocks).
// ai: Each round visits the source blocks in an order of its own (newRound: Fisher-Yates from the rounds' generator, rs), so no
// ai: place in a round holds a source block from one round to the next and no period of the carousel meets a periodic loss.
class Carousel {
  // key: each source block's key (its symbols' order, esiOf); cnt: the symbols each source block sent (a streaming sender's going
  // on from the frames sent while its source blocks were made); rs: the rounds' generator
  constructor(K, key, cnt, rs) {
    const C = K.length;
    this.K = K; this.C = C; this.key = key; this.cnt = cnt; this.i = 0; this.acc = new Uint32Array(C); this.rs = rs; this.perm = new Uint32Array(C);
    this.max = C ? K.reduce((a, b) => Math.max(a, b), 0) : 0;
    this.w = Uint32Array.from(K, (k) => (k >= this.max ? this.max : Math.min(this.max, Math.max(64, k + Math.ceil(2 * Math.sqrt(k))))));
    this.cycle = this.w.reduce((a, b) => a + b, 0);
    this.newRound();
  }
  newRound() {
    const p = this.perm;
    for (let c = 0; c < this.C; c++) p[c] = c;
    for (let i = this.C; i > 1; i--) { this.rs = xorshift(this.rs); const j = this.rs % i, t = p[i - 1]; p[i - 1] = p[j]; p[j] = t; }
  }
  next() {
    let c;
    for (;;) {
      if (this.i >= this.C) { this.i = 0; this.newRound(); }
      c = this.perm[this.i++];
      if ((this.acc[c] += this.w[c]) >= this.max) { this.acc[c] -= this.max; break; }
    }
    return idOf(c, esiOf(this.key[c], this.cnt[c]++));   // past 2^24 a source block, a pass in a new order (esiOf)
  }
}

// A source block's chaining value from its own bytes (BLAKE3's wide hash): a streaming sender's first pass, which makes the root.
export function sbCv(M, plain, c, sbLog2) {
  const W = wasm(M), p = W.put(plain), o = M._malloc(32), rc = M._xfer_sb_cv(p, plain.length, c, sbLog2, o), v = W.take(o, 32);
  M._free(p); M._free(o);
  if (rc) throw new Error(`chaining value of chunk ${c}`);
  return v;
}

// One source block made, the sender's step for each (and a pool's worker's: lizard-web/xfer-pool.mjs): a RaptorQ encoder over
// its bytes (a source block of one block too); its chaining value where the file has two source blocks or more (cv).
export function makeSourceBlock(plain, c, sbLog2, cv, { M, Encoder }) {
  return { K: blocksOf(plain.length), enc: new Encoder(plain, PAYLOAD), cv: cv ? sbCv(M, plain, c, sbLog2) : null };
}

export class XferSender {
  // bytes: the file. M: the codec's wasm module. Encoder: sim/fountain.mjs's, its init() already awaited. sbLog2: the
  // source block size's log2, xfer_sb_log2_for's for the length unless given. Every source block made here, before the
  // constructor returns; or elsewhere (a pool's, lizard-web/xfer-pool.mjs; two source blocks or more, bytes null and length
  // given): source blocks, every source block made (makeSourceBlock's results), or cvs, every source block's chaining value, the source blocks then coming
  // one at a time (add) while frames go. Their fountains held by remote: its fetch(ids) gives a Map of id to block for
  // the ids of those source blocks, its free() lets them go.
  // STREAMING (2026-11-06; the native sender's, core/tx/xfer_tx.h): with cvs the root and the header are made at once; until the last source block is added a frame carries the header (where it has more than one
  // slot, or nothing is in yet, or every other frame of one) and the source blocks added so far in turn, fresh symbols each
  // (warm); then the carousel, the symbols going on. A receiver solves each source block as its symbols come in.
  // SEED (2026-11-14, the native sender's): each source block's first symbol, each round's order of the source blocks, each frame's
  // shuffle and the session's TOI (sessionToi) drawn from `seed` (random where none is given; reseed before the first
  // frame), so no send repeats another's ids or their places, and a send repeated with its seed is the same session.
  // The file's TOI is its content's whatever the seed: every sender's blocks of the same file are one object's.
  // `carouselSeed` the one in use.
  // COMPRESSION (2026-11-22; "encoding" from 2026-11-20): the bytes' compression, an attribute of the description
  // (COMP_ZSTD: the default client's zstd stream, pack.mjs; the name the file's own). Every send is one file (several
  // go as one tar): its description, the file's one CBOR map, in its session header.
  constructor(bytes, { name = "", compression = 0, sbLog2 = null, M, Encoder, sourceBlocks = null, cvs = null, length = bytes?.length ?? 0, remote = null, seed = null }) {
    const W = wasm(M);
    this.compression = compression; this.desc = null;
    sbLog2 ??= W.log2For(length);
    if (sbLog2 < 0) throw new Error(`${length} B is more than the transfer carries (16,383 chunks of 16 MiB)`);
    const size = 2 ** sbLog2;
    if (!(sbLog2 >= W.logMin && sbLog2 <= W.logMax)) throw new Error(`chunk size 2^${sbLog2} is outside 2^${W.logMin} to 2^${W.logMax}`);
    const C = Math.ceil(length / size);
    if (C > MAX_SOURCE_BLOCKS) throw new Error(`${length} B is ${C} chunks of 2^${sbLog2}; the id carries ${MAX_SOURCE_BLOCKS}`);
    if ((sourceBlocks || cvs) && C < 2) throw new Error("chunks made elsewhere: two or more");
    if ((sourceBlocks && sourceBlocks.length !== C) || (cvs && cvs.length !== 32 * C)) throw new Error("chunks made elsewhere: one a chunk");
    this.M = M; this.W = W;
    this.sourceBlocks = C; this.sbLog2 = sbLog2; this.size = size; this.length = length; this.Encoder = Encoder;
    this.len = Array.from({ length: C }, (_, c) => Math.min(size, length - c * size));
    this.K = new Uint32Array(C); this.enc = new Array(C).fill(null);
    this.ready = new Uint8Array(C); this.made = 0; this.cvs = C >= 2 ? new Uint8Array(32 * C) : null; this.remote = remote;
    let ks = null;
    if (cvs) this.cvs.set(cvs);
    else {
      ks = sourceBlocks ?? Array.from({ length: C }, (_, c) => makeSourceBlock(bytes.subarray(c * size, c * size + this.len[c]), c, sbLog2, C >= 2, { M, Encoder }));
      if (this.cvs) ks.forEach((k, c) => this.cvs.set(k.cv, 32 * c));
    }
    // The source blocks' chaining values' root (b3sum's answer) and the header.
    let root;
    const o = M._malloc(32);
    if (C >= 2) {
      const pc = W.put(this.cvs);
      if (M._xfer_root(pc, C, o)) throw new Error("root");
      root = W.take(o, 32);
      M._free(pc);
    } else root = W.b3(bytes);
    M._free(o);
    this.root = root;
    // ai: the name a description carries (src/xfer.h xfer_name_cut; core/tx/xfer_tx.cpp the same): a file alone's
    // ai: description always fits its session header
    this.nm = W.nameCut(name);
    this.hp = { root: W.put(root), out: M._malloc(PAYLOAD) };
    this.carousel = null; this.key = new Uint32Array(C); this.cnt = new Float64Array(C); this.since = Infinity; this.frame = 0; this.turn = 0;
    this.seedWith(seed ?? randomSeed());
    if (ks) ks.forEach((k, c) => this.add(c, k));
    if (!C) this.schedule();
  }
  // Source block c made (makeSourceBlock's result; a pool's has no enc, its fountain remote's): its slots, and with the last source block
  // the carousel.
  add(c, k) {
    if (!this.hp) return;   // let go (free) while a pool still made it
    this.K[c] = k.K; this.enc[c] = k.enc ?? null;
    this.ready[c] = 1; this.made++;
    if (this.made === this.sourceBlocks) this.schedule();
  }
  // ai: The session header (src/xfer.h): the file's description (its own map), under the session's TOI (the seed's,
  // ai: sessionToi). A new seed's rewrites it.
  writeHeader() {
    this.desc?.free();
    this.desc = new Description(this.description(), { M: this.M, toi: sessionToi(this.carouselSeed, this.toi) });
    this.header = this.desc.header;
  }
  // ai: the file's description (src/xfer.h xfer_desc_write: k, length, root, name and compression; its TOI derived)
  description() {
    const M = this.M, W = this.W, pn = W.put(this.nm), out = M._malloc(W.descLongest);
    const n = M._xfer_desc_write_of(out, this.length, this.sbLog2, this.hp.root, pn, this.nm.length, this.compression);
    const e = n > 0 ? W.take(out, n) : null;
    [pn, out].forEach((p) => M._free(p));
    if (!e) throw new Error(`${hex(this.root)}: no description`);
    return e;
  }

  // ai: which TOI an id of this sender's lists goes under (the session header its description's, every other the file's)
  toiFor(id) { return this.desc && id === ID_HEADER ? this.desc.toi : this.toi; }
  // The carousel drawn from seed in place of the one given or drawn; before the first frame only (false after).
  reseed(seed) {
    if (this.frame) return false;
    this.seedWith(seed);
    return true;
  }
  // ai: source block c's key mix32(seed ^ (c + 1) 0x9e3779b1) (its symbols' order, esiOf); the rounds' generator mix32(seed ^ 0x5bd1e995); each frame's shuffle from (frame + 1) 0x9e3779b1 ^ seed;
  // ai: the file's TOI its content's (src/xfer.h xfer_toi_of), whatever the seed; the session's from the seed (writeHeader)
  seedWith(seed) {
    this.carouselSeed = seed >>> 0;
    for (let c = 0; c < this.sourceBlocks; c++) this.key[c] = mix32((this.carouselSeed ^ Math.imul(c + 1, 0x9e3779b1)) >>> 0);
    this.rs = nonzero(mix32((this.carouselSeed ^ 0x5bd1e995) >>> 0));
    if (this.carousel) { this.carousel.rs = this.rs; this.carousel.i = 0; this.carousel.newRound(); }
    this.toi = this.W.toi(this.root, this.length, this.sbLog2);
    this.writeHeader();
  }
  // The carousel, once every source block is in; the session header on a control slot about every eighth of a cycle (at
  // most CONTROL_EVERY data blocks apart).
  schedule() {
    this.carousel = new Carousel(this.K, this.key, this.cnt, this.rs);
    this.every = Math.max(1, Math.min(CONTROL_EVERY, Math.floor(this.carousel.cycle / 8)));
    this.since = Infinity;
  }
  // cycle 0 until every source block is in.
  info() { return { header: b64(this.header ?? new Uint8Array(0)), descToi: this.desc?.toi ?? 0, sourceBlocks: this.sourceBlocks, sbLog2: this.sbLog2, toi: this.toi, root: hex(this.root), cycle: this.carousel?.cycle ?? 0, length: this.length, seed: this.carouselSeed }; }

  // The ids of the next frame of B blocks (each one's TOI toiFor's). Advances the carousel, so a frame asked for
  // twice is two frames.
  frameIds(B, out = new Float64Array(B)) {
    if (!this.sourceBlocks) { out.fill(ID_HEADER); this.frame++; return out; }   // an empty file is its header
    if (!this.carousel) this.warm(B, out);
    else {
      // ai: A control slot every P frames, P odd (core/tx/xfer_tx.cpp frameIds): the frames that carry `every` data
      // ai: blocks between controls, rounded up to an odd count, each control slot the header, so the header's frames are
      // ai: an odd count apart: a camera taking every other frame, in either phase, meets the header.
      let first = 0;
      const per = Math.max(1, B - 1), P = Math.floor((this.every + per - 1) / per) | 1;
      if ((this.since === Infinity || ++this.since >= P) && B > 1) {
        out[first++] = ID_HEADER;
        this.since = 0;
      }
      for (let k = first; k < B; k++) out[k] = this.carousel.next();
    }
    // Fisher-Yates over every slot, the control block's too, from a generator seeded by the frame count and the seed
    // (the native sender's, core/tx/xfer_tx.cpp, byte for byte).
    let s = nonzero(mix32((Math.imul(this.frame + 1, 0x9e3779b1) ^ this.carouselSeed) >>> 0));
    for (let i = B - 1; i > 0; i--) {
      s = xorshift(s);
      const j = s % (i + 1), t = out[i];
      out[i] = out[j]; out[j] = t;
    }
    this.frame++;
    return out;
  }
  // A frame before every source block is in: the header (where it has more than one slot, or nothing is in yet, or every other
  // frame of one), the rest the source blocks in so far in turn, each its next symbol; with none in, every slot the header
  // (as the native sender's: core/tx/xfer_tx.cpp warm).
  warm(B, out) {
    const ready = [];
    for (let c = 0; c < this.sourceBlocks; c++) if (this.ready[c]) ready.push(c);
    let k = 0;
    if (B > 1 || !ready.length || !(this.frame & 1)) out[k++] = ID_HEADER;
    for (; k < B; k++) {
      if (!ready.length) { out[k] = ID_HEADER; continue; }
      const c = ready[this.turn++ % ready.length];
      out[k] = idOf(c, esiOf(this.key[c], this.cnt[c]++));
    }
  }

  // What block `id` carries (an id of this sender's lists): 460 bytes into out.
  block(id, out) {
    const c = idSbn(id), s = idEsi(id);
    if (c === CONTROL) {
      if (s !== 0 || !this.header) { out.fill(0); return out; }
      out.set(this.header);
      return out;
    }
    if (this.enc[c]) return this.enc[c].block(s, out);
    throw new Error(`block ${id}: its chunk's fountain is remote's (fetch it)`);
  }
  free() {
    this.desc?.free(); this.desc = null;
    for (const e of this.enc) e?.free();
    this.enc = []; this.remote?.free(); this.remote = null;
    if (this.hp) { for (const p of Object.values(this.hp)) this.M._free(p); this.hp = null; }
  }
}

// A session's description (src/xfer.h, SPEC 7.7 and 7.11; core/tx/xfer_tx.cpp XferDesc): the send's one file's CBOR map
// (no session-wide keys yet), in the session header, read from one block. Under the session's TOI (the sender's,
// sessionToi); the session header written once, here.
export class Description {
  constructor(desc, { M, toi }) {
    const W = wasm(M);
    this.toi = toi;
    this.bytes = desc;
    const ph = M._malloc(PAYLOAD), pd = W.put(desc), rc = M._xfer_session_write_of(ph, pd, desc.length);
    this.header = W.take(ph, PAYLOAD);
    [ph, pd].forEach((p) => M._free(p));
    if (rc) throw new Error("the session header refused");
  }
  // ai: what a bare id under its TOI carries: the session header (other ids nothing)
  block(id, out) {
    if (idSbn(id) === CONTROL && !idEsi(id)) out.set(this.header); else out.fill(0);
    return out;
  }
  free() {}
}


// The distinct values of 24-bit symbols (the native receiver's radix24 and its count): two stable radix passes of 12 bits.
function distinctOf(ids) {
  let a = Uint32Array.from(ids, (x) => x & ESI_MASK), t = new Uint32Array(a.length);
  for (let shift = 0; shift < 24; shift += 12) {
    const cnt = new Uint32Array(4097);
    for (let k = 0; k < a.length; k++) cnt[((a[k] >>> shift) & 0xfff) + 1]++;
    for (let b = 0; b < 4096; b++) cnt[b + 1] += cnt[b];
    for (let k = 0; k < a.length; k++) t[cnt[(a[k] >>> shift) & 0xfff]++] = a[k];
    [a, t] = [t, a];
  }
  let n = a.length ? 1 : 0;
  for (let k = 1; k < a.length; k++) if (a[k] !== a[k - 1]) n++;
  return n;
}

// A source block keeps no set of the symbols it holds (2026-11-15): a send's ids are each pass of a source block's 2^24 in an order
// of its own (esiOf), so they never repeat within a send; a camera's repeats are the page's dedupe; the rare repeat
// across sends (K / 2^24 of a source block's) is stored as it came and dropped by the decoder at the source block's solve. Its refused
// symbols (a rejected source block's) are a Set: rare. The dense runs of bits held until then (Bits, 2026-11-14) would have
// grown to 2 MiB a source block over shuffled ids. The native receiver (core/rx/xfer_rx_core.cpp) the same.

// Blocks and the finished file, held in memory: the fallback where there is no origin private file system. The
// fountain worker (lizard-web/fountain-worker.mjs) has the OPFS one with the same calls.
// ai: A store that keeps transfers (keeps; 2026-10-22, the OPFS one) holds each in a folder of its own (its id the
// ai: root's 16 hex digits and the TOI's 12, folderOf) and resumes one when its header comes again (open, the notes,
// ai: park, forget: liblizard/core/rx/store.h's). This one keeps one
// ai: transfer at a time, as before: open is begin, park is clear, the notes are nothing.
export class MemoryStore {
  constructor() { this.kind = "memory"; this.keeps = false; this.recs = new Map(); this.out = null; }
  async clear() { this.recs.clear(); this.out = null; }
  async begin(length) { this.out = new Uint8Array(length); }
  async open(id, header, length) { await this.begin(length); return { found: false }; }
  async noteVerified() {}
  async forgetVerified() {}
  async noteMeta() {}
  async park() { await this.clear(); }
  async forget() {}
  async append(c, esi, bytes) {
    let r = this.recs.get(c);
    if (!r) this.recs.set(c, (r = { buf: new Uint8Array(REC * 256), n: 0 }));
    if ((r.n + 1) * REC > r.buf.length) { const b = new Uint8Array(2 * r.buf.length); b.set(r.buf); r.buf = b; }
    const o = r.n++ * REC;
    idPut(r.buf, o, esi);
    r.buf.set(bytes.subarray(0, PAYLOAD), o + ID_BYTES);
  }
  async read(c) { const r = this.recs.get(c); return r ? r.buf.subarray(0, r.n * REC) : new Uint8Array(0); }
  // ai: the records' symbols alone (the native store's ids, store.h): in memory, read off the records
  async ids(c) { const r = this.recs.get(c), v = new Uint32Array(r ? r.n : 0); for (let k = 0; k < v.length; k++) v[k] = symAt(r.buf, k * REC); return v; }
  async drop(c) { this.recs.delete(c); }
  async writeOut(off, bytes) { this.out.set(bytes, off); }
  async finish() { return { bytes: this.out }; }
  held() { let t = this.out?.length ?? 0; for (const r of this.recs.values()) t += r.buf.length; return t; }
}

// ai: meta.json written again every this many blocks stored (about 1 MB) or a second of storing, whichever comes first,
// ai: as well as at each source block verified and at a park, so a page closed mid-source block leaves the list a line at most that
// ai: far behind (liblizard/core/rx/xfer_rx_core.cpp)
const META_EVERY = 2048, META_MS = 1000;
const COLLECTING = 0, HOT = 1, VERIFIED = 3;
// ai: a transfer's folder in a store (core/rx/xfer_rx_core.cpp keyOf): its root's 16 hex digits and its TOI's 6, so a
// ai: file's TOI being its content's, the same file from any sender is one folder
export const folderOf = (rootHex, toi) => rootHex.slice(0, 16) + toi.toString(16).padStart(12, "0");
const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
// ai: The newest data blocks of TOIs no transfer holds yet, kept for a header of theirs (core/rx/xfer_rx_core.cpp
// ai: HOLD_BLOCKS; lizard-web/fountain-worker.mjs held them itself until 2026-11-19): a camera that joins mid-send meets
// ai: data before its header.
const HOLD_BLOCKS = 1024;

// One transfer a receiver holds: a file (or an extended session header, src/xfer.h) from its description on, each source block checked against its
// root, its blocks in its own store.
class TransportObject {
  constructor(rx, store) {
    this.rx = rx; this.M = rx.M; this.W = rx.W; this.Decoder = rx.Decoder; this.store = store;
    this.hdr = null;
    this.sbs = new Map(); this.sinceMeta = 0; this.metaMs = 0;
    this.verified = 0; this.rejected = 0; this.over = 0; this.ms = 0; this.done = false; this.result = null;
  }
  // ai: h: the parsed header (XferReceiver.parse). Its folder resumed or begun; true where that finished it.
  async open(h) {
    const W = this.W;
    this.hdr = h;
    const before = await this.store.open(folderOf(hex(W.take(h.root, 32)), h.toi), h.bytes, h.length);
    if (!h.sourceBlocks) {   // an empty file: its root is BLAKE3 of nothing, which is all there is to check
      if (eq(W.b3(new Uint8Array(0)), W.take(h.root, 32))) { await this.finish(); return true; }
      this.rx.refused++;
      return false;
    }
    return !!(before?.found && (await this.resume(before)));
  }
  // ai: The transfer parked for its header to come again (keep; a store that keeps no transfers lets it go), or let go
  // ai: whole (liblizard/core/rx/xfer_rx_core.cpp close).
  async close(keep = false) {
    const M = this.M;
    if (keep && this.hdr && !this.done && this.store.keeps) await this.saveMeta();
    for (const s of this.sbs.values()) s.dec?.free();
    if (this.hdr) [this.hdr.p, this.hdr.root, this.hdr.cvs].forEach((p) => M._free(p));
    this.hdr = null; this.sbs.clear();
    if (keep) await this.store.park(); else await this.store.clear();
  }
  // ai: The transfer's line for a page's list of unfinished transfers (the store's meta.json).
  async saveMeta() {
    this.sinceMeta = 0; this.metaMs = performance.now();
    const h = this.hdr;
    if (!h || !this.store.keeps) return;
    const p = this.progress();
    await this.store.noteMeta(JSON.stringify({ name: h.name, compression: h.compression, length: h.length, sourceBlocks: h.sourceBlocks, verified: this.verified, root: p.header.root,
      fraction: Math.round(fractionDone(p) * 1e4) / 1e4, bytesIn: this.bytesIn() }));
  }

  state(c) {
    let s = this.sbs.get(c);
    // ai: recs: the blocks the store holds for the source block, every one this transfer's by its TOI
    // ai: dups: of its records, those its last solve found repeating an earlier one's symbol (a repeat across sends)
    if (!s) this.sbs.set(c, (s = { state: COLLECTING, dec: null, recs: 0, dups: 0 }));
    return s;
  }

  // ai: a data block of this transfer's TOI: source block c, its symbol esi
  async add(c, esi, bytes) { return this.addSym(c, esi, bytes); }
  // ai: one symbol of source block c (PAYLOAD bytes)
  async addSym(c, esi, bytes) {
    const h = this.hdr;
    if (this.done || !h || c >= h.sourceBlocks) return;
    const s = this.state(c);
    if (s.state === VERIFIED) return;
    if (s.dec) {
      // ai: kept until the source block is verified, by any store: a resume makes the decoder again from them
      await this.store.append(c, esi, bytes); s.recs++;
      if (s.dec.add(esi, bytes)) await this.recovered(c, s.dec);
      return;
    }
    await this.store.append(c, esi, bytes);
    s.recs++;
    if (++this.sinceMeta >= META_EVERY || performance.now() - this.metaMs >= META_MS) await this.saveMeta();
    if (this.ready() && s.recs >= h.K[c] + s.dups) await this.solve(c);
  }

  // ai: whether a source block may be solved: once the header is in (each source block is checked as it decodes, the
  // ai: file at its last)
  ready() { return !!this.hdr; }
  // ai: A folder of this transfer from before: its verified source blocks taken as verified, their chaining values
  // ai: back (b.cvs), and each collecting source block's symbols marked had from its stored blocks; true when that
  // ai: finished it.
  async resume(b) {
    const h = this.hdr;
    for (const c of b.verified) if (c < h.sourceBlocks && this.state(c).state !== VERIFIED) {
      const s = this.state(c); s.state = VERIFIED; this.verified++;
      if (h.sourceBlocks >= 2 && b.cvs?.get(c)?.length === 32) this.M.HEAPU8.set(b.cvs.get(c), h.cvs + 32 * c);
    }
    for (const c of b.sourceBlocks) {
      if (c >= h.sourceBlocks || this.state(c).state === VERIFIED) { await this.store.drop(c); continue; }
      const s = this.state(c), recs = await this.store.read(c);
      s.recs = recs.length / REC;
    }
    if (this.verified === h.sourceBlocks) { await this.complete(); return this.done; }
    return false;
  }
  // Every source block with its blocks in hand, now that their counts are known.
  async solveReady() {
    const h = this.hdr;
    for (const [c, s] of [...this.sbs]) if (!this.done && s.state === COLLECTING && s.recs >= h.K[c] + s.dups) await this.solve(c);
  }

  async solve(c) {
    this.rx.onSolve?.();   // ai: the page told before the work (the fountain worker's report): the source block in hand, solving
    const t = performance.now();
    await this.solveOnce(c);
    this.ms += performance.now() - t;   // interleaved, most of this falls after the last block: it is the transfer's tail
  }
  async solveOnce(c) {
    const s = this.sbs.get(c);
    // ai: fewer than K symbols once repeats are dropped: no decoder (it would hold the source block's symbols, and near a
    // ai: resumed transfer's end every source block's at once); the source block collects that many more (the native receiver's solve).
    // ai: Counted from the store's ids where it keeps them apart (the memory store's; the OPFS store reads its records:
    // ai: a file of ids a source block would double its open handles, one a source block held for the transfer)
    let recs = null, ids;
    if (this.store.ids) ids = await this.store.ids(c);
    else {
      recs = await this.store.read(c);
      ids = new Uint32Array(recs.length / REC);
      for (let k = 0; k < ids.length; k++) ids[k] = symAt(recs, k * REC);
    }
    const distinct = distinctOf(ids);
    s.dups = ids.length - distinct;
    if (distinct < this.hdr.K[c]) return;
    recs ??= await this.store.read(c);
    const r = this.feed(c, recs);
    // ai: RaptorQ wanted a block more than K (about 1% of source blocks): the blocks fed as they come, and kept until the source block
    // ai: is verified (add)
    if (!r.done) { s.dec = r.d; s.state = HOT; this.over++; return; }
    await this.store.drop(c); s.recs = s.dups = 0;
    await this.recovered(c, r.d);
  }
  // ai: A fresh decoder over a source block's stored blocks, in the order they came: { d, done }.
  feed(c, recs) {
    const h = this.hdr, d = new this.Decoder(h.len[c], PAYLOAD);
    for (let o = 0; o + REC <= recs.length; o += REC) if (d.add(symAt(recs, o), recs.subarray(o + ID_BYTES, o + REC))) return { d, done: true };
    return { d, done: false };
  }
  // recs: the records that went into it where the store has let them go (a solve's), else the store's (a fed-on source block's)
  async recovered(c, d) {
    const s = this.sbs.get(c), out = d.recover();
    d.free(); s.dec = null;
    await this.check(c, out);
  }
  async check(c, out) {
    const h = this.hdr, s = this.sbs.get(c);
    if (this.verify(c, out)) {
      await this.store.writeOut(c * h.size, out);
      await this.store.noteVerified(c, h.sourceBlocks >= 2 ? this.W.take(h.cvs + 32 * c, 32) : null);
      await this.accept(c, s);
    } else await this.reject(c, s);
  }
  // xfer_sb_check as the source block decodes: its length, and for a file of one source block its hash against the root;
  // for two or more its chaining value into the transfer's list (cvs), for the root at the last.
  verify(c, out) {
    const M = this.M, h = this.hdr, p = this.W.put(out), ok = M._xfer_sb_ok(h.p, h.bytes.length, c, p, out.length, h.sourceBlocks >= 2 ? h.cvs + 32 * c : 0) === 0;
    M._free(p);
    return ok;
  }
  async accept(c, s) {
    s.state = VERIFIED; this.verified++;
    await this.store.drop(c); s.recs = s.dups = 0;   // ai: a kept source block's blocks (store keeps), nothing otherwise
    if (this.verified === this.hdr.sourceBlocks) await this.complete(); else await this.saveMeta();
  }
  // The last source block in: a file of one was checked against the root as it decoded; of two or more, the chaining
  // values are combined and held to the root (src/xfer.h xfer_root_check). A file that fails (only a broken sender
  // makes one: every block passed its CRC-64) is refused whole, every source block collected again from the blocks to
  // come: which one was wrong cannot be told, and with no back channel knowing would buy nothing.
  async complete() {
    const h = this.hdr, M = this.M;
    if (h.sourceBlocks >= 2 && M._xfer_root_check(h.cvs, h.sourceBlocks, h.root) !== 0) {
      this.rejected += h.sourceBlocks; this.verified = 0;
      for (const s of this.sbs.values()) s.dec?.free();
      this.sbs.clear();
      await this.store.forgetVerified();
      return;
    }
    await this.finish();
  }
  // A source block that decoded to the wrong bytes: only a broken sender makes one (a wrong block past the CRC-64 is out
  // of reach). Which block was wrong cannot be told, so the symbols that went into it are dropped and the source block
  // is collected again from the sender's next cycles. No list of refused symbols is kept (2026-11-21): a sender outside
  // the model is not defended against beyond memory safety.
  async reject(c, s) {
    s.state = COLLECTING; this.rejected++;
    if (s.dec) { s.dec.free(); s.dec = null; }
    await this.store.drop(c); s.recs = s.dups = 0;
  }
  async finish() {
    const h = this.hdr;
    this.done = true;
    this.result = { ...(await this.store.finish(h)), name: h.name, compression: h.compression, length: h.length, root: hex(this.W.take(h.root, 32)) };
  }
  // The file's bytes in: a verified source block's all, another's blocks in hand to its length.
  bytesIn() {
    const h = this.hdr;
    if (!h) return 0;
    if (this.done) return h.length;
    let got = 0;
    for (let c = 0; c < h.sourceBlocks; c++) { const s = this.sbs.get(c); if (s) got += s.state === VERIFIED ? h.len[c] : Math.min(h.len[c], Math.max(0, s.recs - s.dups) * PAYLOAD); }
    return got;
  }

  // For the page: what is known, and each source block's share of its blocks in hand (0 to 99, 255 verified).
  progress() {
    const h = this.hdr;
    const per = new Uint8Array(h.sourceBlocks), ready = this.ready();
    let solving = 0;
    for (let c = 0; c < h.sourceBlocks; c++) {
      const s = this.sbs.get(c);
      per[c] = this.done || s?.state === VERIFIED ? 255 : !s ? 0 : Math.min(99, Math.floor((100 * Math.max(0, s.recs - s.dups)) / h.K[c]));
      if (ready && s && s.state === COLLECTING && s.recs >= h.K[c] + s.dups) solving++;
    }
    // ai: solving: source blocks whose K symbols are in, their solve under way or next (the native XferProgress.solving);
    // ai: inHand: every source block verified or solving, the file not yet done (the Decoding row)
    const inHand = ready && !this.done && solving > 0 && this.verified + solving === h.sourceBlocks;
    return { header: { name: h.name, compression: h.compression, length: h.length, sourceBlocks: h.sourceBlocks, log2: h.log2, root: hex(this.W.take(h.root, 32)), toi: h.toi }, bytesIn: this.bytesIn(),
      verified: this.verified, rejected: this.rejected, over: this.over, solveMs: this.ms, per, done: this.done, solving, inHand };
  }
}

// Files from blocks alone. add() every block a decoder hands over, control blocks included and in any order, then read
// progress() and, once done is set, results() (result: the file in focus's). Several transfers at once (2026-11-19), a
// TOI each, from an entry of a session's description (src/xfer.h: the session header's base content, read at once, and
// an extended session header's, read when whole): an entry of another TOI opens another transfer beside the others; an
// extended session header is not a file of its own. releaseDone() lets every finished transfer's TOIs go,
// the transfer kept for progress, so the same file in the light again is received again (2026-11-20). The session
// (src/xfer.h, 2026-11-20): every session's description held (descs), its file opened as it is read.
export class XferReceiver {
  // M: the codec's wasm. Decoder: sim/fountain.mjs's. store: a MemoryStore, or the fountain worker's OPFS store: the
  // first transfer's, the next ones' from its make() (another of its class where it has none).
  constructor({ M, Decoder, store }) {
    this.M = M; this.W = wasm(M); this.Decoder = Decoder; this.store = store; this.storeUsed = false;
    this.objs = new Map(); this.order = []; this.focus = null; this.held = []; this.heldAt = 0;
    this.lightSeen = false; this.refused = 0; this.descsRefused = 0; this.descs = new Map();
  }
  makeStore() {
    if (!this.storeUsed) { this.storeUsed = true; return this.store; }
    return this.store.make ? this.store.make() : new this.store.constructor();
  }
  // ai: the transfer in focus (the last file whose header opened one): a single file's view, as before several
  get hdr() { return this.focus?.hdr ?? null; }
  get result() { return this.focus?.result ?? null; }
  // ai: every file this receiver holds done (none: false); an extended session header is no file
  get done() { const fs = this.files(); return fs.length > 0 && fs.every((t) => t.done); }
  files() { return this.order.filter((t) => t.hdr); }
  results() { return this.files().filter((t) => t.done).map((t) => t.result); }
  // ai: a transfer held under TOI
  has(toi) { return this.objs.has(toi); }

  // ai: Every transfer parked for its header to come again (keep), or let go whole.
  async reset(keep = false) {
    for (const t of this.objs.values()) await t.close(keep);
    this.objs.clear(); this.order = []; this.focus = null; this.held = []; this.heldAt = 0; this.lightSeen = false; this.descs.clear();
  }
  // ai: A transfer held let go (its folder parked, keep, or gone), its TOI with it.
  async drop(key, keep) {
    const t = this.objs.get(key);
    if (!t) return;
    await t.close(keep);
    this.objs.delete(key);
    this.order = this.order.filter((o) => o !== t);
    if (this.focus === t) this.focus = this.order[this.order.length - 1] ?? null;
  }
  // ai: Transfers let go by their root (hex, or its first 16 digits: every one of that file, held or parked), or one by
  // ai: its folder's id (28 hex digits, folderOf).
  async forget(root) {
    const folder = root.length === 28;
    for (const [key, t] of [...this.objs]) {
      if (!t.hdr) continue;
      const r = hex(this.W.take(t.hdr.root, 32));
      if (r.startsWith(root.slice(0, 16)) && (!folder || folderOf(r, t.hdr.toi) === root)) await this.drop(key, false);
    }
    await this.store.forget(root);
  }
  // ai: Every finished transfer's TOIs let go, the transfer kept for progress (files(), the file in focus): the same file
  // ai: in the light again is received again, as a download clicked twice (2026-11-20; core/rx/xfer_rx_core.cpp
  // ai: releaseDone). The page's, when its camera stops for a file in.
  releaseDone() {
    for (const [key, t] of [...this.objs]) {
      if (!t.done) continue;
      this.objs.delete(key);
    }
  }

  // ai: a block by its bare id and its TOI: the session header (other control ids reserved, ignored); a data block to
  // ai: its object's transfer, or held for a description naming its object. A block under TOI 0 (none) ignored (SPEC
  // ai: 7.5; src/xfer.h, core/rx/xfer_rx_core.cpp block).
  async add(id, toi, bytes) {
    if (toi === 0) return;
    const c = idSbn(id), esi = idEsi(id);
    if (c === CONTROL) { if (esi === 0) return this.session(bytes, toi); return; }
    if (await this.route(id, toi, bytes)) return;
    const r = [id, toi, bytes.slice()];
    if (this.held.length < HOLD_BLOCKS) this.held.push(r); else { this.held[this.heldAt] = r; this.heldAt = (this.heldAt + 1) % HOLD_BLOCKS; }
  }
  // ai: a data block to the transfer of its object; false where none held names it
  async route(id, toi, bytes) {
    const t = this.objs.get(toi);
    if (!t) return false;
    await t.add(idSbn(id), idEsi(id), bytes);
    return true;
  }
  // ai: the held blocks of a transfer's object into it, oldest first; the rest kept
  async release(key) {
    if (!this.held.length) return;
    const ring = this.held.length < HOLD_BLOCKS ? this.held : [...this.held.slice(this.heldAt), ...this.held.slice(0, this.heldAt)];
    this.held = []; this.heldAt = 0;
    const mine = [];
    for (const r of ring) (r[1] === key ? mine : this.held).push(r);
    for (const [id, t, b] of mine) await this.route(id, t, b);
  }

  // A session header (src/xfer.h) under the session's TOI: its description, the send's one file's map, read at once and
  // its file opened (listed). One whose description does not read, names the session's own TOI as its file's, or carries
  // a key to be understood that this build does not know, opens nothing.
  async session(bytes, toi) {
    this.lightSeen = true;
    const M = this.M, W = this.W, p = W.put(bytes), sc = M._malloc(8);
    const rc = M._xfer_session_read(p, sc);
    // ai: the description's offset and bytes (src/wasm.c xfer_session_read)
    const f = new Int32Array(M.HEAPU8.buffer, sc, 2).slice();
    [p, sc].forEach((q) => M._free(q));
    if (rc || !toi) { this.refused++; return; }
    const desc = bytes.slice(f[0], f[0] + f[1]), t = this.fileOf(desc, toi);
    if (t === null) return;
    this.descs.set(toi, desc);
    await this.listed(desc, t);
  }
  // ai: A description's file's TOI, read under the header's TOI (src/xfer.h xfer_desc_read), or null (descsRefused)
  // ai: where the read refuses it or it carries a key to be understood that this build does not know.
  fileOf(desc, toi) {
    const M = this.M, W = this.W, pd = W.put(desc), sc = M._malloc(44);
    const rc = M._xfer_desc_parse(pd, desc.length, toi, sc, 0, 0), f = new Int32Array(M.HEAPU8.buffer, sc, 11).slice();
    [pd, sc].forEach((q) => M._free(q));
    if (rc || f[8]) { this.descsRefused++; return null; }
    return (f[9] >>> 0) + (f[10] >>> 0) * 2 ** 32;
  }

  // ai: A description's file (under its content's TOI): the same file held again, from this session or another's, opens
  // ai: nothing; a new one opens beside the others.
  async listed(desc, toi) {
    const t = this.objs.get(toi);
    if (t?.hdr && sameFile(this.M, desc, t.hdr.bytes)) return;
    await this.open(desc, toi);
  }
  // ai: An object opened from its description (a session header's): the one held under its key parked,
  // ai: its folder resumed or begun, its held blocks into it. The object, or null where the description is refused.
  async open(desc, toi) {
    const h = this.parse(desc, toi);
    if (!h) return null;
    const key = toi;
    if (this.objs.has(key)) await this.drop(key, true);
    const n = new TransportObject(this, this.makeStore());
    this.objs.set(key, n);
    this.order.push(n);
    this.focus = n;
    if (!(await n.open(h))) {
      // ai: its data blocks held before it
      await this.release(key);
      await n.saveMeta();
      if (n.ready()) await n.solveReady();
    }
    return n;
  }
  // ai: A description (a session header's, src/xfer.h) parsed for an object under TOI:
  // ai: refused under TOI 0 (none), or carrying a critical attribute this build
  // ai: does not know; its source blocks' bytes and source symbols from it alone; null and
  // ai: refused counted where it is not one.
  parse(desc, toi) {
    const M = this.M, W = this.W, ph = W.put(desc), sc = M._malloc(44), root = M._malloc(32), nm = M._malloc(NAME_MAX);
    const rc = M._xfer_desc_parse(ph, desc.length, 0, sc, root, nm);
    // ai: a failed parse writes nothing: a name length read then was heap leftovers, megabytes decoded for nothing.
    if (rc) { this.refused++; [ph, sc, root, nm].forEach((p) => M._free(p)); return null; }
    // ai: hash, log2, the name's length, length's two halves, source blocks, source block size, the compression,
    // ai: critical, the TOI's low 32 bits and high 16 (src/wasm.c xfer_desc_parse)
    const f = new Int32Array(M.HEAPU8.buffer, sc, 11).slice(), name = new TextDecoder().decode(W.take(nm, f[2]));
    [sc, nm].forEach((p) => M._free(p));
    if (!toi || f[8] || (f[9] >>> 0) + (f[10] >>> 0) * 2 ** 32 !== toi) { this.refused++; [ph, root].forEach((p) => M._free(p)); return null; }
    const sourceBlocks = f[5], size = f[6], length = (f[3] >>> 0) + (f[4] >>> 0) * 2 ** 32;
    const len = Uint32Array.from({ length: sourceBlocks }, (_, c) => Math.min(size, length - c * size));
    // ai: cvs: the source blocks' chaining values as they decode (two source blocks or more), for the root at the last
    return { bytes: desc.slice(), p: ph, root, cvs: M._malloc(Math.max(1, 32 * sourceBlocks)),
      name, compression: f[7], log2: f[1], size, length, sourceBlocks, len, K: len.map((l) => blocksOf(l)), toi };
  }
  // For the page: the file in focus's progress as before several (none: no header), and every file's line (files, in
  // the order their transfers opened), the descriptions' count and what was refused.
  progress() {
    const base = this.focus?.hdr ? this.focus.progress() : { header: null, held: this.held.length };
    const files = this.files().map((t) => ({ toi: t.hdr.toi, name: t.hdr.name, compression: t.hdr.compression, length: t.hdr.length, root: hex(this.W.take(t.hdr.root, 32)),
      sourceBlocks: t.hdr.sourceBlocks, verified: t.verified, bytesIn: t.bytesIn(), done: t.done }));
    // ai: descs: the descriptions held (one a session header's TOI)
    return { ...base, lightSeen: this.lightSeen, refused: this.refused, files, descs: this.descs.size,
      descsRefused: this.descsRefused, done: this.done };
  }
}

// ai: fractionDone(p): how much of the file is in, 0 to 1, from progress()'s answer (or the fountain worker's xfer
// ai: message, which carries it): a source block verified (255; 254, decoded before the manifest, until 2026-10-05) counts whole, any other its per / 100, each
// ai: weighted by its length (2^log2 bytes, the last source block the rest). 0 with no header; 1 once done, an empty file too.
export function fractionDone(p) {
  const h = p?.header;
  if (p?.done) return 1;
  if (!h?.length) return 0;
  if (p.bytesIn != null) return Math.min(1, p.bytesIn / h.length);   // ai: the bytes in, each source block's blocks in hand to its length
  const size = 2 ** h.log2;
  let got = 0;
  for (let c = 0; c < h.sourceBlocks; c++) {
    const v = p.per[c], len = Math.min(size, h.length - c * size);
    got += len * (v >= 254 ? 1 : v / 100);
  }
  return got / h.length;
}
