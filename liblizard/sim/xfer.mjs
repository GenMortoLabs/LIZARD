// The transfer in the light for the rig (src/xfer.h holds the layouts and their reasons): the sender's schedule, and a
// receiver that puts a file back together from blocks alone, each chunk checked against the file's BLAKE3 root.
//
// Pure JS. The codec's wasm (sim/ob.mjs init(), for BLAKE3 and the header) and RaptorQ (sim/fountain.mjs) are handed
// in, so sim/phy.mjs and the receiver page take the id test from here without loading either.
//
// THE BYTES (src/xfer.h): a chunk's RaptorQ source block is the chunk's own bytes; the
// transfer compresses nothing (a client compresses a file before it enters). Every chunk's source symbols follow from
// the file's length; the manifest carries the chaining values. A receiver solves a chunk once the list checks against
// the root (one chunk: at once); the control cycle puts the manifest within an eighth of a lap.
//
// THE SCHEDULE: every chunk interleaved with every other, in proportion to its blocks, over the whole file (a lap),
// and laps repeated for as long as the sender runs, each with fresh repair ids. Not chunks in turn, because a camera's
// loss is steady and bursty, not rare: LIZARD degrades instead of stopping, so at a distance the outer rings go and a
// frame brings a share of its blocks, and a hand that moves loses whole seconds. In turn, a chunk gets a fixed budget
// a lap (K plus a margin m, since nothing comes back to say it arrived); a receiver losing more than m / (1 + m) of
// its blocks finishes no chunk in the first lap, and the margin itself is airtime a receiver that loses nothing throws
// away. Interleaved, a file takes 1 / (1 - loss) laps at any loss. Counted over 64 MB at 4 MiB, LIZARD-512, laps of
// airtime from the join to the last chunk (1.000 the least possible; STATUS.md has the table):
//                         interleaved   in turn m=0.1   m=0.25   m=0.5   ideal
//   nothing lost             1.001          1.099        1.246    1.492   1.000
//   joined 50% into a lap    1.001          1.094        1.225    1.459   1.000
//   15% lost at random       1.184          2.183        1.249    1.495   1.176
//   outer 25% of rings       1.347          2.186        2.480    1.497   1.333
//   2 s lost every 10 s      1.271          2.126        3.743    2.977   1.250
// A late join costs neither schedule anything: in turn the chunk being sent when the camera arrived comes round again
// at the end of its lap, which is when the file would end anyway.
// What it costs: no chunk completes before the end of the first lap, so the receiver holds every chunk's blocks at
// once. It keeps them on disk (the origin private file system, OPFS) where the browser has one, so its memory is a
// chunk or two at a solve, not the file. The sender holds every chunk's RaptorQ encoder (its intermediate symbols), about
// 1.02 times the file.
//
// Blocks are spread over a frame's slots by a shuffle drawn from the frame count: slot k of a LIZARD frame is ring k,
// the outer ones the first to go, and a plain round robin would pin chunk c to the same slots whenever the chunk count
// divides the frame's (every chunk on a ring that a distant camera never reads). Control blocks (the header, then the
// manifest) are shuffled with the data, so no one slot carries them: until 2026-10-10 they took slot 0, the lowest
// ring, thought the one read first and lost last, which under the rate profile carries the 7/8 code on the
// innermost sub-channel and for some contents never decodes, the header's among them.

// src/xfer.h, held against the codec's xfer_layout whenever a wasm is handed in. The receiver page has no codec and
// needs the id test anyway.
// A block's id is 40 bits (16 of chunk, 24 of symbol): past JS's 32-bit operators, so it is a number taken apart by
// arithmetic (exact to 2^53), and a list of ids a Float64Array.
export const PAYLOAD = 471, ID_BYTES = 5, SYMBOL_BITS = 24, CHUNK_BITS = 16, CONTROL = 0xffff, PER_BLOCK = 14, MAX_CHUNKS = 0xffff;
const SYMBOLS = 2 ** SYMBOL_BITS;
export const ID_HEADER = CONTROL * SYMBOLS, SYMBOL_MASK = SYMBOLS - 1, MAX_MANIFEST = Math.ceil(MAX_CHUNKS / PER_BLOCK);
const blocksOf = (len) => Math.ceil(len / PAYLOAD);
export const idChunk = (id) => Math.floor(id / SYMBOLS);
export const idSymbol = (id) => id % SYMBOLS;
export const isControlId = (id) => idChunk(id) === CONTROL;
export const idOf = (chunk, symbol) => chunk * SYMBOLS + symbol;
// A block's id from its first ID_BYTES bytes (little-endian), and into them.
export const idGet = (b, o = 0) => b[o] + b[o + 1] * 2 ** 8 + b[o + 2] * 2 ** 16 + b[o + 3] * 2 ** 24 + b[o + 4] * 2 ** 32;
export const idPut = (b, o, id) => { for (let i = 0; i < ID_BYTES; i++, id = Math.floor(id / 256)) b[o + i] = id % 256; };

// 16 MiB, the chunk size a sender takes (src/xfer.h xfer_chunk_log2_for, XFER_LOG2_DEFAULT): a file waits on its
// slowest chunk, so the fewer and larger the chunks the shorter its tail (src/xfer.h has the figures). The receiver holds
// one chunk at a solve several times over (the blocks read back, the decoder of about 35 MB, the recovered bytes and
// their copy for the hash), about 100 MB at the peak of a solve where a 4 MiB chunk took a quarter of it.
export const DEFAULT_LOG2 = 24;
// One control block for this many data blocks (1.6% of the airtime), more often for a short file, so a small file is
// not waiting on its header. A camera that joins sees the header within 128 blocks: 3 frames at LIZARD-512 (61 blocks), 12 at 96 (11).
const CONTROL_EVERY = 63;
const REC = ID_BYTES + PAYLOAD;   // a stored block: its symbol, then its payload
const symAt = (b, o) => idGet(b, o);   // a stored block's symbol

function wasm(M) {
  const put = (bytes) => { const p = M._malloc(Math.max(1, bytes.length)); M.HEAPU8.set(bytes, p); return p; };
  const take = (p, n) => M.HEAPU8.slice(p, p + n);
  const p = M._malloc(4 * 13);
  M._xfer_layout(p);
  const L = new Int32Array(M.HEAPU8.buffer, p, 13).slice();
  M._free(p);
  if (L[0] !== ID_BYTES || L[1] !== PAYLOAD || L[2] !== SYMBOL_BITS || L[3] !== CHUNK_BITS || L[4] !== CONTROL || L[5] !== MAX_MANIFEST || L[6] !== PER_BLOCK || L[11] !== MAX_CHUNKS || L[12] !== 1)
    throw new Error(`src/xfer.h moved (layout ${L.join(",")}): sim/xfer.mjs is out of date`);
  const b3 = (bytes) => { const a = put(bytes), o = M._malloc(32); M._xfer_b3_hash(a, bytes.length, o); const h = take(o, 32); M._free(a); M._free(o); return h; };
  // ai: parses: whether a header payload is one this build reads (a same-file test compares fields of parsed headers)
  const parses = (bytes) => { const p = put(bytes), s = M._malloc(40), rc = M._xfer_hdr_parse(p, s, 0, 0, 0); M._free(p); M._free(s); return rc === 0; };
  return { put, take, b3, parses, logMin: L[9], logMax: L[10], log2For: (length) => M._xfer_log2_for(length) };
}
// Whether two header payloads describe one file's blocks (src/xfer.h xfer_same_file: bytes 0, 1 and 4 to 47, every
// byte but the name's and the type's lengths and bytes): another sender's header for the same file is the same
// transfer.
export const sameFile = (a, b) => a.length === b.length && a[0] === b[0] && a[1] === b[1] && a.subarray(4, 48).every((v, i) => v === b[4 + i]);
export const hex = (a) => Array.from(a, (v) => v.toString(16).padStart(2, "0")).join("");
export const b64 = (a) => { let s = ""; for (const v of a) s += String.fromCharCode(v); return btoa(s); };
// BLAKE3 of a buffer through the vendored C, for a harness that has the codec's wasm to hand.
export const blake3 = (M, bytes) => wasm(M).b3(bytes);

// UTF-8 cut to 255 bytes on a character boundary, and a media type kept to printable ASCII or dropped.
function nameBytes(name) {
  const b = new TextEncoder().encode(name ?? "");
  let n = Math.min(b.length, 255);
  while (n < b.length && n > 0 && (b[n] & 0xc0) === 0x80) n--;
  return b.subarray(0, n);
}
const typeBytes = (t) => (/^[\x20-\x7e]{0,158}$/.test(t ?? "") ? new TextEncoder().encode(t ?? "") : new Uint8Array(0));

// ai: The schedule's draws (core/tx/xfer_tx.cpp's): mix32, murmur3's finaliser, a word's bits mixed; xorshift32 on a
// ai: state that must not be 0; a random seed where a sender is given none.
const mix32 = (x) => {
  x = (x ^ (x >>> 16)) >>> 0; x = Math.imul(x, 0x85ebca6b) >>> 0; x = (x ^ (x >>> 13)) >>> 0; x = Math.imul(x, 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
};
const xorshift = (s) => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; return s >>> 0; };
const nonzero = (s) => (s ? s : 0x9e3779b9);
const randomSeed = () => globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
// ai: A chunk's symbols, pass by pass (2026-11-15; core/tx/xfer_tx.cpp symbolOf, the same arithmetic): its v-th is the
// ai: (v mod 2^24)-th of pass floor(v / 2^24) under a keyed shuffle of the 2^24 ids (four Feistel rounds over two 12-bit
// ai: halves, the round function mix32), the pass's key drawn from the chunk's (the seed's) and the pass: every id once
// ai: a pass, each pass in an order of its own, so no receiver meets the same ids at the same times twice (consecutive
// ai: ids from a start before: each pass repeated the last).
const passKey = (key, pass) => mix32((key ^ mix32((Math.imul(pass >>> 0, 0x85ebca6b) + 0x27d4eb2f) >>> 0)) >>> 0);
const shuffle24 = (key, k) => {
  let l = (k >>> 12) & 0xfff, r = k & 0xfff;
  for (let i = 0; i < 4; i++) { const t = r; r = (l ^ (mix32((key ^ Math.imul(i, 0x9e3779b9) ^ r) >>> 0) & 0xfff)) >>> 0; l = t; }
  return ((l << 12) | r) >>> 0;
};
export const symbolOf = (key, v) => shuffle24(passKey(key, Math.floor(v / SYMBOLS)), v % SYMBOLS);

// Which chunk the next data block is for, and its symbol. A round visits every chunk of the most blocks once; a chunk
// of fewer is visited in proportion to its blocks by an accumulator, plus 2 sqrt(K) visits a lap and at least 64: a
// small chunk's count of blocks received wanders more than a big one's, and the file waits on whichever chunk is
// last. Counted (10 MB, 8.5 MB, 4.2 MB, 64 MB; 0, 15 and 30% lost) when only a last chunk could be short, that costs a
// receiver that loses nothing at most 0.55% and is level with or ahead of plain proportion everywhere else (4.2 MB at
// 30% lost: 1.431 laps against 1.472, the least being 1.429); 4 sqrt(K) + 64 cost 1.4% at 10 MB, which the rig
// measured. Only a last chunk is short (no chunk's blocks depend on its content); it takes the
// margin, 2 / sqrt(K) of its airtime at most (2.6% at 6,000 blocks).
// ai: Each round visits the chunks in an order of its own (newRound: Fisher-Yates from the rounds' generator, rs), so no
// ai: place in a round holds a chunk from one round to the next and no period of the schedule meets a periodic loss.
class Schedule {
  // key: each chunk's key (its symbols' order, symbolOf); cnt: the symbols each chunk sent (a streaming sender's going
  // on from the frames sent while its chunks were made); rs: the rounds' generator
  constructor(K, key, cnt, rs) {
    const C = K.length;
    this.K = K; this.C = C; this.key = key; this.cnt = cnt; this.i = 0; this.acc = new Uint32Array(C); this.rs = rs; this.perm = new Uint32Array(C);
    this.max = C ? K.reduce((a, b) => Math.max(a, b), 0) : 0;
    this.w = Uint32Array.from(K, (k) => (k >= this.max ? this.max : Math.min(this.max, Math.max(64, k + Math.ceil(2 * Math.sqrt(k))))));
    this.lap = this.w.reduce((a, b) => a + b, 0);
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
    return idOf(c, symbolOf(this.key[c], this.cnt[c]++));   // past 2^24 a chunk, a pass in a new order (symbolOf)
  }
}

// A chunk's chaining value from its own bytes (BLAKE3's wide hash): a streaming sender's first pass, which makes the root.
export function chunkCv(M, plain, c, chunkLog2) {
  const W = wasm(M), p = W.put(plain), o = M._malloc(32), rc = M._xfer_chunk_cv(p, plain.length, c, chunkLog2, o), v = W.take(o, 32);
  M._free(p); M._free(o);
  if (rc) throw new Error(`chaining value of chunk ${c}`);
  return v;
}

// One chunk made, the sender's step for each (and a pool's worker's: lizard-web/xfer-pool.mjs): a RaptorQ encoder over
// its bytes (a chunk of one block too); its chaining value where the file has two chunks or more (cv).
export function makeChunk(plain, c, chunkLog2, cv, { M, Encoder }) {
  return { K: blocksOf(plain.length), enc: new Encoder(plain, PAYLOAD), cv: cv ? chunkCv(M, plain, c, chunkLog2) : null };
}

export class XferSender {
  // bytes: the file. M: the codec's wasm module. Encoder: sim/fountain.mjs's, its init() already awaited. chunkLog2: the
  // chunk size's log2, xfer_chunk_log2_for's for the length unless given. Every chunk made here, before the
  // constructor returns; or elsewhere (a pool's, lizard-web/xfer-pool.mjs; two chunks or more, bytes null and length
  // given): chunks, every chunk made (makeChunk's results), or cvs, every chunk's chaining value, the chunks then coming
  // one at a time (add) while frames go. Their fountains held by remote: its fetch(ids) gives a Map of id to block for
  // the ids of those chunks, its free() lets them go.
  // STREAMING (2026-11-06; the native sender's, core/tx/xfer_tx.h): with cvs the root and the header are made at once;
  // until the last chunk is added a frame carries one control block (the header, or a manifest block once its 12 chunks
  // are in) and the chunks added so far in turn, fresh symbols each (warm); then the schedule, the symbols going on. A
  // receiver keeps a chunk's blocks from the header on and solves once the manifest is whole.
  // SEED (2026-11-14, the native sender's): each chunk's first symbol, each round's order of the chunks and each frame's
  // shuffle drawn from `seed` (random where none is given; reseed before the first frame), so no send repeats another's
  // ids or their places. `scheduleSeed` the one in use.
  constructor(bytes, { name = "", type = "", chunkLog2 = null, M, Encoder, chunks = null, cvs = null, length = bytes?.length ?? 0, remote = null, seed = null }) {
    const W = wasm(M);
    chunkLog2 ??= W.log2For(length);
    if (chunkLog2 < 0) throw new Error(`${length} B is more than the transfer carries (16,383 chunks of 16 MiB)`);
    const size = 2 ** chunkLog2;
    if (!(chunkLog2 >= W.logMin && chunkLog2 <= W.logMax)) throw new Error(`chunk size 2^${chunkLog2} is outside 2^${W.logMin} to 2^${W.logMax}`);
    const C = Math.ceil(length / size);
    if (C > MAX_CHUNKS) throw new Error(`${length} B is ${C} chunks of 2^${chunkLog2}; the id carries ${MAX_CHUNKS}`);
    if ((chunks || cvs) && C < 2) throw new Error("chunks made elsewhere: two or more");
    if ((chunks && chunks.length !== C) || (cvs && cvs.length !== 32 * C)) throw new Error("chunks made elsewhere: one a chunk");
    this.M = M; this.W = W;
    this.chunks = C; this.chunkLog2 = chunkLog2; this.size = size; this.length = length; this.Encoder = Encoder;
    this.len = Array.from({ length: C }, (_, c) => Math.min(size, length - c * size));
    this.K = new Uint32Array(C); this.enc = new Array(C).fill(null);
    this.ready = new Uint8Array(C); this.made = 0; this.cvs = C >= 2 ? new Uint8Array(32 * C) : null; this.remote = remote;
    let ks = null;
    if (cvs) this.cvs.set(cvs);
    else {
      ks = chunks ?? Array.from({ length: C }, (_, c) => makeChunk(bytes.subarray(c * size, c * size + this.len[c]), c, chunkLog2, C >= 2, { M, Encoder }));
      if (this.cvs) ks.forEach((k, c) => this.cvs.set(k.cv, 32 * c));
    }
    // The chunks' chaining values' root (b3sum's answer) and the header; the manifest's lists kept in the heap, a
    // group's block written as its chunks come (add).
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
    const nm = nameBytes(name), ty = typeBytes(type), pr = W.put(root), pn = W.put(nm), pt = W.put(ty), ph = M._malloc(PAYLOAD);
    if (M._xfer_hdr_write(ph, length, chunkLog2, pr, pn, nm.length, pt, ty.length)) throw new Error("header refused");
    this.header = W.take(ph, PAYLOAD);
    [pn, pt].forEach((p) => M._free(p));
    this.hp = { root: pr, out: ph, cvs: W.put(this.cvs ?? new Uint8Array(0)) };
    this.mcount = M._xfer_manifest_blocks(C); this.manifest = new Array(this.mcount).fill(null);
    this.left = Uint32Array.from({ length: this.mcount }, (_, m) => Math.min(PER_BLOCK, C - m * PER_BLOCK));
    this.sched = null; this.key = new Uint32Array(C); this.cnt = new Float64Array(C); this.since = Infinity; this.ctl = 0; this.frame = 0; this.turn = 0;
    this.seedWith(seed ?? randomSeed());
    if (ks) ks.forEach((k, c) => this.add(c, k));
    if (!C) this.schedule();
  }
  // Chunk c made (makeChunk's result; a pool's has no enc, its fountain remote's): its slots, its manifest group's block
  // once the group is whole, and with the last chunk the schedule.
  add(c, k) {
    if (!this.hp) return;   // let go (free) while a pool still made it
    const M = this.M, C = this.chunks;
    this.K[c] = k.K; this.enc[c] = k.enc ?? null;
    this.ready[c] = 1; this.made++;
    if (C >= 2) {
      const m = Math.floor(c / PER_BLOCK);
      if (--this.left[m] === 0) {
        if (M._xfer_manifest_write(this.hp.cvs, C, this.hp.root, m, this.hp.out)) throw new Error(`manifest block ${m}`);
        this.manifest[m] = this.W.take(this.hp.out, PAYLOAD);
      }
    }
    if (this.made === C) this.schedule();
  }
  // The schedule drawn from seed in place of the one given or drawn; before the first frame only (false after).
  reseed(seed) {
    if (this.frame) return false;
    this.seedWith(seed);
    return true;
  }
  // ai: chunk c's key mix32(seed ^ (c + 1) 0x9e3779b1) (its symbols' order, symbolOf); the rounds' generator mix32(seed ^ 0x5bd1e995); each frame's shuffle from (frame + 1) 0x9e3779b1 ^ seed
  seedWith(seed) {
    this.scheduleSeed = seed >>> 0;
    for (let c = 0; c < this.chunks; c++) this.key[c] = mix32((this.scheduleSeed ^ Math.imul(c + 1, 0x9e3779b1)) >>> 0);
    this.rs = nonzero(mix32((this.scheduleSeed ^ 0x5bd1e995) >>> 0));
    if (this.sched) { this.sched.rs = this.rs; this.sched.i = 0; this.sched.newRound(); }
  }
  // The schedule, once every chunk is in; the whole control cycle (the header every other block, the manifest between)
  // within an eighth of a lap, so a file of many small chunks does not wait on its manifest.
  schedule() {
    this.sched = new Schedule(this.K, this.key, this.cnt, this.rs);
    this.every = Math.max(1, Math.min(CONTROL_EVERY, Math.floor(this.sched.lap / (8 * Math.max(1, this.mcount)))));
    this.since = Infinity;
  }
  // lap 0 until every chunk is in.
  info() { return { header: b64(this.header), chunks: this.chunks, chunkLog2: this.chunkLog2, manifest: this.mcount, root: hex(this.root), lap: this.sched?.lap ?? 0, length: this.length, seed: this.scheduleSeed }; }

  // The ids of the next frame of B blocks. Advances the schedule, so a frame asked for twice is two frames.
  frameIds(B, out = new Float64Array(B)) {
    if (!this.chunks) { out.fill(ID_HEADER); this.frame++; return out; }   // an empty file is its header
    if (!this.sched) this.warm(B, out);
    else {
      let first = 0;
      if (this.since >= this.every && B > 1) {
        const m = this.mcount, j = this.ctl++;
        out[first++] = !m || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ((j >> 1) % m);
        this.since = 0;
      }
      for (let k = first; k < B; k++) out[k] = this.sched.next();
      this.since += B - first;
    }
    // Fisher-Yates over every slot, the control block's too, from a generator seeded by the frame count and the seed
    // (the native sender's, core/tx/xfer_tx.cpp, byte for byte).
    let s = nonzero(mix32((Math.imul(this.frame + 1, 0x9e3779b1) ^ this.scheduleSeed) >>> 0));
    for (let i = B - 1; i > 0; i--) {
      s = xorshift(s);
      const j = s % (i + 1), t = out[i];
      out[i] = out[j]; out[j] = t;
    }
    this.frame++;
    return out;
  }
  // A frame before every chunk is in (the native sender's warm): a control block (where it has more than one slot, or
  // nothing is in yet, or every other frame of one), the header every other time and the manifest blocks written so far
  // between; the rest the chunks in so far in turn, each its next symbol; with none in, every slot a control block.
  warm(B, out) {
    const ready = [], ms = [];
    for (let c = 0; c < this.chunks; c++) if (this.ready[c]) ready.push(c);
    for (let m = 0; m < this.mcount; m++) if (this.manifest[m]) ms.push(m);
    const ctl = () => { const j = this.ctl++; return !ms.length || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ms[(j >> 1) % ms.length]; };
    let k = 0;
    if (B > 1 || !ready.length || !(this.frame & 1)) out[k++] = ctl();
    for (; k < B; k++) {
      if (!ready.length) { out[k] = ctl(); continue; }
      const c = ready[this.turn++ % ready.length];
      out[k] = idOf(c, symbolOf(this.key[c], this.cnt[c]++));
    }
  }

  // What block `id` carries: 471 bytes into out.
  block(id, out) {
    const c = idChunk(id), s = idSymbol(id);
    if (c === CONTROL) { out.set(s === 0 ? this.header : this.manifest[s - 1]); return out; }
    if (this.enc[c]) return this.enc[c].block(s, out);
    throw new Error(`block ${id}: its chunk's fountain is remote's (fetch it)`);
  }
  free() {
    for (const e of this.enc) e?.free();
    this.enc = []; this.remote?.free(); this.remote = null;
    if (this.hp) { for (const p of Object.values(this.hp)) this.M._free(p); this.hp = null; }
  }
}

// The distinct values of 24-bit symbols (the native receiver's radix24 and its count): two stable radix passes of 12 bits.
function distinctOf(ids) {
  let a = Uint32Array.from(ids, (x) => x & SYMBOL_MASK), t = new Uint32Array(a.length);
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

// A chunk keeps no set of the symbols it holds (2026-11-15): a send's ids are each pass of a chunk's 2^24 in an order
// of its own (symbolOf), so they never repeat within a send; a camera's repeats are the page's dedupe; the rare repeat
// across sends (K / 2^24 of a chunk's) is stored as it came and dropped by the decoder at the chunk's solve. Its refused
// symbols (a rejected chunk's) are a Set: rare. The dense runs of bits held until then (Bits, 2026-11-14) would have
// grown to 2 MiB a chunk over shuffled ids. The native receiver (core/rx/xfer_rx_core.cpp) the same.

// Blocks and the finished file, held in memory: the fallback where there is no origin private file system. The
// fountain worker (lizard-web/fountain-worker.mjs) has the OPFS one with the same calls.
// ai: A store that keeps transfers (keeps; 2026-10-22, the OPFS one) holds each in a folder of its own and resumes one
// ai: when its header comes again (open, the notes, park, forget: liblizard/core/rx/store.h's). This one keeps one
// ai: transfer at a time, as before: open is begin, park is clear, the notes are nothing.
export class MemoryStore {
  constructor() { this.kind = "memory"; this.keeps = false; this.recs = new Map(); this.out = null; }
  async clear() { this.recs.clear(); this.out = null; }
  async begin(length) { this.out = new Uint8Array(length); }
  async open(id, header, length) { await this.begin(length); return { found: false }; }
  async noteManifest() {}
  async noteVerified() {}
  async noteMeta() {}
  async truncate() {}
  async park() { await this.clear(); }
  async forget() {}
  async append(c, sym, bytes) {
    let r = this.recs.get(c);
    if (!r) this.recs.set(c, (r = { buf: new Uint8Array(REC * 256), n: 0 }));
    if ((r.n + 1) * REC > r.buf.length) { const b = new Uint8Array(2 * r.buf.length); b.set(r.buf); r.buf = b; }
    const o = r.n++ * REC;
    idPut(r.buf, o, sym);
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
// ai: as well as at each chunk verified and at a park, so a page closed mid-chunk leaves the list a line at most that
// ai: far behind (liblizard/core/rx/xfer_rx_core.cpp)
const META_EVERY = 2048, META_MS = 1000;
const COLLECTING = 0, HOT = 1, VERIFIED = 3;   // 2 was a chunk decoded before the manifest, which no longer happens: a chunk's block count waits on it
const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// A file from blocks alone. add() every block a decoder hands over, control blocks included and in any order, then
// read progress() and, once done is set, result.
export class XferReceiver {
  // M: the codec's wasm. Decoder: sim/fountain.mjs's. store: a MemoryStore, or the fountain worker's OPFS store.
  constructor({ M, Decoder, store }) {
    this.M = M; this.W = wasm(M); this.Decoder = Decoder; this.store = store;
    this.hdr = null; this.lightSeen = false;
    this.ch = new Map(); this.raw = new Map(); this.replaying = false; this.sinceMeta = 0; this.metaMs = 0;
    this.verified = 0; this.rejected = 0; this.manifestRejected = 0; this.refused = 0; this.over = 0; this.ms = 0; this.done = false; this.result = null;
  }

  // ai: The transfer in hand parked for its header to come again (keep; a store that keeps no transfers lets it go), or
  // ai: let go whole (liblizard/core/rx/xfer_rx_core.cpp reset).
  async reset(keep = false) {
    const M = this.M;
    if (keep && this.hdr && !this.done && this.store.keeps) {
      // ai: the blocks since this transfer's header was last read may be the next file's (the sender moved on, its
      // ai: blocks under the same ids until its header comes): each chunk cut back to its mark
      for (const [c, s] of this.ch) if (s.state !== VERIFIED && s.recs > s.mark) { await this.store.truncate(c, s.mark); s.recs = s.mark; s.dups = 0; }
      await this.saveMeta();
    }
    for (const s of this.ch.values()) s.dec?.free();
    if (this.hdr) [this.hdr.p, this.hdr.root, this.hdr.list].forEach((p) => M._free(p));
    this.hdr = null; this.lightSeen = false; this.ch.clear(); this.raw.clear();
    this.verified = 0; this.rejected = 0; this.manifestRejected = 0; this.over = 0; this.ms = 0; this.done = false; this.result = null; this.sinceMeta = 0;
    if (keep) await this.store.park(); else await this.store.clear();
  }
  // ai: A transfer let go by its root (hex, or its first 16 digits): the one in hand, or one the store parked.
  async forget(root) {
    const id = root.slice(0, 16);
    if (this.hdr && hex(this.W.take(this.hdr.root, 32)).startsWith(id)) return this.reset(false);
    await this.store.forget(id);
  }
  // ai: The transfer's line for a page's list of unfinished transfers (the store's meta.json).
  async saveMeta() {
    this.sinceMeta = 0; this.metaMs = performance.now();
    const h = this.hdr;
    if (!h || !this.store.keeps) return;
    const p = this.progress();
    await this.store.noteMeta(JSON.stringify({ name: h.name, type: h.type, length: h.length, chunks: h.chunks, verified: this.verified, root: p.header.root,
      fraction: Math.round(fractionDone(p) * 1e4) / 1e4, bytesIn: this.bytesIn() }));
  }

  state(c) {
    let s = this.ch.get(c);
    // ai: recs: the blocks the store holds for the chunk; mark: how many when this transfer's header was last read
    // ai: dups: of its records, those its last solve found repeating an earlier one's symbol (a repeat across sends)
    if (!s) this.ch.set(c, (s = { ban: new Set(), state: COLLECTING, dec: null, recs: 0, mark: 0, dups: 0 }));
    return s;
  }

  async add(id, bytes) {
    const c = idChunk(id), sym = idSymbol(id);
    if (c === CONTROL) {
      if (sym === 0) return this.header(bytes);
      if (sym <= MAX_MANIFEST) return this.manifestBlock(sym - 1, bytes);
      return;   // reserved: ignored
    }
    const h = this.hdr;
    // ai: a store that keeps transfers keeps them by their header's folder, so a data block before any header is not
    // ai: kept (the fountain worker holds such blocks for the next header itself)
    if (this.done || (h && c >= h.chunks) || (!h && this.store.keeps)) return;
    const s = this.state(c);
    if (s.state === VERIFIED || s.ban.has(sym)) return;
    if (s.dec) {
      // ai: kept until the chunk is verified, by any store: a resume makes the decoder again from them
      await this.store.append(c, sym, bytes); s.recs++;
      if (s.dec.add(sym, bytes)) await this.recovered(c, s.dec);
      return;
    }
    await this.store.append(c, sym, bytes);
    s.recs++;
    if (++this.sinceMeta >= META_EVERY || performance.now() - this.metaMs >= META_MS) await this.saveMeta();
    if (h && this.ready() && s.recs >= h.K[c] + s.dups) await this.solve(c);
  }

  // A header block, read from the light (the only place a transfer's header comes from): the one in hand again, or the
  // same file's under another name or type (sameFile: another sender's), marks every chunk's blocks so far as its own;
  // another file's is a new transfer, and the one in hand is parked or let go.
  async header(bytes) {
    this.lightSeen = true;
    if (this.hdr && (eq(bytes, this.hdr.bytes) || (bytes.length === this.hdr.bytes.length && sameFile(bytes, this.hdr.bytes) && this.W.parses(bytes)))) { for (const s of this.ch.values()) s.mark = s.recs; return; }
    const M = this.M, W = this.W, ph = W.put(bytes), sc = M._malloc(40), root = M._malloc(32), nm = M._malloc(255), ty = M._malloc(158);
    const rc = M._xfer_hdr_parse(ph, sc, root, nm, ty);
    // ai: a failed parse writes nothing: name and type lengths read then were heap leftovers, megabytes decoded for nothing.
    if (rc) { this.refused++; [ph, sc, root, nm, ty].forEach((p) => M._free(p)); return; }   // not a header this receiver reads
    // ai: hash, log2, name and type lengths, length's two halves, chunks, manifest blocks, chunk size (src/wasm.c)
    const f = new Int32Array(M.HEAPU8.buffer, sc, 9).slice(), name = new TextDecoder().decode(W.take(nm, f[2])), type = new TextDecoder().decode(W.take(ty, f[3]));
    [sc, nm, ty].forEach((p) => M._free(p));
    if (this.hdr) await this.reset(true);
    // ai: every chunk's bytes and source symbols from the header alone (src/xfer.h)
    const chunks = f[6], size = f[8], length = (f[4] >>> 0) + (f[5] >>> 0) * 2 ** 32;
    const len = Uint32Array.from({ length: chunks }, (_, c) => Math.min(size, length - c * size));
    const h = { bytes: bytes.slice(), p: ph, root, list: M._malloc(Math.max(1, 32 * chunks)),
      name, type, log2: f[1], size, length, chunks, mcount: f[7], len, K: len.map(blocksOf),
      mgot: new Uint8Array(f[7]), mhave: 0, listOk: false };
    this.hdr = h;
    for (const c of [...this.ch.keys()]) if (c >= chunks) { this.ch.delete(c); await this.store.drop(c); }   // blocks of another transfer
    const before = await this.store.open(hex(W.take(root, 32)).slice(0, 16), bytes, h.length);
    if (!chunks) {   // an empty file: its root is BLAKE3 of nothing, which is all there is to check
      if (eq(W.b3(new Uint8Array(0)), W.take(root, 32))) await this.finish(); else this.refused++;
      return;
    }
    if (before?.found && (await this.resume(before))) return;
    for (const [m, b] of this.raw) await this.manifestBlock(m, b);
    this.raw.clear();
    await this.saveMeta();
    if (this.ready()) await this.solveReady();
  }
  // ai: whether a chunk may be solved: its chaining value checked against the root (the manifest whole), or a file of
  // ai: one chunk, whose hash is the root
  ready() { const h = this.hdr; return !!h && (h.chunks === 1 || h.listOk); }
  // ai: A folder of this transfer from before: its manifest blocks replayed, its verified chunks taken as verified, and
  // ai: each collecting chunk's symbols marked had from its stored blocks; true when that finished it.
  async resume(b) {
    const h = this.hdr;
    this.replaying = true;
    for (const [m, pay] of b.manifest) await this.manifestBlock(m, pay);
    this.replaying = false;
    for (const c of b.verified) if (c < h.chunks && this.state(c).state !== VERIFIED) { const s = this.state(c); s.state = VERIFIED; s.ban = null; this.verified++; }
    for (const c of b.chunks) {
      if (c >= h.chunks || this.state(c).state === VERIFIED) { await this.store.drop(c); continue; }
      const s = this.state(c), recs = await this.store.read(c);
      s.recs = s.mark = recs.length / REC;
    }
    if (this.verified === h.chunks) { await this.finish(); return true; }
    return false;
  }
  // Every chunk with its blocks in hand, now that their counts are known.
  async solveReady() {
    const h = this.hdr;
    for (const [c, s] of [...this.ch]) if (!this.done && s.state === COLLECTING && s.recs >= h.K[c] + s.dups) await this.solve(c);
  }

  async manifestBlock(m, bytes) {
    const h = this.hdr, M = this.M;
    if (!h) { if (this.raw.size < 4096) this.raw.set(m, bytes.slice()); return; }   // kept until there is a root to check it by
    if (h.listOk || m >= h.mcount || h.mgot[m]) return;
    const p = this.W.put(bytes), ok = M._xfer_mf_parse(p, h.p, m, h.list) === 0;
    M._free(p);
    if (!ok) return;   // another transfer's (its tag), not a manifest block, or padding that is not zero
    h.mgot[m] = 1;
    if (!this.replaying) await this.store.noteManifest(m, bytes);
    if (++h.mhave < h.mcount) return;
    if (M._xfer_manifest_check(h.list, h.chunks, h.root)) {
      // Every block passed its parse and the list still does not make the root: one of them passed its CRC wrongly.
      // Which one cannot be told, so all are collected again; the sender repeats them.
      h.mgot.fill(0); h.mhave = 0; this.manifestRejected++;
      return;
    }
    h.listOk = true;
    await this.solveReady();
  }

  async solve(c) {
    this.onSolve?.();   // ai: the page told before the work (the fountain worker's report): the chunk in hand, solving
    const t = performance.now();
    await this.solveOnce(c);
    this.ms += performance.now() - t;   // interleaved, most of this falls after the last block: it is the transfer's tail
  }
  async solveOnce(c) {
    const s = this.ch.get(c);
    // ai: fewer than K symbols once repeats are dropped: no decoder (it would hold the chunk's symbols, and near a
    // ai: resumed transfer's end every chunk's at once); the chunk collects that many more (the native receiver's solve).
    // ai: Counted from the store's ids where it keeps them apart (the memory store's; the OPFS store reads its records:
    // ai: a file of ids a chunk would double its open handles, one a chunk held for the transfer)
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
    // ai: RaptorQ wanted a block more than K (about 1% of chunks): the blocks fed as they come, and kept until the chunk
    // ai: is verified (add)
    if (!r.done) { s.dec = r.d; s.state = HOT; this.over++; return; }
    await this.store.drop(c); s.recs = s.mark = s.dups = 0;
    await this.recovered(c, r.d, recs);
  }
  // ai: A fresh decoder over a chunk's stored blocks, in the order they came: { d, done }.
  feed(c, recs) {
    const h = this.hdr, d = new this.Decoder(h.len[c], PAYLOAD);
    for (let o = 0; o + REC <= recs.length; o += REC) if (d.add(symAt(recs, o), recs.subarray(o + ID_BYTES, o + REC))) return { d, done: true };
    return { d, done: false };
  }
  // recs: the records that went into it where the store has let them go (a solve's), else the store's (a fed-on chunk's)
  async recovered(c, d, recs) {
    const s = this.ch.get(c), out = d.recover();
    d.free(); s.dec = null;
    await this.check(c, out, recs);
  }
  async check(c, out, recs) {
    const h = this.hdr, s = this.ch.get(c);
    if (this.verify(c, out)) { await this.store.writeOut(c * h.size, out); await this.store.noteVerified(c); await this.accept(c, s); } else await this.reject(c, s, recs);
  }
  // xfer_chunk_check: the chunk's chaining value against the list (already checked against the root), or, for a file
  // of one chunk, its hash against the root.
  verify(c, out) {
    const M = this.M, h = this.hdr, p = this.W.put(out), ok = M._xfer_chunk_ok(h.p, c, p, out.length, h.chunks >= 2 ? h.list : 0) === 0;
    M._free(p);
    return ok;
  }
  async accept(c, s) {
    s.state = VERIFIED; s.ban = null; this.verified++;
    await this.store.drop(c); s.recs = s.mark = s.dups = 0;   // ai: a kept chunk's blocks (store keeps), nothing otherwise
    if (this.verified === this.hdr.chunks) await this.finish(); else await this.saveMeta();
  }
  // A chunk that decoded to the wrong bytes: some block of it passed its CRC wrongly (1 in 2^32), or was sent wrong.
  // Which one cannot be told, so every symbol that went into it is refused from now on and the chunk is collected
  // again from fresh ids; the sender's next laps bring them.
  async reject(c, s, recs) {
    const used = recs ?? (await this.store.read(c));
    for (let o = 0; o + REC <= used.length; o += REC) s.ban.add(symAt(used, o));
    s.state = COLLECTING; this.rejected++;
    if (s.dec) { s.dec.free(); s.dec = null; }
    await this.store.drop(c); s.recs = s.mark = s.dups = 0;
  }
  async finish() {
    this.done = true;
    const h = this.hdr;
    this.result = { ...(await this.store.finish(h)), name: h.name, mediaType: h.type, length: h.length, root: hex(this.W.take(h.root, 32)) };
  }
  // The file's bytes in: a verified chunk's all, another's blocks in hand to its length.
  bytesIn() {
    const h = this.hdr;
    if (!h) return 0;
    let got = 0;
    for (let c = 0; c < h.chunks; c++) { const s = this.ch.get(c); if (s) got += s.state === VERIFIED ? h.len[c] : Math.min(h.len[c], Math.max(0, s.recs - s.dups) * PAYLOAD); }
    return got;
  }

  // For the page: what is known, and each chunk's share of its blocks in hand (0 to 99, 255 verified).
  progress() {
    const h = this.hdr;
    if (!h) return { header: null, lightSeen: this.lightSeen, held: this.ch.size, refused: this.refused };
    const per = new Uint8Array(h.chunks), ready = this.ready();
    let solving = 0;
    for (let c = 0; c < h.chunks; c++) {
      const s = this.ch.get(c);
      per[c] = !s ? 0 : s.state === VERIFIED ? 255 : Math.min(99, Math.floor((100 * Math.max(0, s.recs - s.dups)) / h.K[c]));
      if (ready && s && s.state === COLLECTING && s.recs >= h.K[c] + s.dups) solving++;
    }
    // ai: solving: chunks whose K symbols are in, their solve under way or next (the native XferProgress.solving);
    // ai: inHand: the list checked and every chunk verified or solving, the file not yet done (the Decoding row)
    const inHand = ready && !this.done && solving > 0 && this.verified + solving === h.chunks;
    return { header: { name: h.name, type: h.type, length: h.length, chunks: h.chunks, log2: h.log2, root: hex(this.W.take(h.root, 32)) }, bytesIn: this.bytesIn(), lightSeen: this.lightSeen,
      manifest: h.mcount, manifestHave: h.mhave, listOk: h.listOk, verified: this.verified, rejected: this.rejected, manifestRejected: this.manifestRejected, refused: this.refused, over: this.over, solveMs: this.ms, per, done: this.done, solving, inHand };
  }
}

// ai: fractionDone(p): how much of the file is in, 0 to 1, from progress()'s answer (or the fountain worker's xfer
// ai: message, which carries it): a chunk verified (255; 254, decoded before the manifest, until 2026-10-05) counts whole, any other its per / 100, each
// ai: weighted by its length (2^log2 bytes, the last chunk the rest). 0 with no header; 1 once done, an empty file too.
export function fractionDone(p) {
  const h = p?.header;
  if (p?.done) return 1;
  if (!h?.length) return 0;
  if (p.bytesIn != null) return Math.min(1, p.bytesIn / h.length);   // ai: the bytes in, each chunk's blocks in hand to its length
  const size = 2 ** h.log2;
  let got = 0;
  for (let c = 0; c < h.chunks; c++) {
    const v = p.per[c], len = Math.min(size, h.length - c * size);
    got += len * (v >= 254 ? 1 : v / 100);
  }
  return got / h.length;
}
