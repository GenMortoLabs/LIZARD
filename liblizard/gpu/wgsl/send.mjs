// ai: The sender's GPU encoder (gpu/encoder.mjs; 2026-09-29, as the sender has lagged at LIZARD-1024; bit exact only in
// ai: the ring): PAINT, a block's codeword onto the disc (below: every code of the format's rate profile), then the
// ai: inverse transforms (wgsl/inverse.mjs irows and ipic: PIC, n x n f32 a symbol in 0..1, clipped), then three
// ai: shaders, the C's arithmetic (src/lizard_phy.c resample, lizard_paint_rgba):
// ai: TPOSE, the picture row-major: ipic leaves it quad-major (PICQ[x / 4][y], four columns' rows consecutive, so its
// ai:   stores are whole lines); a tile of 16 quads by 16 rows a workgroup through workgroup memory, read a quad's 16
// ai:   rows (256 B) at a time and written a row's 16 quads at a time, so both sides move whole lines.
// ai: RSV, down the picture: TMP[y][k] = sum over t < 6 of w[6 y + t] pic[idx[6 y + t]][k], y < q, k < n, a thread
// ai:   four adjacent k (vec4f), the taps the C's (sim/ob.mjs resampleGeom), idx its first samples wrapped mod n on the
// ai:   host: the picture read as periodic, so the q pixels cover the picture and its guard interval, which is the
// ai:   picture's own wrap.
// ai: RSH, across and out: a thread one u32 of a frame row, four bytes, the codes symbols side by side (FW = codes W
// ai:   pixels a row, W rows, each row padded to RW words: W need not be a multiple of four, as with pxm 1 in the 256
// ai:   ring, 546 pixels): inside a symbol's square (q pixels from sq) the same six taps along the row of TMP, clamped to 0..1,
// ai:   then (drive 255 + 0.5) truncated and held to 255 as lizard_paint_rgba does; anywhere else the border's byte
// ai:   (BORDER, one symbol W x W, the C's own paint: ring, marks, band, word, quiet margin), so the border is the C's
// ai:   to the byte and nothing border-shaped is drawn here. No integer division: a row a workgroup y, the symbol by
// ai:   subtraction (2026-09-29: with the frame's words flat and the wrap taken mod n a tap, 7.35 ms of the iGPU's
// ai:   13.6 at LIZARD-1024).
// ai: A copy (every pixel's taps the C's on-sample one, [0, 0, 1, 0, 0, 0], wherever the ring's span divides n: every
// ai: picture in the 64 ring but 256, 512, 768, 1024, 1280 and 1536 in the 128, 512, 1024 and 1536 in the 256, 15 of the 24 pairs, SPEC 3.3): RSV is not run and RSH reads the picture at the one tap's sample on both axes, the value the two passes
// ai: would have given exactly (rshSource({ copy: true })).
// ai: PRESENT, a render pass onto the canvas: pixel (x, y) shows the frame's byte (x / whole, y / whole), grey.
// ai: The tolerance: the square within one grey level of the C's (its FFT's rounding differs), the border equal.
// ai: The anchors (SPEC 6.13; src/lizard_phy.c lizard_encode paints them): RSH and its copy form take the border's byte over
// ai: the anchors' squares too, BORDER being the C's paint (core/tx/send_tables.cpp, gpu/encoder.mjs), so the anchors
// ai: are the C's to the byte and nothing here draws them; RSH's uniform two vec4u longer (GA: each anchor's first
// ai: pixel and the side).

import { CRC_SPLIT, PARAMS_STRUCT } from "./back_ldpc.mjs";
import { KNOWN, NC } from "../plan/back_tiers.mjs";

export const TPOSE_TILE = 16;
export const RSV_THREADS = 64;
export const RSH_THREADS = 64;
export const GA_BYTES = 80;

// ai: G: a = (n, q, sq, W): the picture, the square's pixels, its first pixel in a symbol (margin included), the
// ai: symbol's width; c = (whole, f0, gap, 0): the present's whole canvas px a frame px, an encode's first frame, and the gap
// ai: between two codes in px (gpu/encoder.mjs GAP_MODULES modules, 2026-09-30); b = (codes, FW, RW, FS): symbols a frame, the frame's width in pixels, a row's in u32 and a frame's
// ai: in a batch's FRAME (RW W, padded to 256 B so PRESENT binds a frame at its offset); c = (whole, f0, gap, 0): canvas
// ai: pixels a frame pixel (PRESENT) and the frame of FRAME the dispatch's first writes (RSH). TAPS: idx (6 q u32, a
// ai: tap's sample mod n), then w (6 q f32, bitcast), both passes alike. A dispatch is frames x codes symbols of
// ai: scratch, symbol r = frame codes + code, a workgroup z each (RSH: a frame, written to frame f0 + z of FRAME).
const G = /* wgsl */ `
struct G { a: vec4u, b: vec4u, c: vec4u }
`;
// ai: RSH's G: G, then d = the first pixel of anchors 0 to 3 along either axis of a symbol (margin included: the same
// ai: on both axes, a 7 x 7 grid), e = anchors 4 to 6's and the square's side in pixels. 80 B (GA_BYTES); the other
// ai: stages read G's 48 of the same buffer.
const GA = /* wgsl */ `
struct G { a: vec4u, b: vec4u, c: vec4u, d: vec4u, e: vec4u }
`;

// ai: PAINT, a workgroup a (block, symbol), 128 threads (src/lizard_phy.c lizard_encode and map_out, the C's arithmetic
// ai: exactly): the block's 476 payload bytes from BLOCKS, their CRC-32C (the LDPC's split CRC), its code's systematic
// ai: codeword (the data bits, zeros past the 480 B where the code holds more (the 7/8 code's last 128), then the m parity bits: the IRA
// ai: accumulator, a prefix XOR over the checks in check order, a lane ceil(m / 128) of them), its slots (slot i the
// ai: codeword bit its code's bit map names, a slot past the codeword a bit of its frame's count: the pilots; each
// ai: XOR the whitening at its slot of the frame) and the QPSK coefficients (+-a a bit, a = 0.70710678, every
// ai: sub-channel at one amplitude) into the symbol's disc in S at the UV entries. A block's code and first slot come from its tier
// ai: (src/lizard_phy.h lizard_tiers_for: 7/8 blocks of 4496 slots, 3/4 of 5152, 2/3 of 5760, 1/2 of 7680, packed from the
// ai: frame's first slot, the lowest frequencies, out), the codes' shapes and tables from TAB, so the shader holds no code
// ai: of its own. The frame's last block's workgroup also paints the padding past the last block, the whitening alone
// ai: (src/lizard_phy.c lizard_encode), so every coefficient the frame holds is sent.
// ai: Bindings: BLOCKS (ro, 120 words a block, block b of symbol r at (r blocks + b) 120), TAB (ro, gpu/encoder.mjs
// ai: sendTab: word 0 the whitening's offset, then 16 words a code at 16 + 16 c, (n, k, m, z, mb, rows, lay, perm,
// ai: slots, nt): its block's slots, its block rows' first entries, its entries (col | shift << 16), its bit map two
// ai: slots a word; the whitening a bit a slot), UV (ro, a coefficient's S entry), S (rw, the disc), P (uniform,
// ai: wgsl/back_ldpc.mjs PARAMS_AT): sizes[0] = (blocks, UV offset, the slots the blocks take, the coefficients the frame
// ai: holds), sizes[1 + c] = (first block, first slot, blocks, 0) of code c in CODES' order (0 blocks: a code the format
// ai: has none of), dims = (blocks, sStride, symbols, c0 + 4 codes), pw the CRC's powers. Dispatch (blocks, 1,
// ai: symbols): frames x codes of an encode. The pilots (SPEC 7.3): a frame's count is c0 (the encode's first frame's) +
// ai: its place in the encode, mod 4, the frame symbol r / codes; bit 0 on an even block, bit 1 on an odd.
// ai: The test stream on the GPU (2026-11-15, SPEC 9.3): a test block's 471 payload bytes are BLAKE3's extended
// ai: output of its id's five bytes, made in PAINT from the id alone (dims.w's bit 31 set: the host writes the ids, not
// ai: the payloads). Output block t (64 bytes) is one compression of the root's only block (the id, its length 5,
// ai: CHUNK_START | CHUNK_END | ROOT) at counter t, so eight lanes make the eight blocks side by side. B3_PERM is
// ai: BLAKE3's message permutation; the compression is written out with the message words that are always zero (all
// ai: but the id's two) left out. b3Js is the same arithmetic in JS, held to the library's stream (src/xfer.h).
export const B3_IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
const B3_PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
// ai: the compression's statements over v0..v15 (u32), the message m0 (the id's low word) and m1 (its fifth byte), in a
// ai: language's syntax: add(a, b) and rot(x, n) (a rotation right) given by the caller
function b3Steps(add, rot) {
  let pos = [...Array(16).keys()];
  const out = [];
  const g = (a, b, c, d, x, y) => {
    out.push(`v${a} = ${add(add(`v${a}`, `v${b}`), x)};`, `v${d} = ${rot(`v${d} ^ v${a}`, 16)};`, `v${c} = ${add(`v${c}`, `v${d}`)};`,
      `v${b} = ${rot(`v${b} ^ v${c}`, 12)};`, `v${a} = ${add(add(`v${a}`, `v${b}`), y)};`, `v${d} = ${rot(`v${d} ^ v${a}`, 8)};`,
      `v${c} = ${add(`v${c}`, `v${d}`)};`, `v${b} = ${rot(`v${b} ^ v${c}`, 7)};`);
  };
  for (let r = 0; r < 7; r++) {
    const w = (k) => (pos[k] === 0 ? "m0" : pos[k] === 1 ? "m1" : null);
    g(0, 4, 8, 12, w(0), w(1)); g(1, 5, 9, 13, w(2), w(3)); g(2, 6, 10, 14, w(4), w(5)); g(3, 7, 11, 15, w(6), w(7));
    g(0, 5, 10, 15, w(8), w(9)); g(1, 6, 11, 12, w(10), w(11)); g(2, 7, 8, 13, w(12), w(13)); g(3, 4, 9, 14, w(14), w(15));
    pos = pos.map((_, i) => pos[B3_PERM[i]]);
  }
  return out;
}
const B3_WGSL = (() => {
  const add = (a, b) => (b === null ? a : `(${a} + ${b})`), rot = (x, n) => `rotr(${x}, ${n}u)`;
  const iv = (j) => `0x${B3_IV[j].toString(16)}u`;
  return /* wgsl */ `
fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
// ai: output block t of BLAKE3(the id's five bytes) into stw[16 t .. 16 t + 15]
fn b3out(m0: u32, m1: u32, t: u32) {
  var v0 = ${iv(0)}; var v1 = ${iv(1)}; var v2 = ${iv(2)}; var v3 = ${iv(3)};
  var v4 = ${iv(4)}; var v5 = ${iv(5)}; var v6 = ${iv(6)}; var v7 = ${iv(7)};
  var v8 = ${iv(0)}; var v9 = ${iv(1)}; var v10 = ${iv(2)}; var v11 = ${iv(3)};
  var v12 = t; var v13 = 0u; var v14 = 5u; var v15 = 11u;
  ${b3Steps(add, rot).join("\n  ")}
  let o = 16u * t;
  ${[...Array(8).keys()].map((j) => `stw[o + ${j}u] = v${j} ^ v${j + 8};`).join(" ")}
  ${[...Array(8).keys()].map((j) => `stw[o + ${j + 8}u] = v${j + 8} ^ ${iv(j)};`).join(" ")}
}`;
})();
// ai: the JS twin: output block t (16 u32) of the id's BLAKE3, the WGSL's statements run as JS
export const b3Js = (() => {
  const add = (a, b) => (b === null ? a : `((${a} + ${b}) >>> 0)`), rot = (x, n) => `(((${x}) >>> ${n}) | ((${x}) << ${32 - n})) >>> 0`;
  const body = `let v0=${B3_IV[0]},v1=${B3_IV[1]},v2=${B3_IV[2]},v3=${B3_IV[3]},v4=${B3_IV[4]},v5=${B3_IV[5]},v6=${B3_IV[6]},v7=${B3_IV[7]},` +
    `v8=${B3_IV[0]},v9=${B3_IV[1]},v10=${B3_IV[2]},v11=${B3_IV[3]},v12=t>>>0,v13=0,v14=5,v15=11;\n` +
    b3Steps(add, rot).join("\n") + `\nconst o=new Uint32Array(16);` +
    [...Array(8).keys()].map((j) => `o[${j}]=(v${j}^v${j + 8})>>>0;o[${j + 8}]=(v${j + 8}^${B3_IV[j]})>>>0;`).join("") + "return o;";
  return new Function("m0", "m1", "t", body);
})();

export const PAINT_THREADS = 128;
export const CW_WORDS = 240;   // ai: the longest codeword's words, the 1/2 code's 7680 bits
export function paintSource() {
  return /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> BLOCKS: array<u32>;
@group(0) @binding(1) var<storage, read> TAB: array<u32>;
@group(0) @binding(2) var<storage, read> UV: array<u32>;
@group(0) @binding(3) var<storage, read_write> S: array<vec2f>;
@group(0) @binding(4) var<uniform> P: Params;
const KNOWN: u32 = ${KNOWN}u;
const TH: u32 = ${PAINT_THREADS}u;
const A: f32 = 0.70710678;
var<workgroup> packw: array<u32, 120>;
var<workgroup> crcw: atomic<u32>;
var<workgroup> cw: array<atomic<u32>, ${CW_WORDS}>;
var<workgroup> scan: array<u32, ${PAINT_THREADS}>;
var<workgroup> stw: array<u32, 128>;   // ai: a test block's stream, eight BLAKE3 output blocks (B3_WGSL)
${B3_WGSL}

// ai: Data bit j of the codeword: the payload bytes MSB first, then the CRC MSB first, as packw holds them (four
// ai: bytes a word in memory order); zeros past them.
fn dataBit(j: u32) -> u32 {
  if (j >= KNOWN) { return 0u; }
  let t = j & 31u;
  return (packw[j >> 5u] >> (8u * (t >> 3u) + 7u - (t & 7u))) & 1u;
}
fn cwBit(j: u32) -> u32 { return (atomicLoad(&cw[j >> 5u]) >> (j & 31u)) & 1u; }
fn white(s: u32) -> u32 { return (TAB[TAB[0] + (s >> 5u)] >> (s & 31u)) & 1u; }
${CRC_SPLIT()}
@compute @workgroup_size(${PAINT_THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) i: u32) {
  let b: u32 = wg.x;
  let r: u32 = wg.z;
  if (b >= P.sizes[0].x) { return; }
  // ai: the block's tier: its code, and its first slot
  var c: u32 = 0u;
  for (var q: u32 = 0u; q < ${NC}u; q = q + 1u) {
    let t = P.sizes[1u + q];
    if (b >= t.x && b < t.x + t.z) { c = q; }
  }
  let tt = P.sizes[1u + c];
  let h: u32 = 16u + 16u * c;
  let n: u32 = TAB[h];
  let k: u32 = TAB[h + 1u];
  let m: u32 = TAB[h + 2u];
  let z: u32 = TAB[h + 3u];
  let mb: u32 = TAB[h + 4u];
  let rowsAt: u32 = TAB[h + 5u];
  let layAt: u32 = TAB[h + 6u];
  let permAt: u32 = TAB[h + 7u];
  let slots: u32 = TAB[h + 8u];   // ai: the block's slots: its sent bits, then its pilots
  let nt: u32 = TAB[h + 9u];   // ai: the bits sent, n less the code's np never sent (the 7/8 code's 128 known zeros): the slots the codeword fills
  let cpl: u32 = (m + TH - 1u) / TH;
  let slot0: u32 = tt.y + (b - tt.x) * slots;
  // ai: this block's bit of its symbol's frame's count: the pilots' flip
  let pq: u32 = ((((P.dims.w & 3u) + r / max((P.dims.w >> 2u) & 0xffffu, 1u)) & 3u) >> (b & 1u)) & 1u;
  // ai: The payload from BLOCKS: bytes 0 to 475 in words 0 to 118; its CRC-32C big-endian in word 119 (bytes 476 to 479).
  let base: u32 = (r * P.sizes[0].x + b) * 120u;
  if ((P.dims.w >> 31u) != 0u) {
    // ai: the test stream: the payload made here from the id BLOCKS holds (its first five bytes), the block's words the
    // ai: id's then the stream's five bytes on (word w from 2: stream bytes 4 w - 5 to 4 w - 2)
    let m0: u32 = BLOCKS[base];
    let m1: u32 = BLOCKS[base + 1u] & 0xffu;
    if (i < 8u) { b3out(m0, m1, i); }
    workgroupBarrier();
    if (i == 0u) { packw[0] = m0; }
    if (i == 1u) { packw[1] = m1 | (stw[0] << 8u); }
    if (i >= 2u && i < CRC_CHUNKS) { packw[i] = (stw[i - 2u] >> 24u) | (stw[i - 1u] << 8u); }
  } else {
    if (i < CRC_CHUNKS) { packw[i] = BLOCKS[base + i]; }
  }
  if (i == CRC_CHUNKS) { packw[CRC_CHUNKS] = 0u; }
  workgroupBarrier();
  let crc: u32 = crcOf(i);
  if (i == 0u) {
    packw[CRC_CHUNKS] = ((crc >> 24u) & 0xffu) | (((crc >> 16u) & 0xffu) << 8u) | (((crc >> 8u) & 0xffu) << 16u) | ((crc & 0xffu) << 24u);
  }
  workgroupBarrier();
  // ai: The systematic part of the codeword, a bit an index, every word of it (the parity's zeros too); then every
  // ai: lane's cpl checks in check order, the check's parity as the XOR of its data bits, the prefix within them
  // ai: kept a bit each.
  for (var w: u32 = i; w < (n + 31u) / 32u; w = w + TH) {
    var word: u32 = 0u;
    for (var t: u32 = 0u; t < 32u; t = t + 1u) {
      let j = 32u * w + t;
      if (j < k) { word = word | (dataBit(j) << t); }
    }
    atomicStore(&cw[w], word);
  }
  var pl: u32 = 0u;
  var acc: u32 = 0u;
  for (var q: u32 = 0u; q < cpl; q = q + 1u) {
    let chk = cpl * i + q;
    if (chk < m) {
      let ii = chk / mb;
      let rr = chk % mb;
      var t: u32 = 0u;
      let e1: u32 = TAB[rowsAt + rr + 1u];
      // ai: the data bit of entry e in check (rr, ii): col z + (ii - shift) mod z (src/ldpc.c ldpc_init)
      for (var e: u32 = TAB[rowsAt + rr]; e < e1; e = e + 1u) {
        let lc = TAB[layAt + e];
        var s: u32 = ii + z - (lc >> 16u);
        if (s >= z) { s = s - z; }
        t = t ^ dataBit((lc & 0xffffu) * z + s);
      }
      acc = acc ^ t;
    }
    pl = pl | (acc << q);
  }
  scan[i] = acc;
  workgroupBarrier();
  // ai: The prefix XOR across lanes (ldpc_encode's running acc), a read, a barrier and a write a step.
  for (var d: u32 = 1u; d < TH; d = d << 1u) {
    var v: u32 = scan[i];
    if (i >= d) { v = v ^ scan[i - d]; }
    workgroupBarrier();
    scan[i] = v;
    workgroupBarrier();
  }
  let ex: u32 = scan[i] ^ acc;
  for (var q: u32 = 0u; q < cpl; q = q + 1u) {
    let chk = cpl * i + q;
    if (chk < m && (((pl >> q) & 1u) ^ ex) != 0u) {
      let j = k + chk;
      atomicOr(&cw[j >> 5u], 1u << (j & 31u));
    }
  }
  workgroupBarrier();
  // ai: Slots 2 cc and 2 cc + 1 of coefficient cc (every code's nt is even: a pair is inside the sent bits or past them),
  // ai: one bit map word the pair (each entry a codeword bit, the never-sent ones already skipped: back/bitmap.mjs sentBit); the whitening at the frame's slot.
  let uvBase: u32 = P.sizes[0].y + slot0 / 2u;
  let sBase: u32 = r * P.dims.y;
  for (var cc: u32 = i; cc < slots / 2u; cc = cc + TH) {
    let i0: u32 = 2u * cc;
    var b0: u32 = pq;
    var b1: u32 = pq;
    if (i0 < nt) {
      let pe = TAB[permAt + cc];
      b0 = cwBit(pe & 0xffffu);
      b1 = cwBit(pe >> 16u);
    }
    b0 = b0 ^ white(slot0 + i0);
    b1 = b1 ^ white(slot0 + i0 + 1u);
    S[sBase + UV[uvBase + cc]] = vec2f(select(A, -A, b0 != 0u), select(A, -A, b1 != 0u));
  }
  // ai: the padding: coefficients from the slots the blocks take to the frame's last, the whitening alone
  if (b + 1u == P.sizes[0].x) {
    for (var cc: u32 = P.sizes[0].z / 2u + i; cc < P.sizes[0].w; cc = cc + TH) {
      let s: u32 = 2u * cc;
      S[sBase + UV[P.sizes[0].y + cc]] = vec2f(select(A, -A, white(s) != 0u), select(A, -A, white(s + 1u) != 0u));
    }
  }
}
`;
}

// ai: Bindings: PICQ (ro, vec4f), PIC (rw, vec4f), G. Dispatch (n / 4 / TPOSE_TILE, n / TPOSE_TILE, symbols), 64 threads.
export function tposeSource() {
  return G + /* wgsl */ `
@group(0) @binding(0) var<storage, read> PICQ: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> PIC: array<vec4f>;
@group(0) @binding(2) var<uniform> P: G;
var<workgroup> tile: array<vec4f, ${TPOSE_TILE * TPOSE_TILE}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) i: u32) {
  let n: u32 = P.a.x;
  let n4: u32 = n / 4u;
  let base: u32 = wg.z * n4 * n;
  let q0: u32 = wg.x * ${TPOSE_TILE}u;
  let y0: u32 = wg.y * ${TPOSE_TILE}u;
  for (var j: u32 = 0u; j < ${(TPOSE_TILE * TPOSE_TILE) / 64}u; j = j + 1u) {
    let e: u32 = i + 64u * j;
    let qd: u32 = e / ${TPOSE_TILE}u;
    let r: u32 = e % ${TPOSE_TILE}u;
    tile[r * ${TPOSE_TILE}u + qd] = PICQ[base + (q0 + qd) * n + y0 + r];
  }
  workgroupBarrier();
  for (var j: u32 = 0u; j < ${(TPOSE_TILE * TPOSE_TILE) / 64}u; j = j + 1u) {
    let e: u32 = i + 64u * j;
    let r: u32 = e / ${TPOSE_TILE}u;
    let qd: u32 = e % ${TPOSE_TILE}u;
    PIC[base + (y0 + r) * n4 + q0 + qd] = tile[e];
  }
}
`;
}

// ai: Bindings: PIC (ro, vec4f), TAPS (ro), TMP (rw, vec4f), G. Dispatch (ceil(n / 4 / RSV_THREADS), q, symbols).
export function rsvSource() {
  return G + /* wgsl */ `
@group(0) @binding(0) var<storage, read> PIC: array<vec4f>;
@group(0) @binding(1) var<storage, read> TAPS: array<u32>;
@group(0) @binding(2) var<storage, read_write> TMP: array<vec4f>;
@group(0) @binding(3) var<uniform> P: G;
@compute @workgroup_size(${RSV_THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n4: u32 = P.a.x / 4u;
  let q: u32 = P.a.y;
  let k4: u32 = id.x;
  let y: u32 = id.y;
  let r: u32 = id.z;
  if (k4 >= n4) { return; }
  let base: u32 = r * P.a.x * n4 + k4;
  var v: vec4f = bitcast<f32>(TAPS[6u * q + 6u * y]) * PIC[base + TAPS[6u * y] * n4];
  for (var t: u32 = 1u; t < 6u; t = t + 1u) {
    v = v + bitcast<f32>(TAPS[6u * q + 6u * y + t]) * PIC[base + TAPS[6u * y + t] * n4];
  }
  TMP[(r * q + y) * n4 + k4] = v;
}
`;
}

// ai: Bindings: TMP (ro, f32; with copy the picture, PIC, n x n a symbol), TAPS (ro), BORDER (ro, one symbol's bytes,
// ai: four a u32), FRAME (rw, a half's frames FS words apart), G. Dispatch (ceil(RW / RSH_THREADS), W, frames).
// ai: A pixel on an anchor's row and column takes BORDER's byte as the border's do: a row's test once a thread (a
// ai: workgroup is one row, so it is uniform there), a column's only on such a row, each seven wrapped differences
// ai: against the side (integers alone: no device's rounding moves an anchor's edge).
export function rshSource({ copy = false } = {}) {
  const inside = copy ? /* wgsl */ `
    let v: f32 = clamp(TMP[k * P.a.x * P.a.x + TAPS[6u * (y - sq) + 2u] * P.a.x + TAPS[6u * xx + 2u]], 0.0, 1.0);` : /* wgsl */ `
    let base: u32 = (k * q + (y - sq)) * P.a.x;
    var v: f32 = bitcast<f32>(TAPS[6u * q + 6u * xx]) * TMP[base + TAPS[6u * xx]];
    for (var t: u32 = 1u; t < 6u; t = t + 1u) {
      v = v + bitcast<f32>(TAPS[6u * q + 6u * xx + t]) * TMP[base + TAPS[6u * xx + t]];
    }
    v = clamp(v, 0.0, 1.0);`;
  const A = {
    struct: GA,
    fn: /* wgsl */ `
// ai: whether v, a symbol's pixel along either axis, lies on one of the anchors' seven runs (P.d, P.e.xyz from their
// ai: first pixels, P.e.w long)
fn onAnchor(v: u32) -> bool {
  let s: u32 = P.e.w;
  return (v - P.d.x < s) | (v - P.d.y < s) | (v - P.d.z < s) | (v - P.d.w < s) | (v - P.e.x < s) | (v - P.e.y < s) | (v - P.e.z < s);
}`,
    sig: ", ra: bool", test: " && !(ra && onAnchor(xs))", row: /* wgsl */ `
  let ra: bool = onAnchor(y);`, arg: ", ra",
  };
  return A.struct + /* wgsl */ `
@group(0) @binding(0) var<storage, read> TMP: array<f32>;
@group(0) @binding(1) var<storage, read> TAPS: array<u32>;
@group(0) @binding(2) var<storage, read> BORDER: array<u32>;
@group(0) @binding(3) var<storage, read_write> FRAME: array<u32>;
@group(0) @binding(4) var<uniform> P: G;${A.fn}
fn pixel(x: u32, y: u32, f: u32${A.sig}) -> u32 {
  let q: u32 = P.a.y;
  let sq: u32 = P.a.z;
  let W: u32 = P.a.w;
  var xs: u32 = x;
  var k: u32 = f * P.b.x;
  // ai: the codes W wide with a gap of P.c.z between them (gpu/encoder.mjs GAP_MODULES): a gap pixel takes the
  // ai: margin's byte beside it
  loop { if (xs < W + P.c.z) { break; } xs = xs - W - P.c.z; k = k + 1u; }
  if (xs >= W) { xs = W - 1u; }
  if (xs >= sq && xs < sq + q && y >= sq && y < sq + q${A.test}) {
    let xx: u32 = xs - sq;` + inside + /* wgsl */ `
    return min(u32(v * 255.0 + 0.5), 255u);
  }
  let j: u32 = y * W + xs;
  return (BORDER[j >> 2u] >> (8u * (j & 3u))) & 0xffu;
}
@compute @workgroup_size(${RSH_THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let wx: u32 = id.x;
  let y: u32 = id.y;
  let f: u32 = id.z;
  if (wx >= P.b.z) { return; }${A.row}
  var word: u32 = 0u;
  for (var b: u32 = 0u; b < 4u; b = b + 1u) {
    let x: u32 = 4u * wx + b;
    if (x < P.b.y) { word = word | (pixel(x, y, f${A.arg}) << (8u * b)); }
  }
  FRAME[(P.c.y + f) * P.b.w + y * P.b.z + wx] = word;
}
`;
}

// ai: WORDS (2026-11-15, the animated header, SPEC 5.5; core/tx/gpu_painter.cpp encode's car): after rsh, each code's
// ai: word cells painted with its own carousel symbol, so a batch of frames shows its symbols in turn. SPANS: three u32 a
// ai: frame word (4 pixels) the word cells touch, its index within a frame's rows with its code in bit 31, then its four
// ai: bytes' bit indices (16 bits each, bytes 0 and 1, then 2 and 3; 0xffff a byte kept); WALL: the 255 symbols' words as
// ai: bits (U.a.w u32 a symbol, bit i the word's cell side * W + cell, src/fmt.c ob_fmt_bit); POS: each frame's codes'
// ai: positions (U.b.x codes a frame), 255 a code's static word kept (core/tx/sender.cpp draws them); a bit 1 dark (0),
// ai: 0 light (255), the C's border's levels. One invocation a frame word, so no two write one word.
export const WORDS_THREADS = 64;
export function wordsSource() {
  return /* wgsl */ `
struct WU { a: vec4u, b: vec4u }
@group(0) @binding(0) var<storage, read> SPANS: array<u32>;
@group(0) @binding(1) var<storage, read> WALL: array<u32>;
@group(0) @binding(2) var<storage, read_write> FRAME: array<u32>;
@group(0) @binding(3) var<uniform> U: WU;
@group(0) @binding(4) var<storage, read> POS: array<u32>;
@compute @workgroup_size(${WORDS_THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let e: u32 = id.x;
  let f: u32 = id.y;
  if (e >= U.a.x) { return; }
  let s: u32 = SPANS[3u * e];
  let p: u32 = POS[f * U.b.x + (s >> 31u)];
  if (p >= 255u) { return; }
  let at: u32 = (U.a.y + f) * U.a.z + (s & 0x7fffffffu);
  var word: u32 = FRAME[at];
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    let idx: u32 = (SPANS[3u * e + 1u + (k >> 1u)] >> (16u * (k & 1u))) & 0xffffu;
    if (idx != 0xffffu) {
      let bit: u32 = (WALL[p * U.a.w + (idx >> 5u)] >> (idx & 31u)) & 1u;
      let v: u32 = select(255u, 0u, bit == 1u);
      word = (word & ~(0xffu << (8u * k))) | (v << (8u * k));
    }
  }
  FRAME[at] = word;
}
`;
}

// ai: Bindings (fragment): FRAME (ro, one frame: bound at its offset in the batch's), G. A triangle over the canvas; no
// ai: vertex buffer.
export function presentSource() {
  return G + /* wgsl */ `
@group(0) @binding(0) var<storage, read> FRAME: array<u32>;
@group(0) @binding(1) var<uniform> P: G;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p: vec2f = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let whole: u32 = max(P.c.x, 1u);
  let x: u32 = min(u32(pos.x) / whole, P.b.y - 1u);
  let y: u32 = min(u32(pos.y) / whole, P.a.w - 1u);
  let i: u32 = 4u * P.b.z * y + x;
  let g: f32 = f32((FRAME[i >> 2u] >> (8u * (i & 3u))) & 0xffu) / 255.0;
  return vec4f(mix(rgbOf(P.c.y), rgbOf(P.c.w), g), 1.0);
}
// ai: the code's colours (2026-11-16, Send's Colours): the present's G.c.y the ink and G.c.w the paper, 0xRRGGBB, the
// ai: grey painted along the line between them as src/wasm.c lizard_tx_rgba and core/tx/expand.comp do
fn rgbOf(c: u32) -> vec3f {
  return vec3f(f32((c >> 16u) & 0xffu), f32((c >> 8u) & 0xffu), f32(c & 0xffu)) / 255.0;
}
`;
}
