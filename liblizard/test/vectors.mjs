// ai: The SPEC.md conformance vectors (4.7 the track, 4.9 one corner, 5.2 and 5.3 through the painted cells, 6.12 P2 to P4,
// ai: 7.4, 7.10, 7.11) against one build,
// ai: plus digests over the ladder (layout cells at five rates, the drive, the coefficient order) to diff between two
// ai: builds of today's interface (lizard_setup's three arguments, n, subch and span, since 2026-11-06).
//   node test/vectors.mjs [path/to/ob.mjs]      from liblizard/, about 20 s; default build/ob.mjs, the codec's module
import { createHash } from "node:crypto";
import { resolve } from "node:path";

const path = process.argv[2] ?? "build/ob.mjs";
const M = await (await import(resolve(path))).default();
// ai: a block and its payload, bytes (src/xfer.h XFER_BLOCK, XFER_PAYLOAD)
const BLOCK = 472, PAYLOAD = BLOCK - 12;
const sha = (u8) => createHash("sha256").update(u8).digest("hex");
const hex = (u8) => Buffer.from(u8).toString("hex");
let fails = 0;
const want = (name, got, exp) => { const ok = got === exp; if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}${ok ? "" : `  (SPEC ${exp})`}`); };

// ai: span 2B names ring B, 0 the default ring (the 128 since 2026-10-01). SPEC 6.12's vectors and the ladder's digests
// ai: are the 64 ring's, so it is named: span 128.
const setup = (n, subch, span = 128) => M._lizard_setup(n, subch, span);
// ai: The tree's ladder (2^k and 3 x 2^k since 2026-09-27, 1152 and 1280 since 2026-10-29): a build from before it
// ai: cannot build the sizes added since.
const { N_FOR: nFor, VERSIONS } = await import(resolve("sim/lizard_pick.mjs"));
console.log(`${path}: lizard_setup takes ${M._lizard_setup.length} arguments`);

// ai: (int)(v * 255 + 0.5) clamped, in float32 as lizard_paint_rgba's scalar path does it (SPEC 6.8, Painting).
const grey = (v) => { const t = Math.fround(Math.fround(v * 255) + 0.5); const i = Math.trunc(t); return i < 0 ? 0 : i > 255 ? 255 : i; };

// ai: again = false encodes on the codec already set up, whose block bytes the ladder loop holds.
let blockBytes = 0;
function encode(n, subch, fill, again = true, span = 128) {
  const B = again ? setup(n, subch, span) : blockBytes;
  if (B < 0) throw new Error(`setup ${n}/${subch} refused`);
  const T = M._lizard_blocks(), side = M._lizard_side();
  if (M._lizard_dbg(1)) throw new Error("dbg");
  const pB = M._malloc(T * B), pD = M._malloc(4 * side * side);
  const blocks = new Uint8Array(T * B);
  fill(blocks, B, T);
  M.HEAPU8.set(blocks, pB);
  M._lizard_tx(pB, pD);
  const drive = M.HEAPF32.slice(pD >> 2, (pD >> 2) + side * side);
  const sym = new Int8Array(M.HEAPU8.buffer, M._lizard_dbg_sym(), 2 * 320 * subch).slice();
  M._lizard_dbg(0); M._free(pB); M._free(pD);
  return { B, T, side, drive, slots: Uint8Array.from(sym, (s) => (s < 0 ? 1 : 0)) };
}
const pack = (bits) => { const o = new Uint8Array(Math.ceil(bits.length / 8)); bits.forEach((b, i) => { if (b) o[i >> 3] |= 0x80 >> (i & 7); }); return o; };
const greyRow = (e, y, x0, k) => Array.from({ length: k }, (_, i) => grey(e.drive[y * e.side + x0 + i])).join(" ");
const greySquare = (e, a, b) => { const o = new Uint8Array((b - a) * (b - a)); let q = 0; for (let y = a; y < b; y++) for (let x = a; x < b; x++) o[q++] = grey(e.drive[y * e.side + x]); return o; };

// ai: A symbol's cell map (SPEC 4.1: S x S modules, src/layout.h CELL_* a module, CELL_WORD on the word's cells) for
// ai: LIZARD-subch in ring B at a layout (null the default senders'; [6/7, 3/4, 2/3, 1/2] through _lizard_layout_set) and a
// ai: clock [fps, delay] (_lizard_clock_set).
function cellMap(subch, B, layout = null, clock = [60, 1]) {
  if (setup(nFor(subch), subch, 2 * B) < 0) throw new Error(`setup LIZARD-${subch} ring ${B}`);
  if (layout && M._lizard_layout_set(...layout)) throw new Error(`layout ${layout}`);
  if (M._lizard_clock_set(...clock)) throw new Error(`clock ${clock}`);
  const pDims = M._malloc(24);
  M._ob_test_mesh_tables_out(pDims, 0, 0, 0, 0);
  const d = M.HEAP32.slice(pDims >> 2, (pDims >> 2) + 5), S = d[2];
  const pX = M._malloc(4 * d[0]), pY = M._malloc(4 * d[1]), pK = M._malloc(S * S), pMark = M._malloc(4 * d[0] * d[1]);
  M._ob_test_mesh_tables_out(pDims, pX, pY, pK, pMark);
  const kind = M.HEAPU8.slice(pK, pK + S * S);
  [pDims, pX, pY, pK, pMark].forEach((p) => M._free(p));
  return { S, kind };
}
// ai: module t of side s's band at depth 5 (0 top, 1 right, 2 bottom, 3 left; t counted as 4.1 counts a side's modules)
const bandAt = (S, s, t) => [[t, 5], [S - 6, t], [t, S - 6], [5, t]][s];
// ai: each side's word cells in order, dark = 1 (a cell two modules along the band)
function wordCells({ S, kind }) {
  const sides = [];
  for (let s = 0; s < 4; s++) {
    const bits = [];
    for (let t = 0; t < S; t++) { const [x, y] = bandAt(S, s, t), k = kind[y * S + x]; if (k & 8) { bits.push((k & 3) === 2 ? 1 : 0); t++; } }
    sides.push(bits);
  }
  return sides;
}
const nibbles = (bits) => bits.join("").match(/.{1,4}/g).join(" ");

// ai: 4.7, the timing track: the hash's four steps for side 0's pair 0, cells 0 to 15 of each side by the hash, and every
// ai: track cell of LIZARD-64 painted in each ring held to the hash (a band cell from module 15 of its side, OB_THIN_RESERVE,
// ai: two modules a cell, the word's cells skipped, track cell j the j-th of the rest).
{
  const steps = (s, p) => {
    const v1 = (Math.imul(Math.imul(s, 4099) + p, 2654435761) + 40503) >>> 0, v2 = (v1 ^ (v1 >>> 15)) >>> 0;
    const v3 = Math.imul(v2, 2246822519) >>> 0, v4 = (v3 ^ (v3 >>> 13)) >>> 0;
    return [v1, v2, v3, v4];
  };
  const dark = (s, j) => (steps(s, j >> 1)[3] ^ j) & 1;
  want("4.7 side 0 pair 0, v at each step", steps(0, 0).map((v) => v.toString(16).padStart(8, "0")).join(" "), "00009e37 00009e36 c6b2271a c6b4128b");
  ["1010 1010 1001 1010", "1010 0110 1010 0110", "0101 0110 1010 0101", "0110 1001 0101 0110"].forEach((exp, s) =>
    want(`4.7 side ${s} track cells 0..15`, nibbles(Array.from({ length: 16 }, (_, j) => dark(s, j))), exp));
  for (const B of [64, 128, 256]) {
    const L = cellMap(64, B);
    let n = 0, off = 0;
    for (let s = 0; s < 4; s++) {
      let j = 0;
      for (let t = 15; t < L.S - 15; t += 2) {
        const [x, y] = bandAt(L.S, s, t), k = L.kind[y * L.S + x];
        if (k & 8) continue;
        if (((k & 3) === 2 ? 1 : 0) !== dark(s, j)) off++;
        n++; j++;
      }
    }
    want(`4.7 ring ${B}: painted track cells unlike the hash`, `${off} of ${n}`, `0 of ${2 * B}`);
  }
}

// ai: 4.9, one corner: LIZARD-64 in the 64 ring at the default senders' counts, 60 a second, delay 1. The top-left
// ai: corner, modules x 0 to 37 and y 0 to 17, and the band at depth 5 along each side, drawn from the cell map: W / w a
// ai: word cell dark / light; T / t a track cell (the band, depth 5 or 6, from module 15 to S - 16 of its side); # / .
// ai: any other dark / light module; g a guard module (depth 12 to 14, LIZARD_CP), P the picture (depth 15 on).
{
  const L = cellMap(64, 64), S = L.S;
  const band = (x, y) => ([5, 6, S - 7, S - 6].includes(y) && x >= 15 && x < S - 15) || ([5, 6, S - 7, S - 6].includes(x) && y >= 15 && y < S - 15);
  const ch = (x, y) => {
    const k = L.kind[y * S + x], d = Math.min(x, y, S - 1 - x, S - 1 - y);
    if (k & 8) return (k & 3) === 2 ? "W" : "w";
    if ((k & 3) === 2) return band(x, y) ? "T" : "#";
    if ((k & 3) === 1) return band(x, y) ? "t" : ".";
    return d >= 15 ? "P" : d >= 12 ? "g" : "?";
  };
  const corner = Array.from({ length: 18 }, (_, y) => Array.from({ length: 38 }, (_, x) => ch(x, y)).join(""));
  want("4.9 corner, x 0..37, y 0..17", corner.join("|"), [
    "......................................",
    ".#####################################",
    ".#####################################",
    ".##...................................",
    ".##...................................",
    ".##..######....TTttTTttwwWWwwwwTTttTTt",
    ".##..######....TTttTTttwwWWwwwwTTttTTt",
    ".##..######...........................",
    ".##..######...........................",
    ".##..######...........................",
    ".##..######...........................",
    ".##...................................",
    ".##.........gggggggggggggggggggggggggg",
    ".##.........gggggggggggggggggggggggggg",
    ".##.........gggggggggggggggggggggggggg",
    ".##..tt.....gggPPPPPPPPPPPPPPPPPPPPPPP",
    ".##..tt.....gggPPPPPPPPPPPPPPPPPPPPPPP",
    ".##..TT.....gggPPPPPPPPPPPPPPPPPPPPPPP",
  ].join("|"));
  [
    ".##..######....TTttTTttwwWWwwwwTTttTTttWWWWwwwwTTttttTTWWWWWWWWTTttTTttWWwwWWwwTTttttTTwwWWwwwwTTttttTTWWwwWWWWttTTttTTWWwwwwWWttTTTTttwwWWWWWW....######..##.",
    ".##..######....TTttTTttwwwwwwwwttTTTTttwwwwwwwwTTttTTttwwWWWWWWttTTTTttWWwwWWWWTTttttTTWWWWwwwwttTTttTTwwWWWWwwttTTTTttWWWWWWWWTTttttTTwwWWWWww....######..##.",
    ".##..######....ttTTttTTwwwwwwwwttTTTTttwwwwwwwwTTttTTttWWwwwwWWttTTttTTwwWWWWWWTTttTTttWWWWwwwwttTTTTttwwWWWWWWttTTttTTwwwwWWWWttTTttTTWWwwWWww....######..##.",
    ".##..######....ttTTTTttwwwwWWwwTTttttTTWWwwWWWWttTTttTTWWwwwwwwttTTTTttwwWWWWwwttTTTTttwwWWwwwwTTttTTttWWwwWWWWttTTTTttWWwwWWwwttTTTTttWWWWwwWW....######..##.",
  ].forEach((exp, s) => want(`4.9 band at depth 5, side ${s}`, Array.from({ length: S }, (_, t) => ch(...bandAt(S, s, t))).join(""), exp));
}

// ai: 5.3, the placement: 5.2's fourth and fifth vectors (LIZARD-64 at the default senders' counts, 60 a second, delay
// ai: 1, in the 64 and the 128 ring), each side's word cells in order.
for (const [B, exp] of [
  [64, ["0100 1100 1111 1010 0100 1011 1001 0111", "0000 0000 0111 1011 1100 0110 1111 0110", "0000 0000 1001 0111 1100 0111 0011 1010", "0010 1011 1000 0110 0100 1011 1010 1101"]],
  [128, ["0100 1100 1111 1010 1110 0100 1100 1110 0111 0010 1000 1101 1001 0111 1101 0000", "0000 0000 0111 1011 1011 1111 0111 1100 1001 0111 1100 1100 0010 0010 1010 1101",
    "0000 0000 1110 0100 1110 1111 1010 0100 0110 0100 0010 1001 1000 0000 0000 0011", "0010 1011 1001 1001 0000 1010 0000 1101 1100 1110 0111 1101 0000 0010 0010 1101"]],
]) wordCells(cellMap(64, B)).forEach((bits, s) => want(`5.3 ring ${B} side ${s} word cells`, nibbles(bits), exp[s]));

// ai: 5.2's words, each symbol painted in the ring named, the word read back out of the layout's cells: word cell c
// ai: of side s is the c-th CELL_WORD cell along the band (depth 5), dark = 1, and shows bit 7 - c mod 8 of byte
// ai: 4 floor(c / 8) + s (5.3).
{
  // ai: [subch, ring, layout (null the default senders'; [6/7, 3/4, 2/3, 1/2] through _lizard_layout_set), the clock
  // ai: [fps, delay] (60 and 1 the codec's until set, _lizard_clock_set), the codeword]; since 2026-11-23 the 40 bits the
  // ai: index alone, every layout of the four counts, RS(n, 6) (a 35-bit index and its CRC-5, the 3/4 filling, from
  // ai: 2026-11-22; the 32-bit static word and RS(n, 5) before)
  for (const [subch, B, layout, clock, exp] of [
    [512, 64, null, [60, 1], "4c063a383d78d40782800b0278f682d9"],
    [1024, 64, null, [60, 1], "4cb1d6740c644793200d56dba803ca22"],
    [576, 64, null, [60, 4], "4c0ae6c99ca5f564b4c913f656f83dca"],
    [64, 64, null, [60, 1], "4c00002bfa7b97864bc6c74b97f63aad"],
    [64, 128, null, [60, 1], "4c00002bfa7be499e4bfef0ace7ca40d729764ce8dcc297d97228002d0ad032d"],
    [256, 256, null, [30, 2], "4c003df80ea2154994885d3fb06778874fd0afdabe0a5968957866ee53f71745a8008d56bb4022fde941fd0e01638cc6fcd56dad53fe8c28fa709a437c711d54"],
    [1072, 64, [90, 0, 0, 30], [240, 30], "4ce04b3aa62067c2e58f7b025f172a8f"],
    [16, 64, null, [30, 2], "4c000000900de6f08b482b52049b997b"],
    [1072, 128, [152, 0, 0, 0], [0, 1], "4ce0c82eedd5fa520494b962de0eae552d16c5c9c33c062b7d1fad22fcb2c23e"],
  ]) {
    const sides = wordCells(cellMap(subch, B, layout, clock));
    const cw = new Uint8Array(exp.length / 2);
    for (let k = 0; k < cw.length; k++) for (let b = 0; b < 8; b++) cw[k] |= (sides[k % 4][8 * Math.floor(k / 4) + b] ?? 0) << (7 - b);
    want(`5.2 LIZARD-${subch} ring ${B}${layout ? ` layout ${layout}` : ""} clock ${clock} (${sides[0].length} word cells a side)`, hex(cw), exp);
  }
}
// ai: 6.12 P2, LIZARD-64, every payload byte zero: n 384 in the 128 ring: side 572, pxm 2, the picture resampled by 4/3
// ai: from pixel 30, the resampled square [24, 548) (modules 12 to 273: the picture and its 3-module guard,
// ai: LIZARD_CP = 3), the anchors (SPEC 6.13) 5 x 5 of 6 modules from modules 38, 89, 140, 191, 242 (7 x 7 of 4 until
// ai: 2026-11-09).
// ai: (LIZARD-32 at n 256 in the 64 ring until 2026-11-01, LIZARD-16 until 2026-10-31.)
{
  const e = encode(384, 64, () => {}, true, 256);
  want("P2 slot bits sha256", sha(e.slots), "e62f8d3a928e7da162a9c2e3c46751e2c537cd07d5017c5d65f4a7fbf7173d8b");
  want("P2 packed first 16", hex(pack(e.slots).subarray(0, 16)), "0101360807bc03e0f8ffffde008c1c07");
  want("P2 drive side", String(e.side), "572");
  want("P2 drive row 30 cols 30..33", Array.from(e.drive.subarray(30 * 572 + 30, 30 * 572 + 34), (v) => v.toFixed(7)).join(", "), "0.2934176, 0.3214167, 0.5110379, 0.6746191");
  want("P2 grey row 30 cols 30..45", greyRow(e, 30, 30, 16), "75 82 130 172 172 141 129 166 219 237 195 121 64 49 77 117");
  want("P2 grey row 345 cols 345..360", greyRow(e, 345, 345, 16), "157 88 51 48 67 102 141 158 136 86 39 22 34 65 109 156");
  want("P2 grey row 81 cols 74..89 (anchor 0, 0)", greyRow(e, 81, 74, 16), "27 58 255 255 0 0 0 255 255 0 0 0 255 255 24 0");
  const sq = greySquare(e, 24, 548);
  want("P2 square sha256", sha(sq), "6051f5ac85c470a93f7f72c7ec028ee8a543dda04d68e998ef2d6462ec9b5a53");
  want("P2 square zeros / 255s", `${sq.filter((v) => v === 0).length} / ${sq.filter((v) => v === 255).length}`, "5830 / 6634");
  // ai: (2026-11-18: the border's word moved with its 32 bits, version, counts and CRC-4; 9c086111... with a rate code that
  // ai: day, 8dd1f3c8... the RS(n, 5) word before)
  want("P2 whole drive sha256", sha(greySquare(e, 0, 572)), "e3bd57c575094489e2bac4ddb7187b87e9e5b72dc8e46ec85340b534405b8ab1");
}
// ai: 6.12 P3, LIZARD-64, every payload byte zero: n 384 in the 64 ring, side 474, pxm 3 (a copy), the picture
// ai: from pixel 45, the resampled square [36, 438), the anchors (SPEC 6.13) 3 x 3 of 4 modules from modules 34, 77,
// ai: 120 (7 x 7 of 2 until 2026-11-09).
{
  const e = encode(nFor(64), 64, () => {});
  want("P3 slot bits sha256", sha(e.slots), "e62f8d3a928e7da162a9c2e3c46751e2c537cd07d5017c5d65f4a7fbf7173d8b");
  want("P3 packed first 16 (P2's)", hex(pack(e.slots).subarray(0, 16)), "0101360807bc03e0f8ffffde008c1c07");
  want("P3 drive side", String(e.side), "474");
  want("P3 grey row 45 cols 45..60", greyRow(e, 45, 45, 16), "79 113 177 166 119 151 228 224 135 65 64 102 120 90 57 93");
  want("P3 grey row 107 cols 100..115 (anchor 0, 0)", greyRow(e, 107, 100, 16), "91 41 255 255 0 0 0 255 255 0 0 0 255 255 169 180");
  want("P3 square sha256", sha(greySquare(e, 36, 438)), "d9fc59c979f06e47863f17dcc115bb48115631efd9be8e6d7b5fcc65a4402a2b");
  want("P3 whole drive sha256", sha(greySquare(e, 0, 474)), "d8c373a14aa9f024b638034aa94d216608cc5f02b7e7487a8f9c486a5a81a569");   // ai: (0472f801... with a rate code 2026-11-18, 6e1dd56c... before)
}
// ai: 6.12 P4, LIZARD-64, every payload byte zero: n 384 in the 256 ring, side 542, pxm 1, the picture resampled by 4/3
// ai: into 512 pixels from pixel 15 (P2's picture pixels but where the anchors differ), the resampled square [12, 530),
// ai: the anchors (SPEC 6.13) 7 x 7 of 8 modules from modules 48, 121, 194, 267, 340, 413, 486, the word RS(64, 6).
{
  const e = encode(384, 64, () => {}, true, 512);
  const g = M._malloc(32); M._lizard_anchor_geom_get(g);
  want("P4 anchors: side, first modules", Array.from(M.HEAP32.subarray(g >> 2, (g >> 2) + 8)).join(" "), "8 48 121 194 267 340 413 486");
  M._free(g);
  want("P4 slot bits sha256 (P3's)", sha(e.slots), "e62f8d3a928e7da162a9c2e3c46751e2c537cd07d5017c5d65f4a7fbf7173d8b");
  want("P4 drive side", String(e.side), "542");
  want("P4 drive row 15 cols 15..18 (P2's at 30, 30)", Array.from(e.drive.subarray(15 * 542 + 15, 15 * 542 + 19), (v) => v.toFixed(7)).join(", "), "0.2934176, 0.3214167, 0.5110379, 0.6746191");
  want("P4 grey row 15 cols 15..30", greyRow(e, 15, 15, 16), "75 82 130 172 172 141 129 166 219 237 195 121 64 49 77 117");
  want("P4 grey row 300 cols 300..315", greyRow(e, 300, 300, 16), "128 202 188 146 151 217 255 230 134 65 75 124 158 159 155 164");
  want("P4 grey row 51 cols 46..57 (anchor 0, 0)", greyRow(e, 51, 46, 12), "147 109 255 255 0 255 255 0 255 255 187 169");
  const sq = greySquare(e, 12, 530);
  want("P4 square sha256", sha(sq), "a77759cd3d43354d9ccfcbde06c82990cad361e3de4425a61de3681026d06f55");
  want("P4 square zeros / 255s", `${sq.filter((v) => v === 0).length} / ${sq.filter((v) => v === 255).length}`, "4845 / 6946");
  want("P4 whole drive sha256", sha(greySquare(e, 0, 542)), "1922765a807bed6b1680613da519b3750593650d0e7f539521a1134e6f4649a4");
}
// ai: 7.4, LIZARD-64 (6/7 x 3, 3/4 x 2, 2/3 and 1/2 blocks from data slots 0, 4480, 8960, 13,440, 18,560, 23,680 and
// ai: 29,440, the padding from 37,120 to 40,704: the default senders' since 2026-11-23, the 6/7 on the rest; 6/7 x 4 and
// ai: one each of the others before; in every sub-channel coefficient 0 the count pilot and 160 the
// ai: reference, the blocks over the other 318 in order: the pilots apart from the blocks since 2026-11-21, a tail on
// ai: every block from 2026-11-20, the 6/7 and 3/4 blocks' alone before; LIZARD-32 until 2026-11-01, LIZARD-16 until
// ai: 2026-10-31), block b = id b LE32 then byte j = j mod 256: each block's CRC-64 and its first 128 data slots, the
// ai: padding's first 128, the first two sub-channels' four pilot slots at counts 0 and 3, and every frame slot packed.
{
  // ai: data slot ds's frame slot: sub-channel j's data coefficient r is its coefficient 1 + r below the reference, 2 + r
  // ai: past it (src/lizard_phy.c pb_fcoef)
  const frameSlot = (ds) => { const dc = ds >>> 1, j = Math.floor(dc / 318), r = dc - 318 * j; return 2 * (320 * j + (r < 159 ? 1 + r : 2 + r)) + (ds & 1); };
  const fill = (blocks, B) => { for (let b = 0; b < 7; b++) { blocks[b * B] = b; for (let j = 0; j < B - 4; j++) blocks[b * B + 4 + j] = j & 255; } };
  const e = encode(384, 64, fill);
  M._lizard_parity_set(3);
  const e3 = encode(384, 64, fill, false);
  M._lizard_parity_set(0);
  const data = (x, from, n) => Uint8Array.from({ length: n }, (_, i) => x.slots[frameSlot(from + i)]);
  const pilots = (x) => [0, 1].map((j) => [0, 160].map((c) => `${x.slots[2 * (320 * j + c)]}${x.slots[2 * (320 * j + c) + 1]}`).join(" ")).join(" | ");
  const blk = (b) => { const u = new Uint8Array(BLOCK); u[0] = b; for (let j = 0; j < BLOCK - 4; j++) u[4 + j] = j & 255; return u; };
  const WANT = [
    { at: 0, crc: "0x1C630D6B95C449B6", head: "71369823575adf191b84f0346495b798" },
    { at: 4480, crc: "0x0A8741ABAC7C1D92", head: "d384bfcab2af7daf927c3f4ba4402cac" },
    { at: 8960, crc: "0x31AB94EBE6B4E1FE", head: "623dc8e6bcc36e2a04fa1185b1ba216d" },
    { at: 13440, crc: "0x274FD82BDF0CB5DA", head: "cbc746a14ccba9031ff389c57db4c647" },
    { at: 18560, crc: "0x47F23E6B73251926", head: "9adbd16d50173407998af147c161fc0b" },
    { at: 23680, crc: "0x511672AB4A9D4D02", head: "019d64b73ecd500f8cd78b75fe39d1ce" },
    { at: 29440, crc: "0x6A3AA7EB0055B16E", head: "f626e12eb6aa86eaf76e15a985cd437c" },
  ];
  WANT.forEach((w, b) => {
    const p = M._malloc(BLOCK + 8); M.HEAPU8.set(blk(b), p);
    M._ob_test_crc_hash(p, BLOCK, p + BLOCK);
    want(`7.4 block ${b} CRC-64`, "0x" + hex(M.HEAPU8.subarray(p + BLOCK, p + BLOCK + 8)).toUpperCase(), w.crc);
    M._free(p);
    want(`7.4 block ${b} data slots 0..127 (data slots ${w.at}..)`, hex(pack(data(e, w.at, 128))), w.head);
  });
  want("7.4 padding data slots 37120..37247", hex(pack(data(e, 37120, 128))), "483d249f336ef2bd614a224073849c06");
  want("7.4 pilots of sub-channels 0 and 1 (count x y, reference x y) at count 0", pilots(e), "00 11 | 11 00");
  want("7.4 pilots of sub-channels 0 and 1 (count x y, reference x y) at count 3", pilots(e3), "11 11 | 00 00");
  want("7.4 frame slots packed sha256", sha(pack(e.slots)), "a022f5a1ccf3df81c500e9c52b5ec11ba51f4ba9acc70c443848327e96c59f04");
}
// ai: 7.10 and 7.11, the transfer: each of 7.10's files (byte i = i mod 251) a send of its own under its content's TOI,
// ai: its session header under 0x5E55105A17C0 + its index: the TOI, the description's bytes, SHA-256 of the 460-byte
// ai: header, the values made source block by source block combining to the root and not with a bit of the last changed;
// ai: chunk chaining values; 7.11's key rule on vector.bin's description; the packed form; names cut; the ids.
{
  const put = (b) => { const p = M._malloc(Math.max(1, b.length)); M.HEAPU8.set(b, p); return p; };
  const enc = new TextEncoder(), SESSION0 = 0x5e55105a17c0, toiHex = (t) => "0x" + t.toString(16).toUpperCase().padStart(12, "0");
  const file = (n) => Uint8Array.from({ length: n }, (_, i) => i % 251);
  const cvOf = (bytes, c, k) => { const p = put(bytes), o = M._malloc(32); M._xfer_sb_cv(p, bytes.length, c, k, o); const v = M.HEAPU8.slice(o, o + 32); M._free(p); M._free(o); return v; };
  const rootOf = (f, k) => {
    const C = f.length ? Math.ceil(f.length / 2 ** k) : 0, cvs = new Uint8Array(32 * C), o = M._malloc(32);
    for (let c = 0; c < C; c++) cvs.set(cvOf(f.subarray(c * 2 ** k, Math.min(f.length, (c + 1) * 2 ** k)), c, k), 32 * c);
    if (C >= 2) { const pc = put(cvs); M._xfer_root(pc, C, o); M._free(pc); } else { const pf = put(f); M._xfer_b3_hash(pf, f.length, o); M._free(pf); }
    const root = M.HEAPU8.slice(o, o + 32); M._free(o);
    return { root, cvs, C };
  };
  const toiOf = (len, k, root) => { const p = put(root), t = M._xfer_toi_of_wasm(p, len, k); M._free(p); return t; };
  const describe = (len, k, root, name = "", compression = 0) => {
    const pr = put(root), nm = enc.encode(name), pn = put(nm), out = M._malloc(1024), n = M._xfer_desc_write_of(out, len, k, pr, pn, nm.length, compression);
    const d = n < 0 ? new Uint8Array(0) : M.HEAPU8.slice(out, out + n); [pr, pn, out].forEach((p) => M._free(p));
    return d;
  };
  const header = (d) => { const pd = put(d), ph = M._malloc(PAYLOAD); M._xfer_session_write_of(ph, pd, d.length); const h = M.HEAPU8.slice(ph, ph + PAYLOAD); [pd, ph].forEach((p) => M._free(p)); return h; };
  // ai: xfer_desc_read under a session's TOI: "refused", else its critical flag, compression and derived TOI
  const read = (d, self) => {
    const pd = put(d), sc = M._malloc(44), r = M._xfer_desc_parse(pd, d.length, self, sc, 0, 0), v = new Int32Array(M.HEAPU8.buffer, sc, 11).slice(); [pd, sc].forEach((p) => M._free(p));
    return r ? "refused" : `critical ${v[8]} compression ${v[7]} toi ${toiHex((v[10] >>> 0) * 2 ** 32 + (v[9] >>> 0))}`;
  };
  const W = {
    0: ["0x694F3FFCA6EC", "a303160400055820af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262", "2b7cf66f9d8e42b670ce16b25996d159e1bc669415981abdb60119d16d0054b2"],
    1025: ["0xC3BBE69B7462", "a3030b04190401055820d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444", "b108919d659e8b13a1de8ec1b4cdab99b01925b154c8eacd8549cacf35417258"],
    8193: ["0x13520B2A754D", "a3030a04192001055820bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b", "63899f290aac23dae755e22b7aefe9279e94fc6f54d7c6468ed82d90c5d75841"],
    31744: ["0xC0C18F62683C", "a4030a04197c0005582062b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47066a766563746f722e62696e", "62d1fd474e20117e38788b3a755c13ef3262b22f217f7b968ced3ad2bcc1569a"],
    102400: ["0x6501E3FC611D", "a3030a041a00019000055820bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085", "275c6d562715ecab3d1d9bb6acda1473b0ca07b68dd485a8835d1267387fafc5"],
  };
  [[0, 22], [1025, 11], [8193, 10], [31744, 10, "vector.bin"], [102400, 10]].forEach(([len, k, name], i) => {
    const { root, cvs, C } = rootOf(file(len), k), d = describe(len, k, root, name ?? "");
    want(`7.10 ${len} B TOI`, toiHex(toiOf(len, k, root)), W[len][0]);
    want(`7.10 ${len} B description`, hex(d), W[len][1]);
    want(`7.10 ${len} B session header sha256`, sha(header(d)), W[len][2]);
    want(`7.10 ${len} B description read under ${toiHex(SESSION0 + i)}`, read(d, SESSION0 + i), `critical 0 compression 0 toi ${W[len][0]}`);
    if (C < 2) return;
    if (k === 10) want(`7.10 ${len} B source block 1's chaining value`, hex(cvs.subarray(32, 64)), "5b1f23b8390368a2c15095910389d1b716eb52a6dfd34d0aa66a22a21b761ca7");
    const pc = put(cvs), pr = put(root), ok = M._xfer_root_check(pc, C, pr) === 0;
    M.HEAPU8[pc + 32 * (C - 1)] ^= 1;
    const wrong = M._xfer_root_check(pc, C, pr) === 0; [pc, pr].forEach((p) => M._free(p));
    want(`7.10 ${len} B values combine to the root, not with a bit of the last changed`, `${ok} ${wrong}`, "true false");
  });
  const f = file(31744);
  want("7.10 chunk 0's chaining value", hex(cvOf(f.subarray(0, 1024), 0, 10)), "5c9e654411e393d1f4bec710ccd5bc5669ab177d610a0eb691fcfee92fb4e8b1");
  want("7.10 chunk 30 of the 31,744-byte file", hex(cvOf(f.subarray(30720, 31744), 30, 10)), "a6b76fabdd73c356e0a57cd84e53c562b819b21f140425029385c1574fe985f7");
  want("7.10 chunk 8 of the 8,193-byte file", hex(cvOf(file(8193).subarray(8192), 8, 10)), "0c89b23aaf0f396fe25995d42f7d6f3c7b1e18b59d9c778b06688325f44016de");
  // ai: 7.10's ids: the bare id in bytes 0 to 4, the TOI in 5 to 10, byte 11 reserved; the kind by the bare id
  for (const [toi, sbn, esi, bytes, kind] of [[1, 0, 0, "000000000001000000000000", 0], [1, 1, 5, "050000010001000000000000", 0], [1, 65534, 16777215, "fffffffeff01000000000000", 0],
    [1, 65535, 0, "000000ffff01000000000000", 1], [1, 65535, 1, "010000ffff01000000000000", -1], [0xc0c18f62683c, 1, 5, "05000001003c68628fc1c000", 0],
    [0xc0c18f62683c, 0, 31, "1f000000003c68628fc1c000", 0], [0x5e55105a17c3, 65535, 0, "000000ffffc3175a10555e00", 1]]) {
    const bare = M._xfer_id_of(sbn, esi), b = Buffer.alloc(12); b.writeUIntLE(bare, 0, 5); b.writeUIntLE(toi, 5, 6);
    want(`7.10 id TOI ${toiHex(toi)} SBN ${sbn} ESI ${esi}`, `${hex(b)} ${M._xfer_kind_of(bare)}`, `${bytes} ${kind}`);
  }
  // ai: 7.11: vector.bin's description under its session's TOI and under its own as the session's; a pair after its last
  const v = describe(31744, 10, rootOf(f, 10).root, "vector.bin"), S = SESSION0 + 3;
  want("7.11 vector.bin under its own TOI as the session's", read(v, 0xc0c18f62683c), "refused");
  for (const [label, pair, exp] of [["2: true", [0x02, 0xf5], "critical 1 compression 0 toi 0xC0C18F62683C"], ["-2: true", [0x21, 0xf5], "critical 0 compression 0 toi 0xC0C18F62683C"],
    ["-1: 7", [0x20, 0x07], "critical 0 compression 7 toi 0xC0C18F62683C"], ["-1: true", [0x20, 0xf5], "refused"], ["3: 10 again", [0x03, 0x0a], "refused"]]) {
    const x = new Uint8Array(v.length + 2); x.set(v); x[0]++; x.set(pair, v.length);
    want(`7.11 vector.bin with {${label}} after its last`, read(x, S), exp);
  }
  const z = describe(31744, 10, rootOf(f, 10).root, "vector.bin.zst", 1);
  want("7.11 vector.bin.zst's description", `${z.length} ${hex(z.subarray(z.length - 17))}`, "60 6e766563746f722e62696e2e7a73742001");
  want("7.11 vector.bin.zst's session header sha256", sha(header(z)), "d4eb77caa1f398a7bd1e8f1c9ce0130e1cb5158196453d6d586a3005e4bea2eb");
  for (const [nm, exp] of [["a".repeat(296) + ".pdf", "a".repeat(251) + ".pdf"], ["c".repeat(292) + ".tar.zst", "c".repeat(247) + ".tar.zst"], ["b".repeat(300), "b".repeat(255)]]) {
    const b = enc.encode(nm), p = put(b), o = M._malloc(255), n = M._xfer_name_cut_of(p, b.length, o), got = Buffer.from(M.HEAPU8.slice(o, o + n)).toString(); [p, o].forEach((q) => M._free(q));
    want(`7.11 a ${b.length}-byte name (${nm.slice(-8)}) cut`, got === exp ? `${n} B as the SPEC` : got, `${exp.length} B as the SPEC`);
  }
}

// ai: The whole ladder, to diff between builds: every format's cell map (word cells included), its drive on a fixed
// ai: payload, and its coefficient order.
{
  const cells = createHash("sha256"), drives = createHash("sha256"), order = createHash("sha256");
  const pDims = M._malloc(24), pX = M._malloc(4 * 64), pY = M._malloc(4 * 64), pMark = M._malloc(4 * 64 * 64);
  for (const subch of VERSIONS) {
    const n = nFor(subch);
    if ((blockBytes = setup(n, subch)) < 0) throw new Error(`setup ${n}/${subch}`);
    {
      M._ob_test_mesh_tables_out(pDims, 0, 0, 0, 0);
      const d = M.HEAP32.subarray(pDims >> 2, (pDims >> 2) + 5), w = d[2], h = d[3];
      const pK = M._malloc(w * h);
      M._ob_test_mesh_tables_out(pDims, pX, pY, pK, pMark);
      // ai: nx, ny, w, h, corner (the layout's thin flag, always 1, sat before the corner until 2026-10-25)
      const dd = Array.from(M.HEAP32.subarray(pDims >> 2, (pDims >> 2) + 5));
      cells.update(JSON.stringify([subch, dd])).update(M.HEAPU8.subarray(pK, pK + w * h))
        .update(M.HEAPU8.subarray(pX, pX + 4 * dd[0])).update(M.HEAPU8.subarray(pY, pY + 4 * dd[1])).update(M.HEAPU8.subarray(pMark, pMark + 4 * dd[0] * dd[1]));
      M._free(pK);
    }
    const e = encode(n, subch, (blocks) => { for (let k = 0; k < blocks.length; k++) blocks[k] = (k * 37 + subch) & 255; }, false);
    drives.update(new Uint8Array(e.drive.buffer));
    const pp = M._lizard_pos_ptr();
    order.update(M.HEAPU8.subarray(pp, pp + 4 * 320 * subch));
  }
  console.log(`ladder cells ${cells.digest("hex").slice(0, 16)}  drives ${drives.digest("hex").slice(0, 16)}  order ${order.digest("hex").slice(0, 16)}`);
}
console.log(fails ? `${fails} FAILED` : "all SPEC vectors ok");
process.exit(fails ? 1 : 0);
