// ai: The SPEC.md conformance vectors (5.2 through the painted cells, 6.12 P2 and P3, 7.4, 9.3) against one build,
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

// ai: 5.2's words, each symbol painted in the ring named, the word read back out of the layout's cells: word cell c
// ai: of side s is the c-th CELL_WORD cell along the band (depth 5), dark = 1, and shows bit 7 - c mod 8 of byte
// ai: 4 floor(c / 8) + s (5.3).
{
  const pDims = M._malloc(24);
  // ai: [subch, ring, layout (null the default senders'; [6/7, 2/3, 1/2] through _lizard_layout_set), the clock [fps,
  // ai: delay] (60 and 1 the codec's until set, _lizard_clock_set), the codeword]; since 2026-11-22 the word's 35-bit index
  // ai: (configuration x 241 + fps) x 30 + delay - 1 and its CRC-5, RS(n, 6); the 32-bit static word and RS(n, 5) before
  for (const [subch, B, layout, clock, exp] of [
    [512, 64, null, [60, 1], "4c0d8f52f5d8e3acab9b60939a0619b8"],
    [1024, 64, null, [60, 1], "4cd08c9abc0cf6350d2fffc59bbc62a4"],
    [576, 64, null, [60, 4], "4c158a6daefb354d31456e238ebdee77"],
    [64, 64, null, [60, 1], "4c0000dbc183ba98d76e1dbcb5fdb314"],
    [64, 128, null, [60, 1], "4c0000dbc183841dd462708cab75202552c99e25077748d21528cf2bb4118ee5"],
    [256, 256, null, [30, 2], "4c00ea01a67467de15e5d52ef86fbabe493fb716324852613e6e4cf216c2a64cfc6cc26e4df853d7950d959e13beef4576813758e4f3abb9fdecbdce1d79cbd3"],
    [1072, 64, [90, 0, 30], [240, 30], "4cfc4a101d3258e791006e11e7ffad21"],
    [1072, 128, [151, 0, 0], [0, 1], "4cfd4f2dd79dcff6bf932889d78a07c9c2e45e85092334f2be406a74304f94a4"],
  ]) {
    if (setup(nFor(subch), subch, 2 * B) < 0) throw new Error(`setup LIZARD-${subch} ring ${B}`);
    if (layout && M._lizard_layout_set(...layout)) throw new Error(`layout ${layout}`);
    if (M._lizard_clock_set(...clock)) throw new Error(`clock ${clock}`);
    M._ob_test_mesh_tables_out(pDims, 0, 0, 0, 0);
    const d = M.HEAP32.slice(pDims >> 2, (pDims >> 2) + 5), S = d[2];
    const pX = M._malloc(4 * d[0]), pY = M._malloc(4 * d[1]), pK = M._malloc(S * S), pMark = M._malloc(4 * d[0] * d[1]);
    M._ob_test_mesh_tables_out(pDims, pX, pY, pK, pMark);
    const kind = M.HEAPU8.slice(pK, pK + S * S);
    [pX, pY, pK, pMark].forEach((p) => M._free(p));
    const at = [(t) => [t, 5], (t) => [S - 6, t], (t) => [t, S - 6], (t) => [5, t]], sides = [];
    for (let s = 0; s < 4; s++) {
      const bits = [];
      for (let t = 0; t < S; t++) { const [x, y] = at[s](t), k = kind[y * S + x]; if (k & 8) { bits.push((k & 3) === 2 ? 1 : 0); t++; } }
      sides.push(bits);
    }
    const cw = new Uint8Array(exp.length / 2);
    for (let k = 0; k < cw.length; k++) for (let b = 0; b < 8; b++) cw[k] |= (sides[k % 4][8 * Math.floor(k / 4) + b] ?? 0) << (7 - b);
    want(`5.2 LIZARD-${subch} ring ${B}${layout ? ` layout ${layout}` : ""} clock ${clock} (${sides[0].length} word cells a side)`, hex(cw), exp);
  }
  M._free(pDims);
}
// ai: 6.12 P2, LIZARD-64, every payload byte zero: n 384 in the 128 ring: side 572, pxm 2, the picture resampled by 4/3
// ai: from pixel 30, the resampled square [24, 548) (modules 12 to 273: the picture and its 3-module guard,
// ai: LIZARD_CP = 3), the anchors (SPEC 6.13) 5 x 5 of 6 modules from modules 38, 89, 140, 191, 242 (7 x 7 of 4 until
// ai: 2026-11-09).
// ai: (LIZARD-32 at n 256 in the 64 ring until 2026-11-01, LIZARD-16 until 2026-10-31.)
{
  const e = encode(384, 64, () => {}, true, 256);
  want("P2 slot bits sha256", sha(e.slots), "98897095cdce9eb69dd90f9e50c7de831b63afd92cc4e082d1d2e0dc9c4135b1");
  want("P2 packed first 16", hex(pack(e.slots).subarray(0, 16)), "0101360807bc03e0f8ffffde008c1c07");
  want("P2 drive side", String(e.side), "572");
  want("P2 drive row 30 cols 30..33", Array.from(e.drive.subarray(30 * 572 + 30, 30 * 572 + 34), (v) => v.toFixed(7)).join(", "), "0.2779620, 0.4437774, 0.7467755, 0.9889030");
  want("P2 grey row 30 cols 30..45", greyRow(e, 30, 30, 16), "71 113 190 252 255 206 126 93 141 219 254 214 136 70 56 92");
  want("P2 grey row 345 cols 345..360", greyRow(e, 345, 345, 16), "178 87 19 17 77 156 199 201 189 164 116 49 3 20 103 200");
  want("P2 grey row 81 cols 74..89 (anchor 0, 0)", greyRow(e, 81, 74, 16), "80 120 255 255 0 0 0 255 255 0 0 0 255 255 85 87");
  const sq = greySquare(e, 24, 548);
  want("P2 square sha256", sha(sq), "bdccc91d59adc8f5c930b4f30ef6cfda8e968aa2969d42a9e50d2ad3a8d29b21");
  want("P2 square zeros / 255s", `${sq.filter((v) => v === 0).length} / ${sq.filter((v) => v === 255).length}`, "5896 / 6465");
  // ai: (2026-11-18: the border's word moved with its 32 bits, version, counts and CRC-4; 9c086111... with a rate code that
  // ai: day, 8dd1f3c8... the RS(n, 5) word before)
  want("P2 whole drive sha256", sha(greySquare(e, 0, 572)), "925401899ce0a6d28771ccaac2ed7a9d8adbb9c65464e78409e8d7a292e6afca");
}
// ai: 6.12 P3, LIZARD-64, every payload byte zero: n 384 in the 64 ring, side 474, pxm 3 (a copy), the picture
// ai: from pixel 45, the resampled square [36, 438), the anchors (SPEC 6.13) 3 x 3 of 4 modules from modules 34, 77,
// ai: 120 (7 x 7 of 2 until 2026-11-09).
{
  const e = encode(nFor(64), 64, () => {});
  want("P3 slot bits sha256", sha(e.slots), "98897095cdce9eb69dd90f9e50c7de831b63afd92cc4e082d1d2e0dc9c4135b1");
  want("P3 packed first 16 (P2's)", hex(pack(e.slots).subarray(0, 16)), "0101360807bc03e0f8ffffde008c1c07");
  want("P3 drive side", String(e.side), "474");
  want("P3 grey row 45 cols 45..60", greyRow(e, 45, 45, 16), "89 169 255 255 156 87 159 245 220 124 65 88 152 178 133 82");
  want("P3 grey row 107 cols 100..115 (anchor 0, 0)", greyRow(e, 107, 100, 16), "169 133 255 255 0 0 0 255 255 0 0 0 255 255 205 147");
  want("P3 square sha256", sha(greySquare(e, 36, 438)), "e415c3ea96728164142acdecd9177fbf410ddaa65736d9d8274da14ec974514f");
  want("P3 whole drive sha256", sha(greySquare(e, 0, 474)), "c46faa2df19904f3fe845cfbe2e600b2935eeb7c9330b46dfd353c4f23ea4bcb");   // ai: (0472f801... with a rate code 2026-11-18, 6e1dd56c... before)
}
// ai: 7.4, LIZARD-64 (6/7 x 4, 3/4, 2/3 and 1/2 blocks from data slots 0, 4480, 8960, 13,440, 17,920, 23,040 and
// ai: 28,800, the padding from 36,480 to 40,704; in every sub-channel coefficient 0 the count pilot and 160 the
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
    { at: 13440, crc: "0x274FD82BDF0CB5DA", head: "a4b03ec8d2e4a17fb2a1d76df519744c" },
    { at: 17920, crc: "0x47F23E6B73251926", head: "3334a645533364c8c518b8da7795aadb" },
    { at: 23040, crc: "0x511672AB4A9D4D02", head: "124ee8647ae0278c8bd84cb81045318d" },
    { at: 28800, crc: "0x6A3AA7EB0055B16E", head: "5f524a21753f6988a76845081d44666f" },
  ];
  WANT.forEach((w, b) => {
    const p = M._malloc(BLOCK + 8); M.HEAPU8.set(blk(b), p);
    M._ob_test_crc_hash(p, BLOCK, p + BLOCK);
    want(`7.4 block ${b} CRC-64`, "0x" + hex(M.HEAPU8.subarray(p + BLOCK, p + BLOCK + 8)).toUpperCase(), w.crc);
    M._free(p);
    want(`7.4 block ${b} data slots 0..127 (data slots ${w.at}..)`, hex(pack(data(e, w.at, 128))), w.head);
  });
  want("7.4 padding data slots 36480..36607", hex(pack(data(e, 36480, 128))), "f3e66d1e579da948510084c880c53130");
  want("7.4 pilots of sub-channels 0 and 1 (count x y, reference x y) at count 0", pilots(e), "00 11 | 11 00");
  want("7.4 pilots of sub-channels 0 and 1 (count x y, reference x y) at count 3", pilots(e3), "11 11 | 00 00");
  want("7.4 frame slots packed sha256", sha(pack(e.slots)), "848b0b3cc242edf288cda6e62acdbd1826be2bf9defb69cb31003a2207754f50");
}
// ai: 9.3, the test stream (BLAKE3 since 2026-11-15; SHAKE256 before)
{
  const p = M._malloc(PAYLOAD);
  for (const [id, exp] of [[0, "cdc96eca844d7912acdbb3dca677757d"], [1, "7784b17d1d8ff658cfe8f0e111316a96"], [12345, "290fe01f8322abf2dc2bfae8dd2f0dd4"], [4294967295, "ec3b06ed47cba2e0d61a795520003d72"], [1099511627775, "5a8078d3321f90ffb3bc9aef9ce8ea65"]]) {
    M._stream_fill_of(id, p, PAYLOAD);
    want(`9.3 id ${id}`, hex(M.HEAPU8.subarray(p, p + 16)), exp);
  }
  M._stream_fill_of(0, p, PAYLOAD);
  want("9.3 id 0 tail", hex(M.HEAPU8.subarray(p + PAYLOAD - 8, p + PAYLOAD)), "169b3c33d27f0651");
  M._free(p);
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
