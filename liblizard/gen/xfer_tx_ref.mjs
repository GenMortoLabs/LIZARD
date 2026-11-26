// ai: The sending end's ids and blocks for a file through the wasm build of the C (lizard.h liz_tx over the plain module,
// ai: core/tx/xfer_tx.cpp as emscripten builds it; the JS sender, sim/xfer.mjs, until 2026-11-25), frame after frame:
// ai: each frame's ids as 12 bytes (the bare id and the 48-bit TOI, the file's or its session header's, little-endian, and
// ai: a reserved byte), then each block's 460 bytes, the layout of `tx_check ids`, so the native build of the same C is
// ai: held to the wasm build's byte for byte (two compilers, two targets):
// ai:   node gen/xfer_tx_ref.mjs <file> <frames> <blocks a frame> <out> [seed]    (the carousel's seed, 1 unless given)
import { readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
const LIB = new URL("../", import.meta.url);
const { init: initOb, hex, ID_BYTES, BLOCK } = await import(new URL("sim/ob.mjs", LIB).href);

const M = await initOb();
const [file, frames, T, out, seed = "1"] = process.argv.slice(2);
const bytes = new Uint8Array(readFileSync(file)), name = basename(file);
const fp = M._malloc(Math.max(1, bytes.length)), nn = M.lengthBytesUTF8(name) + 1, np = M._malloc(nn);
M.HEAPU8.set(bytes, fp);
M.stringToUTF8(name, np, nn);
// ai: the plain module has no threads: every source block made before liz_tx_new returns, so no frame is a warm one
const t = M._liz_tx_new(fp, bytes.length, np, 0);
if (!t || M._liz_tx_seed(t, Number(seed) >>> 0) < 0) throw new Error(M.UTF8ToString(M._liz_last_error()));
const n = +T, bp = M._malloc(n * BLOCK), parts = [];
for (let f = 0; f < +frames; f++) {
  if (M._liz_tx_next(t, n, bp) < 0) throw new Error(M.UTF8ToString(M._liz_last_error()));
  const b = M.HEAPU8.slice(bp, bp + n * BLOCK), ids = Buffer.alloc(n * ID_BYTES), pay = Buffer.alloc(n * (BLOCK - ID_BYTES));
  for (let k = 0; k < n; k++) { ids.set(b.subarray(k * BLOCK, k * BLOCK + ID_BYTES), k * ID_BYTES); pay.set(b.subarray(k * BLOCK + ID_BYTES, (k + 1) * BLOCK), k * (BLOCK - ID_BYTES)); }
  parts.push(ids, pay);
}
writeFileSync(out, Buffer.concat(parts));
const ip = M._malloc(64);
M._liz_tx_info_get(t, ip);
const dv = new DataView(M.HEAPU8.buffer, ip, 64);
console.log(`ref: ${frames} frames of ${T}, source blocks ${dv.getUint32(16, true)}, cycle ${dv.getUint32(20, true)}, root ${hex(M.HEAPU8.slice(ip + 24, ip + 56))}, seed ${dv.getUint32(56, true)}`);
