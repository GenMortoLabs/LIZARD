// ai: The JS sender's ids and blocks for a file (2026-10-01): liblizard/sim/xfer.mjs XferSender, frame after frame (the ids
// ai: as 12 bytes: the bare id and the 48-bit TOI (the file's, or its session header's), little-endian, and a reserved byte; then each block's 460 bytes), the same layout as `tx_check ids`, so the native sending end
// ai: (core/tx/xfer_tx.cpp) is held to it byte for byte: node gen/xfer_tx_ref.mjs <file> <frames> <blocks a frame> <out>
// ai: [seed] (the carousel's, 1 unless given: `tx_check ids`'s, 2026-11-14).
import { readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
const LIB = new URL("../", import.meta.url);
const imp = (p) => import(new URL(p, LIB).href);
const { init: initRq, Encoder } = await imp("sim/fountain.mjs");
const { init: initOb } = await imp("sim/ob.mjs");
const { XferSender, PAYLOAD, ID_BYTES, idPut } = await imp("sim/xfer.mjs");

const [M] = await Promise.all([initOb(), initRq()]);
const [file, frames, T, out, seed = "1"] = process.argv.slice(2);
const bytes = new Uint8Array(readFileSync(file));
const xs = new XferSender(bytes, { name: basename(file), type: "", M, Encoder, seed: Number(seed) >>> 0 });
const parts = [], b = new Uint8Array(PAYLOAD);
for (let f = 0; f < +frames; f++) {
  const ids = xs.frameIds(+T), idb = new Uint8Array(ids.length * ID_BYTES);
  ids.forEach((id, k) => idPut(idb, k * ID_BYTES, id, xs.toiFor(id)));   // ai: the file's TOI, or its session header's
  parts.push(Buffer.from(idb));
  for (const id of ids) parts.push(Buffer.from(xs.block(id, b)));
}
writeFileSync(out, Buffer.concat(parts));
console.log(`ref: ${frames} frames of ${T}, source blocks ${xs.sourceBlocks}, cycle ${xs.carousel.cycle}, root ${xs.info().root}, seed ${xs.carouselSeed}`);
