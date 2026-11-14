// ai: The JS sender's ids and blocks for a file (2026-10-01): liblizard/sim/xfer.mjs XferSender, frame after frame (the ids
// ai: little-endian, then each block's 472 bytes), the same layout as `tx_check ids`, so the native sending end
// ai: (core/tx/xfer_tx.cpp) is held to it byte for byte: node gen/xfer_tx_ref.mjs <file> <frames> <blocks a frame> <out>
// ai: [seed] (the schedule's, 1 unless given: `tx_check ids`'s, 2026-11-14)
import { readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
const LIB = new URL("../", import.meta.url);
const imp = (p) => import(new URL(p, LIB).href);
const { init: initWh, Encoder } = await imp("sim/fountain.mjs");
const { init: initOb } = await imp("sim/ob.mjs");
const { XferSender } = await imp("sim/xfer.mjs");

const [file, frames, T, out, seed = "1"] = process.argv.slice(2);
const [M] = await Promise.all([initOb(), initWh()]);
const bytes = new Uint8Array(readFileSync(file));
const xs = new XferSender(bytes, { name: basename(file), type: "", M, Encoder, seed: Number(seed) >>> 0 });
const parts = [], b = new Uint8Array(472);
for (let f = 0; f < +frames; f++) {
  const ids = xs.frameIds(+T);
  parts.push(Buffer.from(new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength)));
  for (const id of ids) parts.push(Buffer.from(xs.block(id, b)));
}
writeFileSync(out, Buffer.concat(parts));
console.log(`ref: ${frames} frames of ${T}, chunks ${xs.chunks}, lap ${xs.sched.lap}, root ${xs.info().root}, seed ${xs.scheduleSeed}`);
