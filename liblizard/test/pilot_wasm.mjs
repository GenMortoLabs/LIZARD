// ai: The pilots' two readings (SPEC 7.3): every sub-channel carries a count pilot and a reference (since 2026-11-21; the
// ai: blocks' tails before), so the told C (Lizard) and the blind C the receiver runs (LizardAny) read r over the even
// ai: sub-channels and r2 over the odd, every sub-channel of the format. A clean capture painted at count c reads r by bit 0 (+1 at c = 0 and
// ai: 2, -1 at 1 and 3) and r2 by bit 1 (+1 at 0 and 1, -1 at 2 and 3), |r| >= 0.9, the bound scripts/exp/pilot_check.mjs
// ai: holds a clean capture to. The symbol is painted by the codec at a pixel a drive pixel on white, no camera.
//   node test/pilot_wasm.mjs      from liblizard/, a few seconds; OB_BUILD=<path> for another build; exit 1 on a failure
import { init, Lizard, LizardAny } from "../sim/ob.mjs";
import { N_FOR } from "../sim/ob.mjs";

await init();
let fails = 0;
const want = (name, ok, got) => { if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}`); };
const P = 24;   // ai: white round the symbol
const fa = new LizardAny(1536);
// ai: the smallest formats (the fewest pilots a reading) and the largest
for (const subch of [64, 80, 96, 1072]) {
  const n = N_FOR(subch), fc = new Lizard(n, subch, 1);
  for (const c of [0, 1, 2, 3]) {
    const blocks = new Uint8Array(fc.blocks * fc.blockBytes);
    for (let i = 0; i < blocks.length; i++) blocks[i] = ((i + 7919 * (c + 1)) * 2654435761) >>> 13 & 255;
    fc.setParity(c);
    const { w, rgba } = fc.encodeRGBA(blocks), W = w + 2 * P, img = new Uint8Array(W * W).fill(255);
    for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) img[(y + P) * W + x + P] = rgba[4 * (y * w + x)];
    const s0 = c & 1 ? -1 : 1, s1 = c & 2 ? -1 : 1;
    for (const [who, p] of [["told", fc.decode(img, W, W).pilot], ["blind", fa.decode(img, W, W).pilot]]) {
      const tag = `LIZARD-${subch} c=${c} ${who}`;
      want(`${tag} sub-channels`, !!p && p.subch === subch, p ? `${p.subch} of ${subch}` : "null");
      if (!p) continue;
      want(`${tag} r`, Math.sign(p.r) === s0 && Math.abs(p.r) >= 0.9 && Number.isFinite(p.sd), `${p.r.toFixed(3)} sd ${p.sd.toFixed(4)}`);
      want(`${tag} r2`, Math.sign(p.r2) === s1 && Math.abs(p.r2) >= 0.9 && Number.isFinite(p.sd2), `${p.r2.toFixed(3)} sd ${p.sd2.toFixed(4)}`);
    }
  }
  fc.free();
}
console.log(fails ? `FAILED: ${fails}` : "all ok");
process.exit(fails ? 1 : 0);
