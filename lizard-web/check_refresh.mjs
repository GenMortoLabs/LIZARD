// ai: Rate refresh's hold rule (refresh.mjs, as send-worker.mjs runs it), in node, no page and no codec:
// ai:   1. a made-up profile (7/8 x 3, 3/4 x 2, 2/3 x 2, 1/2 x 1) at 1 and 2 codes, every [1,2,3,4] and [2,3,4,4], 60
// ai:      frames from seq 0 and from seq 7: the first frame new throughout; then a block held exactly where its rate's
// ai:      every > 1 and the seq is no multiple of it, a held block the frame before's id at its place, a new one an id
// ai:      never painted before; the new ids asked equal to the new places (none asked where there are none);
// ai:   2. a run cut in batches of 5 (the GPU encoder's "frames"): the same ids as one run, and every id of a batch among
// ai:      batchIds' (what the worker fetches from the pool);
// ai:   3. every [1,1,1,1]: each frame's ids the test stream's run from its base, as before the setting;
// ai:   4. setFormat and reset forget the frame before; a profile laying other than the symbol's blocks holds nothing;
// ai:   5. everyOf's bounds; ratesFor over real formats (LIZARD-64 to -1024, every fifth version) the tiers' counts in order.
// ai: node lizard-web/check_refresh.mjs (under a second). Exit 1 on a failure.
import { Refresh, ratesOf, ratesFor, everyOf, batchIds, REFRESH_MAX } from "./refresh.mjs";
import { tiersFor, blocksFor, VERSIONS } from "../liblizard/sim/lizard_pick.mjs";

let failed = 0, passed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (ok) passed++; else failed++; };

const TIERS = [[6, 3, 4496], [4, 2, 5152], [3, 2, 5760], [2, 1, 7680]], V = 8, RATE = [0, 0, 0, 1, 1, 2, 2, 3];
check(ratesOf(TIERS, V)?.join() === RATE.join(), `the made-up profile's rates ${ratesOf(TIERS, V)?.join()}`);

// ai: one run: frames seq0 .. seq0 + n - 1 in order, new ids from a counter; each frame's ids, new marks and ids asked
function runOf(every, codes, seq0, n, cut = n) {
  const r = new Refresh();
  r.setEvery(every); r.rates = ratesOf(TIERS, V); r.reset();
  const T = V * codes, frames = [], batches = [];
  let next = 1000;
  for (let b = 0; b < n; b += cut) {
    const batch = [];
    for (let f = b; f < Math.min(n, b + cut); f++) {
      let asked = null;
      const got = r.next(seq0 + f, T, (m) => { asked = m; return Uint32Array.from({ length: m }, () => next++); });
      batch.push({ ...got, asked, seq: seq0 + f });
    }
    batches.push(batch); frames.push(...batch);
  }
  return { frames, batches, T };
}

for (const codes of [1, 2]) for (const every of [[1, 2, 3, 4], [2, 3, 4, 4]]) for (const seq0 of [0, 7]) {
  const tag = `codes ${codes}, every ${every.join()}, from seq ${seq0}`, { frames, T } = runOf(every, codes, seq0, 60);
  const seen = new Set();
  let bad = 0, held = 0, fresh = 0, askBad = 0;
  frames.forEach((fr, f) => {
    let want = 0;
    for (let t = 0; t < T; t++) {
      const e = every[RATE[t % V]], hold = f > 0 && e > 1 && fr.seq % e !== 0;
      if (hold) { held++; if (fr.fresh[t] || fr.ids[t] !== frames[f - 1].ids[t]) bad++; }
      else { fresh++; want++; if (!fr.fresh[t] || seen.has(fr.ids[t])) bad++; }
    }
    for (const id of fr.ids) seen.add(id);
    if (fr.m !== want || fr.asked !== (want ? want : null)) askBad++;
  });
  check(bad === 0, `${tag}: ${held} held places repeat and ${fresh} new ones change (${bad} wrong)`);
  check(askBad === 0, `${tag}: the new ids asked equal the new places in every frame (${askBad} frames wrong)`);
  check(frames[0].m === T, `${tag}: the first frame new throughout (${frames[0].m} of ${T})`);
  // ai: the share held per rate over the run, against the rule's own count
  const share = [0, 1, 2, 3].map((r) => frames.slice(1).filter((fr) => every[r] > 1 && fr.seq % every[r] !== 0).length);
  const got = [0, 1, 2, 3].map((r) => frames.slice(1).filter((fr, f) => fr.ids.some((id, t) => RATE[t % V] === r && id === frames[f].ids[t])).length);
  check(share.join() === got.join(), `${tag}: frames holding each rate ${got.join()} (the rule ${share.join()})`);

  // ai: in batches of 5, as the GPU encoder asks: the same ids, and each batch's ids among batchIds'
  const cut = runOf(every, codes, seq0, 60, 5);
  check(cut.frames.every((fr, f) => fr.ids.join() === frames[f].ids.join()), `${tag}: batches of 5 the same ids as one run`);
  let missing = 0;
  for (const batch of cut.batches) {
    const need = new Set(batchIds(batch));
    for (const fr of batch) for (const id of fr.ids) if (!need.has(id)) missing++;
  }
  check(missing === 0, `${tag}: every id of a batch among batchIds' (${missing} missing)`);
}

// ai: 3. the default: every frame the test stream's run from its base, as before the setting
{
  const r = new Refresh();
  r.setFormat(480, blocksFor(480));
  const T = 2 * blocksFor(480);
  let same = 0;
  for (let f = 0; f < 20; f++) {
    const base = 5000 + f * T, { ids } = r.next(f, T, (m) => Uint32Array.from({ length: m }, (_, i) => base + i));
    if (ids.every((id, t) => id === base + t)) same++;
  }
  check(same === 20, `every 1,1,1,1 at LIZARD-480 x 2: ${same} of 20 frames the run from their base`);
}

// ai: 4. forgetting: setFormat and reset make the next frame new throughout; a profile that lays other than the
// ai: symbol's blocks holds nothing
{
  const r = new Refresh(), take = (m) => new Uint32Array(m);
  r.setEvery([4, 4, 4, 4]); r.setFormat(480, blocksFor(480));
  const T = blocksFor(480);
  r.next(0, T, take);
  check(r.next(1, T, take).m === 0, "every 4: the frame after the first all held");
  r.reset();
  check(r.next(2, T, take).m === T, "after reset: the next frame new throughout");
  r.setFormat(480, blocksFor(480));
  check(r.next(3, T, take).m === T, "after setFormat: the next frame new throughout");
  check(r.next(4, 2 * T, take).m === 2 * T, "a frame of another size than the one before: new throughout");
  r.setFormat(480, T + 1);
  r.next(5, T + 1, take);
  check(r.rates === null && r.next(6, T + 1, take).m === T + 1, "tiers not laying the symbol's blocks: nothing held");
}

// ai: 5. bounds and the real profiles
check(everyOf([0, 11, "3", NaN]).join() === `1,${REFRESH_MAX},3,1` && everyOf(undefined).join() === "1,1,1,1", `everyOf ${everyOf([0, 11, "3", NaN]).join()}`);
{
  let bad = 0, n = 0;
  for (let i = 0; i < VERSIONS.length; i += 5) {
    const subch = VERSIONS[i], tiers = tiersFor(subch), rates = ratesFor(subch, blocksFor(subch));
    n++;
    const runs = [];
    if (rates) for (let b = 0; b < rates.length; b++) if (!b || rates[b] !== rates[b - 1]) runs.push([rates[b], 1]); else runs[runs.length - 1][1]++;
    const want = tiers.map(([rate, k]) => [{ 6: 0, 4: 1, 3: 2, 2: 3 }[rate], k]);
    if (!rates || runs.join() !== want.join() || runs.some((x, k) => k && x[0] <= runs[k - 1][0])) { bad++; console.log(`     LIZARD-${subch}: ${runs.join(" ")} against ${want.join(" ")}`); }
  }
  check(bad === 0, `ratesFor over ${n} formats from LIZARD-64 to -1024: the tiers' counts in order from 7/8 out (${bad} wrong)`);
}

console.log(`${passed} ok${failed ? `, ${failed} FAILED` : ", all passed"}`);
process.exit(failed ? 1 : 0);
