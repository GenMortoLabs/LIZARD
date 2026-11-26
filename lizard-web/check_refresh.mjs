// ai: Refresh's hold rule, the C's (core/tx/refresh.h; lizard.h liz_refresh_places, liz_refresh_held), through the
// ai: wasm as the web's sender runs it (sim/ob.mjs Refresh, lizard-web/send-worker.mjs), in node, no page:
// ai:   1. a made-up layout (6/7 x 3, 3/4 x 2, 2/3 x 2, 1/2 x 1) at 1 and 2 codes, delays [1,2,3,4] and [2,3,4,4], 60
// ai:      frames from seq 0 and from seq 7: the first frame new throughout; then a block held exactly where its place's
// ai:      delay is past 1 and the seq no multiple of it, a held block the frame before's id at its place, a new one an
// ai:      id never painted before; the new ids asked equal to the new places (none asked where there are none); the
// ai:      frames holding each rate the rule's count; the ids past 2^32, as a bare block id is 40 bits;
// ai:   2. the same run cut in batches of 5 (the GPU encoder's "frames"): the same ids as one run;
// ai:   3. every delay 1: each frame's ids the test stream's run from its base;
// ai:   4. setFormat and reset forget the frame before; a frame of another size than the one before is new throughout; a
// ai:      layout laying other than the symbol's blocks holds nothing;
// ai:   5. everyOf's bounds (REFRESH_MAX the word's largest delay); refreshPlaces over real formats (every fifth version
// ai:      from LIZARD-16): each place its tier's rate's delay, the tiers' counts in order from the 6/7 out.
// ai: node lizard-web/check_refresh.mjs (about a second). Exit 1 on a failure.
import { init, Refresh, refreshPlaces, everyOf, REFRESH_MAX, tiersFor, blocksFor, VERSIONS } from "../liblizard/sim/ob.mjs";

await init();
let failed = 0, passed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (ok) passed++; else failed++; };

// ai: the made-up layout's places by rate index (0 6/7, 1 3/4, 2 2/3, 3 1/2)
const V = 8, RATE = [0, 0, 0, 1, 1, 2, 2, 3];
const placesOf = (every) => Int32Array.from(RATE, (r) => every[r]);

// ai: one run: frames seq0 .. seq0 + n - 1 in order, new ids from a counter; each frame's ids, new marks and ids asked
function runOf(every, codes, seq0, n, cut = n) {
  const r = new Refresh();
  r.setEvery(every); r.places = placesOf(r.every); r.reset();
  const T = V * codes, frames = [];
  let next = 2 ** 39 + 1000;   // ai: 40-bit ids (src/xfer.h), past what 32 bits hold
  for (let b = 0; b < n; b += cut) for (let f = b; f < Math.min(n, b + cut); f++) {
    let asked = null;
    const got = r.next(seq0 + f, T, (m) => { asked = m; return Float64Array.from({ length: m }, () => next++); });
    frames.push({ ...got, asked, seq: seq0 + f });
  }
  return { frames, T };
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
  check(frames.some((fr) => fr.ids.some((id) => id > 2 ** 32)), `${tag}: ids past 2^32 kept whole`);
  // ai: the frames holding each rate over the run, against the rule's own count
  const share = [0, 1, 2, 3].map((r) => frames.slice(1).filter((fr) => every[r] > 1 && fr.seq % every[r] !== 0).length);
  const got = [0, 1, 2, 3].map((r) => frames.slice(1).filter((fr, f) => fr.ids.some((id, t) => RATE[t % V] === r && id === frames[f].ids[t])).length);
  check(share.join() === got.join(), `${tag}: frames holding each rate ${got.join()} (the rule ${share.join()})`);
  // ai: in batches of 5, as the GPU encoder asks: the same ids
  const cut = runOf(every, codes, seq0, 60, 5);
  check(cut.frames.every((fr, f) => fr.ids.join() === frames[f].ids.join()), `${tag}: batches of 5 the same ids as one run`);
}

// ai: 3. the default: every frame the test stream's run from its base
{
  const r = new Refresh();
  r.setFormat(480, blocksFor(480));
  const T = 2 * blocksFor(480);
  let same = 0;
  for (let f = 0; f < 20; f++) {
    const base = 5000 + f * T, { ids } = r.next(f, T, (m) => Float64Array.from({ length: m }, (_, i) => base + i));
    if (ids.every((id, t) => id === base + t)) same++;
  }
  check(same === 20, `every 1,1,1,1 at LIZARD-480 x 2: ${same} of 20 frames the run from their base`);
}

// ai: 4. forgetting: setFormat and reset make the next frame new throughout; a layout that lays other than the symbol's
// ai: blocks holds nothing
{
  const r = new Refresh(), take = (m) => new Float64Array(m);
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
  check(r.places === null && r.next(6, T + 1, take).m === T + 1, "a layout not laying the symbol's blocks: nothing held");
}

// ai: 5. bounds and the real layouts
check(REFRESH_MAX === 30 && everyOf([0, REFRESH_MAX + 7, "3", NaN]).join() === `1,${REFRESH_MAX},3,1` && everyOf([30, 13, 1, 2]).join() === "30,13,1,2" && everyOf(undefined).join() === "1,1,1,1", `everyOf ${everyOf([0, REFRESH_MAX + 7, "3", NaN]).join()}, REFRESH_MAX ${REFRESH_MAX}`);
{
  const every = [2, 3, 5, 7], byRate = { 6: 2, 4: 3, 3: 5, 2: 7 };
  let bad = 0, n = 0;
  for (let i = 0; i < VERSIONS.length; i += 5) {
    const subch = VERSIONS[i], tiers = tiersFor(subch), places = refreshPlaces(subch, blocksFor(subch), every);
    n++;
    const runs = [];
    if (places) for (let b = 0; b < places.length; b++) if (!b || places[b] !== places[b - 1]) runs.push([places[b], 1]); else runs[runs.length - 1][1]++;
    const want = tiers.map(([rate, k]) => [byRate[rate], k]);
    if (!places || runs.join() !== want.join()) { bad++; console.log(`     LIZARD-${subch}: ${runs.join(" ")} against ${want.join(" ")}`); }
  }
  check(bad === 0, `refreshPlaces over ${n} formats, every fifth from LIZARD-${VERSIONS[0]}: each place its tier's delay, the tiers' counts in order from 6/7 out (${bad} wrong)`);
  check(refreshPlaces(480, blocksFor(480) + 1, every) === null && refreshPlaces(17, 4, every) === null, "refreshPlaces refuses a count its layout does not lay and a count that is no format");
}

console.log(`${passed} ok${failed ? `, ${failed} FAILED` : ", all passed"}`);
process.exit(failed ? 1 : 0);
