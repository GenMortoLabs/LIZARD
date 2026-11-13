// ai: Rate refresh, the sender's setting (the native sender's alike: liblizard/core/tx/sender.cpp Sender::produce and
// ai: sender.h setRefresh): every[r] for the code rate r (0 7/8, 1 3/4, 2 2/3, 3 1/2), 1 to REFRESH_MAX, 1 the default.
// ai: A frame's blocks lie per code in the format's tier order from the inside out (sim/lizard_pick.mjs tiersFor: the
// ai: 7/8 blocks first, then 3/4, 2/3, 1/2), code k's after code k - 1's. In the frame painted seq-th a block of rate r
// ai: repeats the id at its place in the frame before where there is one in this run, every[r] > 1 and seq is no
// ai: multiple of every[r]; every other block takes a new id. No format change: a receiver reads each frame as ever,
// ai: and a repeated block is one it has. Pure (no codec, no page): check_refresh.mjs runs it in node.
import { tiersFor } from "../liblizard/sim/lizard_pick.mjs";

export const REFRESH_MAX = 4;
// ai: a tier's rate code (src/lizard_phy.h: 6 7/8, 4 3/4, 3 2/3, 2 1/2) to its index in every
const INDEX = { 6: 0, 4: 1, 3: 2, 2: 3 };

// ai: a symbol's block b's rate index in the order of the tiers ([[rate code, blocks, slots], ...], tiersFor's); null
// ai: where they do not lay `blocks` blocks (a guard: then every block is new)
export function ratesOf(tiers, blocks) {
  const out = [];
  for (const [rate, n] of tiers) for (let k = 0; k < n; k++) out.push(INDEX[rate] ?? 3);
  return out.length === blocks ? Uint8Array.from(out) : null;
}
export const ratesFor = (subch, blocks) => ratesOf(tiersFor(subch), blocks);

// ai: four whole values 1 to REFRESH_MAX, 1 for anything else
export const everyOf = (every) => [0, 1, 2, 3].map((r) => Math.max(1, Math.min(REFRESH_MAX, Math.round(+every?.[r]) || 1)));

// ai: One run's state: the setting, the format's rates and the frame before's ids. A new format (start, a re-pick) or a
// ai: stop forgets the frame before, so the next frame is new throughout.
export class Refresh {
  constructor() { this.every = [1, 1, 1, 1]; this.rates = null; this.prev = null; }
  setEvery(every) { this.every = everyOf(every); }
  setFormat(subch, blocks) { this.rates = ratesFor(subch, blocks); this.prev = null; }
  reset() { this.prev = null; }
  // ai: the ids of the frame painted seq-th, `total` blocks (codes x the symbol's blocks); take(m) gives the m new ids
  // ai: in order, asked only for m > 0. Returns { ids, fresh, m }: fresh[t] 1 where block t took a new id.
  next(seq, total, take) {
    const { rates, prev, every } = this, V = rates ? rates.length : 0, fresh = new Uint8Array(total);
    const hold = !!rates && !!prev && prev.length === total;
    let m = 0;
    for (let t = 0; t < total; t++) {
      const e = hold ? every[rates[t % V]] : 1;
      fresh[t] = e <= 1 || seq % e === 0 ? 1 : 0;
      m += fresh[t];
    }
    const ids = new Uint32Array(total);
    if (m) {
      const got = take(m);
      for (let t = 0, k = 0; t < total; t++) ids[t] = fresh[t] ? got[k++] : prev[t];
    } else ids.set(prev);
    this.prev = ids;
    return { ids, fresh, m };
  }
}

// ai: every id a run of consecutive frames (next's answers, in order) paints: the first frame's, then each later
// ai: frame's new ones, since a repeat is an id of the frame before
export function batchIds(frames) {
  const need = frames.length ? [...frames[0].ids] : [];
  for (let f = 1; f < frames.length; f++) frames[f].ids.forEach((id, t) => { if (frames[f].fresh[t]) need.push(id); });
  return Uint32Array.from(need);
}
