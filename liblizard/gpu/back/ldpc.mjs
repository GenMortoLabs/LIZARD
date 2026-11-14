// The sizes of the decoded blocks' lane buffers, which the rate profile's stage (tiers.mjs) writes and the native host
// makes and reads back: V (u32 a block: 0 not run, 1 verified, 2 declined, 3 given up by the learned stop or at the cap, 4 CRC), ITS (i32 a block,
// the C's codes) and REC (u32 count, then 484-byte records: frame + 65536 block, iterations, the 476 payload bytes). Every stride comes from dims.mjs derived().
import { derived } from "./dims.mjs";

export const REC_WORDS = 121;
export const REC_BYTES = 4 * REC_WORDS;

// ai: sizes[s] = { n, subch, blocks } at the size's index (sel[f].z), null where not served. REC holds every block of
// ai: every frame (B x blocksMax records).
export async function build(device, { B, sizes }) {
  const dims = derived(sizes, { B }), { served, blocksMax, recCap } = dims;
  const bytes = { V: 4 * B * blocksMax, ITS: 4 * B * blocksMax, REC: 4 + REC_BYTES * recCap };
  return { B, sizes, served, dims, blocksMax, recCap, bytes };
}
