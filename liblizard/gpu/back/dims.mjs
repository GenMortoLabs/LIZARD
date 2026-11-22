// The per batch strides every stage of the back half sizes its buffers by, from one function, so the soft-value
// stage (B), the LDPC (C) and the composition (back.mjs) cannot disagree on where frame f's slots sit. Before this
// each stage recomputed them from its own view of the size list (DESIGN.md section 9).
//
import { blocksMaxFor, SUB_SLOTS } from "../../sim/lizard_pick.mjs";
import { PILOT_READ } from "../plan/back_tiers.mjs";

// ai: sizes[s] = { n, subch, blocks, ... } at the size's index (sel[f].z, the picture slot), null where the size is not
// ai: built. A size with no blocks is not served.
// ai: blocksMax: the most blocks a layout of a served size's sub-channels may state (2026-11-18, its word's counts:
// ai: sim/lizard_pick.mjs blocksMaxFor; 149 at 1072 sub-channels), which every per-frame block stride holds.
export function derived(sizes, { B } = {}) {
  const served = sizes.map((z, s) => (z && z.blocks > 0 ? s : -1)).filter((s) => s >= 0);
  if (!served.length) throw new Error("no size served");
  const subchMax = Math.max(...served.map((s) => sizes[s].subch));
  const blocksMax = Math.max(...served.map((s) => blocksMaxFor(sizes[s].subch)));
  return {
    B, served, subchMax, blocksMax,
    sStride: Math.max(...served.map((s) => sizes[s].subch * 320)),   // S entries a frame (the largest disc)
    lStride: subchMax * SUB_SLOTS / 4,                                // L words a frame: int8 data-slot order, 636 a sub-channel
    pilotRead: PILOT_READ,                                            // PILOT's floats a frame before the shifts (r, sd, r2, sd2)
    blkStride: blocksMax,                                             // BLK, V and ITS entries a frame
    recCap: (B ?? 1) * blocksMax,                                     // records the result buffer holds a batch: every block
  };
}
