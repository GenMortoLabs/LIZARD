// F1, the pyramid (src/pyramid.slang): level l + 1 the 2 x 2 mean of level l, levels 1 to 4 from one 64 x 64 block of
// a frame's layer a workgroup.
import { slang } from "./load.mjs";
import { PYRAMID_BLOCK } from "../plan/pyramid.mjs";

if (PYRAMID_BLOCK !== 64) throw new Error(`the pyramid is written for a block of 64, not ${PYRAMID_BLOCK}`);

export const PYRAMID = { slang: slang("pyramid.slang") };
