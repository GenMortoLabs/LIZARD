// F3 SELECT (src/select.slang): the strongest cap peaks of a frame from its tiles' slots, one workgroup a frame, by two
// cuts and the slots' order, ties the lower slot's (2026-11-16: a cut bin taken by its atomics' order kept a peak or its
// 9% weaker neighbour as the lanes happened to arrive, and on a real replay 7 of 1,165 frames were found or lost by that
// order alone).
import { slang } from "./load.mjs";

export const SELECT = { slang: slang("select.slang") };
