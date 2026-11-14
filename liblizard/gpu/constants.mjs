// ai: Constants the decoder's stages share with it, in a module that imports nothing (2026-10-01): the classifier and
// ai: proposer plans (plan/classify_gemm.mjs, plan/bank_fcn2.mjs) and the bank's host (bank_fcn2.mjs) take these
// ai: from here, not from decoder.mjs, which imports them; decoder.mjs re-exports them.
export const LEVELS = 5;        // level 0 is the capture; 1 to 4 halve it each time
// ai: SELECT's histogram origin (a peak's bin is 8 log2(score / TAU), glsl/bank.mjs), and the proposer's floor where its
// ai: weights file states none (plan/bank_fcn2.mjs).
export const TAU = 0.02;
