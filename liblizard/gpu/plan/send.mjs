// ai: The sender's constants and uniform layouts, which its Slang (../slang/send.mjs, ../slang/src/send_*.slang), the web's
// ai: encoder (../encoder.mjs, over the WGSL compiled from that Slang) and the native painter (../../gen/sender.mjs into
// ai: send.json, ../../core/tx/gpu_painter.cpp) share. No kernel text here.
import { SLOTS } from "./common.mjs";
import { PW_WORDS } from "./back_tiers.mjs";
import { PAYLOAD } from "../back/ref_ldpc.mjs";

// ai: the payload is whole words (472 B, 118), its CRC-64 the two words after it; the split CRC (../slang/src/lib/crc64.slang)
// ai: a lane a word
if (PAYLOAD % 4) throw new Error(`the block's ${PAYLOAD} B are no whole words`);
export const CRC_CHUNKS = PAYLOAD / 4;
// ai: The paint's uniform P in u32 words: sizes, a uvec4 a picture slot; dims; pw, PW_VECS uvec4 (the CRC-64 powers,
// ai: ../back/ref_ldpc.mjs crcPowers(PAYLOAD): each chunk's two words, high then low, then the init term's).
export const PW_VECS = Math.ceil((2 * CRC_CHUNKS + 2) / 4);
if (4 * PW_VECS !== PW_WORDS) throw new Error(`the CRC powers take ${4 * PW_VECS} words, the LDPC's uniform ${PW_WORDS}`);
export const PARAMS_AT = { sizes: 0, dims: 4 * SLOTS, pw: 4 * SLOTS + 4, words: 4 * SLOTS + 4 + 4 * PW_VECS };

// ai: PAINT's lanes and the longest codeword's words (the 1/2 code's 7680 bits); RSV's and RSH's lanes;
// ai: RSH's uniform (G and the anchors' two uvec4, 80 B)
export const PAINT_THREADS = 128;
export const CW_WORDS = 240;
export const RSV_THREADS = 64;
export const RSH_THREADS = 64;
export const GA_BYTES = 80;
// ai: BLAKE3's IV (the test stream's payloads, made in PAINT from the ids) and its message permutation
export const B3_IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
export const B3_PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

// ai: RaptorQ's blocks (../slang/src/send_rq_blocks.slang): TAB's layout (V0 to V3 at 0, 256, 512, 768; Deg at TAB_DEG; a
// ai: plan's (K, K', J, W, P, P1) TAB_PLAN words apart from TAB_PLANS; ../../core/tx/gpu_stream.cpp its twin) and a
// ai: block's words in the painter's BLOCKS
export const TAB_DEG = 1024, TAB_PLANS = 1056, TAB_PLAN = 8;
export const RQ_OUT_WORDS = 120;
