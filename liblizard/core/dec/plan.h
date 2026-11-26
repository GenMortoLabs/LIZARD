// ai: The GPU decoder's plan as the C++ host states it: the stages' constants build.cpp makes the setup from and the
// ai: kernels' recipes state (since 2026-11-25 the one definition: the JavaScript setup host it was held to is gone).
#pragma once
#include <cstdint>

namespace lizard::plan {

// ai: The tree's version: what the host makes and reads of a variant (core/dec/build.cpp) against gen's index of it
// ai: (setup/<variant>.json treeVersion, liblizard/gen/gen.mjs TREE_VERSION): raised in both places whenever what the
// ai: host takes from gen changes, so an index of another version is refused before any of it is read.
// ai: 25: the host makes the setup itself (build.cpp): gen's setup/<variant>.json the variant's index (each module's
// ai: recipe, name, lanes and needs; the nets' files as blobs/<hash>.bin), no recorded tree or buffer blobs read.
// ai: 24: the squares' fill's sums in a lane buffer of their own (ANCH_FIND's group binding 8, ANCH_FIT's 10).
// ai: 23: no REFIT_POST (the tree's front.refitPost); RESID the lens record alone.
// ai: 22: the word's four counts, every layout (no CRC): the configuration two words, record word 29 the version << 24 |
// ai: the 6/7 count << 16 | the 3/4 << 8 | the 2/3, word 30 the clock | the 1/2 count << 16; LISTS a seventh word a frame.
// ai: 21: the soft pass's four steps a size (tnoise, each sub-channel's 1 / N into EP's fourth float, before talign,
// ai: which weights the pilots by it and binds EP at 6).
// ai: 20: versions 1 to 3 in the word's list (VERSION_MIN 1, the configurations' sums from version 1).
// ai: 19: the word's index and the clock (RS(n, 6), five unknown bytes; record word 30 the clock, word 6 the word's
// ai: last 32 bits), the pilots apart from the blocks (PILOT a frame's four readings, then its shift; EP a vec4 a
// ai: sub-channel; L in data-slot order, dims.lStride 159 a sub-channel), the block's CRC-64/LIZARD (REC 120 words).
constexpr int TREE_VERSION = 25;

// ai: The nets' files (the library's paths, gen's index naming each one's blob): the classifier installed and its
// ai: int8 twin, the proposer in float and int8, each code's learned LDPC stop by its checks (the 6/7's 640, the
// ai: 3/4's 1280, the 2/3's 1920, the 1/2's 3840)
constexpr const char* CLASSIFIER = "gpu/cnn/weights-v3.safetensors";
constexpr const char* CLASSIFIER_INT8 = "gpu/cnn/weights-v3-q8.safetensors";
constexpr const char* PROPOSER = "gpu/cnn/proposer/weights2.safetensors";
constexpr const char* PROPOSER_INT8 = "gpu/cnn/proposer/weights2-q8.safetensors";
constexpr struct { int m; const char* file; } STOPS[] = {{640, "gpu/back/stop/stop_67.safetensors"}, {1280, "gpu/back/stop/stop.safetensors"},
                                                          {1920, "gpu/back/stop/stop_23.safetensors"}, {3840, "gpu/back/stop/stop_12.safetensors"}};
// ai: the workgroup memory every stage is planned for (the classifier's and the proposer's forms, pass 1's sizes, the
// ai: LDPC's forms): 32 KB, the floor a device is held to (core/dec/setup.cpp variantsFor)
constexpr int WG_BYTES = 32768;

// ai: The pyramid's levels (level 0 the capture), the counters a frame, the finder's result
// ai: floats a frame, sel's words, the timestamps a readback, the ring's slack, the batch's ceiling and cap, the twins'
// ai: measurement, SELECT's histogram origin, F6's track threshold, SELECT's cap
constexpr int LEVELS = 5, COUNTERS = 16, RESULT = 32, SEL_WORDS = 4, TS_MAX = 80, TS_PLAIN = 2, RING_SLACK = 4;
constexpr int B_CEIL = 32, CAP_MS = 400, TWIN_FRAMES = 2, TWIN_ROUNDS = 3, CAP = 256;
constexpr double TAU = 0.02, TAU_TRACK = 0.15;
// ai: the finder's
constexpr double CENTRE[2] = {8, 6};
constexpr int K = 4, CELL = 16, ORIENT = 8, BENDS = 2, VOTE_READS = 16, NEAR = 40, SUPPRESS = 10, PEAK_LIST = 512, STATIONS = 32, R2_N = 13;
constexpr double SMIN = 0.35, RAY_LO = 0.45, RAY_HI = 2.2, RAY_SPREAD = 0.3;
// ai: F7's (nodes.slang, refit.slang), the pyramid's block, the bank's keep and tile
constexpr int NODES_MAX = 136, LENS_SLOTS = 5, RESID = LENS_SLOTS, PYRAMID_BLOCK = 64, KEEP = 4, TILE = 16;
// ai: the back half's (its ARGS, counts, a block's record) and the anchors' (anch_*.slang)
constexpr int ARGS_WORDS = 12, COUNTS = 8, PAYLOAD = 472, REC_WORDS = 2 + PAYLOAD / 4, REC_BYTES = 4 * REC_WORDS, NC = 4, ANCH_SIDE = 7, ANCH_K = ANCH_SIDE * ANCH_SIDE, ANCH_GS = 16;
constexpr int ANCH_SURF = 4 + (ANCH_K + 1) / 2, ANCH_GAIN = ANCH_SURF + 2, ANCH_REC = ANCH_SURF + 4;
constexpr int ARGS_PASS1 = 0, ARGS_REDUCE = 3, ARGS_PASS2 = 6, ARGS_SOFT = 9;
// ai: the back half's: the rule stage's blocks a workgroup, the reference pilot's
// ai: coefficient, a frame's pilot readings, the bits a block's payload and CRC fill (the k of the 3/4, 2/3 and 1/2
// ai: codes), the CRC's powers' words, pass 1's columns a workgroup, the samples the anchors' erase takes round a square
constexpr int TIER_BLOCKS = 160, PB_REF = 160, PILOT_READ = 4, KNOWN = 3840, PW_WORDS = 240, CPW = 4, ANCH_ERASE = 2;

}  // namespace lizard::plan
