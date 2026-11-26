// ai: The GPU sender's plan: its kernels' constants and uniforms' layouts, and its modules' recipes (a kernel's file
// ai: under gpu/slang/src and its constants, each a -D to slangc), the one definition the native painter
// ai: (gpu_painter.cpp, send_tables.cpp, gpu_stream.cpp) and gen's listing (tools/gpu_tree.cpp recipes: gen compiles
// ai: each module to SPIR-V and MSL for the native hosts and to WGSL for the web's encoder, gpu/send_wgsl.mjs) share.
// ai: The format-level constants it shares with the decoder (the block, the reference pilot, the CRC's powers' words,
// ai: the tree's version) are dec/plan.h's. Header-only.
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "../dec/plan.h"

extern "C" {
#include "fmt.h"
#include "lizard_phy.h"
}

namespace lizard::send {

// ai: a block in the painter's BLOCKS: its 472 bytes (the id first), then two words the paint fills with its CRC; the
// ai: CRC-64 split a lane a payload word (src/lib/crc64.slang), its powers PW_VECS vec4u (dec/codes.h crcPowers)
constexpr int BLOCK_BYTES = 480, CRC_CHUNKS = plan::PAYLOAD / 4, PW_VECS = (2 * CRC_CHUNKS + 2 + 3) / 4;
static_assert(plan::PAYLOAD % 4 == 0 && 4 * PW_VECS == plan::PW_WORDS, "the paint's CRC words are the LDPC's");
// ai: the picture sizes, which irows<n> and ipic<n> are built for (every one some version's picture), the paint's
// ai: uniform a uvec4 a size
constexpr int SLOTS = LIZARD_NPICTURES;
// ai: The paint's uniform P in u32 words: sizes (a uvec4 a picture slot: slot 0 the frame's blocks, the slots they take
// ai: and its coefficients, slot 1 + c code c's tier), dims, the CRC-64's powers
constexpr int P_SIZES = 0, P_DIMS = 4 * SLOTS, P_PW = 4 * SLOTS + 4, P_WORDS = 4 * SLOTS + 4 + 4 * PW_VECS;
// ai: PAINT's lanes and the longest codeword's words (the 1/2 code's 7680 bits); RSH's lanes (a lane a word, both
// ai: forms), the resampling form's band (its rows a workgroup: a pixel's taps read once for them) and a code's strip
// ai: (the samples a row's vertical pass makes for LANES words: n at most 2 q, so 2 x 4 LANES + 8), CODES_MAX codes a
// ai: frame; RSH's uniform G (12 words, then the anchors' 8: G_WORDS)
constexpr int PAINT_THREADS = 128, CW_WORDS = 240, RSH_THREADS = 64, RSH_BAND = 32, CODES_MAX = 2, RSH_STRIP = 2 * 4 * RSH_THREADS + 8, G_WORDS = 20;
// ai: a sub-channel's data coefficients (636 data slots, two a coefficient)
constexpr int DCOEF = OB_FMT_SLOTS / LIZARD_STEP / 2;
// ai: RaptorQ's blocks (src/send_rq_blocks.slang): TAB's layout (V0 to V3 at 0, 256, 512, 768; Deg at TAB_DEG; a plan's
// ai: (K, K', J, W, P, P1) TAB_PLAN words apart from TAB_PLANS: gpu_stream.cpp) and a block's words in BLOCKS
constexpr int TAB_DEG = 1024, TAB_PLANS = 1056, TAB_PLAN = 8, RQ_OUT_WORDS = 120;
static_assert(RQ_OUT_WORDS * 4 == BLOCK_BYTES, "the RQ blocks kernel writes a block as the paint reads it");

// ai: A sender's module: its name (send_<k>: the painter loads spv/<name>.spv, the web's encoder SEND_WGSL[<k>]), its
// ai: recipe, the stage the web runs it at ("" where only the native hosts run it) and whether the native hosts do
struct Module {
  std::string name, src;
  std::vector<std::pair<std::string, std::string>> defs;
  std::string webStage;
  bool native = true;
};

inline std::string uDef(int v) { return std::to_string(v) + "u"; }
inline std::string iDef(int v) { return std::to_string(v); }

// ai: Every module, in gen's order: paint, rsh in its two forms, RaptorQ's blocks (native alone), each picture size's
// ai: irows, ipic and ipic's frame form (the copy form's), then the web's present (a render pipeline's two stages, the
// ai: web's alone)
inline std::vector<Module> modules() {
  const std::vector<std::pair<std::string, std::string>> irows = {{"SLOT_COUNT", iDef(SLOTS)}, {"PW_VECS", iDef(PW_VECS)}, {"DCOEF", uDef(DCOEF)}, {"PB_REF", uDef(plan::PB_REF)}};
  std::vector<Module> m = {
      {"send_paint", "send_paint.slang",
       {{"SLOT_COUNT", iDef(SLOTS)}, {"PW_VECS", iDef(PW_VECS)}, {"PAINT_KNOWN", uDef(plan::KNOWN)}, {"CW_WORDS", uDef(CW_WORDS)}, {"NCODES", uDef(plan::NC)}, {"DCOEF", uDef(DCOEF)},
        {"PB_REF", uDef(plan::PB_REF)}, {"CRC64_CHUNKS", uDef(CRC_CHUNKS)}},
       "compute"},
      {"send_rsh", "send_rsh.slang", {{"BORDER_ONLY", "0"}, {"LANES", iDef(RSH_THREADS)}, {"BAND", uDef(RSH_BAND)}, {"STRIP", uDef(RSH_STRIP)}, {"CODES_MAX", uDef(CODES_MAX)}}, "compute"},
      {"send_rshBorder", "send_rsh.slang", {{"BORDER_ONLY", "1"}, {"LANES", iDef(RSH_THREADS)}}, "compute"},
      {"send_rqBlocks", "send_rq_blocks.slang", {{"TAB_DEG", uDef(TAB_DEG)}, {"TAB_PLANS", uDef(TAB_PLANS)}, {"TAB_PLAN", uDef(TAB_PLAN)}, {"OUT_WORDS", uDef(RQ_OUT_WORDS)}}, ""},
  };
  for (int s = 0; s < LIZARD_NPICTURES; s++) {
    const int n = LIZARD_PICTURES[s];
    std::vector<std::pair<std::string, std::string>> d = {{"LANES", iDef(n / 8)}, {"PICTURE_N", uDef(n)}};
    d.insert(d.end(), irows.begin(), irows.end());
    m.push_back({"send_irows" + iDef(n), "send_irows.slang", d, "compute"});
    m.push_back({"send_ipic" + iDef(n), "send_ipic.slang", {{"LANES", iDef(n / 8)}, {"PICTURE_N", uDef(n)}, {"FRAME_OUT", "0"}}, "compute"});
    m.push_back({"send_ipicFrame" + iDef(n), "send_ipic.slang", {{"LANES", iDef(n / 8)}, {"PICTURE_N", uDef(n)}, {"FRAME_OUT", "1"}}, "compute"});
  }
  m.push_back({"send_presentVs", "send_present_vs.slang", {}, "vertex", false});
  m.push_back({"send_presentFs", "send_present_fs.slang", {}, "fragment", false});
  return m;
}

}  // namespace lizard::send
