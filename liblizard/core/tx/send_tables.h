// ai: The GPU sender's tables a format (2026-10-02, gpu_painter.h), from the C's own codec, the native painter's and the
// ai: web's (gpu/encoder.mjs takes these through the wasm since 2026-11-25: bindings/wasm/tables.cpp, sim/ob.mjs sendTables): the disc's rows and each coefficient's entry (the spectrum
// ai: transposed), the twiddles, the inverse's sizes and the clip (SU), the resampler's taps (lizard_resample_geom), the
// ai: geometry word G, the paint's uniform (send_plan.h P_*: the format's block count, each tier's first block, first
// ai: slot and blocks (the format's rate profile), dims, and the CRC powers), and the border, the C's paint of the
// ai: symbol (its pixels outside the square: ring, marks, band, word, margin) as grey bytes. Held to the C by `tx_check
// ai: gputables` (every format through the GPU's kernels against the C's paint: the border and the anchors to the byte,
// ai: the square within a grey level).
// ai: The border holds the anchors' squares too (SPEC 6.13: the C paints them over the picture), and anch says where
// ai: they are.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include "lizard_phy.h"
}

namespace lizard {

// ai: What no format changes, made from the C codec (dec/codes.h; gen wrote it as setup/send.json and a blob until
// ai: 2026-11-25): the paint's codes in TAB's order (fmt.h OB_TIER_RATE's: ldpc.h rate, slots a block), its TAB and the
// ai: CRC-64's split powers (the paint's uniform's pw). TAB: word 0 the whitening's offset; 16 words a code at 16 + 16 c
// ai: (n, k, m, z, mb, rows, lay, perm, slots a block, nt); a code's block rows' first entries (mb + 1), its entries
// ai: (col | shift << 16), its bit map (slot i's codeword bit, codes.h sentBit of LINEAR's perm[i], in the half i & 1 of
// ai: word i >> 1, over the nt sent bits); the whitening, a bit a slot of the frame over the ladder's largest and a word
// ai: past them (src/send_paint.slang reads it).
struct SendConsts {
  std::vector<uint32_t> pw;
  std::vector<int> codeRate, codeSlots;
  std::vector<uint32_t> tab;
  SendConsts();
};

struct SendTables {
  int n = 0, subch = 0, blocks = 0, blockBytes = 0, npos = 0, Vr = 0;   // ai: Vr: the disc's rows (the web's tb.V)
  int sdWords = 0;              // ai: paint's data bits a symbol: 16 data coefficients (LIZARD_SUB - 2 a sub-channel) a word
  int q = 0, sq = 0, W = 0, FW = 0, RW = 0, FS = 0;                       // ai: the square, a symbol's side, the frame
  int codes = 1, gap = 0;       // ai: symbols a frame side by side, and the pixels between them (gpu/encoder.mjs shape)
  bool copy = false;            // ai: every tap the C's on-sample copy: ipic's frame form and rsh's border form, no PIC
  std::vector<uint32_t> rows;   // ai: 2048 words (the uniform's size): (off, count, nonnegative, umaxN) a disc row
  std::vector<uint32_t> uv;     // ai: npos: each coefficient's entry of S
  std::vector<float> tw;        // ai: 2 n: cos, sin of -2 pi m / n
  std::vector<uint32_t> su;     // ai: 12 words: n, Vr, Vr n, npos, n n / 4, 0, 0, 0, then lim and 0.5 / lim as f32
  std::vector<uint32_t> g;      // ai: 12 words: n, q, sq, W, codes, FW, RW, FS, 1, the encode's first frame, gap, 0
  std::vector<uint32_t> taps;   // ai: 12 q words: 6 q sample indices (mod n), then 6 q weights as f32
  std::vector<uint8_t> border;  // ai: W x W grey, padded to a whole word
  std::vector<uint32_t> pu;     // ai: send_plan.h P_WORDS: sizes[0] (blocks, 0, the slots they take, npos), sizes[1 + c] (code c's tier: first block, first slot, blocks), dims (blocks, npos, symbols an encode, 0), pw
  // ai: the anchors (SPEC 6.13): the 8 words that follow g in rsh's G (gpu/slang/src/send_rsh.slang's G, the manifest's gWords
  // ai: 20): each anchor's first pixel along either axis of a symbol (margin included), then the square's side in pixels
  std::vector<uint32_t> anch;
};


// ai: The gap between two codes of a frame, in modules, where none is set (liblizard/gpu/encoder.mjs GAP_MODULES,
// ai: lizard-web/send-worker.mjs the same, since 2026-09-30); TxFormat.gap sets it
constexpr int GAP_MODULES = 12;

// ai: The tables of f's format (lizard_init'd, its rate set: the border's word states it), encodes of up to `frames`
// ai: frames of `codes` symbols side by side (2026-10-03, the desktop sender's two codes: the frame codes W + gap wide, the
// ai: gap the margin's byte, rsh stepping W + gap; the paint's symbols frames x codes, a frame's codes one pilot count),
// ai: `gap` modules apart (2026-10-04). "" or why not (the disc not as the shader reads it).
std::string sendTables(const lizard_t& f, const SendConsts& k, int frames, SendTables& t, int codes = 1, int gap = GAP_MODULES);
// ai: The border's grey bytes alone (sendTables' border, W x W padded to a whole word): again after the word changes
// ai: (lizard_clock: the sender's clock, src/fmt.h).
void sendBorder(const lizard_t& f, int W, std::vector<uint8_t>& border);

}  // namespace lizard
