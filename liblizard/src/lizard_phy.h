// FOCUS (Hermans et al., MobiSys 2016)'s modulator under soft decoding: LIZARD's picture.
//
// Data lives in the spectrum of a grey image: QPSK symbols (phase only) on the coefficients of
// the upper half-plane, taken in order of increasing frequency, 320 to a sub-channel. The
// lower half-plane is the conjugate reflection, the inverse FFT is real, and it is clipped to
// tame the peak-to-average ratio and shown as grey levels. Per-coefficient LLRs go into the QC-IRA LDPC, a block its
// code's sent bits by its rate, each block with a CRC-64 (the paper's RS(80, 64) fragments with hard QPSK lost
// to it on 2026-09-20, research/08, and left the code on 2026-10-25).
// What the paper leaves open and this file chooses: image size n (the paper says "e.g. 512"),
// the clipping ratio, equal magnitude on every used coefficient, and a checksum per block. The paper's 16 circle
// markers and calibrated lens undistortion are replaced by the border (layout.h: the ring, four corner marks, and a
// band of timing track and format word), which is all the receiver registers on.
#ifndef OB_LIZARD_H
#define OB_LIZARD_H
#include "ob.h"
#include "fmt.h"

// The code rate follows the frequency: a frame's data slots (two a coefficient, in order of rising frequency, subch x 636
// of them: every sub-channel's other two coefficients are its pilots, lizard_parity) carry LDPC blocks at four rates from the lowest frequencies outwards, 6/7, then 3/4, then 2/3, then 1/2, because a
// capture's signal-to-noise ratio falls with the frequency: the low frequencies carry more than 3/4 needs and the high
// ones less. Every block is the same LIZARD_BLOCK bytes whatever its rate (the
// transfer's 472: xfer.h XFER_BLOCK), then its LIZARD_CRC bytes of CRC-64 (layout.h ob_crc64), so a profile moves a frame's capacity by its block count. A block takes its
// code's sent bits (lizard_block_slots: 4480 slots at 6/7, 5120 at 3/4, 5760 at 2/3, 7680 at 1/2), the blocks packed one
// after another from the frame's first data slot whatever the sub-channels; the slots past the last
// block are padded (the whitening alone). A frame's layout is its word's: the 6/7, 2/3 and 1/2 blocks counted there, the
// 3/4 filling the rest (fmt.h ob_fmt_layout; lizard_layout), chosen by the sender, read by every receiver (2026-11-18;
// the profile was a function of the version alone before, which every receiver derived). The default senders' choice
// (lizard_tiers_for) puts the 6/7 on about the inner LIZARD_TIER_IN thousandths, the 2/3 on LIZARD_TIER_23 and the 1/2 on
// the outer LIZARD_TIER_OUT, each rounded to whole blocks. LIZARD_RATE is the 3/4 tier's rate, the one that fills.
enum { LIZARD_RATE = 4, LIZARD_TIER_IN = 390, LIZARD_TIER_23 = 150, LIZARD_TIER_OUT = 200, LIZARD_BLOCK = 472, LIZARD_CRC = 8 };

// ai: THE RINGS (64, 128 and 256 band cells a side since 2026-10-25; the 32 ring as well from 2026-09-27, the 96 in the
// ai: 256's place until 2026-10-10). The dotted band holds LIZARD_RING[r] cells of 2 x 2 modules on a side, the border
// ai: round it is 15 modules (a 12-module mark, a 3-module guard) and the corner clearance 15, so a side is S = 2 B + 30
// ai: modules (158, 286, 542) and the picture spans 2 B (128, 256, 512). The ring only locates the symbol and syncs its
// ai: grid, as 5G's sync block does; which picture is inside is the format word's to say (sub-channels, so n), and any
// ai: ring may carry any picture: a caller names a ring by its span. With none named the ring is LIZARD_RING_DEFAULT,
// ai: the 128, whatever the picture: no ring follows the version, even by default.
//
// The ring and the picture are independent grids. The ring is painted at pxm = lizard_pxm_for(n, span) pixels a
// module (ceil(n / span)), a whole number, so no module is ever split; the picture keeps its n samples and is resampled to fill its span
// modules (lizard_encode, Lanczos-3), never shrunk. The decoder reads the picture on its own n-sample grid in module
// coordinates, a step of 1 / scale modules, and never sees the painter's pixels.
// ai: A span that divides n has scale == pxm and the resampling is an exact copy (which ones, below).
// ai: The 64 ring's span (128) divides every picture size, so its pictures are exact copies; the 128 ring's (256)
// ai: divides all but 384 and 1152; the 256 ring's (512) divides 512, 1024 and 1536 and resamples 384, 768, 1152 and
// ai: 1280 (scale under 1 at 384: one pixel a module, the picture resampled up).
enum { LIZARD_RINGS = 3, LIZARD_RING_DEFAULT = 1 };
extern const int LIZARD_RING[LIZARD_RINGS];

// How a block's codeword lies on its coefficients: the bit map. A block is 4480 to 7680 slots by its rate, a thin
// ring of the spectrum taken in order of rising frequency, two slots a coefficient. Until 2026-09-23 slot i carried
// codeword bit i and nothing else: the payload's own bits were the picture's phases, so a payload of runs (a small
// file padded out to its blocks) painted thousands of coefficients at one phase, a peak the clip flattened, and the
// frame did not decode (LIZARD-64, every block 100 payload bytes then zeros: 6 of 64 blocks on a clean capture,
// against 64); and the codeword being systematic, data first, its 1272 weakest bits, the degree-2 parity, sat on the
// ring's outermost slots, the ones blur and distance take first. The map (LIZARD_BITMAP, LINEAR, the format's one):
// slot i carries bit i P mod n XOR the frame's whitening sequence at that slot (lizard_phy.c whiten_fill, PRBS-23), P near
// n over the golden ratio, so every run of the code is spread over the ring.
//
// Whitening is what fixes the small payload (every block 100 bytes then zeros: 6 of 64 blocks to 64 of 64, and
// all zero the same). Where the bits go matters only where a frame is losing blocks, and most of all where only
// its first one or two survive: every format's first block runs from the centre of the spectrum out to 40 cycles,
// so its reliability falls steeply across it. At 8 px of motion blur, 48 frames, LIZARD-64 read 674 B a frame with
// the bits in order and 870 linear, LIZARD-128 557 and 791; parity on the innermost slots did as well (899, 723) and
// over the whole race (2026-09-23) four whitened maps were within 1% of each other elsewhere. LINEAR over the inner
// parity because it spreads every run of the code over the ring rather than betting on which way the damage runs,
// which also covers the narrow-band damage the simulator cannot make: moire at one spatial frequency lands on
// consecutive slots of one block. The other maps raced (none, whitened only, parity spread, parity inner) left the
// code on 2026-10-25; lizard_t.bitmap stays LIZARD_BITMAP, the number a spec names (sim/ob.mjs Focus).
enum { LIZARD_BITMAP = 3 };
// LIZARD_SUB: the coefficients a sub-channel (640 slots: 636 data slots and its two pilots), the unit a format counts and
// the soft values' moments are taken over (a block's slots run across sub-channels as they fall). LIZARD_STEP: the sub-channels a version steps by: a
// format is a multiple of 16 sub-channels from 64 to 1072 and its word's version is subch / LIZARD_STEP, 4 to 67
// (fmt.h OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX; OB_FMT_SLOTS a version's data slots).
enum { LIZARD_SUB = 320, LIZARD_STEP = 16, LIZARD_STRIP = 64, LIZARD_TIERS = 4 };

// A run of blocks sent at one code rate. Low frequencies survive blur and distance and high ones do
// not, so a frame can spend a high rate where a capture is nearly always clean and a low one further out
// (research/10). Every block carries the same payload, because the fountain above wants blocks of one size: a block
// at a lower rate is a longer codeword and takes more slots (lizard_block_slots).
typedef struct { int rate, blocks, slots; } lizard_tier_t;   // LDPC rate index (ldpc.h), how many blocks, slots a block
// ai: A block's slots at a rate index: its code's sent bits (ldpc.h nt): 4480, 5120, 5760 and 7680; 0 for a rate with no
// ai: code.
int lizard_block_slots(int rate);

// Named, so internal.h can forward-declare it for the two accessors that hand back what lizard_acquire is holding.
typedef struct lizard_s {
  int n, subch, blocks, block_bytes;
  int blocks_max;         // the most blocks a layout of this frame holds (lizard_layout), its buffers' size
  float scale;            // picture samples a frame module, n / span (span = 2 LIZARD_RING[ring])
  int pxm, px;            // the painter's whole pixels a module, and the drive's side in them
  int margin;             // border modules between the symbol's edge and the picture: the mark and its guard, 15
  // The transform never holds the whole plane. The picture goes through a strip of LIZARD_STRIP columns at a time
  // (transformed along y), and only the rows v < n / 2 are kept, since the lower half-plane is the upper's mirror.
  // They are kept as blocks of bw columns v, each n rows u, which is the shape the second pass (along x) wants:
  // its butterflies then run over contiguous memory. Blocks past the last used coefficient do not exist.
  int bw, vblocks;
  int *pos;               // subch * 320 indices into that array ((v / bw) * n * bw + u * bw + v % bw, u wrapped), low frequency first
  ob_layout_t frame;      // the border round the picture: ring, corner marks, band (layout.h)
  int tiers;                                  // the four codes, fmt.h OB_TIER_RATE's order (inner first)
  lizard_tier_t tier[LIZARD_TIERS];           // each code's blocks in the frame's layout (0 where it carries none)
  ldpc_t code[LIZARD_TIERS];
  int *block_tier, *block_slot;               // per block: its tier, and its first data slot among the frame's (two a coefficient, pos order, the pilots left out)
  int used;                                   // the data slots the blocks take; the frame's other subch x 636 - used are padded
  struct lizard_ws *ws;    // ai: transform arrays and tables: allocated once, written by encode and decode
  ob_fmt_t fmt;           // what the frame's format word says (fmt.h); the PHY half restates this struct
  uint8_t fmt_cw[OB_FMT_BYTES_MAX];   // the word as painted, ob_fmt_bytes of it, all zero where the configuration has none
  int bitmap;             // LIZARD_BITMAP, the map init builds (the only one)
  int parity;             // ai: the painted picture's count mod 4, which its count pilots carry (lizard_parity)
} lizard_t;

// The word the last decode read out of the symbol, or 0 where it read none.
const ob_fmt_t *lizard_fmt_rx(const lizard_t *f);
// The frame's layout (fmt.h ob_fmt_layout): count[0] 6/7 blocks, count[1] 2/3, count[2] 1/2, the 3/4 blocks filling the
// rest; the next encode paints it and its word states it, and a decode finishes a frame at its own static word's (the
// word read before the frame). lizard_init sets the default senders' (lizard_tiers_for). 0, or -1 where it is no frame of
// this codec's version. Cheap: no table, no search.
int lizard_layout(lizard_t *f, const int count[3]);
// The sender's clock the word states (fmt.h, 2026-11-22): fps the pictures a second painted, 0 to OB_FMT_FPS_MAX, 240
// (0 a still: a print, or one picture with no clock), delay the largest of its block places' Refresh delays, 1 to
// OB_FMT_DELAY_MAX, 30. lizard_init sets LIZARD_CLOCK_FPS and 1; the next encode paints it. 0, or -1 for a value out of range.
enum { LIZARD_CLOCK_FPS = 60 };
int lizard_clock(lizard_t *f, int fps, int delay);
// ai: The pilots (SPEC 7.3; 2026-09-30, apart from the blocks since 2026-11-21): in every sub-channel j coefficient 0 is a
// ai: count pilot, the whitening XOR bit j & 1 of c, the sender's count of painted pictures mod 4, on both axes, and
// ai: coefficient 160 a reference pilot, the whitening alone. Bit 0 (the even sub-channels) flips sign picture to
// ai: picture, bit 1 (the odd) every second picture. So a capture of picture k that holds a share a of the picture before
// ai: and b of the one after reads, in size, 1 - 2 a - 2 b on the even sub-channels and 1 - 2 a (k even) or 1 - 2 b (k
// ai: odd) on the odd ones: which neighbour leaks, and from the two signs k mod 4, so a picture shown twice or skipped is
// ai: read, not guessed. No block's slot depends on c, so a held block (Refresh) never touches the readings.
// ai: lizard_parity sets the c the next encode paints (0 the default). 0, or -1.
int lizard_parity(lizard_t *f, int c);
// ai: What the last decode read from them (the soft values its own): r[0], bit 0's, the even sub-channels' count pilots
// ai: (each against its whitening, x + y) summed over the references of every sub-channel scaled to their count, and r[1]
// ai: the odd's; +1 where the picture's bit is 0 and -1 where it is 1, whatever the signal's scale. On a capture a share f
// ai: of whose light came from pictures of the other bit: 1 - 2 f, the reference the same in every picture; 0 at an even
// ai: mix and largest in size unmixed. sd[0], sd[1]: their standard errors by the delta method (the root of the sum of
// ai: (c_j - r ref_j)^2 over the denominator). A parity with none read reads NaN, its error too; none is read where the
// ai: soft values were made elsewhere. Returns the sub-channels read (0: none).
int lizard_pilot(const lizard_t *f, float r[2], float sd[2]);
// ai: The grid's shift the last decode read off the pilots and turned back before reading its soft values (lizard_phy.c
// ai: pilot_align): samples along u and v, 0, 0 where the pilots did not tell one from none (or soft values made
// ai: elsewhere). A decoder's own doing, not the format's: the pilots are known symbols at every radius.
void lizard_align(const lizard_t *f, float d[2]);
// The clipping ratio c (SPEC 6.7), a format constant: the picture is clipped at c times its rms (L = c sigma, the rms
// known from the coefficients sent, every one at the same magnitude) and that range mapped onto black to white.
#define LIZARD_CLIP 2.0f
// The format at subch sub-channels (a multiple of 16 from 64 to 1072), with the default senders' layout
// (lizard_tiers_for; lizard_layout another), in a picture of n (LIZARD_PICTURES, lizard_n_for(subch) for the format's own),
// in the ring whose span (2 LIZARD_RING[r]) span names, 0 the default. 0, or -1 for a configuration that is no format's.
int lizard_init(lizard_t *f, int n, int subch, int span);
// The same with blocks at the four codes given (an experiment's: each run's rate one of the four, in their order, each
// once, with its blocks; the 3/4's as given, not filling), in a frame of the fewest sub-channels that hold them. Its
// band is blank unless the word could state it (lizard_layout's own).
int lizard_init_tiers(lizard_t *f, int n, const lizard_tier_t *tier, int tiers, int span);
// The default senders' layout (2026-11-18: no receiver derives it; a frame's word states its counts) for subch
// sub-channels (a multiple of LIZARD_STEP, 64 to 1072), over its S = subch x 636 data
// slots: a 6/7 blocks, m 3/4 blocks, t 2/3 blocks and c 1/2 blocks, inner first, of s7, s3, s2 and s1 slots
// (lizard_block_slots), with a s7 + t s2 + c s1 + s3 <= S, m = floor((S - a s7 - t s2 - c s1) / s3) >= 1 (the 3/4 tier
// fills, the remainder padded), a, t and c each LIZARD_TIER_IN, LIZARD_TIER_23 and LIZARD_TIER_OUT thousandths of S in
// whole blocks, rounded half up, then c, t and a trimmed in that order until one 3/4 block fits (in integers).
// LIZARD-656: 6/7 x 36, 3/4 x 21, 2/3 x 11, 1/2 x 11, 576 slots padded; LIZARD-64: 6/7 x 4, 3/4 x 1, 2/3 x 1, 1/2 x 1.
// Writes the tiers present (all four at every format) to tier[] and returns their count, or 0 for a count that is no
// format's.
int lizard_tiers_for(int subch, lizard_tier_t tier[LIZARD_TIERS]);
// The blocks a frame of subch sub-channels carries under that profile (0 for no format).
int lizard_blocks_for(int subch);
// The most blocks any layout a word may state gives a frame of subch sub-channels (6/7 blocks, the smallest, to
// OB_FMT_COUNT67_MAX, and 3/4 blocks filling the rest; 152 at version 67): what a receiver's buffers hold.
int lizard_blocks_max_for(int subch);
// ai: Where the camera's optical axis falls in the image the calling thread decodes next, image px (2026-10-17): where
// ai: the lens model's centre is first taken (acquire.c lensm_axis) and where the final fit's search for it starts
// ai: (lens_centre, 2026-11-10: the centre is fitted in every frame, so a wrong or missing axis costs little). NaN, the
// ai: default, is the image's centre with a coarse search over the image first; a crop centred on the camera's frame has
// ai: its axis there, a 2:1 half's is the middle of its inner edge. Kept for the thread's later decodes until set again.
void lizard_set_axis(float x, float y);
// ai: The anchors' geometry in f's symbol, in modules, for a painter elsewhere (the senders' GPU painters,
// ai: core/tx/send_tables.cpp and the wasm's lizard_anchor_geom_get, 2026-10-26): out[0] a square's side, out[1 + i]
// ai: square i's first module along either axis (i < k, the ring's grid: 3, 5, 7; the grid is the same on both; out[1 + k]
// ai: to out[7] repeat the last, so a painter testing seven runs paints the same square again). Returns k. Nothing kept,
// ai: any thread.
int lizard_anchor_geom(const lizard_t *f, int out[8]);
void lizard_free(lizard_t *f);
// blocks: blocks * block_bytes. drive: px^2 values in 0..1, the frame at pxm pixels a module and the picture resampled into it.
void lizard_encode(const lizard_t *f, const uint8_t *blocks, float *drive);
// ai: The resampler lizard_encode paints the picture with, for an encoder elsewhere (the sender's GPU encoder,
// ai: 2026-09-29): out[0] q, the pixels across the picture and its guard interval; out[1] their first pixel in the drive
// ai: on either axis, (margin - cp) pxm; out[2] n; out[3] px. *i0 is q first samples (unwrapped: read mod n) and *w
// ai: q x 6 weights, the taps both passes read. 0, or -1.
int lizard_resample_geom(const lizard_t *f, int *out, const int **i0, const float **w);
// Measurement: keeps the bit pair (+1 / -1 per axis) the last encode put on each coefficient and the coefficient
// the last decode read there, both in pos order, two values a coefficient. For experiments, off by default.
int lizard_debug(lizard_t *f, int on, const int8_t **sym, const float **coef);
// Per block of the last decode: LDPC iterations used (-1 never converged, -2 declined, 0 not tried), and
// the information per coded bit its sub-channels were estimated to hold before decoding.
void lizard_block_stats(const lizard_t *f, const int8_t **its, const float **est);
// ok[b] = 1 where block b passed its checksum. Returns the count.
int lizard_decode(const lizard_t *f, const uint8_t *img, int iw, int ih, float gamma, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// The same decode in two halves, so the sampler can run somewhere the caller cannot wait on inline: a GPU fence
// costs more than the stage does unless several frames are outstanding, and one blocking call can have one.
// lizard_acquire registers the symbol and holds the result; lizard_finish takes it up again, sampling the held
// image itself when grid is NULL or reading the caller's grid when it is not. lizard_decode is the two in a row.
// The image the caller passed must outlive the matching lizard_finish, and a supplied grid is scratch: the
// transform runs in place over it. grid layout: strip s of LIZARD_STRIP columns at s * n * LIZARD_STRIP, n rows.
int lizard_acquire(const lizard_t *f, const uint8_t *img, int iw, int ih, float gamma, ob_result_t *res);
// ai: A receiver told nothing (2026-09-27). rings: a
// ai: codec a ring (LIZARD_RING), whose layouts the finder registers against, a bounded search over the side counts
// ai: like the eight orientations; the one that registered comes back in *which, holding the registration, with the
// ai: format word read (lizard_fmt_rx). The word names the sub-channels, so the picture; the caller hands the
// ai: registration to the codec for that picture in that ring (lizard_hand_over) and finishes there.
int lizard_acquire_ring(const lizard_t *const *rings, int k, const uint8_t *img, int iw, int ih, float gamma, ob_result_t *res, int *which);
// ai: The registration `from` holds, moved to `to`, a codec of the same ring (the same layout) at the picture the word
// ai: named; `to` reads the word again and finishes the frame. 0 when from holds nothing or the rings differ.
int lizard_hand_over(const lizard_t *from, const lizard_t *to);
// ai: A held registration let go of unfinished (a frame whose picture is not known).
void lizard_release(const lizard_t *f);
// The picture size a format takes (sim/lizard_pick.mjs N_FOR).
// ai: LIZARD_PICTURES: the sizes the ladder takes, smallest first (sim/lizard_pick.mjs PICTURE_SIZES): 384 to 1536.
enum { LIZARD_NPICTURES = 7 };
extern const int LIZARD_PICTURES[LIZARD_NPICTURES];
int lizard_n_for(int subch);
// The painter's whole pixels a module for picture n in a ring of span modules: ceil(n / span), so no module is split
// and the picture is never shrunk (every size gives an anchor's square, 4, 6 or 8 modules by the ring, at least 8
// pixels a side; SPEC 3.3, 6.13). lizard_init's pxm; the library's geometry takes it from here.
int lizard_pxm_for(int n, int span);
// The margin: modules of white round the frame, part of the symbol (2026-09-23) and painted by the codec, not
// asked of the page. With the frame's own outermost module, which is light, the ring the decoder reads has three
// modules of light outside it, and the finder assumes light there: noise painted right up to the ring left LIZARD-256
// 64 to 388 of its 1446 blocks (archive/build-scratch-2026-09/ring_race.mjs). It lies outside the module grid, whose origin stays the ring's
// outer corner, so nothing a decoder reads moves with it.
enum { LIZARD_QUIET = 2 };
// The symbol as a page paints it, RGBA, grey levels round(drive * 255), the margin round it (LIZARD_QUIET modules at
// pxm pixels a module; lizard-web/send.mjs through lizard-web/send-worker.mjs). rgba holds W^2 * 4 bytes, W = px + 2 LIZARD_QUIET pxm.
void lizard_paint_rgba(const lizard_t *f, const float *drive, uint8_t *rgba);
// ai: The same symbol as grey levels, one byte a pixel (what the GPU painter writes; the senders keep frames grey,
// ai: 2026-10-07): grey holds W^2 bytes, the margin 255, every level as lizard_paint_rgba's.
void lizard_paint_grey(const lizard_t *f, const float *drive, uint8_t *grey);
// The same with the quad found elsewhere: eight floats in image pixels, its orientation and its track score.
int lizard_acquire_quad(const lizard_t *f, const uint8_t *img, int iw, int ih, float gamma, ob_result_t *res,
                       const float *quad, int orient, float score);
int lizard_finish(const lizard_t *f, float *grid, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// The same with the soft values already quantised somewhere else: ext_llr is subch runs of 2 * LIZARD_SUB int8,
// one per global sub-channel, and ext_est the information estimate of each, which the decline gate reads. The
// grid is then unused and the transform is not run, so a caller supplying these supplies the spectrum's whole
// product. Everything from the decline gate onwards is shared with the ordinary path.
int lizard_finish_ext(const lizard_t *f, float *grid, const int8_t *ext_llr, const float *ext_est, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// The format word from soft values alone.
int lizard_fmt_check(const lizard_t *f, const float *q);
// ai: The last acquire's soft word values into out (4 x the word cells a side); returns the cells a side, 0 for none.
int lizard_fmt_soft(const lizard_t *f, float *out);
// What lizard_acquire is holding is declared in internal.h, where image_t and ob_reg_t live: it is the sampler's
// business, not the format's, and this header is the format's.
// For a caller running the transform elsewhere: the detrend basis it needs.
void lizard_tables(const lizard_t *f, float *out);          // cx[n], qx[n], x1r[n], x1i[n], x2r[n], x2i[n], sx2, sq2

#endif
