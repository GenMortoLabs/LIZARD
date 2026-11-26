// The codec's own exports in the library's WebAssembly module, beside the C API (CMakeLists.txt's list): for the tree's
// pages, the native generator and the lab through sim/ob.mjs, over one lizard_t and one lizard_any_t. No stable interface.
#include "ob.h"
#include "internal.h"
#include "lizard_phy.h"
#include "any.h"
#include "ldpc.h"
#include "xfer.h"
#include "fmt.h"
#include "layout.h"
#include "lizard.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static ob_result_t R;
// Camera frames arrive as RGBA. Converting here keeps the 4 bytes a pixel out of JS loops, a strip at a time, so that
// the heap never holds a frame at four bytes a pixel: n pixels of RGBA go to luma[offset ..] of a buffer of `total`.
// Returns the buffer, which the last strip completes.
static uint8_t *lum; static int lum_cap;
const uint8_t *ob_luma_strip(const uint8_t *rgba, int n, int offset, int total) {
  if (total > lum_cap) { free(lum); lum = malloc((size_t)total); lum_cap = lum ? total : 0; }
  if (!lum || offset < 0 || n < 0 || offset + n > total) return NULL;
  PROF(PROF_LUMA, ob_luma(rgba, n, lum + offset));
  return lum;
}
const ob_result_t *ob_last(void) { return &R; }
// Test hook (scripts/exp/simd_check.mjs, lizard_bench.mjs): nonzero asks a decode to leave a hash of its floats and soft
// values here (lizard_phy.c). The host defines it (core/cpu/codec.c, in this module too).
extern uint32_t ob_debug_hash;
uint32_t ob_dbg_hash(int arm) { uint32_t v = ob_debug_hash; ob_debug_hash = (uint32_t)arm; return v; }
double ob_prof(int i) { return i >= 0 && i < PROF_N ? ob_prof_ms[i] : 0; }
void ob_prof_reset(void) { for (int i = 0; i < PROF_N; i++) ob_prof_ms[i] = 0; }

static lizard_t F;
static int fready;
// ai: The format at n and subch in the ring span names (lizard_phy.h lizard_init).
int lizard_setup(int n, int subch, int span) {
  if (fready) { lizard_free(&F); fready = 0; }
  if (lizard_init(&F, n, subch, span)) return -1;
  fready = 1;
  return F.block_bytes;
}
// ai: the codec lizard_setup or lizard_setup_tiers made last, NULL before one (bindings/wasm/tables.cpp, the GPU sender's
// ai: tables of it)
const lizard_t *lizard_current(void) { return fready ? &F : NULL; }
// Up to four runs (LIZARD_TIERS) of blocks at different code rates (lizard_phy.h): rate index and blocks (a block's slots are
// its rate's, lizard_block_slots); a run of 0 blocks ends the list.
int lizard_setup_tiers(int n, int span, int r0, int b0, int r1, int b1, int r2, int b2, int r3, int b3) {
  const lizard_tier_t all[LIZARD_TIERS] = { { r0, b0, 0 }, { r1, b1, 0 }, { r2, b2, 0 }, { r3, b3, 0 } };
  int tiers = 0;
  while (tiers < LIZARD_TIERS && all[tiers].blocks > 0) tiers++;
  if (fready) { lizard_free(&F); fready = 0; }
  if (lizard_init_tiers(&F, n, all, tiers, span)) return -1;
  fready = 1;
  return F.block_bytes;
}
// ai: A receiver told nothing: the module's one blind receiver (any.h), behind the exports.
static lizard_any_t ANY = { .which = -1 };
// ai: nmax: the largest picture this receiver decodes (1536, the ladder's top, in every receiver since 2026-09-29). Returns the
// ai: block bytes (472), or -1.
int lizard_any_setup(int nmax) { return lizard_any_init(&ANY, nmax); }
// ai: blocks and ok hold lizard_any_max_blocks() blocks. Returns the blocks decoded; lizard_any_which says which ring
// ai: registered (-1 none), lizard_any_n the picture finished (0 none).
int lizard_any_rx(const uint8_t *img, int iw, int ih, float gamma, uint8_t *blocks, uint8_t *ok) {
  return lizard_any_frame(&ANY, img, iw, ih, gamma, blocks, ok, &R);
}
int lizard_any_which(void) { return ANY.which; }
int lizard_any_n(void) { return ANY.n; }
// ai: Pictures built since the setup, for a caller timing its first frames (the rig's workersFirst).
int lizard_any_builds(void) { return ANY.builds; }
int lizard_any_blocks(void) { return ANY.last ? ANY.last->blocks : 0; }
int lizard_any_max_blocks(void) { return lizard_any_top(&ANY); }
// ai: The word this frame's ring read, or 0.
const ob_fmt_t *lizard_any_fmt(void) { return lizard_any_word(&ANY); }
// ai: the pilots the blind receiver's last frame read: out[0], out[1] r of the even and of the odd sub-channels, out[2],
// ai: out[3] their standard errors; the sub-channels read, 0 for none
int lizard_any_pilot_get(float *out) { return lizard_any_pilot(&ANY, out, out + 2); }

// The bit map of the codec lizard_setup made (lizard_phy.h LIZARD_BITMAP, the only one), which a spec may name (sim/ob.mjs
// Focus refuses any other).
int lizard_bitmap_get(void) { return fready ? F.bitmap : -1; }

// What the last decode read out of the band, an ob_fmt_t (fmt.h: the version, count[4], fps and delay, as ints), or 0
// for nothing read.
const ob_fmt_t *lizard_fmt_rx_ptr(void) { return fready ? lizard_fmt_rx(&F) : 0; }
// ai: The frame's layout the next encode paints (lizard_phy.h lizard_layout): 6/7, 3/4, 2/3 and 1/2 blocks, the rest padded;
// ai: 0, or -1 where it is no frame. lizard_blocks() then says the frame's blocks; lizard_blocks_max() the most any layout
// ai: of this codec holds (what a told receiver's buffers hold, a frame's word choosing its own).
int lizard_layout_set(int a, int b, int t, int c) { const int k[OB_TIERS] = { a, b, t, c }; return fready ? lizard_layout(&F, k) : -1; }
// ai: the sender's clock the word states (lizard_phy.h lizard_clock): fps 0 to 240 (0 a still), delay 1 to 30
int lizard_clock_set(int fps, int delay) { return fready ? lizard_clock(&F, fps, delay) : -1; }
int lizard_blocks_max(void) { return F.blocks_max; }
// ai: The pilots (lizard_phy.h lizard_parity, lizard_pilot): the count mod 4 the next encode paints, and what the last told
// ai: decode read, out[0], out[1] r of the even and of the odd sub-channels, out[2], out[3] their standard errors; the
// ai: sub-channels read, 0 for none.
int lizard_parity_set(int c) { return fready ? lizard_parity(&F, c) : -1; }
int lizard_pilot_get(float *out) { return fready ? lizard_pilot(&F, out, out + 2) : 0; }
// ai: the grid's shift the last decode read off the pilots and turned back (lizard_phy.h lizard_align), for a measurement
// ai: that re-reads the coefficients lizard_dbg_coef holds, which are the ones before the turn (2026-10-07)
int lizard_align_get(float *out) { if (!fready) return 0; lizard_align(&F, out); return 1; }
int lizard_subch(void) { return F.subch; }
int lizard_blocks(void) { return F.blocks; }
int lizard_side(void) { return F.px; }
int lizard_cell(void) { return F.pxm; }   // whole pixels a module in the painted symbol
int lizard_quiet(void) { return LIZARD_QUIET; }   // the margin the codec paints, in modules, so the JS has no copy of its own
void lizard_tx(const uint8_t *blocks, float *drive) { lizard_encode(&F, blocks, drive); }
// The same encode and the page's pixels in one call, margin included: drive is the scratch the encode writes, rgba
// what is painted. ai: ink and paper (0xRRGGBB; 2026-11-16, Send's Colours, the native senders' Sender::setColours):
// ai: each grey level v painted along the line between them, each channel apart, (ink (255 - v) + paper v + 127) / 255,
// ai: the margin the paper; 0x000000 and 0xFFFFFF leave lizard_paint_rgba's grey as it is.
void lizard_tx_rgba(const uint8_t *blocks, float *drive, uint8_t *rgba, uint32_t ink, uint32_t paper) {
  lizard_encode(&F, blocks, drive);
  lizard_paint_rgba(&F, drive, rgba);
  ink &= 0xFFFFFFu; paper &= 0xFFFFFFu;
  if (ink == 0 && paper == 0xFFFFFFu) return;
  uint32_t lut[256];
  for (uint32_t v = 0; v < 256; v++) {
    uint32_t px = 0xFF000000u;
    for (int k = 0; k < 3; k++) {
      const uint32_t a = (ink >> (16 - 8 * k)) & 0xFFu, b = (paper >> (16 - 8 * k)) & 0xFFu;
      px |= ((a * (255 - v) + b * v + 127) / 255) << (8 * k);
    }
    lut[v] = px;
  }
  const int W = F.px + 2 * LIZARD_QUIET * F.pxm;
  uint32_t *p = (uint32_t *)(void *)rgba;
  for (size_t i = 0, n = (size_t)W * W; i < n; i++) p[i] = lut[p[i] & 0xFFu];
}
// ai: The resampler's geometry and taps (lizard_phy.h lizard_resample_geom): out[0..3] q, the square's first drive pixel, n
// ai: and px, out[4] and out[5] the heap offsets of the q first samples and the q x 6 weights. 0, or -1 before a setup.
int lizard_rs_geom(int *out) {
  const int *i0; const float *w;
  if (!fready || lizard_resample_geom(&F, out, &i0, &w)) return -1;
  out[4] = (int)(intptr_t)i0; out[5] = (int)(intptr_t)w;
  return 0;
}
int lizard_rx(const uint8_t *img, int iw, int ih, float gamma, uint8_t *blocks, uint8_t *ok) { return lizard_decode(&F, img, iw, ih, gamma, blocks, ok, &R); }
static const int8_t *dbg_sym; static const float *dbg_coef;
int lizard_dbg(int on) { dbg_sym = 0; dbg_coef = 0; return fready ? lizard_debug(&F, on, &dbg_sym, &dbg_coef) : -1; }
const int8_t *lizard_dbg_sym(void) { return dbg_sym; }
const float *lizard_dbg_coef(void) { return dbg_coef; }
const int8_t *lizard_blk_its(void) { const int8_t *i; const float *e; lizard_block_stats(&F, &i, &e); return i; }
const float *lizard_blk_est(void) { const int8_t *i; const float *e; lizard_block_stats(&F, &i, &e); return e; }
// The top of the heap. sbrk never gives memory back here, so this is the most the module has had in use.
unsigned ob_heap_top(void) { return (unsigned)(uintptr_t)sbrk(0); }

// Test hook for a second implementation of ob_sample_grid (the GPU back half's checks, scripts/gpu/back/). Acquires
// one capture with the current LIZARD setup, writes out everything the sampler reads, then the grid it produces,
// so the two can be held to the same floats. A GPU cannot be byte-identical to the scalar build the way
// scripts/exp/simd_check.mjs holds the vector build, so what this exists for is a tolerance and an ulp count.
//   dims:  nx, ny, n, found
//   state: H[9], g0, step, node_x[nx], node_y[ny], nx * ny pairs of (dx, dy), lut[256], the sampler's own ms
//   grid:  n * n samples, row major, pitch n
// Returns n, or 0 if the capture did not register. The caller sizes the buffers: nx and ny are at most
// OB_MAX_NODES, so 9 + 2 + 2 * 40 + 2 * 40 * 40 + 256 + 1 floats is always enough.
static void state_dump(const image_t *im, const ob_reg_t *reg, int32_t *dims, float *state) {
  const ob_layout_t *L2 = &F.frame;
  const int nx = L2->nx, ny = L2->ny;
  dims[0] = nx; dims[1] = ny; dims[2] = F.n; dims[3] = 1;
  float *s = state;
  for (int i = 0; i < 9; i++) *s++ = reg->H.h[i];
  *s++ = F.margin + 0.5f / F.scale; *s++ = 1.0f / F.scale;
  for (int i = 0; i < nx; i++) *s++ = L2->node_x[i];
  for (int i = 0; i < ny; i++) *s++ = L2->node_y[i];
  for (int i = 0; i < nx * ny; i++) { *s++ = reg->nodes[i].dx; *s++ = reg->nodes[i].dy; }
  for (int i = 0; i < 256; i++) *s++ = im->lut[i];
}

int ob_test_sample(const uint8_t *img, int iw, int ih, float gamma, int32_t *dims, float *state, float *grid) {
  if (!fready) return 0;
  const int n = F.n;
  dims[0] = F.frame.nx; dims[1] = F.frame.ny; dims[2] = n; dims[3] = 0;
  image_t im;
  ob_image_init(&im, img, iw, ih, gamma);
  ob_reg_t reg;
  if (!ob_acquire(&F.frame, &im, &reg, &R, NULL)) return 0;
  state_dump(&im, &reg, dims, state);
  const float g0 = F.margin + 0.5f / F.scale, step = 1.0f / F.scale;
  const double t0 = ob_now_ms();
  ob_sample_grid(&F.frame, &im, &reg, g0, g0, step, 0, n, 0, n, grid, n);
  state[9 + 2 + F.frame.nx + F.frame.ny + 2 * F.frame.nx * F.frame.ny + 256] = (float)(ob_now_ms() - t0);
  ob_reg_free(&reg);
  return n;
}

// The decode in two halves (lizard_phy.h), for a receiver that samples on the GPU. lizard_rx_acquire registers and
// writes out what the sampler needs, in state_dump's layout above minus the trailing ms; lizard_rx_finish takes
// the grid back. A caller that passes grid = 0 gets the sampler the decoder has always had, which is how the
// two arms are compared on one build.
int lizard_rx_acquire(const uint8_t *img, int iw, int ih, float gamma, int32_t *dims, float *state) {
  if (!fready) return 0;
  dims[0] = F.frame.nx; dims[1] = F.frame.ny; dims[2] = F.n; dims[3] = 0;
  if (!lizard_acquire(&F, img, iw, ih, gamma, &R)) return 0;
  state_dump(lizard_held_image(&F), lizard_held_reg(&F), dims, state);
  return 1;
}
// Acquisition from a quad found elsewhere, so a GPU finder can be held to the decode it produces. Writes the
// same state lizard_rx_acquire does.
int lizard_rx_acquire_quad(const uint8_t *img, int iw, int ih, float gamma, const float *quad, int orient, float score, int32_t *dims, float *state) {
  if (!fready) return 0;
  dims[0] = F.frame.nx; dims[1] = F.frame.ny; dims[2] = F.n; dims[3] = 0;
  if (!lizard_acquire_quad(&F, img, iw, ih, gamma, &R, quad, orient, score)) return 0;
  state_dump(lizard_held_image(&F), lizard_held_reg(&F), dims, state);
  return 1;
}
int lizard_rx_finish(float *grid, uint8_t *blocks, uint8_t *ok) { return fready ? lizard_finish(&F, grid, blocks, ok, &R) : 0; }

// ai: The mesh tables (the node lattice, the per-module kind map, which nodes carry a mark), for the checks that read a
// ai: symbol's cells (test/vectors.mjs, scripts/gpu/word_check.mjs, scripts/exp/fmt_word.mjs) and the lab's GPU tables
// ai: (scripts/gpu/native/format.mjs, host.mjs).
int ob_test_mesh_tables_out(int32_t *dims, float *nodeX, float *nodeY, uint8_t *kind, int32_t *mark) {
  return fready ? ob_test_mesh_tables(&F.frame, dims, nodeX, nodeY, kind, mark) : 0;
}
int ob_test_fmt_plan(int cells, float *pts, int32_t *meta, int32_t *widx) {
  return fready ? ob_thin_fmt_plan(&F.frame, cells, pts, meta, widx) : 0;
}
// The track read's sample plan (the lab's GPU tables, scripts/gpu/native/format.mjs and host.mjs), and a single
// hypothesis through it and what the finder settled on last time (scripts/exp/fmt_word.mjs).
int ob_test_thin_plan(float *pts, int32_t *meta, int32_t *signs) { return fready ? ob_thin_score_plan(&F.frame, pts, meta, signs) : 0; }
int ob_test_thin_score_out(const uint8_t *img, int w, int h, const float *centres, int hyp, int merged, float *out) {
  return fready ? ob_test_thin_score(&F.frame, img, w, h, centres, hyp, merged, out) : 0;
}
void ob_test_finder_out(float *out) { for (int k = 0; k < 8; k++) out[k] = ob_finder_info[k]; }
// ai: the anchors (SPEC 6.13): how many the last acquisition counted (-1 where it made no mesh), and their geometry in
// ai: the current symbol for the lab (the web's GPU encoder takes them in bindings/wasm/tables.cpp; lizard_phy.h lizard_anchor_geom: out[0] the side, out[1..7] the
// ai: first modules, the ring's k and the last repeated past them; k (3, 5 or 7), or 0 before a setup)
int lizard_anchors_found(void) { return ob_anch_found; }
int lizard_anchor_geom_get(int *out) { return fready ? lizard_anchor_geom(&F, out) : 0; }
int ob_test_fmt_cells(void) { return fready ? ob_fmt_side_cells(F.frame.w < F.frame.h ? F.frame.w : F.frame.h, F.frame.reserve) : 0; }

// The current layout's mark geometry (acquire.c ob_mark_geom), for the lab's GPU tables (scripts/gpu/native/format.mjs,
// host.mjs) and scripts/exp/fmt_word.mjs.
int ob_test_marks(float *out) { if (!fready) return 0; ob_mark_geom(&F.frame, out); return 1; }
// Shapes the GPU pipeline has to allocate for: n, bw, vblocks, subch, blocks, and the coefficient index table's
// length. The table itself is lizard_pos_ptr: subch * LIZARD_SUB ints into the spectrum (src/lizard_phy.c).
int lizard_shape(int32_t *out) {
  if (!fready) return 0;
  out[0] = F.n; out[1] = F.bw; out[2] = F.vblocks; out[3] = F.subch; out[4] = F.blocks; out[5] = F.subch * LIZARD_SUB;
  return 1;
}
const int *lizard_pos_ptr(void) { return fready ? F.pos : 0; }
void lizard_tables_out(float *out) { if (fready) lizard_tables(&F, out); }
// The decode finished from soft values quantised elsewhere: llr is subch runs of
// 2 * LIZARD_SUB int8, est one information estimate per sub-channel. The transform is not run.
int lizard_rx_ldpc(const int8_t *llr, const float *est, uint8_t *blocks, uint8_t *ok) {
  return fready ? lizard_finish_ext(&F, 0, llr, est, blocks, ok, &R) : 0;
}
// Which slot of the frame each block starts at, and how many it spans: what maps a flat per-slot soft-value array
// (subch runs of 2 * LIZARD_SUB) onto the blocks.
void lizard_block_runs(int32_t *first, int32_t *count) {
  if (!fready) return;
  for (int b = 0; b < F.blocks; b++) { first[b] = F.block_slot[b]; count[b] = F.tier[F.block_tier[b]].slots; }
}

// The LDPC code of a tier, for a decoder running elsewhere: tier is an index into the frame's tiers (lizard_phy.h
// lizard_tiers_for gives each one's blocks). See ldpc.c ldpc_tables for the layout (12 dims).
void lizard_ldpc_tables(int tier, int32_t *dims, int32_t *lay) {
  if (fready && tier >= 0 && tier < LIZARD_TIERS) ldpc_tables(&F.code[tier], dims, lay);
}
int lizard_fmt_check_out(const float *q) { return fready ? lizard_fmt_check(&F, q) : 0; }
int lizard_fmt_soft_out(float *out) { return fready ? lizard_fmt_soft(&F, out) : 0; }
// The sampled grid's origin and step in module coordinates, what state_dump hands a sampler after an
// acquisition, for a sampler that runs without one.
void lizard_grid_out(float *out) { out[0] = F.margin + 0.5f / F.scale; out[1] = 1.0f / F.scale; }

// The decline gate against the live configuration, for the GPU back half's to be held to (scripts/gpu/back/frames.mjs).
// The CRC itself needs no state, so ob_test_crc_hash is exported straight out of src/lizard_phy.c.
int ob_test_crc_gate_out(const float *est, float *blk_est, float *bar, uint8_t *declined) {
  return fready ? ob_test_crc_gate(&F, est, blk_est, bar, declined) : 0;
}

// ---- The transfer (xfer.h): the block id, the header, the file's check, BLAKE3 ----
// The C functions with plain arguments are exported as they are (xfer_root_check, xfer_root, xfer_sb_cv,
// xfer_b3_hash); these wrap the rest for JS. A uint32 comes back as a signed int32, so JS
// reads it with >>> 0; a block's bare id (its source block and symbol, 40 bits), a TOI (48 bits, xfer.h) and lengths past
// 2^32 cross as doubles, exact to 2^53.

// The layout's numbers, so the JS keeps no copy: id bytes, payload bytes, symbol bits, source block bits, the control
// source block, name bytes at most, sb_log2 least and most, source blocks at most, BLAKE3's
// hash byte, TOI bits, the TOI's first byte in the id, the longest description, a description's bytes at most.
void xfer_layout(int32_t *out) {
  const int32_t v[] = { XFER_ID_BYTES, XFER_PAYLOAD, XFER_ESI_BITS, XFER_SBN_BITS, XFER_CONTROL, XFER_NAME_MAX,
                        XFER_LOG2_MIN, XFER_LOG2_MAX, XFER_MAX_SOURCE_BLOCKS, XFER_HASH_BLAKE3, XFER_TOI_BITS, XFER_ID_TOI,
                        XFER_DESC_LONGEST, XFER_DESC_MAX };
  for (unsigned i = 0; i < sizeof v / sizeof *v; i++) out[i] = v[i];
}
double xfer_id_of(uint32_t sbn, uint32_t symbol) { return (double)xfer_id(sbn, symbol); }
// The test stream's payload for a block id (xfer.h stream_fill), the id a double (40 bits, exact).
void stream_fill_of(double id, uint8_t *out, uint32_t len) { stream_fill(id >= 0 && id < 1099511627776.0 ? (uint64_t)id : 0, out, len); }
// 1 where payload is the test stream's for the id (xfer.h stream_matches: a file's block stops at its first 64 bytes)
int stream_matches_of(double id, const uint8_t *payload, uint32_t len) { return stream_matches(id >= 0 && id < 1099511627776.0 ? (uint64_t)id : 0, payload, len); }
int xfer_kind_of(double id) { return id >= 0 && id < 1099511627776.0 ? xfer_id_kind((xfer_id_t)id) : XFER_KIND_RESERVED; }

// ai: a description's fields from the JS's: the length (a double, exact to 2^53), sb_log2, the root
static int desc_of(xfer_header_t *h, double length, int sb_log2, const uint8_t *root) {
  if (!(length >= 0 && length <= 9007199254740992.0) || (double)(uint64_t)length != length) return -1;
  if (xfer_header_init(h, (uint64_t)length, sb_log2)) return -1;
  memcpy(h->root, root, XFER_CV);
  return 0;
}
// The session header's 460 bytes: its description (desc, len bytes), xfer_session_write.
int xfer_session_write_of(uint8_t *payload, const uint8_t *desc, uint32_t len) {
  return xfer_session_write(payload, desc, len);
}
// The name a description carries for a name of len bytes (xfer_name_cut: cut in its stem, its extension kept): its
// bytes into out (room for 255), their count.
int xfer_name_cut_of(const uint8_t *name, uint32_t len, uint8_t *out) { return (int)xfer_name_cut(name, len, out); }
// A session header read (xfer_session_parse). scalars, 2 int32: the description's offset and bytes. 0, or XFER_ERR (-1).
int xfer_session_read(const uint8_t *payload, int32_t *scalars) {
  xfer_session_t r;
  const int rc = xfer_session_parse(payload, &r);
  if (rc) return rc;
  scalars[0] = (int32_t)r.at; scalars[1] = (int32_t)r.len;
  return 0;
}
// A file's TOI (xfer_toi_of), its length a double (exact to 2^53): the TOI as a double (exact), 0 for a length out of range.
double xfer_toi_of_wasm(const uint8_t *root, double length, int sb_log2) {
  if (!(length >= 0 && length < 1099511627776.0)) return 0;
  return (double)xfer_toi_of(root, (uint64_t)length, sb_log2);
}
// A file's description (xfer_desc_write) from its fields, its name (0 for none) and its compression (XFER_COMP_*): its
// byte count, or -1 for fields the layout refuses.
int xfer_desc_write_of(uint8_t *out, double length, int sb_log2, const uint8_t *root, const uint8_t *name, int name_len, int compression) {
  xfer_header_t h;
  if (name_len < 0 || name_len > XFER_NAME_MAX || compression < 0 || compression > 255) return -1;
  if (desc_of(&h, length, sb_log2, root)) return -1;
  h.name_len = (uint8_t)name_len; if (name_len) memcpy(h.name, name, (size_t)name_len);
  h.compression = (uint8_t)compression;
  return xfer_desc_write(&h, out);
}
// A description (len bytes, whole; xfer_desc_read under the session header's TOI self_toi, 0 none) read back. scalars,
// 11 int32: hash, sb_log2, name_len,
// length's low 32 bits, its high 32 bits, source blocks, the source block size in bytes, the
// compression, critical (1: an attribute this build does not know and must not present the file under), the TOI's low 32
// bits and its high 16. root
// (32) and name (255) take the bytes, or may be 0. 0, or -1, and nothing is written then.
int xfer_desc_parse(const uint8_t *desc, uint32_t len, double self_toi, int32_t *scalars, uint8_t *root, uint8_t *name) {
  xfer_header_t h;
  xfer_toi_t toi = 0;
  if (xfer_desc_read(desc, len, self_toi >= 0 && self_toi < 281474976710656.0 ? (xfer_toi_t)self_toi : 0, &h, &toi)) return -1;
  const int32_t v[] = { h.hash, h.sb_log2, h.name_len, (int32_t)(uint32_t)h.length, (int32_t)(uint32_t)(h.length >> 32),
                        (int32_t)h.source_blocks, 1 << h.sb_log2, h.compression, h.critical, (int32_t)(uint32_t)toi, (int32_t)(toi >> 32) };
  for (unsigned i = 0; i < sizeof v / sizeof *v; i++) scalars[i] = v[i];
  if (root) memcpy(root, h.root, XFER_CV);
  if (name) memcpy(name, h.name, XFER_NAME_MAX);
  return 0;
}
// 1 where the descriptions a and b describe one transfer's bytes (xfer_same_file), 0 where not, -1
// where either does not parse.
int xfer_same_file_of(const uint8_t *a, uint32_t alen, const uint8_t *b, uint32_t blen) {
  xfer_header_t x, y;
  return xfer_desc_read(a, alen, 0, &x, NULL) || xfer_desc_read(b, blen, 0, &y, NULL) ? -1 : xfer_same_file(&x, &y);
}
// The source block size's log2 a sender picks for a file of `length` bytes (xfer_sb_log2_for), -1 past what the layout holds.
int xfer_sb_log2_of(double length) {
  return length >= 0 && length <= 9007199254740992.0 ? xfer_sb_log2_for((uint64_t)length) : -1;
}
// Source block `index` as it decodes, for the object the description (desc_len bytes) describes (xfer_sb_check): 0 where
// its length is right and, with one source block, its hash the root; with two or more its chaining value into cv (32
// bytes, may be 0 with one). -1 otherwise.
int xfer_sb_ok(const uint8_t *desc, uint32_t desc_len, uint32_t index, const uint8_t *data, uint32_t len, uint8_t *cv) {
  xfer_header_t h;
  return xfer_desc_read(desc, desc_len, 0, &h, NULL) ? -1 : xfer_sb_check(&h, index, data, len, cv);
}
// xfer_b3_subtree with the 1 KiB chunk counter as a double (xfer_sb_cv is the same thing by chunk index).
void xfer_b3_sub(const uint8_t *in, uint32_t len, double counter, int root, uint8_t *out) {
  xfer_b3_subtree(in, len, counter > 0 && counter < 18446744073709551616.0 ? (uint64_t)counter : 0, root, out);
}

// ai: The format's arithmetic as one JSON object, for the web's and the lab's JS (build.sh writes it into build/format.mjs
// ai: after the module, gen/format.mjs; sim/ob.mjs reads it, so a module needs no codec instance for it): the constants,
// ai: each ring's cells, span and modules (lizard.h liz_ring_modules), each version 1 to 67's picture, tiers (lizard_tiers_for), blocks, most blocks, radius and room and camera pixels in
// ai: each ring (lizard.h), and pixels a module for each picture size in each ring. Returns its length, or what it would
// ai: need where cap is too small.
static int jput(char *out, int cap, int at, const char *s) {
  const int n = (int)strlen(s);
  if (at + n < cap) memcpy(out + at, s, (size_t)n + 1);
  return at + n;
}
int lizard_format_json(char *out, int cap) {
  char b[256];
  int at = 0, i, r, v;
  snprintf(b, sizeof b, "{\"STEP\":%d,\"VERSION_MIN\":%d,\"VERSION_MAX\":%d,\"QUIET\":%d,\"OB_MARGIN\":%d,\"BITMAP\":%d,"
           "\"SUB_SLOTS\":%d,\"PAYLOAD\":%d,\"BLOCKS_DEFAULT\":%d,\"GAP_MODULES\":%d,\"TAR_END\":%d,\"RING_DEFAULT\":%d,",
           LIZARD_STEP, OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX, LIZARD_QUIET, OB_MARGIN, LIZARD_BITMAP, 2 * LIZARD_SUB - 4,
           XFER_PAYLOAD, LIZ_BLOCKS_DEFAULT, LIZ_GAP_MODULES, LIZ_TAR_END, LIZARD_RING_DEFAULT);
  at = jput(out, cap, at, b);
  /* ai: a block's id (src/xfer.h): its bytes, the symbol's bits under the source block's, the control source block, the
   * ai: TOI's first byte and its bytes; a sender's source block size where it names none (2^24) */
  snprintf(b, sizeof b, "\"ID_BYTES\":%d,\"ESI_BITS\":%d,\"CONTROL\":%d,\"ID_TOI\":%d,\"TOI_BYTES\":%d,\"LOG2_DEFAULT\":%d,\"DELAY_MAX\":%d,\"RINGS\":[",
           XFER_ID_BYTES, XFER_ESI_BITS, XFER_CONTROL, XFER_ID_TOI, XFER_TOI_BYTES, XFER_LOG2_DEFAULT, OB_FMT_DELAY_MAX);
  at = jput(out, cap, at, b);
  for (r = 0; r < LIZARD_RINGS; r++) { snprintf(b, sizeof b, "%s%d", r ? "," : "", LIZARD_RING[r]); at = jput(out, cap, at, b); }
  at = jput(out, cap, at, "],\"SPANS\":[");
  for (r = 0; r < LIZARD_RINGS; r++) { snprintf(b, sizeof b, "%s%d", r ? "," : "", 2 * LIZARD_RING[r]); at = jput(out, cap, at, b); }
  at = jput(out, cap, at, "],\"MODULES\":[");
  for (r = 0; r < LIZARD_RINGS; r++) { snprintf(b, sizeof b, "%s%d", r ? "," : "", liz_ring_modules(r)); at = jput(out, cap, at, b); }
  at = jput(out, cap, at, "],\"BLOCK_SLOTS\":{");
  for (i = 0; i < 4; i++) {
    static const int RATES[4] = {6, 4, 3, 2};
    snprintf(b, sizeof b, "%s\"%d\":%d", i ? "," : "", RATES[i], lizard_block_slots(RATES[i]));
    at = jput(out, cap, at, b);
  }
  at = jput(out, cap, at, "},\"PICTURE_SIZES\":[");
  for (i = 0; i < LIZARD_NPICTURES; i++) { snprintf(b, sizeof b, "%s%d", i ? "," : "", LIZARD_PICTURES[i]); at = jput(out, cap, at, b); }
  at = jput(out, cap, at, "],\"pxm\":[");
  for (i = 0; i < LIZARD_NPICTURES; i++) {
    for (r = 0; r < LIZARD_RINGS; r++) {
      snprintf(b, sizeof b, "%s%d", r ? "," : i ? ",[" : "[", lizard_pxm_for(LIZARD_PICTURES[i], 2 * LIZARD_RING[r]));
      at = jput(out, cap, at, b);
    }
    at = jput(out, cap, at, "]");
  }
  at = jput(out, cap, at, "],\"versions\":[");
  for (v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX; v++) {
    const int subch = LIZARD_STEP * v;
    lizard_tier_t tier[LIZARD_TIERS];
    const int tiers = lizard_tiers_for(subch, tier);
    snprintf(b, sizeof b, "%s{\"n\":%d,\"blocks\":%d,\"max\":%d,\"radius\":%.17g,\"tiers\":[", v > OB_FMT_VERSION_MIN ? "," : "",
             lizard_n_for(subch), lizard_blocks_for(subch), lizard_blocks_max_for(subch), liz_radius_for(v));
    at = jput(out, cap, at, b);
    for (i = 0; i < tiers; i++) {
      snprintf(b, sizeof b, "%s[%d,%d,%d]", i ? "," : "", tier[i].rate, tier[i].blocks, tier[i].slots);
      at = jput(out, cap, at, b);
    }
    at = jput(out, cap, at, "],\"room\":[");
    for (r = 0; r < LIZARD_RINGS; r++) { snprintf(b, sizeof b, "%s%.17g", r ? "," : "", liz_room_for(v, r)); at = jput(out, cap, at, b); }
    at = jput(out, cap, at, "],\"camera\":[");
    for (r = 0; r < LIZARD_RINGS; r++) { snprintf(b, sizeof b, "%s%.17g", r ? "," : "", liz_camera_for(v, r)); at = jput(out, cap, at, b); }
    at = jput(out, cap, at, "]}");
  }
  return jput(out, cap, at, "]}");
}
