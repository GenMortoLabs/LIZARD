// RaptorQ's solve as a plan (src/rq.c): the schedule, made from the rows alone (which symbols are held, never their
// bytes), and the symbol work it leaves. rq.c's decoder and encoder solve through it; a receiver that keeps its symbols
// in records of its own (core/rx/xfer_rx_core.cpp) solves over them where they lie, with no copy into a decoder.
// Internal to the library: its API is rq.h.
//
// The symbol work, in order (rq.c plan_exec is the reference):
//   1. every inactive column zero; then each peeled column, level by level, its row's right side XOR its peeled
//      columns (pass A: the value with every inactive column taken as zero);
//   2. the dense system's right sides: each unused binary row's right side XOR its peeled columns; the H HDPC rows
//      through MT GAMMA's recurrence (acc = alpha acc ^ C[j] for j < K' + S, acc added to MT's two rows of column j,
//      alpha^i acc to row i at the last column);
//   3. Gauss-Jordan over those R right sides (the plan's op list), unknown u then row row_of[u], into the inactive
//      columns;
//   4. each peeled column again, level by level, its row's right side XOR every other column of its row (pass B);
//   5. each source symbol, the XOR of its row's columns (rq_plan_recover).
#ifndef RQ_PLAN_H
#define RQ_PLAN_H
#include "rq.h"

// A block's parameters (RFC 6330 5.3.3.3).
typedef struct { uint32_t K, Kp, J, S, H, W, L, P, P1, B; } rq_par_t;

typedef struct {
  rq_par_t p;
  uint32_t sym0;   // rows from sym0 on are the held symbols in hold order
  // The peeled columns by level, a column's level one past the deepest peeled column its row reads (so the columns of
  // a level read only earlier levels): targets lev_start[l] .. lev_start[l + 1] - 1 of level l. Target t: column
  // tgt[t], its row tgt_row[t], the row's other columns src[src_start[t] .. src_start[t + 1]), its peeled ones first
  // (src_peeled[t] of them).
  uint32_t npeel, nlev;
  uint32_t *lev_start, *tgt, *tgt_row, *src_start, *src_peeled, *src;
  // The dense system: R = Nu + H rows over I unknowns. Unused binary row k: row u_row[k], its peeled columns
  // u_src[u_start[k] .. u_start[k + 1]); then the H HDPC rows. hpair: for each column j < K' + S - 1, the two HDPC rows
  // (0 to H - 1) MT adds to. Unknown u is column inact[u].
  uint32_t Nu, R, I;
  uint32_t *u_row, *u_start, *u_src, *hpair, *inact;
  // Gauss-Jordan over the right sides as the coefficients took it: op k multiplies row op_dst[k] by op_c[k]
  // (op_src[k] == UINT32_MAX) or adds op_c[k] times row op_src[k] to it; unknown u is then row row_of[u].
  uint32_t nops;
  uint32_t *op_dst, *op_src, *row_of;
  uint8_t *op_c;
} rq_plan_t;

// The plan of a solve over n held symbols named by their ESIs (distinct, in the caller's order): 1 (the plan in *out),
// 0 a rank short (more symbols wanted), -1 out of memory or a length rq_symbols refuses.
int rq_plan_for(size_t len, uint32_t T, uint32_t n, const uint32_t *esi, rq_plan_t **out);
// Steps 1 to 4 over the symbols where the caller holds them (sym[k] the T bytes of the plan's k-th ESI): C (L x T).
// 0, or -1 out of memory.
int rq_plan_exec(const rq_plan_t *pl, uint32_t T, uint32_t n, const uint8_t *const *sym, uint8_t *C);
// Step 5: the block's len bytes from C, each source symbol XORed into out where it lies. 0, or -1 out of memory.
int rq_plan_recover(const rq_plan_t *pl, const uint8_t *C, uint32_t T, size_t len, uint8_t *out);
void rq_plan_free(rq_plan_t *pl);

#endif
