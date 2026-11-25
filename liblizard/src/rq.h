// RaptorQ (RFC 6330), one source block, for the transfer's source blocks (src/xfer.h): a source block of len bytes is
// K = ceil(len / T) source symbols of T bytes (the last zero padded), sent as encoding symbols by ESI: 0 .. K - 1 the
// source symbols themselves, K and up repair symbols (internal symbol id ESI + K' - K, RFC 6330 5.3.1). One source
// block and one sub-block (Z = N = 1): each of the transfer's source blocks is RaptorQ's one, and nothing of the RFC's
// object partitioning is used.
//
// The encoder solves for the L intermediate symbols once (rq_enc_new) and makes any symbol after (rq_enc_symbol). The
// decoder keeps every distinct symbol it is given and, from K of them on, solves after each new one until the block is
// whole: it never refuses a symbol, and a symbol past what it needs is ignored. Both solve the RFC's constraint matrix
// by inactivation: the sparse binary rows peeled, a column inactivated where no row has one unknown left, the S LDPC,
// H HDPC and received rows then reduced to a dense system over the inactive columns (GF(256)), back-substituted in peel
// order. Any solver gives the same symbols: the RFC fixes the matrix, not how it is solved.
#ifndef RQ_H
#define RQ_H
#include <stddef.h>
#include <stdint.h>

enum { RQ_KMAX = 56403 };  // K'_max, RFC 6330 5.1.2

typedef struct rq_enc rq_enc_t;
typedef struct rq_dec rq_dec_t;

// The source symbols of len bytes at T, ceil(len / T), or 0 where len is 0 or that passes RQ_KMAX.
uint32_t rq_symbols(size_t len, uint32_t T);

// An encoder over data[0 .. len), symbols of T bytes (T > 0). NULL on a length rq_symbols refuses, or out of memory.
rq_enc_t *rq_enc_new(const uint8_t *data, size_t len, uint32_t T);
// Encoding symbol `esi` (below 2^24) into out, T bytes.
void rq_enc_symbol(const rq_enc_t *e, uint32_t esi, uint8_t *out);
void rq_enc_free(rq_enc_t *e);

// A decoder for len bytes at T, ESIs below esi_limit (at most 2^24). NULL as rq_enc_new, or esi_limit out of range.
rq_dec_t *rq_dec_new(size_t len, uint32_t T, uint32_t esi_limit);
// Symbol `esi`, T bytes: 1 the block is whole (now or before), 0 more wanted, -1 an ESI past the limit or out of memory.
// From K symbols on it solves after each new one (rq_dec_hold, then rq_dec_solve).
int rq_dec_add(rq_dec_t *d, uint32_t esi, const uint8_t *sym);
// Symbol `esi` kept and no solve tried: 0, or -1 as rq_dec_add (a caller holding many symbols at once, then one solve).
int rq_dec_hold(rq_dec_t *d, uint32_t esi, const uint8_t *sym);
// A solve over the symbols held: 1 whole (now or before), 0 more wanted (fewer than K, or a rank short), -1 out of memory.
int rq_dec_solve(rq_dec_t *d);
// The block's len bytes into out, once rq_dec_add has said 1: 0, or -1 before.
int rq_dec_recover(const rq_dec_t *d, uint8_t *out);
void rq_dec_free(rq_dec_t *d);

#endif
