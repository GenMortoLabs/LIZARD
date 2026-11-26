// The sender's IROWS, the first inverse: a workgroup a (disc row, symbol), N / 8 lanes. The disc is the spectrum
// transposed (core/tx/send_tables.cpp sendTables), so its rows are the picture's columns and this pass runs along the
// picture's y; IPIC then runs along its rows. The row's coefficients at (u + N) mod N, zeros elsewhere, conjugated in,
// the inverse (the decoder's forward FFT on the conjugate, unscaled, f32: lib/fft_body.slang, halves exchanged),
// conjugated out, the row stored in Y quad-major (Y[x / 4][v][x mod 4], x along this pass: IPIC's
// workgroup reads its quad's rows as one run), each complex value two binary16 halves in a word (lib/pack.slang: Y the
// largest round trip of a frame, half its bytes; about 0.015 of a grey level a pixel). Built a picture size each.
// A disc entry's coefficient is made here (src/lizard_phy.c lizard_encode, the C's arithmetic): its frame coefficient
// fc from INV, sub-channel fc / 320's count pilot (coefficient 0: the whitening XOR bit (j & 1) of the frame's count on
// both axes) or reference (PB_REF: the whitening alone), else data coefficient dc (past the two pilots): past the slots
// the blocks take the padding (the whitening alone), else its two bits from SD (../send_paint.slang: its block's
// codeword through the bit map, whitened), the QPSK (+-a a bit, a = 0.70710678), conjugated where INV's bit 31 says. An
// entry past the frame's coefficients (a picture size's disc larger than its format's) is zero.
// Bindings: SD (ro, the data coefficients' bits, 16 a word, a symbol's PP.dims.y words), the twiddles, Y (rw, two
// halves a word), the rows table (a row's (start, length, first run, gap)), the size's uniform (a = (0, the disc's rows,
// Y a symbol, 0), b, c), TAB (ro, the paint's: word 0 the whitening's offset, the whitening a bit a frame slot), INV
// (ro, a disc entry's frame coefficient, bit 31 conjugated), PP (the paint's uniform: the slots the blocks take, the
// coefficients the frame holds, dims.y SD a symbol, dims.w the count).
// Defines (core/tx/send_plan.h): LANES (N / 8, int), PICTURE_N (uint), SLOT_COUNT, PW_VECS (int), DCOEF, PB_REF
// (uint).
[[vk::binding(0, 0)]] StructuredBuffer<uint> SD;
[[vk::binding(1, 0)]] StructuredBuffer<float2> tw;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> yb;
struct RowsBlock { uint4 r[512]; };
[[vk::binding(3, 0)]] ConstantBuffer<RowsBlock> rows;
struct SizeParamsBlock { uint4 a; uint4 b; float4 c; };
[[vk::binding(4, 0)]] ConstantBuffer<SizeParamsBlock> P;
[[vk::binding(5, 0)]] StructuredBuffer<uint> TAB;
[[vk::binding(6, 0)]] StructuredBuffer<uint> INV;
struct PaintBlock { uint4 sizes[SLOT_COUNT]; uint4 dims; uint4 pw[PW_VECS]; };
[[vk::binding(7, 0)]] ConstantBuffer<PaintBlock> PP;
static const float A = 0.70710678;
static const uint N = PICTURE_N;
static const uint T = PICTURE_N / 8u;
// the exchange buffer, a complex value two halves a word (lib/pack.slang)
groupshared uint buf[N];
#include "lib/butterflies.slang"
#define LD(X) unpackHalf2x16(X)
#define ST(X) packHalf2x16(X)
#define FFT_LANES_2 0
#define FFT_BUF0 buf
#include "lib/fft.slang"
#include "lib/pack.slang"
[numthreads(LANES, 1, 1)]
void main(uint3 wg : SV_GroupID, uint tid : SV_GroupIndex) {
  uint r = wg.z;
  uint vv = wg.x;
  uint4 row = rows.r[vv];
  uint ustart = vv == 0u ? 1u : 0u;
  uint eBase = row.x;
  // the symbol's frame's count (the pilots, SPEC 7.3): c0 (the encode's first frame's) + its place in the encode, mod 4
  uint count = ((PP.dims.w & 3u) + r / max((PP.dims.w >> 2u) & 0xffffu, 1u)) & 3u;
  // The row's eight values a lane, register K the row's entry at u = x (x < N / 2) or u = x - N, x = tid + K T,
  // conjugated in; zero off the row and past the frame's coefficients. In phases across the eight, branch-free, so each
  // phase's loads go out together: the entries' frame coefficients (INV), then their bits (SD) and the whitening's word
  // (the pilots' and the padding's).
  uint woff = TAB[0];
  uint used = PP.sizes[0].z;
  uint nfc = PP.sizes[0].w;
  uint sdBase = r * PP.dims.y;
  uint iv[8];
  bool on[8];
  [ForceUnroll] for (uint K = 0u; K < 8u; K = K + 1u) {
    uint x = tid + K * T;
    uint un = N - x;
    bool lo = x < N / 2u;
    on[K] = lo ? (x >= ustart && x < ustart + row.z) : (un <= row.w);
    uint e = lo ? x - ustart : row.z + row.w - un;
    iv[K] = INV[eBase + (on[K] ? e : 0u)];
  }
  uint sd[8];
  uint wh[8];
  uint kind[8];
  uint sh[8];
  [ForceUnroll] for (uint K = 0u; K < 8u; K = K + 1u) {
    uint fc = iv[K] & 0x7fffffffu;
    uint j = fc / 320u;
    uint rr = fc - 320u * j;
    uint dc = DCOEF * j + (rr < PB_REF ? rr - 1u : rr - 2u);
    // kind: 0 off, 1 a count pilot, 2 the whitening alone (the reference, the padding), 3 data
    bool live = on[K] && fc < nfc;
    bool data = live && rr != 0u && rr != PB_REF && 2u * dc < used;
    kind[K] = !live ? 0u : (rr == 0u ? 1u : (data ? 3u : 2u));
    sd[K] = SD[sdBase + (data ? dc >> 4u : 0u)];
    sh[K] = data ? 2u * (dc & 15u) : (2u * fc) & 31u;
    wh[K] = data ? 0u : TAB[woff + (live ? fc >> 4u : 0u)];
  }
  float2 v[8];
  [ForceUnroll] for (uint K = 0u; K < 8u; K = K + 1u) {
    // a count pilot's bit: bit (j & 1) of the frame's count on both axes
    uint fc = iv[K] & 0x7fffffffu;
    uint cp = kind[K] == 1u ? ((count >> ((fc / 320u) & 1u)) & 1u) * 3u : 0u;
    uint q = kind[K] == 3u ? (sd[K] >> sh[K]) & 3u : ((wh[K] >> sh[K]) & 3u) ^ cp;
    float2 val = float2((q & 1u) != 0u ? -A : A, (q & 2u) != 0u ? -A : A);
    // conjugated where INV says, then conjugated in for the inverse (the forward FFT on the conjugate)
    val = (iv[K] >> 31u) != 0u ? float2(val.x, -val.y) : val;
    v[K] = kind[K] == 0u ? float2(0.0) : float2(val.x, -val.y);
  }
#define FFT_V0 v
#include "lib/fft_body.slang"
  // quad-major: a quad of columns spans the disc's rows (P.a.y) four values each
  uint yBase = r * P.a.z + vv * 4u;
  uint QS = P.a.y * 4u;
  for (uint x = tid; x < N; x = x + T) {
    float2 z = unpackHalf2x16(buf[x]);
    yb[yBase + (x >> 2u) * QS + (x & 3u)] = packHalf2x16(float2(z.x, -z.y));
  }
}
