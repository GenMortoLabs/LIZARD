// The sender's PAINT, a workgroup a (block, symbol), 128 lanes (src/lizard_phy.c lizard_encode, the C's arithmetic
// exactly): the block's 472 payload bytes from BLOCKS, their CRC-64/LIZARD (lib/crc64.slang, the LDPC's split CRC), and
// its code's systematic codeword (the data bits, zeros past the 480 B where the code holds more (the 6/7 code's last
// 128), then the m parity bits: the IRA accumulator, a prefix XOR over the checks in check order, a lane ceil(m / 128) of
// them), its data slots (slot i the codeword bit its code's bit map names, XOR the whitening at its frame coefficient's
// slot), and into SD each data coefficient's two bits (the QPSK's signs: 1 for -a), data coefficient dc's at bits 2 (dc
// mod 16) of word dc / 16 from the symbol's r dims.y: a block's data coefficients start on a word and fill whole words
// (every code's slots a block a multiple of 32), so each word has one writer. IROWS makes the disc's coefficients from
// them (../send_irows.slang: the pilots and the padding there too), so no coefficient is scattered onto the disc (on the
// S26 the disc's scattered stores were 0.5 of the paint's 0.67 ms a frame at LIZARD-1024 x 2, 2026-11-24). A block's
// code and first data slot come from its tier (src/lizard_phy.h lizard_tiers_for: 6/7 blocks of 4480 slots, 3/4 of
// 5120, 2/3 of 5760, 1/2 of 7680, packed from the frame's first data slot, the lowest frequencies, out), the codes'
// shapes and tables from TAB, so the shader holds no code of its own.
// Bindings: BLOCKS (ro, 120 words a block, block b of symbol r at (r blocks + b) 120), TAB (ro, ../../encoder.mjs
// sendTab: word 0 the whitening's offset, then 16 words a code at 16 + 16 c, (n, k, m, z, mb, rows, lay, perm, slots,
// nt): its block's slots, its block rows' first entries, its entries (col | shift << 16), its bit map two slots a word;
// the whitening a bit a frame slot), SD (rw, the data coefficients' bits), P (uniform, core/tx/send_plan.h P_SIZES .. P_WORDS):
// sizes[0] = (blocks, 0, the data slots the blocks take, the coefficients the frame holds, 320 a sub-channel), sizes[1
// + c] = (first block, first data slot, blocks, 0) of code c in CODES' order (0 blocks: a code the format has none of),
// dims = (blocks, SD's words a symbol, symbols, c0 + 4 codes), pw the CRC's powers. Dispatch (blocks, 1, symbols):
// frames x codes of an encode.
// The test stream (the tools', src/xfer.h stream_fill, no part of the format): a test block's 460 payload bytes are
// BLAKE3's extended output of its id's low five bytes (its source block and symbol), made here from the id alone
// (dims.w's bit 31 set: the host writes the ids, packed in BLOCKS, 12 bytes a block, the bare id and the 48-bit TOI).
// Output block t (64 bytes) is one compression of the root's only block (the id, its length 5, CHUNK_START | CHUNK_END |
// ROOT) at counter t, so eight lanes make the eight blocks side by side; the compression unrolled (b3out), the message
// words that are always zero (all but the id's two) adding nothing.
// Every workgroup variable is written before the barrier that precedes its reads.
// Defines (core/tx/send_plan.h): SLOT_COUNT, PW_VECS (int), PAINT_KNOWN, CW_WORDS, NCODES, DCOEF, PB_REF (uint),
// CRC64_CHUNKS (lib/crc64.slang).
[[vk::binding(0, 0)]] StructuredBuffer<uint> BLOCKS;
[[vk::binding(1, 0)]] StructuredBuffer<uint> TAB;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> SD;
struct ParamsBlock { uint4 sizes[SLOT_COUNT]; uint4 dims; uint4 pw[PW_VECS]; };
[[vk::binding(3, 0)]] ConstantBuffer<ParamsBlock> P;
static const uint KNOWN = PAINT_KNOWN;
static const uint TH = 128u;
groupshared uint packw[120];
groupshared Atomic<uint> crcw[2];
groupshared Atomic<uint> cw[CW_WORDS];
groupshared uint scan[128];
// a test block's stream, eight BLAKE3 output blocks
groupshared uint stw[128];
uint rotr(uint x, uint n) { return (x >> n) | (x << (32u - n)); }
// output block t of BLAKE3(the id's five bytes) into stw[16 t .. 16 t + 15]
// BLAKE3's compression (seven rounds of the column and diagonal G steps, the message permuted between rounds), the
// message all zeros but its first two words: m0 and m1 sit at words p0 and p1 of a round's message, which the
// permutation moves (B3_INV, its inverse, maps a word's place to its next); a step adding a zero word adds none.
static const int B3_G[8][4] = { { 0, 4, 8, 12 }, { 1, 5, 9, 13 }, { 2, 6, 10, 14 }, { 3, 7, 11, 15 },
                                { 0, 5, 10, 15 }, { 1, 6, 11, 12 }, { 2, 7, 8, 13 }, { 3, 4, 9, 14 } };
static const int B3_INV[16] = { 5, 8, 0, 2, 6, 11, 1, 4, 15, 12, 3, 9, 10, 7, 13, 14 };
void b3out(uint m0, uint m1, uint t) {
  uint v[16] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
                 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, t, 0u, 5u, 11u };
  int p0 = 0, p1 = 1;
  [ForceUnroll] for (int r = 0; r < 7; r++) {
    [ForceUnroll] for (int g = 0; g < 8; g++) {
      const int a = B3_G[g][0], b = B3_G[g][1], c = B3_G[g][2], d = B3_G[g][3];
      const int x = 2 * g, y = 2 * g + 1;
      v[a] = x == p0 ? ((v[a] + v[b]) + m0) : x == p1 ? ((v[a] + v[b]) + m1) : (v[a] + v[b]);
      v[d] = rotr(v[d] ^ v[a], 16u);
      v[c] = (v[c] + v[d]);
      v[b] = rotr(v[b] ^ v[c], 12u);
      v[a] = y == p0 ? ((v[a] + v[b]) + m0) : y == p1 ? ((v[a] + v[b]) + m1) : (v[a] + v[b]);
      v[d] = rotr(v[d] ^ v[a], 8u);
      v[c] = (v[c] + v[d]);
      v[b] = rotr(v[b] ^ v[c], 7u);
    }
    p0 = B3_INV[p0];
    p1 = B3_INV[p1];
  }
  uint o = 16u * t;
  stw[o + 0u] = v[0] ^ v[8]; stw[o + 1u] = v[1] ^ v[9]; stw[o + 2u] = v[2] ^ v[10]; stw[o + 3u] = v[3] ^ v[11];
  stw[o + 4u] = v[4] ^ v[12]; stw[o + 5u] = v[5] ^ v[13]; stw[o + 6u] = v[6] ^ v[14]; stw[o + 7u] = v[7] ^ v[15];
  stw[o + 8u] = v[8] ^ 0x6a09e667u; stw[o + 9u] = v[9] ^ 0xbb67ae85u; stw[o + 10u] = v[10] ^ 0x3c6ef372u; stw[o + 11u] = v[11] ^ 0xa54ff53au;
  stw[o + 12u] = v[12] ^ 0x510e527fu; stw[o + 13u] = v[13] ^ 0x9b05688cu; stw[o + 14u] = v[14] ^ 0x1f83d9abu; stw[o + 15u] = v[15] ^ 0x5be0cd19u;
}

// Data bit j of the codeword: the payload bytes MSB first, then the CRC MSB first, as packw holds them (four bytes a
// word in memory order); zeros past them.
uint dataBit(uint j) {
  if (j >= KNOWN) { return 0u; }
  uint t = j & 31u;
  return (packw[j >> 5u] >> (8u * (t >> 3u) + 7u - (t & 7u))) & 1u;
}
uint cwBit(uint j) { return (cw[j >> 5u].load() >> (j & 31u)) & 1u; }
uint white(uint s) { return (TAB[TAB[0] + (s >> 5u)] >> (s & 31u)) & 1u; }
// data coefficient dc's coefficient in the frame: past each sub-channel's count pilot (0) and reference (PB_REF)
// (src/lizard_phy.c pb_fcoef)
uint fcoef(uint dc) { uint j = dc / DCOEF; uint r = dc - DCOEF * j; return 320u * j + (r < PB_REF - 1u ? 1u + r : 2u + r); }
#define CRC_PW P.pw
#include "lib/crc64.slang"
[numthreads(128, 1, 1)]
void main(uint3 wg : SV_GroupID, uint i : SV_GroupIndex) {
  uint b = wg.x;
  uint r = wg.z;
  if (b >= P.sizes[0].x) { return; }
  // the block's tier: its code, and its first slot
  uint c = 0u;
  for (uint q = 0u; q < NCODES; q = q + 1u) {
    uint4 t = P.sizes[1u + q];
    if (b >= t.x && b < t.x + t.z) { c = q; }
  }
  uint4 tt = P.sizes[1u + c];
  uint h = 16u + 16u * c;
  uint n = TAB[h];
  uint k = TAB[h + 1u];
  uint m = TAB[h + 2u];
  uint z = TAB[h + 3u];
  uint mb = TAB[h + 4u];
  uint rowsAt = TAB[h + 5u];
  uint layAt = TAB[h + 6u];
  uint permAt = TAB[h + 7u];
  // the block's slots: its sent bits (n less the code's np never sent, the 6/7 code's 128 known zeros)
  uint slots = TAB[h + 8u];
  uint slot0 = tt.y + (b - tt.x) * slots;
  uint cpl = (m + TH - 1u) / TH;
  // The payload from BLOCKS: bytes 0 to 471 in words 0 to 117; its CRC-64 big-endian in words 118 and 119 (bytes 472
  // to 479).
  uint base = (r * P.sizes[0].x + b) * 120u;
  if ((P.dims.w >> 31u) != 0u) {
    // the test stream: the payload made here from the id BLOCKS holds packed (three words a block, the block's 12-byte
    // id), the block's words the id's three, then the stream's from word 3 (word w: stream bytes 4 w - 12 to 4 w - 9)
    uint ib = (r * P.sizes[0].x + b) * 3u;
    uint m0 = BLOCKS[ib];
    uint m1 = BLOCKS[ib + 1u] & 0xffu;
    if (i < 8u) { b3out(m0, m1, i); }
    GroupMemoryBarrierWithGroupSync();
    if (i == 0u) { packw[0] = m0; }
    if (i == 1u) { packw[1] = BLOCKS[ib + 1u]; }
    if (i == 2u) { packw[2] = BLOCKS[ib + 2u]; }
    if (i >= 3u && i < CRC_CHUNKS) { packw[i] = stw[i - 3u]; }
  } else {
    if (i < CRC_CHUNKS) { packw[i] = BLOCKS[base + i]; }
  }
  if (i == CRC_CHUNKS || i == CRC_CHUNKS + 1u) { packw[i] = 0u; }
  GroupMemoryBarrierWithGroupSync();
  uint2 crc = crcOf(i);
  if (i == 0u) {
    packw[CRC_CHUNKS] = ((crc.x >> 24u) & 0xffu) | (((crc.x >> 16u) & 0xffu) << 8u) | (((crc.x >> 8u) & 0xffu) << 16u) | ((crc.x & 0xffu) << 24u);
    packw[CRC_CHUNKS + 1u] = ((crc.y >> 24u) & 0xffu) | (((crc.y >> 16u) & 0xffu) << 8u) | (((crc.y >> 8u) & 0xffu) << 16u) | ((crc.y & 0xffu) << 24u);
  }
  GroupMemoryBarrierWithGroupSync();
  // The systematic part of the codeword, a bit an index, every word of it (the parity's zeros too); then every lane's
  // cpl checks in check order, the check's parity as the XOR of its data bits, the prefix within them kept a bit each.
  for (uint w = i; w < (n + 31u) / 32u; w = w + TH) {
    uint word = 0u;
    for (uint t = 0u; t < 32u; t = t + 1u) {
      uint j = 32u * w + t;
      if (j < k) { word = word | (dataBit(j) << t); }
    }
    cw[w].store(word);
  }
  uint pl = 0u;
  uint acc = 0u;
  for (uint q = 0u; q < cpl; q = q + 1u) {
    uint chk = cpl * i + q;
    if (chk < m) {
      uint ii = chk / mb;
      uint rr = chk % mb;
      uint t = 0u;
      uint e1 = TAB[rowsAt + rr + 1u];
      // the data bit of entry e in check (rr, ii): col z + (ii - shift) mod z (src/ldpc.c ldpc_init)
      for (uint e = TAB[rowsAt + rr]; e < e1; e = e + 1u) {
        uint lc = TAB[layAt + e];
        uint s = ii + z - (lc >> 16u);
        if (s >= z) { s = s - z; }
        t = t ^ dataBit((lc & 0xffffu) * z + s);
      }
      acc = acc ^ t;
    }
    pl = pl | (acc << q);
  }
  scan[i] = acc;
  GroupMemoryBarrierWithGroupSync();
  // The prefix XOR across lanes (ldpc_encode's running acc), a read, a barrier and a write a step.
  for (uint d = 1u; d < TH; d = d << 1u) {
    uint v = scan[i];
    if (i >= d) { v = v ^ scan[i - d]; }
    GroupMemoryBarrierWithGroupSync();
    scan[i] = v;
    GroupMemoryBarrierWithGroupSync();
  }
  uint ex = scan[i] ^ acc;
  for (uint q = 0u; q < cpl; q = q + 1u) {
    uint chk = cpl * i + q;
    if (chk < m && (((pl >> q) & 1u) ^ ex) != 0u) {
      uint j = k + chk;
      cw[j >> 5u].or(1u << (j & 31u));
    }
  }
  GroupMemoryBarrierWithGroupSync();
  // The block's data coefficients' bits, a word of 16 a lane: data slots 2 cc and 2 cc + 1 of data coefficient cc (one
  // bit map word the pair, each entry a codeword bit, the never-sent ones already skipped: core/dec/codes.h
  // sentBit), each XOR the whitening at its frame coefficient's slots.
  uint sdBase = r * P.dims.y + slot0 / 32u;
  for (uint w = i; w < slots / 32u; w = w + TH) {
    uint word = 0u;
    for (uint t = 0u; t < 16u; t = t + 1u) {
      uint cc = 16u * w + t;
      uint pe = TAB[permAt + cc];
      uint fc = fcoef(slot0 / 2u + cc);
      word = word | ((cwBit(pe & 0xffffu) ^ white(2u * fc)) << (2u * t)) | ((cwBit(pe >> 16u) ^ white(2u * fc + 1u)) << (2u * t + 1u));
    }
    SD[sdBase + w] = word;
  }
}
