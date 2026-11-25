// The sender's PAINT, a workgroup a (block, symbol), 128 lanes (src/lizard_phy.c lizard_encode and map_out, the C's
// arithmetic exactly): the block's 472 payload bytes from BLOCKS, their CRC-64/LIZARD (lib/crc64.slang, the LDPC's split
// CRC), its code's systematic codeword (the data bits, zeros past the 480 B where the code holds more (the 6/7 code's
// last 128), then the m parity bits: the IRA accumulator, a prefix XOR over the checks in check order, a lane
// ceil(m / 128) of them), its data slots (slot i the codeword bit its code's bit map names, XOR the whitening at its
// frame coefficient's slot) and the QPSK coefficients (+-a a bit, a = 0.70710678, every sub-channel at one amplitude)
// into the symbol's disc in S at the UV entries of their frame coefficients (fcoef: each sub-channel's coefficients 0
// and PB_REF are its pilots, apart from the blocks), the disc the spectrum transposed (put). A block's code and first data slot come from its tier
// (src/lizard_phy.h lizard_tiers_for: 6/7 blocks of 4480 slots, 3/4 of 5120, 2/3 of 5760, 1/2 of 7680, packed from the
// frame's first data slot, the lowest frequencies, out), the codes' shapes and tables from TAB, so the shader holds no
// code of its own. The frame's last block's workgroup also paints the padding past the last block, the whitening alone,
// and every sub-channel's two pilots, so every coefficient the frame holds is sent.
// Bindings: BLOCKS (ro, 120 words a block, block b of symbol r at (r blocks + b) 120), TAB (ro, ../../encoder.mjs
// sendTab: word 0 the whitening's offset, then 16 words a code at 16 + 16 c, (n, k, m, z, mb, rows, lay, perm, slots,
// nt): its block's slots, its block rows' first entries, its entries (col | shift << 16), its bit map two slots a word;
// the whitening a bit a frame slot), UV (ro, a frame coefficient's S entry, bit 31 conjugated), S (rw, the disc), P (uniform,
// ../../plan/send.mjs PARAMS_AT): sizes[0] = (blocks, UV offset, the data slots the blocks take, the coefficients the
// frame holds, 320 a sub-channel), sizes[1 + c] = (first block, first data slot, blocks, 0) of code c in CODES' order
// (0 blocks: a code the format has none of), dims = (blocks, sStride, symbols, c0 + 4 codes), pw the CRC's powers.
// Dispatch (blocks, 1, symbols): frames x codes of an encode. The pilots (SPEC 7.3): a frame's count is c0 (the encode's
// first frame's) + its place in the encode, mod 4, the frame symbol r / codes; bit 0 on an even sub-channel's count
// pilot, bit 1 on an odd's.
// The test stream (the tools', src/xfer.h stream_fill, no part of the format): a test block's 460 payload bytes are
// BLAKE3's extended output of its id's low five bytes (its source block and symbol), made here from the id alone
// (dims.w's bit 31 set: the host writes the ids, packed in BLOCKS, 12 bytes a block, the bare id and the 48-bit TOI).
// Output block t (64 bytes) is one compression of the root's only block (the id, its length 5, CHUNK_START | CHUNK_END |
// ROOT) at counter t, so eight lanes make the eight blocks side by side; the compression is written out (gen/b3,
// ../send.mjs b3Steps) with the message words that are always zero (all but the id's two) left out.
// Every workgroup variable is written before the barrier that precedes its reads.
// Defines (../send.mjs paintSource): SLOT_COUNT, PW_VECS (int), PAINT_KNOWN, CW_WORDS, NCODES, DCOEF, PB_REF (uint),
// CRC64_CHUNKS (lib/crc64.slang).
[[vk::binding(0, 0)]] StructuredBuffer<uint> BLOCKS;
[[vk::binding(1, 0)]] StructuredBuffer<uint> TAB;
[[vk::binding(2, 0)]] StructuredBuffer<uint> UV;
[[vk::binding(3, 0)]] RWStructuredBuffer<float2> S;
struct ParamsBlock { uint4 sizes[SLOT_COUNT]; uint4 dims; uint4 pw[PW_VECS]; };
[[vk::binding(4, 0)]] ConstantBuffer<ParamsBlock> P;
static const uint KNOWN = PAINT_KNOWN;
static const uint TH = 128u;
static const float A = 0.70710678;
groupshared uint packw[120];
groupshared Atomic<uint> crcw[2];
groupshared Atomic<uint> cw[CW_WORDS];
groupshared uint scan[128];
// a test block's stream, eight BLAKE3 output blocks
groupshared uint stw[128];
uint rotr(uint x, uint n) { return (x >> n) | (x << (32u - n)); }
// output block t of BLAKE3(the id's five bytes) into stw[16 t .. 16 t + 15]
void b3out(uint m0, uint m1, uint t) {
  uint v0 = 0x6a09e667u; uint v1 = 0xbb67ae85u; uint v2 = 0x3c6ef372u; uint v3 = 0xa54ff53au;
  uint v4 = 0x510e527fu; uint v5 = 0x9b05688cu; uint v6 = 0x1f83d9abu; uint v7 = 0x5be0cd19u;
  uint v8 = 0x6a09e667u; uint v9 = 0xbb67ae85u; uint v10 = 0x3c6ef372u; uint v11 = 0xa54ff53au;
  uint v12 = t; uint v13 = 0u; uint v14 = 5u; uint v15 = 11u;
#include "gen/b3"
  uint o = 16u * t;
  stw[o + 0u] = v0 ^ v8; stw[o + 1u] = v1 ^ v9; stw[o + 2u] = v2 ^ v10; stw[o + 3u] = v3 ^ v11;
  stw[o + 4u] = v4 ^ v12; stw[o + 5u] = v5 ^ v13; stw[o + 6u] = v6 ^ v14; stw[o + 7u] = v7 ^ v15;
  stw[o + 8u] = v8 ^ 0x6a09e667u; stw[o + 9u] = v9 ^ 0xbb67ae85u; stw[o + 10u] = v10 ^ 0x3c6ef372u; stw[o + 11u] = v11 ^ 0xa54ff53au;
  stw[o + 12u] = v12 ^ 0x510e527fu; stw[o + 13u] = v13 ^ 0x9b05688cu; stw[o + 14u] = v14 ^ 0x1f83d9abu; stw[o + 15u] = v15 ^ 0x5be0cd19u;
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
// frame coefficient c's QPSK value: its two bits (x, y) XOR the whitening at its frame slots and q
float2 qpsk(uint c, uint q) { return float2((white(2u * c) ^ q) != 0u ? -A : A, (white(2u * c + 1u) ^ q) != 0u ? -A : A); }
// frame coefficient c's value into the symbol's disc at its UV entry, conjugated where the entry's bit 31 says (the
// spectrum transposed, ../back/disc.mjs sendDiscLayout)
void put(uint sBase, uint c, float2 val) {
  uint at = UV[P.sizes[0].y + c];
  S[sBase + (at & 0x7fffffffu)] = (at >> 31u) != 0u ? float2(val.x, -val.y) : val;
}
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
  // the block's slots: its sent bits (nt)
  uint slots = TAB[h + 8u];
  // the bits sent, n less the code's np never sent (the 6/7 code's 128 known zeros): the slots the codeword fills
  uint nt = TAB[h + 9u];
  uint cpl = (m + TH - 1u) / TH;
  uint slot0 = tt.y + (b - tt.x) * slots;
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
  // Data slots 2 cc and 2 cc + 1 of the block's data coefficient cc (its slots are its code's nt sent bits, an even
  // count), one bit map word the pair (each entry a codeword bit, the never-sent ones already skipped:
  // ../../back/bitmap.mjs sentBit), on the frame coefficient fcoef gives, the whitening at its frame slots.
  uint sBase = r * P.dims.y;
  for (uint cc = i; cc < nt / 2u; cc = cc + TH) {
    uint pe = TAB[permAt + cc];
    uint fc = fcoef(slot0 / 2u + cc);
    put(sBase, fc, float2((cwBit(pe & 0xffffu) ^ white(2u * fc)) != 0u ? -A : A, (cwBit(pe >> 16u) ^ white(2u * fc + 1u)) != 0u ? -A : A));
  }
  // the last block's workgroup: the padding, the data coefficients from the slots the blocks take to the frame's last,
  // the whitening alone; then every sub-channel's pilots (src/lizard_phy.c lizard_encode): its count pilot (coefficient
  // 0) the whitening XOR bit j & 1 of its frame's count on both axes, its reference (PB_REF) the whitening alone. The
  // frame's count: c0 (the encode's first frame's) + its place in the encode, mod 4, the frame the symbol r / codes.
  if (b + 1u == P.sizes[0].x) {
    uint subch = P.sizes[0].w / 320u;
    for (uint cc = P.sizes[0].z / 2u + i; cc < DCOEF * subch; cc = cc + TH) {
      uint fc = fcoef(cc);
      put(sBase, fc, qpsk(fc, 0u));
    }
    uint count = ((P.dims.w & 3u) + r / max((P.dims.w >> 2u) & 0xffffu, 1u)) & 3u;
    for (uint j = i; j < subch; j = j + TH) {
      put(sBase, 320u * j, qpsk(320u * j, (count >> (j & 1u)) & 1u));
      put(sBase, 320u * j + PB_REF, qpsk(320u * j + PB_REF, 0u));
    }
  }
}
