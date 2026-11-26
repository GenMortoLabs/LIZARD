// The split CRC-64 (CRC-64/LIZARD) over a block's payload, the LDPC's check (tldpc.slang) and the paint's
// (send_paint.slang; its powers core/dec/codes.h crcPowers). crcOf(i) is called by every lane of the
// workgroup with its lane index (it holds two barriers) once the 472 payload bytes sit in the workgroup array packw
// (four bytes a word in memory order, words 0 to 117), and returns the CRC to every lane as (high, low): a lane a
// 4-byte chunk, its CRC from a zero register, times its power from CRC_PW (chunk w's two words at CRC_PW[w / 2], .xy
// where w is even, .zw where odd), lane 0 adding the init term, folded into crcw by atomics. A kernel including it
// declares packw (uint) and crcw (two Atomic<uint>) and defines CRC_PW and CRC64_CHUNKS (118u).
static const uint CRC_CHUNKS = CRC64_CHUNKS;
static const uint CRC_POLY_HI = 0xc02c38b4u;
static const uint CRC_POLY_LO = 0x7d32d307u;
uint2 crcStep(uint2 c) {
  uint m = 0u - (c.y & 1u);
  return uint2((c.x >> 1u) ^ (CRC_POLY_HI & m), ((c.y >> 1u) | (c.x << 31u)) ^ (CRC_POLY_LO & m));
}
uint2 crcPow(uint w) { uint4 v = CRC_PW[w >> 1u]; return (w & 1u) != 0u ? v.zw : v.xy; }
uint2 crcOf(uint i) {
  if (i == 0u) { crcw[0].store(0u); crcw[1].store(0u); }
  GroupMemoryBarrierWithGroupSync();
  if (i < CRC_CHUNKS) {
    uint w = packw[i];
    uint2 c = uint2(0u);
    for (uint t = 0u; t < 4u; t = t + 1u) {
      c.y = c.y ^ ((w >> (8u * t)) & 0xffu);
      for (uint q = 0u; q < 8u; q = q + 1u) { c = crcStep(c); }
    }
    uint2 p = crcPow(i);
    uint2 acc = uint2(0u);
    uint2 tt = c;
    for (uint j = 0u; j < 32u; j = j + 1u) {
      if (((p.x >> (31u - j)) & 1u) != 0u) { acc = acc ^ tt; }
      tt = crcStep(tt);
    }
    for (uint j = 0u; j < 32u; j = j + 1u) {
      if (((p.y >> (31u - j)) & 1u) != 0u) { acc = acc ^ tt; }
      tt = crcStep(tt);
    }
    if (i == 0u) { acc = acc ^ crcPow(CRC_CHUNKS); }
    crcw[0].xor(acc.x);
    crcw[1].xor(acc.y);
  }
  GroupMemoryBarrierWithGroupSync();
  return uint2(crcw[0].load(), crcw[1].load());
}
