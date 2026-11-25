// The native senders' RaptorQ blocks: the CPU solves a file's chunks for their intermediate symbols (src/rq.c, a chunk a
// core, core/tx/xfer_tx.h), the host hands them up (core/tx/gpu_stream.h), and each frame's blocks are made here from
// their ids alone: each block the XOR of its tuple's columns (RFC 6330 5.3.5.3, LT and PI; the tuple of 5.3.5.4 made
// here, rqCols, src/rq.c rq_row's twin), a control block's payload one symbol the host wrote, written as the painter's
// BLOCKS hold a block (core/tx/gpu_painter.cpp: 480 B apart, the 12-byte id: the bare id and its TOI's low 24 bits in
// the first two words, then the TOI's high 24 bits and a reserved byte (PAR's), then the 460-byte payload,
// word-aligned), so the painter paints them in the same submission and no block's bytes cross from the host.
// Bindings: SYM (a chunk's L intermediate symbols from its first, 480 B each: 30 uint4, the 460-byte symbol zero padded,
// then the control payloads), IDS (each block's id, its low 32 bits and its high 32), TAB (RFC 6330's V0 to V3 at 0,
// 256, 512 and 768, Deg's table at TAB_DEG (31 words), then a plan's parameters TAB_PLAN words apart from TAB_PLANS (K,
// K', J, W, P, P1); the host's tables are src/rq_tables.h), CHUNKS (a chunk's first symbol and its plan), OUT (the
// painter's BLOCKS: OUT_WORDS words a block, 118 written: the id's 3, the payload's 115), PAR (the blocks, the control
// payloads' first symbol, the source blocks, the controls, the file's TOI's high 24 bits, a data block's word 2, and
// the session's, the session header's, each word's top byte 0). Two blocks a workgroup: its first lane the block's
// columns (a data block its tuple's, a control block its payload's one symbol; none past the file: zeros), then 30
// lanes XORing 16 bytes each, then all 32 writing the shifted words from workgroup memory. Every workgroup variable is
// written before the barrier that precedes its reads.
// Defines (../send.mjs rqBlocksSource): TAB_DEG, TAB_PLANS, TAB_PLAN, OUT_WORDS (uint).
[[vk::binding(0, 0)]] StructuredBuffer<uint4> sym;
[[vk::binding(1, 0)]] StructuredBuffer<uint2> ids;
[[vk::binding(2, 0)]] StructuredBuffer<uint> tab;
[[vk::binding(3, 0)]] StructuredBuffer<uint2> chunks;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> outw;
[[vk::binding(5, 0)]] StructuredBuffer<uint> par;
groupshared uint cols[2][33];
groupshared uint ncol[2];
groupshared uint pay[2][120];
uint rqRand(uint y, uint i, uint m) {
  return (tab[(y + i) & 255u] ^ tab[256u + (((y >> 8u) + i) & 255u)] ^ tab[512u + (((y >> 16u) + i) & 255u)] ^ tab[768u + (((y >> 24u) + i) & 255u)]) % m;
}
// the columns of internal symbol isi's row, src/rq.c rq_row (RFC 6330 5.3.5.4's tuple walked as 5.3.5.3's Enc does),
// into the workgroup's cols[s] (an index at run time into a private array read zeros on the S26's Adreno), their count;
// pl the plan's J, W, P, P1
uint rqCols(uint4 pl, uint isi, uint s) {
  uint A = 53591u + pl.x * 997u;
  if ((A & 1u) == 0u) { A = A + 1u; }
  uint y = 10267u * (pl.x + 1u) + isi * A;
  uint v = rqRand(y, 0u, 1048576u);
  uint d = 1u;
  while (v >= tab[TAB_DEG + d]) { d = d + 1u; }
  d = min(d, pl.y - 2u);
  uint a = 1u + rqRand(y, 1u, pl.y - 1u);
  uint b = rqRand(y, 2u, pl.y);
  uint d1 = 2u;
  if (d < 4u) { d1 = 2u + rqRand(isi, 3u, 2u); }
  uint a1 = 1u + rqRand(isi, 4u, pl.w - 1u);
  uint b1 = rqRand(isi, 5u, pl.w);
  cols[s][0] = b;
  uint n = 1u;
  for (uint j = 1u; j < d; j++) { b = (b + a) % pl.y; cols[s][n] = b; n = n + 1u; }
  while (b1 >= pl.z) { b1 = (b1 + a1) % pl.w; }
  cols[s][n] = pl.y + b1;
  n = n + 1u;
  for (uint j = 1u; j < d1; j++) {
    b1 = (b1 + a1) % pl.w;
    while (b1 >= pl.z) { b1 = (b1 + a1) % pl.w; }
    cols[s][n] = pl.y + b1;
    n = n + 1u;
  }
  return n;
}
[numthreads(64, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint li = lid.x;
  uint s = li >> 5u;
  uint w = li & 31u;
  uint b = wg.x * 2u + s;
  bool live = b < par[0];
  uint2 id = uint2(0u);
  if (live) { id = ids[b]; }
  if (live && w == 0u) {
    uint c = (id.x >> 24u) | ((id.y & 255u) << 8u);
    uint e = id.x & 0xffffffu;
    uint n = 0u;
    if (c == 0xffffu) {
      if (e < par[3]) { cols[s][0] = 0u; n = 1u; }
    } else if (c < par[2]) {
      uint2 ch = chunks[c];
      uint at = TAB_PLANS + ch.y * TAB_PLAN;
      uint isi = e;
      if (e >= tab[at]) { isi = e + tab[at + 1u] - tab[at]; }
      n = rqCols(uint4(tab[at + 2u], tab[at + 3u], tab[at + 4u], tab[at + 5u]), isi, s);
    }
    uint base = par[1] + e;
    if (c != 0xffffu) { base = chunks[min(c, par[2] - 1u)].x; }
    for (uint j = 0u; j < n; j++) { cols[s][j] = cols[s][j] + base; }
    ncol[s] = n;
  }
  GroupMemoryBarrierWithGroupSync();
  if (live && w < 30u) {
    uint4 a = uint4(0u);
    for (uint j = 0u; j < ncol[s]; j++) { a = a ^ sym[cols[s][j] * 30u + w]; }
    pay[s][4u * w] = a.x; pay[s][4u * w + 1u] = a.y; pay[s][4u * w + 2u] = a.z; pay[s][4u * w + 3u] = a.w;
  }
  GroupMemoryBarrierWithGroupSync();
  if (!live) { return; }
  for (uint j = w; j < 118u; j += 32u) {
    uint v = 0u;
    if (j == 0u) { v = id.x; }
    else if (j == 1u) { v = id.y; }
    else if (j == 2u) { v = (((id.x >> 24u) | ((id.y & 255u) << 8u)) == 0xffffu ? par[5] : par[4]) & 0xffffffu; }
    else { v = pay[s][j - 3u]; }
    outw[b * OUT_WORDS + j] = v;
  }
}
