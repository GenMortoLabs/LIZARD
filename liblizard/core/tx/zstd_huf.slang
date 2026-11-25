// ai: Stream files packed on the GPU (gpu_zstd.h), stage 3: each block's literals as zstd's four Huffman streams
// ai: (RFC 8878 3.1.1.3.1: the block's residuals split in four, the first three (n + 3) / 4 each), a workgroup a block,
// ai: by the code the host built from the block's counts (tab[(q nb + b) 256 + v]: the code's bits | its length << 16).
// ai: A stream's bits run from its last symbol to its first (the decoder reads it backwards), each code at its bit,
// ai: least significant first: a thread a run of each stream's symbols, its place the bits of the runs after it (a
// ai: suffix sum), its whole words stored, the two it may share with its neighbours OR-ed in; then the end mark, a 1
// ai: past the last code. The four streams back to back from the block's place (mode[blk].y, bytes, a multiple of 4:
// ai: the host's bound of the block's bytes from its counts, so the output holds the bytes that go and no more), each
// ai: its bytes (size); mode[blk].x: 0 Huffman, 1 raw (the block's residuals copied to its place as they are), 2 none
// ai: (one value throughout: the host writes an RLE block).
// ai: a: squares, n, wps, nb
struct P { uint4 a; };
[[vk::binding(0, 0)]] StructuredBuffer<uint> res;
[[vk::binding(1, 0)]] StructuredBuffer<uint> tab;
[[vk::binding(2, 0)]] StructuredBuffer<uint2> mode;
[[vk::binding(3, 0)]] globallycoherent RWStructuredBuffer<uint> outw;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> size;
[[vk::binding(5, 0)]] ConstantBuffer<P> p;
groupshared uint lens[4][256];
groupshared uint offs[4][256];
groupshared uint tot[4];
groupshared uint base[4];
static const uint BLOCK = 131072u;
uint sym(uint rb, uint i) { return (res[rb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint t : SV_GroupIndex) {
  let b = wg.x; let q = wg.y;
  let blk = q * p.a.w + b;
  let md = mode[blk];
  let start = b * BLOCK; let bn = min(p.a.y - start, BLOCK); let rbase = q * p.a.z;
  if (md.x == 2u) { if (t < 4u) size[blk * 4u + t] = 0u; return; }
  if (md.x == 1u) {
    let bw = (bn + 3u) >> 2u; let o = md.y >> 2u; let r = rbase + (start >> 2u);
    for (uint i = t; i < bw; i += 256u) outw[o + i] = res[r + i];
    if (t < 4u) size[blk * 4u + t] = t == 0u ? bn : 0u;
    return;
  }
  let seg = (bn + 3u) / 4u; let tb = blk * 256u;
  for (uint s = 0u; s < 4u; s++) {
    let s0 = start + s * seg; let len = s < 3u ? seg : bn - 3u * seg;
    let per = (len + 255u) / 256u; let i0 = min(len, t * per); let i1 = min(len, i0 + per);
    uint sum = 0u;
    for (uint i = i0; i < i1; i++) sum += tab[tb + sym(rbase, s0 + i)] >> 16u;
    lens[s][t] = sum;
  }
  GroupMemoryBarrierWithGroupSync();
  if (t < 4u) {
    uint acc = 0u;
    for (int u = 255; u >= 0; u--) { offs[t][u] = acc; acc += lens[t][u]; }
    tot[t] = acc;
  }
  GroupMemoryBarrierWithGroupSync();
  if (t == 0u) {
    uint at = md.y;
    for (uint s = 0u; s < 4u; s++) {
      base[s] = at * 8u;
      let bytes = (tot[s] >> 3u) + 1u;
      size[blk * 4u + s] = bytes;
      at += bytes;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint s = 0u; s < 4u; s++) {
    let s0 = start + s * seg; let len = s < 3u ? seg : bn - 3u * seg;
    let per = (len + 255u) / 256u; let i0 = min(len, t * per); let i1 = min(len, i0 + per);
    // ai: this thread's symbols from its last to its first, from its place up
    uint pos = base[s] + offs[s][t], word = pos >> 5u, acc = 0u;
    let firstWord = word;
    for (uint i = i1; i > i0;) {
      i--;
      let e = tab[tb + sym(rbase, s0 + i)]; let l = e >> 16u; let code = e & 0xffffu; let sh = pos & 31u;
      acc |= code << sh;
      if (sh + l >= 32u) {
        if (word == firstWord) InterlockedOr(outw[word], acc); else outw[word] = acc;
        word++;
        acc = sh + l > 32u ? code >> (32u - sh) : 0u;
      }
      pos += l;
    }
    if ((pos & 31u) != 0u) InterlockedOr(outw[word], acc);
  }
  if (t < 4u) {
    let T = base[t] + tot[t];
    InterlockedOr(outw[T >> 5u], 1u << (T & 31u));
  }
}
