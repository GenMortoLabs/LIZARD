// ai: The native senders' blocks made on the GPU (2026-11-16): the CPU solves a file's chunks for their intermediate
// ai: symbols (src/rq.c, a chunk a core, core/tx/xfer_tx.h), the host hands them up (core/tx/gpu_stream.h), and each
// ai: frame's blocks are made here from their ids alone: BLOCKS (rqBlocksSource), each block the XOR of its tuple's
// ai: columns (RFC 6330 5.3.5.3, LT and PI; the tuple of 5.3.5.4 made here, TUPLE's rqCols, src/rq.c rq_row's twin), a
// ai: control block's payload one symbol the host wrote, written as the painter's BLOCKS hold a block
// ai: (core/tx/gpu_painter.cpp: 480 B apart, the 12-byte id: the bare id and its TOI's low 24 bits in the first two words,
// ai: then the TOI's high 24 bits and a reserved byte (PAR's), then the 460-byte payload, word-aligned), so the painter paints them in the
// ai: same submission and no block's bytes cross from the host. SYM: a chunk's L intermediate symbols from its first,
// ai: 480 B each (30 vec4u, the 460-byte symbol zero padded), then the control payloads.
// ai: The solve on the GPU (XOR's levels, HDPC, the Gauss-Jordan in dependency levels, a source check) was built and
// ai: measured the same day and taken back out: the cores make the chunks as fast or faster on every device measured,
// ai: once, at the start (STATUS "RaptorQ's encoder on the GPU in the sender").

// ai: TAB: RFC 6330's V0 to V3 at 0, 256, 512 and 768, Deg's table at TAB_DEG (31 words), then
// ai: a plan's parameters TAB_PLAN words apart from TAB_PLANS (K, K', J, W, P, P1); the host's tables are src/rq_tables.h
export const TAB_DEG = 1024, TAB_PLANS = 1056, TAB_PLAN = 8;
// ai: the columns of internal symbol isi's row, src/rq.c rq_row (RFC 6330 5.3.5.4's tuple walked as 5.3.5.3's Enc does),
// ai: into the workgroup's cols[s] (an index at run time into a private array read zeros on the S26's Adreno), their
// ai: count; pl the plan's J, W, P, P1
const TUPLE = `
fn rqRand(y: u32, i: u32, m: u32) -> u32 {
  return (tab[(y + i) & 255u] ^ tab[256u + (((y >> 8u) + i) & 255u)] ^ tab[512u + (((y >> 16u) + i) & 255u)] ^ tab[768u + (((y >> 24u) + i) & 255u)]) % m;
}
fn rqCols(pl: vec4u, isi: u32, s: u32) -> u32 {
  var A = 53591u + pl.x * 997u;
  if ((A & 1u) == 0u) { A = A + 1u; }
  let y = 10267u * (pl.x + 1u) + isi * A;
  let v = rqRand(y, 0u, 1048576u);
  var d = 1u;
  while (v >= tab[${TAB_DEG}u + d]) { d = d + 1u; }
  d = min(d, pl.y - 2u);
  let a = 1u + rqRand(y, 1u, pl.y - 1u);
  var b = rqRand(y, 2u, pl.y);
  var d1 = 2u;
  if (d < 4u) { d1 = 2u + rqRand(isi, 3u, 2u); }
  let a1 = 1u + rqRand(isi, 4u, pl.w - 1u);
  var b1 = rqRand(isi, 5u, pl.w);
  cols[s][0] = b;
  var n = 1u;
  for (var j = 1u; j < d; j++) { b = (b + a) % pl.y; cols[s][n] = b; n = n + 1u; }
  while (b1 >= pl.z) { b1 = (b1 + a1) % pl.w; }
  cols[s][n] = pl.y + b1;
  n = n + 1u;
  for (var j = 1u; j < d1; j++) {
    b1 = (b1 + a1) % pl.w;
    while (b1 >= pl.z) { b1 = (b1 + a1) % pl.w; }
    cols[s][n] = pl.y + b1;
    n = n + 1u;
  }
  return n;
}`;

// ai: BLOCKS: SYM, IDS (each block's id, its low 32 bits and its high 32), TAB, CHUNKS (a chunk's first symbol and its
// ai: plan), OUT (the painter's BLOCKS: OUT_WORDS words a block, 118 written: the id's 3, the payload's 115), PAR (the blocks, the control payloads'
// ai: first symbol, the source blocks, the controls, the file's TOI's high 24 bits, a data block's word 2, and the session's,
// ai: the session header's, each word's top byte 0). Two blocks a workgroup: its first lane the block's columns (a data
// ai: block its tuple's, a control block its payload's one symbol; none past the file: zeros), then 30 lanes XORing 16
// ai: bytes each, then all 32 writing the shifted words from workgroup memory.
export const RQ_OUT_WORDS = 120;
export function rqBlocksSource({ outWords = RQ_OUT_WORDS } = {}) {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> sym: array<vec4u>;
@group(0) @binding(1) var<storage, read> ids: array<vec2u>;
@group(0) @binding(2) var<storage, read> tab: array<u32>;
@group(0) @binding(3) var<storage, read> chunks: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> outw: array<u32>;
@group(0) @binding(5) var<storage, read> par: array<u32>;
var<workgroup> cols: array<array<u32, 33>, 2>;
var<workgroup> ncol: array<u32, 2>;
var<workgroup> P: array<array<u32, 120>, 2>;
${TUPLE}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let s = li.x >> 5u;
  let w = li.x & 31u;
  let b = wg.x * 2u + s;
  let live = b < par[0];
  var id = vec2u(0u);
  if (live) { id = ids[b]; }
  if (live && w == 0u) {
    var c = (id.x >> 24u) | ((id.y & 255u) << 8u);
    let e = id.x & 0xffffffu;
    var n = 0u;
    if (c == 0xffffu) {
      if (e < par[3]) { cols[s][0] = 0u; n = 1u; }
    } else if (c < par[2]) {
      let ch = chunks[c];
      let at = ${TAB_PLANS}u + ch.y * ${TAB_PLAN}u;
      var isi = e;
      if (e >= tab[at]) { isi = e + tab[at + 1u] - tab[at]; }
      n = rqCols(vec4u(tab[at + 2u], tab[at + 3u], tab[at + 4u], tab[at + 5u]), isi, s);
    }
    var base = par[1] + e;
    if (c != 0xffffu) { base = chunks[min(c, par[2] - 1u)].x; }
    for (var j = 0u; j < n; j++) { cols[s][j] = cols[s][j] + base; }
    ncol[s] = n;
  }
  workgroupBarrier();
  if (live && w < 30u) {
    var a = vec4u(0u);
    for (var j = 0u; j < ncol[s]; j++) { a = a ^ sym[cols[s][j] * 30u + w]; }
    P[s][4u * w] = a.x; P[s][4u * w + 1u] = a.y; P[s][4u * w + 2u] = a.z; P[s][4u * w + 3u] = a.w;
  }
  workgroupBarrier();
  if (!live) { return; }
  for (var j = w; j < 118u; j += 32u) {
    var v = 0u;
    if (j == 0u) { v = id.x; }
    else if (j == 1u) { v = id.y; }
    else if (j == 2u) { v = select(par[4], par[5], ((id.x >> 24u) | ((id.y & 255u) << 8u)) == 0xffffu) & 0xffffffu; }
    else { v = P[s][j - 3u]; }
    outw[b * ${outWords}u + j] = v;
  }
}
`;
}
