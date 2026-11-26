// The back half's gate: a batch of B frames listed by picture slot (each listed frame's configuration, rows and ring),
// the passes' indirect dispatches sized (pass 1's workgroups, pass 2's rows, tsoft's chunks, each code's LDPC blocks),
// and the chain's counters, verdicts, iteration counts and record count zeroed. One workgroup of 64. Built a batch
// size. Bindings: 0 frames, 1 sel, 2 LISTS, 3 ARGS, 4 the gate's uniform (at[s] a slot's (workgroups of pass 1, rows of
// pass 2, top version, 0), then the rows table four versions' words a float4), 5 BCOUNTS, 6 V, 7 ITS, 8 REC. The LISTS
// and ARGS layouts and the defines (core/dec/build.cpp backHalf): BATCH, SLOT_COUNT,
// LISTS_HEAD, ARGS_LDPC, ARGS_WORDS, NC, CHUNKS, BLK_STRIDE (uint), GATE_ROWS; lib/cfg.slang's.
#include "lib/frame.slang"
[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> sel;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> lists;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> args;
struct GateBlock { uint4 at[SLOT_COUNT]; uint4 rows[GATE_ROWS]; };
[[vk::binding(4, 0)]] ConstantBuffer<GateBlock> G;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> V;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> ITS;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> rec;
static const uint B = BATCH;
static const uint LS = BATCH + 1u;
static const uint SLOTS = SLOT_COUNT;
static const uint HEAD = LISTS_HEAD;
static const uint LDPC = ARGS_LDPC;
#include "lib/cfg.slang"
[numthreads(64, 1, 1)]
void main(uint lix : SV_GroupIndex) {
  uint t = lix;
  for (uint i = t; i < 8u * B; i += 64u) { counts[i] = 0u; }
  for (uint i = t; i < B * BLK_STRIDE; i += 64u) { V[i] = 0u; ITS[i] = 0u; }
  if (t == 0u) { rec[0] = 0u; }
  if (t != 0u) { return; }
  // pass 2's workgroups a frame gather here as the frames are listed: the most rows any frame of the size needs.
  // and tsoft's, the chunks of the most version any frame of the size takes.
  for (uint s = 0u; s < SLOTS; s++) { lists[s * LS] = 0u; args[s * ARGS_WORDS + 6u] = 0u; args[s * ARGS_WORDS + 9u] = 0u; }
  // and each code's, the most blocks of it a listed frame's configuration takes, the frames listed counted in m.
  for (uint c = 0u; c < NC; c++) { args[LDPC + 3u * c] = 0u; }
  uint m = 0u;
  for (uint f = 0u; f < B; f++) {
    lists[HEAD + 2u * f] = frames[f].w;
    lists[HEAD + 2u * f + 1u] = frames[f].h;
    lists[HEAD + 2u * B + f] = 0u;
    lists[HEAD + 3u * B + f] = 0u;
    lists[HEAD + 4u * B + f] = 0u;
    lists[HEAD + 6u * B + f] = 0u;
    // F8's configuration (sel.y = 1 + its first word, the version << 24 | the 6/7, 3/4 and 2/3 counts, sel.w its
    // 1/2 count; lib/cfg.slang, src/fmt.h), the frame's layout
    if (frames[f].valid == 0u || sel[f].y < 2u) { continue; }
    uint2 g = uint2(sel[f].y - 1u, sel[f].w);
    uint s = sel[f].z;
    if (s >= SLOTS || G.at[s].x == 0u || !cfgOk(g)) { continue; }
    uint c = lists[s * LS];
    lists[s * LS + 1u + c] = f;
    lists[s * LS] = c + 1u;
    uint k = min(cfgVersion(g), G.at[s].z);
    uint i = k * SLOTS + s;
    uint r = G.rows[i >> 2u][i & 3u];
    lists[HEAD + 2u * B + f] = g.x;
    lists[HEAD + 6u * B + f] = g.y;
    lists[HEAD + 3u * B + f] = r;
    lists[HEAD + 4u * B + f] = sel[f].x;
    args[s * ARGS_WORDS + 6u] = max(args[s * ARGS_WORDS + 6u], r);
    args[s * ARGS_WORDS + 9u] = max(args[s * ARGS_WORDS + 9u], CHUNKS * k);
    lists[HEAD + 5u * B + m] = f;
    m = m + 1u;
    uint4 cc = cfgCounts(g);
    args[LDPC] = max(args[LDPC], cc.x);
    args[LDPC + 3u] = max(args[LDPC + 3u], cc.y);
    args[LDPC + 6u] = max(args[LDPC + 6u], cc.z);
    args[LDPC + 9u] = max(args[LDPC + 9u], cc.w);
  }
  for (uint c = 0u; c < NC; c++) { args[LDPC + 3u * c + 1u] = m; args[LDPC + 3u * c + 2u] = 1u; }
  for (uint s = 0u; s < SLOTS; s++) {
    uint c = lists[s * LS];
    uint a = s * ARGS_WORDS;
    args[a] = G.at[s].x; args[a + 1u] = 1u; args[a + 2u] = c;
    args[a + 3u] = 1u; args[a + 4u] = 1u; args[a + 5u] = c;
    args[a + 7u] = 1u; args[a + 8u] = c;
    args[a + 10u] = 1u; args[a + 11u] = c;
  }
}
