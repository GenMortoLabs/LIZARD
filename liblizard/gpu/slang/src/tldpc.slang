// tldpc, the rule stage's LDPC: a workgroup a (block of this code, listed frame), the frame from the gate's list of
// every listed frame; the block's soft values through its code's inverse bit map (TAB's INV) into the posteriors,
// layered integer min-sum (norm 13/16, caps 8191 and 127) up to MAXIT iterations, the violated checks a popcount of the
// packed signs every iteration, the code's learned stop read after each (its MLP's words in TAB at NET_AT), then the
// CRC-64 over the decided payload and a REC record for a verified block. Forms differing in workgroup memory alone (the
// posteriors' store, ../../plan/back_tiers.mjs LDPC_FORMS): LW_PAIR two int16 halves a uint, updated by workgroup
// atomic XOR (two lanes may write the two halves of one word in a layer); LW_I32 an int a posterior; LW_I16 an int16_t
// a posterior, plain loads and stores (a 16-bit location is its own memory location), declared in whole 32-bit words (2
// ceil(N / 2)): the Adreno 840 (driver E031.50.19.27) read an odd-length int16_t workgroup array wrongly. A check's
// slot loops are unrolled by slangc itself ([ForceUnroll]): left to the driver's unroll hint, RADV once left the 6/7
// and 3/4 codes' (NS 24 and 17) rolled and took 1.24 times as long, and the Adreno 840 left Slang's rolled (the 6/7
// code's int16 form 0.087 against 0.055 ms a frame, half the instructions). The learned stop's sums are `precise`: no contraction into fused multiply-adds, so the kernel keeps the float32
// order of stop/rule.mjs scoreF32 and the trainer's. The arithmetic: ../../back/DESIGN.md 6 and 7. Defines
// (back_tiers.mjs tldpcSource): LW_PAIR, LW_I32, LW_I16, ONE (0 or 1: a check's record one uint, else a uint2); TH
// (lanes), LDPC_CODE, LDPC_INV_AT, LDPC_N, LDPC_K, LDPC_Z, LDPC_MB, LDPC_NS, LDPC_KNOWN, LDPC_NET_H, LDPC_NET_AT,
// CRC64_CHUNKS, PACK_WORDS, ROWV, PAIRV, PW_VECS, SLOT_COUNT (uint); lib/tab.slang's.
#if LW_I16
#define LW_INT(X) int(X)
#else
#define LW_INT(X) X
#endif
#if ONE
#define REC_T uint
#define RECORD(AT) Rr[AT] = uint(m1) | (uint(m2) << 8u) | (arg << 16u) | (signs << 20u);
#else
#define REC_T uint2
#define RECORD(AT) Rr[AT] = uint2(uint(m1) | (uint(m2) << 8u) | (arg << 16u), signs);
#endif
#include "lib/sync.slang"
#include "lib/lists.slang"
[[vk::binding(0, 0)]] StructuredBuffer<uint> Lin;
[[vk::binding(1, 0)]] StructuredBuffer<uint2> BLK;
[[vk::binding(2, 0)]] StructuredBuffer<uint> LISTS;
[[vk::binding(3, 0)]] StructuredBuffer<uint> TAB;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> V;
[[vk::binding(5, 0)]] RWStructuredBuffer<int> ITS;
// the record: its count, then its words (REC[0], REC[1 + i])
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> REC;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> BCOUNTS;
struct TierCodeBlock { uint4 dims; uint4 rows[ROWV]; uint4 pairs[PAIRV]; uint4 pw[PW_VECS]; };
[[vk::binding(8, 0)]] ConstantBuffer<TierCodeBlock> T;
#include "lib/tab.slang"
static const uint CODE = LDPC_CODE;
static const uint INV_AT = LDPC_INV_AT;
static const uint N = LDPC_N;
static const uint K = LDPC_K;
static const uint Z = LDPC_Z;
static const uint MB = LDPC_MB;
static const uint NS = LDPC_NS;
static const uint KNOWN = LDPC_KNOWN;
static const int NORM = 13;
static const uint MAXIT = 30u;
static const int MAG_CAP = 1016;
static const uint GROW_MAX = 65535u;
static const uint DATA_WORDS = CRC64_CHUNKS + 2u;
static const uint REC_WORDS = CRC64_CHUNKS + 2u;

#if LW_I32
groupshared int Lw[N];
#elif LW_I16
groupshared int16_t Lw[2u * ((N + 1u) / 2u)];
#else
groupshared Atomic<uint> Lw[(N + 1u) / 2u];
#endif
groupshared Atomic<uint> badc;
groupshared Atomic<uint> magc;
groupshared uint flag;
groupshared uint packw[PACK_WORDS];
groupshared Atomic<uint> crcw[2];
static const uint NET_H = LDPC_NET_H;
static const uint NET_AT = LDPC_NET_AT;
groupshared float netX[6];
groupshared float netH[NET_H];
groupshared float netP[NET_H];
float netW(uint q) { return asfloat(TAB[NET_AT + q]); }

#if LW_I32
int ldL(int idx) { return Lw[uint(idx)]; }
void stL(int idx, int old, int nl) { Lw[uint(idx)] = nl; }
#elif LW_I16
int ldL(int idx) { return int(Lw[uint(idx)]); }
void stL(int idx, int old, int nl) { Lw[uint(idx)] = int16_t(nl); }
#else
int ldL(int idx) {
  uint w = Lw[uint(idx) >> 1u].load();
  return int(w << (16u * (1u - (uint(idx) & 1u)))) >> 16u;
}
void stL(int idx, int old, int nl) {
  Lw[uint(idx) >> 1u].xor((uint(old ^ nl) & 0xffffu) << (16u * (uint(idx) & 1u)));
}
#endif
uint rowAt(uint r) { return T.rows[r >> 2u][r & 3u]; }
int slotIdx(uint r, uint i, uint e) {
  uint p0 = rowAt(r);
  uint nd = rowAt(r + 1u) - p0;
  if (e < nd) {
    uint q = p0 + e;
    uint lc = T.pairs[q >> 2u][q & 3u];
    uint t = i + Z - (lc >> 16u);
    if (t >= Z) { t = t - Z; }
    return int((lc & 0xffffu) * Z + t);
  }
  if (e == nd) {
    if (r > 0u) { return int(K + (r - 1u) * Z + i); }
    if (i > 0u) { return int(K + (MB - 1u) * Z + i - 1u); }
    return -1;
  }
  if (e == nd + 1u) { return int(K + r * Z + i); }
  return -1;
}

int oldMsg(REC_T rec, uint e) {
#if ONE
  int m = e == ((rec >> 16u) & 0xfu) ? int((rec >> 8u) & 0xffu) : int(rec & 0xffu);
  return ((rec >> (20u + e)) & 1u) != 0u ? -m : m;
#else
  int m = e == ((rec.x >> 16u) & 0x1fu) ? int((rec.x >> 8u) & 0xffu) : int(rec.x & 0xffu);
  return ((rec.y >> e) & 1u) != 0u ? -m : m;
#endif
}

static const uint CH = (Z + 31u) / 32u;
static const uint KCOL = K / Z;
uint get32(uint p) {
  uint w = p >> 5u;
  uint s = p & 31u;
  uint lo = packw[w] >> s;
  return s == 0u ? lo : lo | (packw[w + 1u] << (32u - s));
}
uint win(uint c, uint t0) {
  uint a = get32(c * Z + t0);
  if (t0 + 32u <= Z) { return a; }
  uint n1 = Z - t0;
  return (a & ((1u << n1) - 1u)) | (get32(c * Z) << n1);
}
void signsToBitmap(uint i) {
  uint mg = 0u;
  for (uint w = i; w < (N + 31u) / 32u; w = w + TH) {
    uint b = 0u;
#if !LW_PAIR
    for (uint t = 0u; t < 32u; t = t + 1u) {
      uint idx = 32u * w + t;
      if (idx < N) {
        int v = LW_INT(Lw[idx]);
        if (v < 0) { b = b | (1u << t); }
        mg = mg + uint(min(abs(v), MAG_CAP));
      }
    }
#else
    for (uint h = 0u; h < 16u; h = h + 1u) {
      uint q = 16u * w + h;
      if (q < (N + 1u) / 2u) {
        uint v = Lw[q].load();
        b = b | (((v >> 15u) & 1u) << (2u * h)) | ((v >> 31u) << (2u * h + 1u));
        mg = mg + uint(min(abs(int(v << 16u) >> 16u), MAG_CAP)) + uint(min(abs(int(v) >> 16u), MAG_CAP));
      }
    }
#endif
    packw[w] = b;
  }
  if (mg != 0u) { magc.add(mg); }
  if (i == 0u) { packw[(N + 31u) / 32u] = 0u; }
}
uint violatedChunks(uint i) {
  uint c = 0u;
  for (uint u = i; u < MB * CH; u = u + TH) {
    uint r = u / CH;
    uint i0 = 32u * (u - r * CH);
    uint acc = win(KCOL + r, i0);
    if (r > 0u) { acc = acc ^ win(KCOL + r - 1u, i0); }
    else { acc = acc ^ (win(KCOL + MB - 1u, (i0 + Z - 1u) % Z) & (i0 == 0u ? 0xfffffffeu : 0xffffffffu)); }
    uint q1 = rowAt(r + 1u);
    for (uint q = rowAt(r); q < q1; q = q + 1u) {
      uint lc = T.pairs[q >> 2u][q & 3u];
      uint t0 = i0 + Z - (lc >> 16u);
      if (t0 >= Z) { t0 = t0 - Z; }
      acc = acc ^ win(lc & 0xffffu, t0);
    }
    uint m = min(32u, Z - i0);
    c = c + uint(countbits(acc & (m == 32u ? 0xffffffffu : (1u << m) - 1u)));
  }
  return c;
}
#define CRC_PW T.pw
#include "lib/crc64.slang"
[numthreads(TH, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint i = lix;
  uint B = T.dims.x;
  uint lStride = T.dims.y;
  uint blocksMax = T.dims.z;
  // a workgroup a (block of this code, listed frame): the frame from the gate's list of every listed frame (listsAll)
  uint f = LISTS[SLOT_COUNT * (B + 1u) + 5u * B + wg.y];
  uint2 v = CFG_AT(LISTS, B, f);
  uint4 tt = ttOf(v, CODE);
  uint x = wg.x;
  if (x >= tt.z) { return; }
  uint b = tt.x + x;
  uint slotV = f * blocksMax + b;
  if (BLK[slotV].y != 0u) {
    if (i == 0u) { V[slotV] = 2u; ITS[slotV] = -2; }
    return;
  }
  // the block's first slot (its tier's first and the blocks before it, a multiple of 16), as L's word: four slots a
  // word
  uint lBase = f * lStride + (tt.y + x * slotsOf(CODE)) / 4u;
  for (uint wd = i; wd < (N + 1u) / 2u; wd = wd + TH) {
    uint pk = 0u;
    for (uint h = 0u; h < 2u; h = h + 1u) {
      uint d = 2u * wd + h;
      if (d >= N) { continue; }
      uint j = d;
      if (d >= K) { uint t = d - K; j = K + MB * (t % Z) + t / Z; }
      int vv = 127;
      if (j < KNOWN || j >= K) {
        uint slot = (TAB[INV_AT + (j >> 1u)] >> (16u * (j & 1u))) & 0xffffu;
        if (slot == 0xffffu) { vv = 0; }
        else {
          uint w = Lin[lBase + (slot >> 2u)];
          vv = int(w << (8u * (3u - (slot & 3u)))) >> 24u;
        }
      }
      #if LW_I32
      if (d < N) { Lw[d] = vv; }
#elif LW_I16
      if (d < N) { Lw[d] = int16_t(vv); }
#else
      pk = pk | ((uint(vv) & 0xffffu) << (16u * h));
#endif
    }
    #if LW_PAIR
    Lw[wd].store(pk);
#endif
  }
  REC_T Rr[MB];
  for (uint q = 0u; q < MB; q = q + 1u) { Rr[q] = REC_T(0u); }
  if (i == 0u) { badc.store(0u); magc.store(0u); flag = 0u; }
  GroupMemoryBarrierWithGroupSync();

  uint first0 = 0u;
  uint low = 0xffffffffu;
  uint prev = 0u;
  uint mag1 = 0u;
  uint it = 0u;
  uint exitCode = 0u;
  for (it = 0u; it < MAXIT; it = it + 1u) {
    for (uint st = 0u; st < MB; st = st + 1u) {
      uint r = st;
      if ((it & 1u) != 0u) { r = MB - 1u - st; }
      if (i < Z) {
        REC_T rec = Rr[r];
        int min1 = 32767;
        int min2 = 32767;
        uint arg = 0u;
        uint sgn = 0u;
        int idxs[NS];
        int qs[NS];
        [ForceUnroll] for (uint e = 0u; e < NS; e = e + 1u) {
          int idx = slotIdx(r, i, e);
          idxs[e] = idx;
          int q = 8191;
          if (idx >= 0) { q = ldL(idx) - oldMsg(rec, e); }
          qs[e] = q;
        }
        [ForceUnroll] for (uint e = 0u; e < NS; e = e + 1u) {
          int idx = idxs[e];
          if (idx < 0) { continue; }
          int q = qs[e];
          int a = abs(q);
          sgn = sgn ^ uint(q < 0);
          bool lt = a < min1;
          min2 = min(min2, max(min1, a));
          arg = lt ? e : arg;
          min1 = min(min1, a);
        }
        int m1 = min(127, (min1 * NORM) >> 4u);
        int m2 = min(127, (min2 * NORM) >> 4u);
        uint signs = 0u;
        [ForceUnroll] for (uint e = 0u; e < NS; e = e + 1u) {
          int idx = idxs[e];
          if (idx < 0) { continue; }
          int q = qs[e];
          int mag = e == arg ? m2 : m1;
          uint s = sgn ^ uint(q < 0);
          int rn = s != 0u ? -mag : mag;
          signs = signs | (s << e);
          int nl = clamp(q + rn, -8191, 8191);
          stL(idx, q + oldMsg(rec, e), nl);
        }
        RECORD(r)
      }
      GroupMemoryBarrierWithGroupSync();
    }
    signsToBitmap(i);
    GroupMemoryBarrierWithGroupSync();
    uint c = violatedChunks(i);
    if (c != 0u) { badc.add(c); }
    GroupMemoryBarrierWithGroupSync();
    if (i == 0u) {
      uint bad = badc.load();
      badc.store(0u);
      uint mg = magc.load();
      magc.store(0u);
      uint d = 0u;
      if (bad == 0u) { d = 1u; }
      else {
        if (it == 0u) { first0 = bad; prev = bad; mag1 = mg; }
        low = min(low, bad);
        uint grow = 256u;
        if (mag1 != 0u) { grow = min((256u * mg) / mag1, GROW_MAX); }
        netX[0] = float(bad); netX[1] = float(low); netX[2] = float(first0); netX[3] = float(prev); netX[4] = float(it + 1u); netX[5] = float(grow);
        prev = bad;
      }
      flag = d;
    }
    UNIFORM_LOAD(uint d, flag)
    if (d != 0u) { exitCode = d; break; }
    if (it + 1u < MAXIT) {
      if (i < NET_H) {
        precise float a = netW(6u * NET_H + i);
        for (uint k = 0u; k < 6u; k++) { a = a + netW(k * NET_H + i) * netX[k]; }
        netH[i] = max(a, 0.0);
      }
      GroupMemoryBarrierWithGroupSync();
      if (i < NET_H) {
        precise float a = netW((7u + NET_H) * NET_H + i);
        for (uint k = 0u; k < NET_H; k++) { a = a + netW((7u + k) * NET_H + i) * netH[k]; }
        netP[i] = netW((8u + NET_H) * NET_H + i) * max(a, 0.0);
      }
      GroupMemoryBarrierWithGroupSync();
      if (i == 0u) {
        precise float sum = netP[0];
        for (uint k = 1u; k < NET_H; k++) { sum = sum + netP[k]; }
        flag = sum < netW((9u + NET_H) * NET_H) ? 2u : 0u;
      }
      UNIFORM_LOAD(uint g, flag)
      if (g != 0u) { exitCode = g; break; }
    }
  }
  int its = -1;
  uint ran = MAXIT;
  if (exitCode == 1u) { its = int(it) + 1; ran = it + 1u; }
  if (exitCode == 2u) { its = -3; ran = it + 1u; }
  if (i == 0u) {
    ITS[slotV] = its;
    InterlockedAdd(BCOUNTS[f * 8u + 3u], 1u);
    InterlockedAdd(BCOUNTS[f * 8u + 7u], ran);
  }
  if (its < 0) {
    if (i == 0u) { V[slotV] = 3u; }
    return;
  }
  if (i < DATA_WORDS) {
    uint w = 0u;
    for (uint t = 0u; t < 32u; t = t + 1u) {
      uint j = 32u * i + t;
      if (j < K && ldL(int(j)) < 0) { w = w | (1u << (8u * (t >> 3u) + 7u - (t & 7u))); }
    }
    packw[i] = w;
  }
  uint2 crc = crcOf(i);
  if (i == 0u) {
    // the CRC-64 the block carries, big-endian in the two words after its payload's (bytes 472 to 479)
    uint wh = packw[CRC_CHUNKS], wl = packw[CRC_CHUNKS + 1u];
    uint2 expect = uint2(((wh & 0xffu) << 24u) | (((wh >> 8u) & 0xffu) << 16u) | (((wh >> 16u) & 0xffu) << 8u) | (wh >> 24u),
                         ((wl & 0xffu) << 24u) | (((wl >> 8u) & 0xffu) << 16u) | (((wl >> 16u) & 0xffu) << 8u) | (wl >> 24u));
    uint sl = 0u;
    if (all(crc == expect)) {
      V[slotV] = 1u;
      InterlockedAdd(BCOUNTS[f * 8u + 4u], 1u);
      uint n;
      InterlockedAdd(REC[0], 1u, n);
      if (n < T.dims.w) { sl = n + 1u; InterlockedAdd(BCOUNTS[f * 8u + 5u], 1u); }
    } else {
      V[slotV] = 4u;
    }
    flag = sl;
  }
  UNIFORM_LOAD(uint sl, flag)
  if (sl == 0u || i >= REC_WORDS) { return; }
  uint base = (sl - 1u) * REC_WORDS;
  uint word = 0u;
  if (i == 0u) { word = f + 65536u * b; }
  else if (i == 1u) { word = uint(its); }
  else { word = packw[i - 2u]; }
  REC[1u + base + i] = word;
}
