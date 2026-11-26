// F4, the classifier: each kept peak's 32 x 32 patch through the net (cnn-v3) as convolutions tiled as GEMMs, a
// register tile a lane and several patches a workgroup: is it a corner mark, which way is out, how big a module. Its
// float forms (f16 and f32) and its int8 form (int8-v1: per-channel weights, per-layer activation scales, requantised
// by one f32 multiply, so its codes are the trainer's integer forward's), dispatched ceil(cap / PATCHES) workgroups a
// frame. A patch is cut from the pyramid level its size reads at (a lane four samples of a row, bilinear), normalised
// by its mean and spread into X, then the net's convs, X and Y taking turns as a layer's input and output, and its two
// fc layers; the last one's five outputs are the reading. The plan (patches a workgroup, each layer's tile, the
// buffers, the scratch, the weights' layout) is the host's: core/dec/build.cpp classifyPlan, its recipe's constants
// below, the weights packed there by the same layout. Every workgroup variable is written before the barrier that
// precedes its reads, so none is zeroed. A bitcast of u32 words to f16 is unpackFloat2x16, to f32 asfloat; the packed
// int8 dot product OpSDot (lib/nets.slang dotPacked4x8); pack4xI8 the low bytes shifted together (the Adreno 840 got an
// 8-bit vector form wrong); round is roundEven; a scalar choice is select (both operands made, one OpSelect). Every
// loop over a tile's channels, positions, taps and patches is unrolled, so no private array is indexed at run time.
// Defines: INT8, F16, SUBGROUPS (0 or 1); PATCHES, GROUP (lanes a patch, 256 / PATCHES), PATCH (the patch's side),
// PATCH_PITCH (the patch's rows in X, vec4s apart: the first conv's input pitch), Z_AT (the last layer's outputs in red),
// BUF_X, BUF_Y, RED, REDI (uint); PATCH_KAPPA, Q8_MAX, Q8_INV (float); VEC4_T
// (half4 or float4); CONVS (the conv layers, 1 to 4) and each layer's arguments, a list of ints:
//   CONV1 .. CONV<CONVS>: in channels, in side, out channels, out side, stride; the input's and the output's row pitch
//     (vec4s a row); the tile: TC channels x 4 TX positions a lane, the input channels in KS slices on adjacent lanes;
//     the weights: the taps' first vec4, vec4s a tap, the biases' first vec4, the scales' (int8; 0 in the float forms)
//   FC1: in, out, the last conv's side and row pitch; NB blocks of TCF outputs, KSF slices of SLICE inputs (int8: vec4s
//     of 16 codes); the rows' first vec4, vec4s a row, the biases', the scales' (int8); its outputs' place in red
//   FC2: in, out; the rows' first vec4, vec4s a row, the biases'; its inputs' and its outputs' places in red
#include "lib/nets.slang"
#include "lib/round.slang"
#include "lib/frame.slang"
#include "lib/convert.slang"
#include "lib/level_bindings.slang"
#include "lib/levels.slang"
[[vk::binding(7, 0)]] StructuredBuffer<float4> peaks;
[[vk::binding(8, 0)]] StructuredBuffer<uint> counts;
[[vk::binding(9, 0)]] RWStructuredBuffer<float4> readings;
struct P4Block { uint cap; uint p0; uint p1; uint p2; };
[[vk::binding(10, 0)]] ConstantBuffer<P4Block> P;
[[vk::binding(11, 0)]] StructuredBuffer<uint4> W;
static const float KAPPA = PATCH_KAPPA;
static const float PI = 3.14159265;
static const uint WG = 256u;
// outputs a lane owns in the first fc layer
static const uint TCF = 4u;
// a patch's share of X and Y
static const uint XS = BUF_X / PATCHES;
static const uint YS = BUF_Y / PATCHES;

#if INT8
static const float QMAX = Q8_MAX;
static const float INV = Q8_INV;
// int8 codes: PATCHES patches a workgroup, X and Y their activations (uvec4s, four codes a word), red and redi scratch
typealias Act = uint4;
groupshared uint4 bufX[BUF_X];
groupshared uint4 bufY[BUF_Y];
groupshared float red[RED];
groupshared int redi[REDI];
int4 d4(uint4 v, uint w) { return int4(dotPacked4x8(int(v.x), int(w)), dotPacked4x8(int(v.y), int(w)), dotPacked4x8(int(v.z), int(w)), dotPacked4x8(int(v.w), int(w))); }
int4 d4l(uint4 v, uint w) { return int4(0, dotPacked4x8(int(v.y), int(w)), dotPacked4x8(int(v.z), int(w)), dotPacked4x8(int(v.w), int(w))); }
int4 d4r(uint4 v, uint w) { return int4(dotPacked4x8(int(v.x), int(w)), dotPacked4x8(int(v.y), int(w)), dotPacked4x8(int(v.z), int(w)), 0); }
int4 rq(int4 a, float m) { return f2i(clamp(roundEven(float4(a) * m), float4(0.0), float4(QMAX))); }
uint4 packq(int4 c0, int4 c1, int4 c2, int4 c3) {
  return uint4(pack4xI8(int4(c0.x, c1.x, c2.x, c3.x)), pack4xI8(int4(c0.y, c1.y, c2.y, c3.y)), pack4xI8(int4(c0.z, c1.z, c2.z, c3.z)), pack4xI8(int4(c0.w, c1.w, c2.w, c3.w)));
}
#else
// f16 or f32: PATCHES patches a workgroup, X and Y their activations (vec4s), red scratch
typealias Act = VEC4_T;
groupshared VEC4_T bufX[BUF_X];
groupshared VEC4_T bufY[BUF_Y];
groupshared float red[RED];
#endif

// X (B 0) or Y (B 1) at i
Act ldAct<let B : int>(uint i) { if (B == 0) { return bufX[i]; } return bufY[i]; }
#define PUT_ACT(B, I, V) if (B == 0) { bufX[I] = V; } else { bufY[I] = V; }

// The float weights: E elements a vec4 of words (8 halves or 4 floats; the int8 form's fc2 is f32). A run of TC
// consecutive weights from element sub of a vec4 (all of it where TC is E or more) as its block holds them, and its
// c-th as f32 at its use.
#if F16
static const int E = 8;
typealias Elt = half;
struct Blk { half4 a; half4 b; half2 h; half s; };
Blk block<let TC : int>(uint4 v, uint sub) {
  Blk k = {};
  if (TC >= 8) { k.a = half4(unpackFloat2x16(v.x), unpackFloat2x16(v.y)); k.b = half4(unpackFloat2x16(v.z), unpackFloat2x16(v.w)); }
  else if (TC == 4) { uint2 w = ((sub == 4u) ? v.zw : v.xy); k.a = half4(unpackFloat2x16(w.x), unpackFloat2x16(w.y)); }
  else {
    k.h = unpackFloat2x16(select(sub >= 4u, select((sub >> 1u) == 3u, v.w, v.z), select((sub >> 1u) == 1u, v.y, v.x)));
    if (TC == 1) { k.s = select((sub & 1u) == 1u, k.h.y, k.h.x); }
  }
  return k;
}
float at<let TC : int>(Blk k, int c) {
  if (TC >= 8) { if (c < 4) { return float(k.a[c]); } return float(k.b[c - 4]); }
  if (TC == 4) { return float(k.a[c]); }
  if (TC == 2) { return float(k.h[c]); }
  return float(k.s);
}
// element I (a lane's value, 0 to 7) of V as f32
#define ELEM(V, I) float(select(((I) & 1u) == 1u, unpackFloat2x16(select((I) >= 4u, select(((I) >> 1u) == 3u, V.w, V.z), select(((I) >> 1u) == 1u, V.y, V.x))).y, unpackFloat2x16(select((I) >= 4u, select(((I) >> 1u) == 3u, V.w, V.z), select(((I) >> 1u) == 1u, V.y, V.x))).x))
#else
static const int E = 4;
typealias Elt = float;
struct Blk { float4 a; float2 h; float s; };
Blk block<let TC : int>(uint4 v, uint sub) {
  Blk k = {};
  if (TC >= 4) { k.a = asfloat(v); }
  else if (TC == 2) { k.h = asfloat(((sub == 2u) ? v.zw : v.xy)); }
  else { k.s = asfloat(select(sub >= 2u, select(sub == 3u, v.w, v.z), select(sub == 1u, v.y, v.x))); }
  return k;
}
float at<let TC : int>(Blk k, int c) {
  if (TC >= 4) { return k.a[c]; }
  if (TC == 2) { return k.h[c]; }
  return k.s;
}
#define ELEM(V, I) asfloat(select((I) >= 2u, select((I) == 3u, V.w, V.z), select((I) == 1u, V.y, V.x)))
#endif

#if !INT8
// One conv layer (3 x 3, padding 1) in a float form: a lane's tile of TC channels x 4 TX positions from its slice of
// the input channels, the slices' partials summed (by shuffles with subgroups, else through red), ReLU, into the other
// buffer. SRC: the input X (0) or Y (1).
void conv<let IN_C : int, let IN_S : int, let OUT_C : int, let OUT_S : int, let STRIDE : int, let PIN : int, let POUT : int,
          let TC : int, let TX : int, let KS : int, let WO : int, let KX : int, let BO : int, let MO : int, let SRC : int>(uint li) {
  const int NX = OUT_S / (4 * TX), NB = OUT_C / TC, CI_PER = IN_C / KS, ROW_G = IN_S / 4;
  const int VPB = (TC + E - 1) / E, TE = TC < E ? TC : E;
  uint ks = li % uint(KS);
  uint r0 = li / uint(KS);
  uint xg = r0 % uint(NX);
  uint r1 = r0 / uint(NX);
  uint y = r1 % uint(OUT_S);
  uint r2 = r1 / uint(OUT_S);
  uint cb = r2 % uint(NB);
  uint p = r2 / uint(NB);
  uint cbE = cb * uint(TC);
  uint wvec = cbE / uint(E);
  uint wsub = cbE % uint(E);
  uint x4 = xg * uint(TX);
  uint src0 = p * (SRC == 0 ? XS : YS);
  uint dst0 = p * (SRC == 0 ? YS : XS);
  float4 a[TC][TX];
  [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { a[c][t] = float4(0.0); } }
  if (ks == 0u) {
    [ForceUnroll] for (int vb = 0; vb < VPB; vb++) {
      uint4 bv = W[uint(BO) + wvec + uint(vb)];
      Blk k = block<TC>(bv, wsub);
      [ForceUnroll] for (int c = 0; c < TE; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { a[vb * E + c][t] = float4(at<TC>(k, c)); } }
    }
  }
  for (uint ci = ks * uint(CI_PER); ci < (ks + 1u) * uint(CI_PER); ci += 1u) {
    [ForceUnroll] for (int ky = 0; ky < 3; ky++) {
      int iy = int(y) * STRIDE - 1 + ky;
      Elt okm = select(iy >= 0 && iy < IN_S, Elt(1.0), Elt(0.0));
      uint r = src0 + (ci * uint(IN_S) + uint(clamp(iy, 0, IN_S - 1))) * uint(PIN);
      float4 iv[3][TX];
      if (STRIDE == 1) {
        float4 m[TX];
        [ForceUnroll] for (int t = 0; t < TX; t++) { m[t] = float4(ldAct<SRC>(r + x4 + uint(t)) * okm); }
        float lf = select(x4 > 0u, float(ldAct<SRC>(r + max(x4, 1u) - 1u).w * okm), 0.0);
        float rt = select(x4 + uint(TX) < uint(ROW_G), float(ldAct<SRC>(r + min(x4 + uint(TX), uint(ROW_G - 1))).x * okm), 0.0);
        [ForceUnroll] for (int t = 0; t < TX; t++) { if (t == 0) { iv[0][t] = float4(lf, m[t].xyz); } else { iv[0][t] = float4(m[max(t - 1, 0)].w, m[t].xyz); } }
        [ForceUnroll] for (int t = 0; t < TX; t++) { iv[1][t] = m[t]; }
        [ForceUnroll] for (int t = 0; t < TX; t++) { if (t < TX - 1) { iv[2][t] = float4(m[t].yzw, m[min(t + 1, TX - 1)].x); } else { iv[2][t] = float4(m[t].yzw, rt); } }
      } else {
        float4 ma[TX], mb[TX];
        [ForceUnroll] for (int t = 0; t < TX; t++) { ma[t] = float4(ldAct<SRC>(r + 2u * (x4 + uint(t))) * okm); mb[t] = float4(ldAct<SRC>(r + 2u * (x4 + uint(t)) + 1u) * okm); }
        float lf = select(x4 > 0u, float(ldAct<SRC>(r + 2u * max(x4, 1u) - 1u).w * okm), 0.0);
        [ForceUnroll] for (int t = 0; t < TX; t++) { if (t == 0) { iv[0][t] = float4(lf, ma[t].y, ma[t].w, mb[t].y); } else { iv[0][t] = float4(mb[max(t - 1, 0)].w, ma[t].y, ma[t].w, mb[t].y); } }
        [ForceUnroll] for (int t = 0; t < TX; t++) { iv[1][t] = float4(ma[t].x, ma[t].z, mb[t].x, mb[t].z); }
        [ForceUnroll] for (int t = 0; t < TX; t++) { iv[2][t] = float4(ma[t].y, ma[t].w, mb[t].y, mb[t].w); }
      }
      uint wb = uint(WO) + ((ci * 3u + uint(ky)) * 3u) * uint(KX) + wvec;
      Blk k[3][2];
      [ForceUnroll] for (int kx = 0; kx < 3; kx++) { [ForceUnroll] for (int vb = 0; vb < VPB; vb++) { uint4 w = W[wb + uint(kx * KX + vb)]; k[kx][vb] = block<TC>(w, wsub); } }
      [ForceUnroll] for (int kx = 0; kx < 3; kx++) {
        [ForceUnroll] for (int vb = 0; vb < VPB; vb++) {
          [ForceUnroll] for (int c = 0; c < TE; c++) {
            [ForceUnroll] for (int t = 0; t < TX; t++) { a[vb * E + c][t] = fma(float4(at<TC>(k[kx][vb], c)), iv[kx][t], a[vb * E + c][t]); }
          }
        }
      }
    }
  }
#define CONV_FINISH \
  [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { \
    PUT_ACT(1 - SRC, dst0 + ((cbE + uint(c)) * uint(OUT_S) + y) * uint(POUT) + x4 + uint(t), VEC4_T(max(a[c][t], float4(0.0)))) \
  } }
  if (KS == 1) {
    CONV_FINISH
  } else if (SUBGROUPS == 1) {
    [ForceUnroll] for (int i = 0; i < 2; i++) {
      if ((1 << i) < KS) { [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { a[c][t] += shuffleXor(a[c][t], uint(1 << i)); } } }
    }
    if (ks == 0u) {
      CONV_FINISH
    }
  } else {
    uint ti = li / uint(KS);
    if (ks > 0u) {
      [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { [ForceUnroll] for (int e = 0; e < 4; e++) {
        red[((ks - 1u) * uint(WG / KS) + ti) * uint(4 * TC * TX) + uint(4 * (c * TX + t) + e)] = a[c][t][e];
      } } }
    }
    GroupMemoryBarrierWithGroupSync();
    if (ks == 0u) {
      [ForceUnroll] for (int s = 1; s < KS; s++) { [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) {
#define PART(E_) red[((uint(s) - 1u) * uint(WG / KS) + ti) * uint(4 * TC * TX) + uint(4 * (c * TX + t) + E_)]
        a[c][t] += float4(PART(0), PART(1), PART(2), PART(3));
#undef PART
      } } }
      CONV_FINISH
    }
  }
#undef CONV_FINISH
}

// The first fc layer as a GEMM over the last conv's buffer (SRC): a lane TCF outputs of a patch over its slice of the
// inputs, the slices summed, ReLU.
void fc1<let IN_N : int, let OUT_N : int, let S : int, let PITCH : int, let NB : int, let KSF : int, let SLICE : int, let WO : int,
         let KX : int, let BO : int, let MO : int, let OUT : int, let SRC : int>(uint li) {
  {
    uint ks = li % uint(KSF);
    uint r0 = li / uint(KSF);
    uint ob = r0 % uint(NB);
    uint p = r0 / uint(NB);
    uint src0 = p * (SRC == 0 ? XS : YS);
    uint o0 = ob * TCF;
    float f[4];
    [ForceUnroll] for (int c = 0; c < 4; c++) { f[c] = 0.0; }
    for (uint v = 0u; v < uint(SLICE / E); v++) {
      uint k = ks * uint(SLICE) + v * uint(E);
      uint kv = k / uint(E);
      float4 x[E / 4];
      [ForceUnroll] for (int q = 0; q < E / 4; q++) { uint kq = k + uint(4 * q); x[q] = float4(ldAct<SRC>(src0 + ((kq / uint(S * S)) * uint(S) + (kq % uint(S * S)) / uint(S)) * uint(PITCH) + (kq % uint(S)) / 4u)); }
      Blk w[4];
      [ForceUnroll] for (int c = 0; c < 4; c++) { uint4 wv = W[uint(WO) + (o0 + uint(c)) * uint(KX) + kv]; w[c] = block<E>(wv, 0u); }
      [ForceUnroll] for (int c = 0; c < 4; c++) {
        [ForceUnroll] for (int q = 0; q < E / 4; q++) { f[c] += dot(x[q], float4(at<E>(w[c], 4 * q), at<E>(w[c], 4 * q + 1), at<E>(w[c], 4 * q + 2), at<E>(w[c], 4 * q + 3))); }
      }
    }
    [ForceUnroll] for (int c = 0; c < 4; c++) { red[li * TCF + uint(c)] = f[c]; }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < PATCHES * uint(OUT_N)) {
    uint p = li / uint(OUT_N);
    uint o = li % uint(OUT_N);
    uint4 bv = W[uint(BO) + o / uint(E)];
    float acc = ELEM(bv, (o % uint(E)));
    for (uint s = 0u; s < uint(KSF); s++) { acc += red[((p * uint(NB) + o / TCF) * uint(KSF) + s) * TCF + o % TCF]; }
    red[uint(OUT) + li] = max(acc, 0.0);
  }
  GroupMemoryBarrierWithGroupSync();
}
#else
// One conv layer (3 x 3, padding 1) in int8: a lane's tile of TC channels x 4 TX positions, four input channels a
// word (the first layer's one channel three taps a word), the slices' partials summed (by shuffles with subgroups,
// else through redi), requantised by the channel's scale into the other buffer. A row's edge products are left out
// where a lane's positions span the whole row (NX 1). SRC: the input X (0) or Y (1).
void conv<let IN_C : int, let IN_S : int, let OUT_C : int, let OUT_S : int, let STRIDE : int, let PIN : int, let POUT : int,
          let TC : int, let TX : int, let KS : int, let WO : int, let KX : int, let BO : int, let MO : int, let SRC : int>(uint li) {
  const int NX = OUT_S / (4 * TX), NB = OUT_C / TC, VB = TC / 4, ROW_G = IN_S / 4;
  const bool EDGE = NX == 1;
  uint ks = li % uint(KS);
  uint r0 = li / uint(KS);
  uint xg = r0 % uint(NX);
  uint r1 = r0 / uint(NX);
  uint y = r1 % uint(OUT_S);
  uint r2 = r1 / uint(OUT_S);
  uint cb = r2 % uint(NB);
  uint p = r2 / uint(NB);
  uint wvec = cb * uint(VB);
  uint x4 = xg * uint(TX);
  uint src0 = p * (SRC == 0 ? XS : YS);
  uint dst0 = p * (SRC == 0 ? YS : XS);
  int4 a[TC][TX];
  [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { a[c][t] = int4(0); } }
  if (ks == 0u) {
    [ForceUnroll] for (int vb = 0; vb < VB; vb++) {
      int4 bv = int4(W[uint(BO) + wvec + uint(vb)]);
      [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb][t] = int4(bv.x); }
      [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 1][t] = int4(bv.y); }
      [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 2][t] = int4(bv.z); }
      [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 3][t] = int4(bv.w); }
    }
  }
  if (IN_C == 1) {
    [ForceUnroll] for (int ky = 0; ky < 3; ky++) {
      int iy = int(y) * STRIDE - 1 + ky;
      uint okm = select(iy >= 0 && iy < IN_S, 0xffffffffu, 0u);
      uint r = src0 + uint(clamp(iy, 0, IN_S - 1)) * uint(PIN);
      uint A[2], B[2], L0;
      uint4 v = ldAct<SRC>(r + x4 / 2u) & uint4(okm);
      if (TX == 1) {
        bool hi = (x4 & 1u) == 1u;
        A[0] = select(hi, v.z, v.x);
        B[0] = select(hi, v.w, v.y);
        L0 = select(hi, v.y, select(x4 > 0u, ldAct<SRC>(r + max(x4 / 2u, 1u) - 1u).w & okm, 0u));
      } else {
        A[0] = v.x;
        B[0] = v.y;
        A[1] = v.z;
        B[1] = v.w;
        L0 = select(x4 > 0u, ldAct<SRC>(r + max(x4 / 2u, 1u) - 1u).w & okm, 0u);
      }
      uint4 tp[TX];
      [ForceUnroll] for (int t = 0; t < TX; t++) {
        uint lo = t == 0 ? L0 : B[max(t - 1, 0)];
        tp[t] = uint4((lo >> 24u) | (A[t] << 8u), A[t] >> 8u, (A[t] >> 24u) | (B[t] << 8u), B[t] >> 8u);
      }
      uint4 w[VB];
      [ForceUnroll] for (int vb = 0; vb < VB; vb++) { w[vb] = W[uint(WO + ky * KX) + wvec + uint(vb)]; }
      [ForceUnroll] for (int vb = 0; vb < VB; vb++) {
        [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb][t] += d4(tp[t], w[vb].x); }
        [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 1][t] += d4(tp[t], w[vb].y); }
        [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 2][t] += d4(tp[t], w[vb].z); }
        [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 3][t] += d4(tp[t], w[vb].w); }
      }
    }
  } else {
    const int G_PER = IN_C / 4 / KS;
    for (uint g = ks * uint(G_PER); g < (ks + 1u) * uint(G_PER); g++) {
      [ForceUnroll] for (int ky = 0; ky < 3; ky++) {
        int iy = int(y) * STRIDE - 1 + ky;
        uint okm = select(iy >= 0 && iy < IN_S, 0xffffffffu, 0u);
        uint r = src0 + (g * uint(IN_S) + uint(clamp(iy, 0, IN_S - 1))) * uint(PIN);
        uint4 iv[3][TX];
        if (STRIDE == 1) {
          uint4 m[TX];
          [ForceUnroll] for (int t = 0; t < TX; t++) { m[t] = ldAct<SRC>(r + x4 + uint(t)) & uint4(okm); }
          uint lf = 0u, rt = 0u;
          if (!EDGE) {
            lf = select(x4 > 0u, ldAct<SRC>(r + max(x4, 1u) - 1u).w & okm, 0u);
            rt = select(x4 + uint(TX) < uint(ROW_G), ldAct<SRC>(r + min(x4 + uint(TX), uint(ROW_G - 1))).x & okm, 0u);
          }
          [ForceUnroll] for (int t = 0; t < TX; t++) { if (t == 0) { iv[0][t] = uint4(lf, m[t].xyz); } else { iv[0][t] = uint4(m[max(t - 1, 0)].w, m[t].xyz); } }
          [ForceUnroll] for (int t = 0; t < TX; t++) { iv[1][t] = m[t]; }
          [ForceUnroll] for (int t = 0; t < TX; t++) { if (t < TX - 1) { iv[2][t] = uint4(m[t].yzw, m[min(t + 1, TX - 1)].x); } else { iv[2][t] = uint4(m[t].yzw, rt); } }
        } else {
          uint4 ma[TX], mb[TX];
          [ForceUnroll] for (int t = 0; t < TX; t++) { ma[t] = ldAct<SRC>(r + 2u * (x4 + uint(t))) & uint4(okm); mb[t] = ldAct<SRC>(r + 2u * (x4 + uint(t)) + 1u) & uint4(okm); }
          uint lf = 0u;
          if (!EDGE) { lf = select(x4 > 0u, ldAct<SRC>(r + 2u * max(x4, 1u) - 1u).w & okm, 0u); }
          [ForceUnroll] for (int t = 0; t < TX; t++) { if (t == 0) { iv[0][t] = uint4(lf, ma[t].y, ma[t].w, mb[t].y); } else { iv[0][t] = uint4(mb[max(t - 1, 0)].w, ma[t].y, ma[t].w, mb[t].y); } }
          [ForceUnroll] for (int t = 0; t < TX; t++) { iv[1][t] = uint4(ma[t].x, ma[t].z, mb[t].x, mb[t].z); }
          [ForceUnroll] for (int t = 0; t < TX; t++) { iv[2][t] = uint4(ma[t].y, ma[t].w, mb[t].y, mb[t].w); }
        }
        uint wb = uint(WO) + ((g * 3u + uint(ky)) * 3u) * uint(KX) + wvec;
        uint4 w[3][VB];
        [ForceUnroll] for (int kx = 0; kx < 3; kx++) { [ForceUnroll] for (int vb = 0; vb < VB; vb++) { w[kx][vb] = W[wb + uint(kx * KX + vb)]; } }
#define CONV_DOT(KXI, T, WC) (EDGE && KXI == 0 && T == 0 ? d4l(iv[KXI][T], WC) : EDGE && STRIDE == 1 && KXI == 2 && T == TX - 1 ? d4r(iv[KXI][T], WC) : d4(iv[KXI][T], WC))
        [ForceUnroll] for (int kx = 0; kx < 3; kx++) {
          [ForceUnroll] for (int vb = 0; vb < VB; vb++) {
            [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb][t] += CONV_DOT(kx, t, w[kx][vb].x); }
            [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 1][t] += CONV_DOT(kx, t, w[kx][vb].y); }
            [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 2][t] += CONV_DOT(kx, t, w[kx][vb].z); }
            [ForceUnroll] for (int t = 0; t < TX; t++) { a[4 * vb + 3][t] += CONV_DOT(kx, t, w[kx][vb].w); }
          }
        }
#undef CONV_DOT
      }
    }
  }
#define CONV_FINISH \
  [ForceUnroll] for (int vb = 0; vb < VB; vb++) { \
    float4 mv = asfloat(W[uint(MO) + wvec + uint(vb)]); \
    [ForceUnroll] for (int t = 0; t < TX; t++) { \
      PUT_ACT(1 - SRC, dst0 + ((wvec + uint(vb)) * uint(OUT_S) + y) * uint(POUT) + x4 + uint(t), packq(rq(a[4 * vb][t], mv.x), rq(a[4 * vb + 1][t], mv.y), rq(a[4 * vb + 2][t], mv.z), rq(a[4 * vb + 3][t], mv.w))) \
    } \
  }
  if (KS == 1) {
    CONV_FINISH
  } else if (SUBGROUPS == 1) {
    [ForceUnroll] for (int i = 0; i < 2; i++) {
      if ((1 << i) < KS) { [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { a[c][t] += shuffleXor(a[c][t], uint(1 << i)); } } }
    }
    if (ks == 0u) {
      CONV_FINISH
    }
  } else {
    uint ti = li / uint(KS);
    if (ks > 0u) {
      [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) { [ForceUnroll] for (int e = 0; e < 4; e++) {
        redi[((ks - 1u) * uint(WG / KS) + ti) * uint(4 * TC * TX) + uint(4 * (c * TX + t) + e)] = a[c][t][e];
      } } }
    }
    GroupMemoryBarrierWithGroupSync();
    if (ks == 0u) {
      [ForceUnroll] for (int s = 1; s < KS; s++) { [ForceUnroll] for (int c = 0; c < TC; c++) { [ForceUnroll] for (int t = 0; t < TX; t++) {
#define PART(E_) redi[((uint(s) - 1u) * uint(WG / KS) + ti) * uint(4 * TC * TX) + uint(4 * (c * TX + t) + E_)]
        a[c][t] += int4(PART(0), PART(1), PART(2), PART(3));
#undef PART
      } } }
      CONV_FINISH
    }
  }
#undef CONV_FINISH
}

// The first fc layer in int8 over the last conv's words (SRC; its rows unpadded): a lane TCF outputs of a patch over
// its slice of the codes, 16 a vec4, the slices summed, dequantised by the output's scale, ReLU.
void fc1<let IN_N : int, let OUT_N : int, let S : int, let PITCH : int, let NB : int, let KSF : int, let SLICE : int, let WO : int,
         let KX : int, let BO : int, let MO : int, let OUT : int, let SRC : int>(uint li) {
  {
    uint ks = li % uint(KSF);
    uint r0 = li / uint(KSF);
    uint ob = r0 % uint(NB);
    uint p = r0 / uint(NB);
    uint src0 = p * (SRC == 0 ? XS : YS);
    uint o0 = ob * TCF;
    int f[4];
    [ForceUnroll] for (int c = 0; c < 4; c++) { f[c] = 0; }
    for (uint v = 0u; v < uint(SLICE); v++) {
      uint kv = ks * uint(SLICE) + v;
      uint4 x = ldAct<SRC>(src0 + kv);
      uint4 w[4];
      [ForceUnroll] for (int c = 0; c < 4; c++) { w[c] = W[uint(WO) + (o0 + uint(c)) * uint(KX) + kv]; }
      [ForceUnroll] for (int c = 0; c < 4; c++) { f[c] += dotPacked4x8(int(x.x), int(w[c].x)) + dotPacked4x8(int(x.y), int(w[c].y)) + dotPacked4x8(int(x.z), int(w[c].z)) + dotPacked4x8(int(x.w), int(w[c].w)); }
    }
    [ForceUnroll] for (int c = 0; c < 4; c++) { redi[li * TCF + uint(c)] = f[c]; }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li < PATCHES * uint(OUT_N)) {
    uint p = li / uint(OUT_N);
    uint o = li % uint(OUT_N);
    int acc = int(W[uint(BO) + o / 4u][o % 4u]);
    for (uint s = 0u; s < uint(KSF); s++) { acc += redi[((p * uint(NB) + o / TCF) * uint(KSF) + s) * TCF + o % TCF]; }
    red[uint(OUT) + li] = max(float(acc) * asfloat(W[uint(MO) + o / 4u][o % 4u]), 0.0);
  }
  GroupMemoryBarrierWithGroupSync();
}
#endif

// The second fc layer (floats; the int8 form's in f32): a lane an output of a patch, from the first's outputs in red.
void fc2<let IN_N : int, let OUT_N : int, let WO : int, let KX : int, let BO : int, let IN_AT : int, let OUT : int>(uint li) {
  if (li < PATCHES * uint(OUT_N)) {
    uint p = li / uint(OUT_N);
    uint o = li % uint(OUT_N);
    uint4 bv = W[uint(BO) + o / uint(E)];
    float acc = ELEM(bv, (o % uint(E)));
    for (uint v = 0u; v < uint((IN_N + E - 1) / E); v++) {
      uint k = v * uint(E);
      uint4 wv = W[uint(WO) + o * uint(KX) + v];
      Blk w = block<E>(wv, 0u);
      [ForceUnroll] for (int e = 0; e < E; e++) { if (uint(e) < uint(IN_N) - k) { acc += at<E>(w, e) * red[uint(IN_AT) + p * uint(IN_N) + k + uint(e)]; } }
    }
    red[uint(OUT) + li] = acc;
  }
  GroupMemoryBarrierWithGroupSync();
}

// a sample of pyramid level L at (x, y) of frame f (row: the level's row of y, levels 1 to 4)
float cutPix<let L : int>(int x, int y, uint row, uint f) {
  if (L == 0) { return img.Load(int4(x, y, int(f), 0)); }
  if (L == 1) { return lv1[row + uint(x)]; }
  if (L == 2) { return lv2[row + uint(x)]; }
  if (L == 3) { return lv3[row + uint(x)]; }
  return lv4[row + uint(x)];
}
// the cut at level L: a lane's four samples of a row of each of its patches, bilinear, and their sums
#define CUT(L) \
  [ForceUnroll] for (int k = 0; k < int(PATCHES); k++) { \
    uint q = lq + GROUP * uint(k); \
    float sy = float(q / 8u); \
    float sx = float((q % 8u) * 4u); \
    float py = (pk.y + u * (sy - 15.5)) * inv - 0.5; \
    float fy = floor(py); \
    float ay = py - fy; \
    int jy = f2i(fy); \
    int iy0 = clamp(jy, 0, lh - 1); \
    int iy1 = clamp(jy + 1, 0, lh - 1); \
    uint row0 = lbase + uint(iy0) * lsx; uint row1 = lbase + uint(iy1) * lsx; \
    [ForceUnroll] for (int t = 0; t < 4; t++) { \
      float px = (pk.x + u * (sx + float(t) - 15.5)) * inv - 0.5; \
      float fx = floor(px); \
      float ax = px - fx; \
      int jx = f2i(fx); \
      int ix0 = clamp(jx, 0, lw - 1); \
      int ix1 = clamp(jx + 1, 0, lw - 1); \
      s[k][t] = lerp(lerp(cutPix<L>(ix0, iy0, row0, f), cutPix<L>(ix1, iy0, row0, f), ax), lerp(cutPix<L>(ix0, iy1, row1, f), cutPix<L>(ix1, iy1, row1, f), ax), ay); \
    } \
    ssum += s[k].x + s[k].y + s[k].z + s[k].w; \
    ssq += dot(s[k], s[k]); \
  }

[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.y;
  if (frames[f].valid != 1u) { return; }
  uint n = min(counts[f * 16u + 2u], P.cap);
  uint j0 = wg.x * PATCHES;
  if (j0 >= n) { return; }
  uint cp = li / GROUP;
  uint lq = li % GROUP;
  uint jc = j0 + cp;
  uint ic = select(jc < n, jc, 0u);
  float4 pk = peaks[f * P.cap + ic];
  float u = pk.z / KAPPA;
  uint l = levelFor(u);
  float4 s[PATCHES];
  [ForceUnroll] for (int k = 0; k < int(PATCHES); k++) { s[k] = float4(0.0); }
  float ssum = 0.0;
  float ssq = 0.0;
  {
    Frame F = frames[f];
    int lw = int((F.w + (1u << l) - 1u) >> l);
    int lh = int((F.h + (1u << l) - 1u) >> l);
    uint4 ls = LD.d[l];
    uint lsx = ls.x;
    uint lbase = f * ls.y * lsx;
    float inv = 1.0 / float(1u << l);
    switch (l) {
      case 0u: { CUT(0) break; }
      case 1u: { CUT(1) break; }
      case 2u: { CUT(2) break; }
      case 3u: { CUT(3) break; }
      default: { CUT(4) }
    }
  }
  red[li] = ssum;
  red[256u + li] = ssq;
  GroupMemoryBarrierWithGroupSync();
  if (li < 32u) {
    float t1 = 0.0;
    float t2 = 0.0;
    for (uint k = 0u; k < 8u; k++) { t1 += red[li * 8u + k]; t2 += red[256u + li * 8u + k]; }
    red[512u + li] = t1;
    red[544u + li] = t2;
  }
  GroupMemoryBarrierWithGroupSync();
  float tsum = 0.0;
  float tsq = 0.0;
  for (uint k = 0u; k < (GROUP / 8u); k++) { tsum += red[512u + cp * (GROUP / 8u) + k]; tsq += red[544u + cp * (GROUP / 8u) + k]; }
  float mu = tsum / float(PATCH * PATCH);
  float sd = sqrt(max(tsq / float(PATCH * PATCH) - mu * mu, 0.0));
  // the patch normalised by its mean and spread into X (int8: its codes, four a word)
  [ForceUnroll] for (int k = 0; k < int(PATCHES); k++) {
    uint q = lq + GROUP * uint(k);
#if INT8
    bufX[cp * XS + (q / 8u) * PATCH_PITCH + (q % 8u) / 4u][q % 4u] = pack4xI8(f2i(clamp(roundEven(((s[k] - float4(mu)) / (sd + 0.02)) * INV), float4(-QMAX), float4(QMAX))));
#else
    bufX[cp * XS + (q / 8u) * PATCH_PITCH + (q % 8u)] = VEC4_T((s[k] - float4(mu)) / (sd + 0.02));
#endif
  }
  GroupMemoryBarrierWithGroupSync();
  conv<CONV1, 0>(li);
  GroupMemoryBarrierWithGroupSync();
#if CONVS > 1
  conv<CONV2, 1>(li);
  GroupMemoryBarrierWithGroupSync();
#endif
#if CONVS > 2
  conv<CONV3, 0>(li);
  GroupMemoryBarrierWithGroupSync();
#endif
#if CONVS > 3
  conv<CONV4, 1>(li);
  GroupMemoryBarrierWithGroupSync();
#endif
  fc1<FC1, CONVS % 2>(li);
  fc2<FC2>(li);
  if (li < PATCHES) {
    uint j = j0 + li;
    if (j < n) {
      uint i = j;
      float4 pw = peaks[f * P.cap + i];
      float uw = pw.z / KAPPA;
      float z0 = red[Z_AT + li * 5u];
      float z1 = red[Z_AT + li * 5u + 1u];
      float z2 = red[Z_AT + li * 5u + 2u];
      float z3 = red[Z_AT + li * 5u + 3u];
      float z4 = red[Z_AT + li * 5u + 4u];
      float t = atan2(z2, z1);
      float pr = 1.0 / (1.0 + exp(-z0));
      uint o3 = 3u * (f * P.cap + i);
      readings[o3] = float4(pw.xy, uw * exp(z3), t - PI / 4.0);
      readings[o3 + 1u] = float4(pr, 0.0, 0.0, 0.0);
      readings[o3 + 2u] = float4(select(z4 > 0.0, 1.0, 0.0), pw.w, float(levelFor(uw)), 1.0);
    }
  }
}
