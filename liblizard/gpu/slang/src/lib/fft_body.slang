// The FFT's stages (lib/fft.slang says what they are), included in the pass's body where its lanes' points (FFT_V0,
// and FFT_V1 where FFT_LANES_2: arrays of eight), their workgroup buffers (FFT_BUF0, FFT_BUF1) and its lane index tid
// are. A lane's statements are macros of its points and its index (FFT_EACH runs one for each lane), so that each
// lane's points stay an array of their own and each buffer access reads as written.
#define FFT_LD(L, I) LD(FFT_BUF##L[I])
#define FFT_ST(L, I, X) FFT_BUF##L[I] = ST(X)
// an odd butterfly of R (3 or 5) points p, its outputs written D, D + Ns, ... as it makes them
#define FFT_BFLY(L, D) \
  if (R == 3) { \
    float2 t1 = p[1] + p[2]; float2 t2 = mneg_i(p[1] - p[2]) * S3; float2 mm = p[0] - 0.5 * t1; \
    FFT_ST(L, D, p[0] + t1); FFT_ST(L, D + uint(Ns), mm + t2); FFT_ST(L, D + uint(2 * Ns), mm - t2); \
  } else { \
    float2 t1 = p[1] + p[4]; float2 t2 = p[2] + p[3]; float2 t3 = p[1] - p[4]; float2 t4 = p[2] - p[3]; \
    float2 m1 = p[0] + FFT_C5A * t1 - FFT_C5B * t2; float2 m2 = p[0] - FFT_C5B * t1 + FFT_C5A * t2; \
    float2 n1 = mneg_i(FFT_S5A * t3 + FFT_S5B * t4); float2 n2 = mneg_i(FFT_S5B * t3 - FFT_S5A * t4); \
    FFT_ST(L, D, p[0] + t1 + t2); FFT_ST(L, D + uint(Ns), m1 + n1); FFT_ST(L, D + uint(4 * Ns), m1 - n1); \
    FFT_ST(L, D + uint(2 * Ns), m2 + n2); FFT_ST(L, D + uint(3 * Ns), m2 - n2); \
  }
#if FFT_LANES_2
#define FFT_EACH(M) M(FFT_V0, 0) M(FFT_V1, 1)
#else
#define FFT_EACH(M) M(FFT_V0, 0)
#endif
#define FFT_FIRST(V, L) dft8(V); [ForceUnroll] for (int r = 0; r < 8; r++) FFT_ST(L, tid * 8u + uint(r), V[r]);
#define FFT_LOADS(V, L) [ForceUnroll] for (int r = 0; r < 8; r++) if (r < R) V[q * R + r] = FFT_LD(L, (tid + uint(q) * T) + uint(r * span));
#define FFT_POW2(V, L) \
  [ForceUnroll] for (int r = 1; r < 8; r++) if (r < R) V[q * R + r] = cmul(V[q * R + r], w[r]); \
  if (R == 8) dft8(V); else if (R == 4) dft4(V, uint(q * R)); else dft2(V, uint(q * R)); \
  [ForceUnroll] for (int r = 0; r < 8; r++) if (r < R) FFT_ST(L, d + uint(r * Ns), V[q * R + r]);
#define FFT_GDECL(V, L) float2 g##L[16];
#define FFT_GLOADS(V, L) [ForceUnroll] for (int r = 0; r < 5; r++) if (r < R) g##L[q * R + r] = FFT_LD(L, (tid + uint(q) * T) + uint(r * M));
#define FFT_ODD(V, L) { \
  float2 p[5]; \
  [ForceUnroll] for (int r = 0; r < 5; r++) if (r < R) { if (r == 0) p[r] = g##L[q * R]; else p[r] = cmul(g##L[q * R + r], w[r]); } \
  FFT_BFLY(L, d) }
#define FFT_LAST(V, L) { \
  float2 p[5]; \
  [ForceUnroll] for (int r = 0; r < 5; r++) if (r < R) { if (r == 0) p[r] = FFT_LD(L, j); else p[r] = cmul(FFT_LD(L, j + uint(r * Ns)), w[r]); } \
  FFT_BFLY(L, j) }
{
  // stage 1: radix 8
  FFT_EACH(FFT_FIRST)
  GroupMemoryBarrierWithGroupSync();
  int m = int(N) / 8, Ns = 8;
  [ForceUnroll] for (int s = 1; s < 8; s++) {
    if (m > 1) {
      const int R = m % 2 == 0 ? (m % 8 == 0 ? 8 : m % 4 == 0 ? 4 : 2) : m % 5 == 0 ? 5 : 3;
      if (R != 3 && R != 5) {
        // a power-of-two stage: each lane its 8 / R butterflies of R points
        const int per = 8 / R, span = int(N) / R, step = int(N) / (Ns * R);
        [ForceUnroll] for (int q = 0; q < 4; q++) if (q < per) { FFT_EACH(FFT_LOADS) }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (int q = 0; q < 4; q++) if (q < per) {
          uint j = tid + uint(q) * T;
          uint k = j & uint(Ns - 1);
          float2 w[8];
          w[1] = tw[k * uint(step)];
          if (R >= 4) { w[2] = cmul(w[1], w[1]); w[3] = cmul(w[2], w[1]); }
          if (R == 8) { w[4] = cmul(w[2], w[2]); w[5] = cmul(w[4], w[1]); w[6] = cmul(w[3], w[3]); w[7] = cmul(w[4], w[3]); }
          uint d = (j - k) * uint(R) + k;
          FFT_EACH(FFT_POW2)
        }
        GroupMemoryBarrierWithGroupSync();
      } else if (m != R) {
        // an odd stage that is not the last: its N / R butterflies over the lanes, Q a lane
        const int M = int(N) / R, Q = (M + int(T) - 1) / int(T), step = int(N) / (Ns * R);
        {
          FFT_EACH(FFT_GDECL)
          [ForceUnroll] for (int q = 0; q < 4; q++) if (q < Q) {
            if ((q + 1) * int(T) <= M || tid + uint(q) * T < uint(M)) { FFT_EACH(FFT_GLOADS) }
          }
          GroupMemoryBarrierWithGroupSync();
          [ForceUnroll] for (int q = 0; q < 4; q++) if (q < Q) {
            if ((q + 1) * int(T) <= M || tid + uint(q) * T < uint(M)) {
              uint j = tid + uint(q) * T;
              uint k;
              if ((Ns & (Ns - 1)) == 0) k = j & uint(Ns - 1); else k = j % uint(Ns);
              uint d = (j - k) * uint(R) + k;
              float2 w[5];
              [ForceUnroll] for (int r = 1; r < 5; r++) if (r < R) w[r] = tw[k * uint(r * step)];
              FFT_EACH(FFT_ODD)
            }
          }
          GroupMemoryBarrierWithGroupSync();
        }
      } else {
        // the last stage, odd: in place over the buffer
        for (uint j = tid; j < uint(Ns); j += T) {
          float2 w[5];
          [ForceUnroll] for (int r = 1; r < 5; r++) if (r < R) w[r] = tw[uint(r) * j];
          FFT_EACH(FFT_LAST)
        }
        GroupMemoryBarrierWithGroupSync();
      }
      Ns *= R;
      m /= R;
    }
  }
}
#undef FFT_LD
#undef FFT_ST
#undef FFT_BFLY
#undef FFT_EACH
#undef FFT_FIRST
#undef FFT_LOADS
#undef FFT_POW2
#undef FFT_GDECL
#undef FFT_GLOADS
#undef FFT_ODD
#undef FFT_LAST
