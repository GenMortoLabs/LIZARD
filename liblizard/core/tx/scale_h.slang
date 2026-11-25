// ai: The desktop presenter's stretch by Lanczos-3 (2026-11-17, Scaling's Lanczos beside the blit's linear filter;
// ai: lizard-desktop/native/presenter.cpp), its first pass, across, on the grey picture: the slot's buffer (the sender's
// ai: ring frame as painted, four pixels a word, rows rw words apart, sw x sh) to dw x sh floats (0 to 255). Each
// ai: destination column's taps from the presenter's table (stretchAxis: the column's first source pixel, then its
// ai: weights: Lanczos-3 widened by the source's pixels a destination pixel where it shrinks, so the shrink does not
// ai: alias, normalised, zero past its support); the source's edge pixels repeated past it. A workgroup is 128
// ai: destination columns over ROWS rows: their weights read together into shared memory once and each thread's own
// ai: then held in registers (each thread reading its own from the buffer, 13 floats from its neighbour's, and a loop
// ai: of dependent shared-memory reads: 7 ms on the iGPU), each row's source pixels loaded once into shared memory;
// ai: read straight from the buffers only where a shrink past about 2.5 gives a column more than TR taps or one past
// ai: about 15 makes the span longer than SPAN. Scaling Lanczos's kernel (presenter.h Scaling), the grey picture alone: a
// ai: code in Colours coloured in scale_v.slang's last step. One channel: an RGBA picture took four times the memory and
// ai: three times the arithmetic (10 ms across on the iGPU).
struct P { int sw; int sh; int dw; int dh; int tx; int ty; int yo; int rw; uint ink; uint paper; };
[[vk::push_constant]] ConstantBuffer<P> p;
[[vk::binding(0, 0)]] StructuredBuffer<uint> g;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> m;
[[vk::binding(2, 0)]] StructuredBuffer<float> t;
static const int SPAN = 1024;
static const int ROWS = 16;
static const int TR = 16;
groupshared float row[SPAN];
float px(int x, int y) {
  x = clamp(x, 0, p.sw - 1);
  return float((g[y * p.rw + (x >> 2)] >> (8u * uint(x & 3))) & 0xffu);
}
[numthreads(128, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
  int l = int(li), o0 = int(wg.x) * 128, ox = o0 + l, n = min(128, p.dw - o0), st = p.tx + 1;
  // ai: each thread's own weights into registers once for ROWS rows: a fixed TR taps unrolled, past its own taps weight
  // ai: 0 (a column with more taps than TR, a shrink past about 2.5, takes the plain loop over the buffers); shared
  // ai: memory holds the row alone, so more groups run at once
  bool wIn = p.tx <= TR;
  int lo0 = int(t[o0 * st]), span = int(t[(o0 + n - 1) * st]) - lo0 + p.tx;
  bool rIn = span <= SPAN;
  float w[TR] = {};   // ai: read only where written (wIn, l < n); zero so the compiler sees every path
  int off = 0;
  if (wIn && l < n) {
    off = int(t[ox * st]) - lo0;
    [ForceUnroll] for (int k = 0; k < TR; k++) w[k] = k < p.tx ? t[ox * st + 1 + k] : 0.0;
  }
  int y0 = int(wg.y) * ROWS;
  for (int r = 0; r < ROWS; r++) {
    int y = y0 + r;
    if (y >= p.sh) break;   // ai: the same for the whole group
    GroupMemoryBarrierWithGroupSync();   // ai: the row before read
    if (rIn) {
      if (lo0 >= 0 && lo0 + span <= p.sw) {
        // ai: a word a thread, its four pixels into the row (each pixel's own load fetched every word four times: 1.8 ms
        // ai: of the pass on the iGPU)
        int w0 = lo0 >> 2, w1 = (lo0 + span - 1) >> 2, base = y * p.rw;
        for (int i = w0 + l; i <= w1; i += 128) {
          uint wd = g[base + i];
          for (int j = 0; j < 4; j++) {
            int q = 4 * i + j - lo0;
            if (q >= 0 && q < span) row[q] = float((wd >> (8u * uint(j))) & 0xffu);
          }
        }
      } else {
        for (int i = l; i < span; i += 128) row[i] = px(lo0 + i, y);   // ai: at the picture's edges, each pixel clamped
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (l >= n) continue;
    float acc = 0.0;
    if (wIn && rIn) {
      // ai: past a column's own taps (weight 0) the index held inside the row as loaded: shared memory past it holds
      // ai: whatever an earlier dispatch left, and 0 times a NaN there is a NaN
      [ForceUnroll] for (int k = 0; k < TR; k++) acc += w[k] * row[min(off + k, span - 1)];
    } else {
      int b = ox * st, lo = int(t[b]);
      for (int k = 0; k < p.tx; k++) acc += t[b + 1 + k] * px(lo + k, y);
    }
    m[y * p.dw + ox] = acc;
  }
}
