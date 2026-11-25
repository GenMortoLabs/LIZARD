// ai: A camera frame's crop developed to the 8-bit luma the decoder reads (develop.h, whose CPU develop this matches
// ai: byte for byte): the crop's samples as the host staged them (stageRaw: a header, then rows `pitch` bytes apart),
// ai: each to light by its 2 x 2 position's black level and scale, clipped to [0, 4096], a 2 x 2 bin summed in the
// ai: CPU's order and quartered, then the gamma table's step; P010 its samples' top bytes. Four pixels an invocation,
// ai: in raster order, into a word of the output (w x h bytes), which the batch copies into the slot's layer. The light's
// ai: arithmetic `precise` (no fused multiply-add), as the CPU's.
[[vk::binding(0, 0)]] StructuredBuffer<uint> st;
[[vk::binding(1, 0)]] StructuredBuffer<uint> lut;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> dst;

static uint kind, w, h, bin, pitch, ps, off, cfaRow;
static float4 black, scale;

uint byteAt(uint b) { return (st[b >> 2] >> ((b & 3u) * 8u)) & 0xffu; }
uint lutAt(uint i) { return (lut[i >> 2] >> ((i & 3u) * 8u)) & 0xffu; }

// ai: sample x of staged row y (develop.cpp rawAt's layouts)
uint sampleAt(uint y, uint x) {
  let b = 64u + y * pitch;
  if (kind == 1u) { let p = b + x * ps; return byteAt(p) | (byteAt(p + 1u) << 8); }
  if (kind == 2u) { let g = b + (x >> 2) * 5u; let k = x & 3u; return (byteAt(g + k) << 2) | ((byteAt(g + 4u) >> (2u * k)) & 3u); }
  // ai: RAW12 on the S26's Adreno reads wrong (2026-11-24, the GLSL's form and this alike, exact on the 4090): 7.3 M of
  // ai: 31.6 M bytes apart from the CPU's at bin 1 (most by 255), 563 blocks against 696; written as raw10's shift
  // ai: (byte x & 1, byte 2 >> 4 (x & 1)) 696 blocks but 4.3 M bytes apart (most by 27), at -O0 the same. Open.
  let g = b + (x >> 1) * 3u;
  return (x & 1u) != 0u ? ((byteAt(g + 1u) << 4) | (byteAt(g + 2u) >> 4)) : ((byteAt(g) << 4) | (byteAt(g + 2u) & 15u));
}

float lin(uint v, uint p) {
  precise float f = (float(v) - black[p]) * scale[p];
  return f <= 0.0 ? 0.0 : (f >= 4096.0 ? 4096.0 : f);
}

uint pixel(uint u, uint v) {
  if (kind == 0u) return byteAt(64u + v * pitch + (off + u) * ps + 1u);
  if (bin == 2u) {
    let r0 = 2u * v; let r1 = r0 + 1u; let c0 = off + 2u * u; let c1 = c0 + 1u;
    let p0 = ((cfaRow + r0) & 1u) << 1; let p1 = ((cfaRow + r1) & 1u) << 1;
    precise float s = lin(sampleAt(r0, c0), p0 | (c0 & 1u)) + lin(sampleAt(r0, c1), p0 | (c1 & 1u));
    s = s + lin(sampleAt(r1, c0), p1 | (c0 & 1u));
    s = s + lin(sampleAt(r1, c1), p1 | (c1 & 1u));
    return lutAt(uint(s * 0.25));
  }
  let c = off + u;
  return lutAt(uint(lin(sampleAt(v, c), (((cfaRow + v) & 1u) << 1) | (c & 1u))));
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  kind = st[0]; w = st[1]; h = st[2]; bin = st[3]; pitch = st[4]; ps = st[5]; off = st[6]; cfaRow = st[7];
  black = float4(asfloat(st[8]), asfloat(st[9]), asfloat(st[10]), asfloat(st[11]));
  scale = float4(asfloat(st[12]), asfloat(st[13]), asfloat(st[14]), asfloat(st[15]));
  let g = gid.x; let n = w * h;
  if (g * 4u >= n) return;
  uint word = 0u;
  for (uint j = 0u; j < 4u; j++) {
    let i = g * 4u + j;
    if (i < n) word |= pixel(i % w, i / w) << (8u * j);
  }
  dst[g] = word;
}
