// ai: The packed int8 dot product's ceiling (today's int8 kernels, dot4I8Packed): each lane 8 independent chains of R
// ai: dot products of 4 int8 pairs, each accumulated with saturation (OpSDotAccSat). MACs a dispatch = invocations x 8 x R
// ai: x 4. Built by hand into the tree's spv/ (gen writes no bench): slangc bench_dot.slang -target spirv -profile spirv_1_3
// ai: -entry main -stage compute -O2 -o <out>/spv/bench_dot.spv
[[vk::constant_id(1)]] const uint R = 256;
[[vk::binding(0, 0)]] RWStructuredBuffer<int> s;
[[vk::binding(1, 0)]] RWStructuredBuffer<int> d;
int dotAcc(int a, int b, int c) {
  return spirv_asm {
    OpCapability DotProduct; OpCapability DotProductInput4x8BitPacked; OpExtension "SPV_KHR_integer_dot_product";
    result:$$int = OpSDotAccSat $a $b $c PackedVectorFormat4x8Bit;
  };
}
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  int x = s[i & 255u], y = s[(i + 7u) & 255u];
  int a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
  for (uint r = 0; r < R; r++) {
    a0 = dotAcc(x, y, a0); a1 = dotAcc(x ^ 1, y, a1); a2 = dotAcc(x ^ 2, y, a2); a3 = dotAcc(x ^ 3, y, a3);
    a4 = dotAcc(x, y ^ 1, a4); a5 = dotAcc(x, y ^ 2, a5); a6 = dotAcc(x, y ^ 3, a6); a7 = dotAcc(x ^ 5, y, a7);
  }
  d[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
