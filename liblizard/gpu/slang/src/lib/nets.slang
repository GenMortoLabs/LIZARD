// The nets' word-level operations (the classifier's and the proposer's generated code): four signed bytes packed low byte
// first (pack4xI8: the low bytes shifted together, the form the Adreno 840 reads right), their dot product
// (dotPacked4x8: SPIR-V's OpSDot, packed 4 x 8, on the two words), two halves from a word and back (bit casts), and a
// lane's value from the lane whose index differs by m (shuffleXor: the subgroup's shuffle at the lane index XOR m).
uint pack4xI8(int4 v) { uint4 u = uint4(v) & uint4(0xffu); return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u); }
// the dot product as SPIR-V 1.3 with SPV_KHR_integer_dot_product (slangc's dot4add_i8packed raises a module to 1.4,
// past the host's Vulkan 1.1 floor); WGSL's dot4I8Packed elsewhere
int dotPacked4x8(int a, int b) {
  __target_switch {
  case spirv:
    return spirv_asm {
      OpCapability DotProduct;
      OpCapability DotProductInput4x8BitPacked;
      OpExtension "SPV_KHR_integer_dot_product";
      result:$$int = OpSDot $a $b PackedVectorFormat4x8Bit;
    };
  default:
    return dot4add_i8packed(uint(a), uint(b), 0);
  }
}
half2 unpackFloat2x16(uint w) { return bit_cast<half2>(w); }
uint packFloat2x16(half2 h) { return bit_cast<uint>(h); }
float4 shuffleXor(float4 v, uint m) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ m); }
int4 shuffleXor(int4 v, uint m) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ m); }
float shuffleXor(float v, uint m) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ m); }
int shuffleXor(int v, uint m) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ m); }
// GLSL.std.450's UnpackHalf2x16 and PackHalf2x16: two binary16 halves of a word as floats (the low half x) and back
float2 unpackHalf2x16(uint w) {
  __target_switch {
  case spirv: return spirv_asm { OpExtInst $$float2 result glsl450 UnpackHalf2x16 $w };
  default: return float2(f16tof32(w & 0xffffu), f16tof32(w >> 16u));
  }
}
