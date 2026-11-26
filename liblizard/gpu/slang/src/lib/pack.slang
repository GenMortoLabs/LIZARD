// GLSL.std.450's PackHalf2x16 and UnpackHalf2x16: two floats as one word of two binary16 halves (round to nearest
// even, x in the low half) and back; WGSL's own pack and unpack elsewhere. The stages' f16 stores (pass 1), the nets'
// half words (lib/nets.slang) and the sender's Y between its two inverses (send_irows.slang, send_ipic.slang).
uint packHalf2x16(float2 v) {
  __target_switch {
  case spirv: return spirv_asm { OpExtInst $$uint result glsl450 PackHalf2x16 $v };
  default: return f32tof16(v.x) | (f32tof16(v.y) << 16u);
  }
}
float2 unpackHalf2x16(uint w) {
  __target_switch {
  case spirv: return spirv_asm { OpExtInst $$float2 result glsl450 UnpackHalf2x16 $w };
  default: return float2(f16tof32(w & 0xffffu), f16tof32(w >> 16u));
  }
}
