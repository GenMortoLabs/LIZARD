// Round to nearest, ties to even (SPIR-V's GLSL.std.450 RoundEven; WGSL's round is that): Slang's round is
// GLSL.std.450 Round, whose tie direction is the driver's.
float roundEven(float x) {
  __target_switch {
  case spirv: return spirv_asm { OpExtInst $$float result glsl450 RoundEven $x };
  case wgsl: return round(x);
  }
}
float2 roundEven(float2 x) {
  __target_switch {
  case spirv: return spirv_asm { OpExtInst $$float2 result glsl450 RoundEven $x };
  case wgsl: return round(x);
  }
}
float4 roundEven(float4 x) {
  __target_switch {
  case spirv: return spirv_asm { OpExtInst $$float4 result glsl450 RoundEven $x };
  case wgsl: return round(x);
  }
}
