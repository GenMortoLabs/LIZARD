// GLSL.std.450's PackHalf2x16 (each component to binary16, round to nearest even, x in the low half), the stages' f16
// stores of two floats as one word.
uint packHalf2x16(float2 v) {
  return spirv_asm { OpExtInst $$uint result glsl450 PackHalf2x16 $v };
}
