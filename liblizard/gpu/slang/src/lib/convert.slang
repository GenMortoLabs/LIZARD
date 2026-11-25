// Float to integer, saturating: a float clamped in f32 to [-2147483648, 2147483520] for int, [0, 4294967040] for uint
// before the conversion (a conversion of a float out of range is undefined), so a degenerate value (a frame's map gone
// to infinity) converts to the bound; the stages first written in WGSL, whose conversions saturate so, kept it. The
// overloads a stage never calls cost nothing.
int f2i(float x) { return int(clamp(x, -2147483648.0, 2147483520.0)); }
int2 f2i(float2 x) { return int2(clamp(x, float2(-2147483648.0), float2(2147483520.0))); }
int4 f2i(float4 x) { return int4(clamp(x, float4(-2147483648.0), float4(2147483520.0))); }
uint f2u(float x) { return uint(clamp(x, 0.0, 4294967040.0)); }
uint2 f2u(float2 x) { return uint2(clamp(x, float2(0.0), float2(4294967040.0))); }
