// The web sender's PRESENT, its vertex stage: one triangle over the canvas, no vertex buffer (the fragment stage,
// send_present_fs.slang, shows the frame). The web only: a native sender presents through core/tx/.
[shader("vertex")]
float4 main(uint i : SV_VertexID) : SV_Position {
  float2 p = float2(float((i << 1u) & 2u), float(i & 2u));
  return float4(p * 2.0 - 1.0, 0.0, 1.0);
}
