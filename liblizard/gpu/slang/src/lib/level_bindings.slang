// The pyramid's levels and the frames, bindings 0 to 6 (register's level reader binds them too). levels.slang reads
// every level by level-0 coordinates; a stage including it includes convert.slang and frame.slang.
[[vk::binding(0, 0)]] Texture2DArray<float> img;
[[vk::binding(1, 0)]] StructuredBuffer<float> lv1;
[[vk::binding(2, 0)]] StructuredBuffer<float> lv2;
[[vk::binding(3, 0)]] StructuredBuffer<float> lv3;
[[vk::binding(4, 0)]] StructuredBuffer<float> lv4;
[[vk::binding(5, 0)]] StructuredBuffer<Frame> frames;
struct LevelDimsBlock { uint4 d[5]; };
[[vk::binding(6, 0)]] ConstantBuffer<LevelDimsBlock> LD;
