// A frame of a batch: (w, h, axis, valid), 16 B; axis the camera's optical axis in the frame (plan/common.mjs
// axisWord), 0 the frame's centre.
struct Frame { uint w; uint h; uint axis; uint valid; };
