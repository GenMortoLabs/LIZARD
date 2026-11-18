// ai: A stream file (2026-11-18): a configured sender's frames painted ahead into a .lizard whose ZIP comment is
// ai: "LIZARD stream 1" (rx/lizfile.h Kind::Stream), with a meta.json a player shows them by. What liz_sender_export
// ai: (the C API) and the apps' Export call; a player shows the frames in turn at fps and loops, painting nothing, and
// ai: any receiver's Upload decodes the file as frames.
#pragma once
#include <cstdint>
#include <functional>
#include <string>

#include "lizfile.h"

namespace lizard {

class Sender;

// ai: what meta.json says of the stream beyond what the sender knows: the file's name and type as the sender was
// ai: made with them, the format as the caller names it (blocks, ring, fps, codes) and its blocks a frame
struct StreamInfo {
  std::string name, type;
  int blocks = 0, ring = 0, fps = 0, codes = 1, frameBlocks = 0;
};

// ai: The sender's frames as a stream file at path (Sender::exportStart with no screen: from the first frame made once
// ai: the file's lap is known, `frames` of them, made odd, or 0: 2.5 laps), every frame packed on the machine's cores
// ai: as `packing` says. progress(done, frames) as frames are written (frames 0 until the lap is known): false stops the
// ai: export (the .part removed). Returns the frames written, or -1 with why. Blocks: not on a UI thread.
int64_t exportStream(Sender& s, const StreamInfo& info, const std::string& path, int frames, const std::function<bool(int, int)>& progress,
                     std::string& why, const lizfile::Packing& packing = {});

// ai: a stream file's meta.json for the sender's last `frames` frames (its geometry, stats, transfer and colours): a
// ai: replay's fields (w, h, frames, sizes, ms, layout; axes at 2:1), so every reader of a .lizard reads a stream, and
// ai: the stream's own (kind, pictures, gap, fps, format, file, test, colours, seed, painter)
std::string streamMeta(Sender& s, const StreamInfo& info, int frames);

}  // namespace lizard
