// ai: The default client's compression (2026-11-14, header version 4: the transfer carries a file as it is and
// ai: compresses nothing; how a client compresses is the client's). The apps' rule and the web's (the threaded wasm module's,
// ai: since 2026-11-25; `sim/pack.mjs` its JS twin before): a file goes as a zstd stream of independent frames, each of PACK_FRAME bytes of the file
// ai: (liblizard/zstd/shim.h, the level and parameters pinned, so every client packs a file to the same bytes and the
// ai: same root), named <name>.zst and its description saying compression 1 (src/xfer.h XFER_COMP_ZSTD, an elective
// ai: key), where it pays: a sample of the file (three slices of 1 MiB, its start, middle and end; the whole file
// ai: under 3 MiB) compressed to 90% or less, and the whole stream to 97% or less; else the file goes as it is. Any
// ai: zstd reads the stream (a receiver that does not know the key keeps the .zst); a LIZARD receiver unpacks a file
// ai: whose description says compression 1 (unpack) and files it under its name less the .zst. The frames are made a worker a physical core, in order on the
// ai: disk, at most two a worker held in memory.
#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace lizard {

constexpr size_t PACK_FRAME = size_t{1} << 22;     // ai: 4 MiB of the file a frame
constexpr size_t PACK_FRAME_MAX = size_t{1} << 26; // ai: the most a frame may hold for unpack (another client's frames)

struct Packed {
  bool packed = false;            // ai: the stream at path (compression 1), else the file goes as it is (path and name the caller's)
  std::string path, name;         // ai: what the transfer carries: packed, the name <name>.zst
  uint64_t size = 0;              // ai: its bytes
  uint64_t original = 0;          // ai: the file's own bytes
};

// ai: The file at `path` (UTF-8), its name as the caller would send it, packed into a new file in `dir`
// ai: where that pays (the caller removes it after the send), else returned as it is. Throws std::runtime_error where
// ai: the file cannot be read or the stream not written. `stopped`, where given, is asked before the sample and before
// ai: each round of frames (a worker's 4 MiB each): true stops the pack, its stream removed, std::runtime_error
// ai: ("cancelled") thrown (a make the app let go, 2026-11-16).
Packed pack(const std::string& path, const std::string& name, const std::string& dir, const std::function<bool()>& stopped = {});
// ai: The same over a reader (2026-11-25, the web's picked file, which no path names): read(off, n, into) puts the n
// ai: bytes of the file at off into `into`, true when whole, from the pack's threads, several at once (xfer_tx.h
// ai: XferTx::Reader); length the file's bytes. path is the stream's where it pays, else "" (the file goes as it is,
// ai: read as the caller reads it).
Packed pack(const std::function<bool(uint64_t, size_t, uint8_t*)>& read, uint64_t length, const std::string& name, const std::string& dir,
            const std::function<bool()>& stopped = {});

// ai: The stream at `from` unpacked into a new file `to` (UTF-8 paths): "" or why not (a frame damaged or cut short,
// ai: one not saying its bytes or holding more than PACK_FRAME_MAX, a write refused; `to` is removed then).
std::string unpack(const std::string& from, const std::string& to);

// ai: A packed stream read back as the file it holds, in order, one frame in memory at a time (unpack's reader; the
// ai: Android app streams a take through it straight into Downloads, 2026-11-15). open walks the frames' headers and
// ai: their blocks' (3 bytes a block of 128 KiB at most; none of the frames' bytes read) for each frame's place and the
// ai: file's size; take decompresses a frame when the one in hand is spent.
class Unpacker {
 public:
  // ai: The stream at `path` (UTF-8): "" or why not, as unpack's (a frame not saying its bytes or holding more than
  // ai: PACK_FRAME_MAX, one cut short or past zstd's bound for its bytes, the file not read).
  std::string open(const std::string& path);
  // ai: the file's bytes, the frames' sum
  uint64_t size() const { return total_; }
  // ai: Up to n (at least 1) bytes of the file at the read point, valid until the next call; n set to their count, 0
  // ai: at the end. Throws std::runtime_error where a frame is damaged or the stream cannot be read.
  const uint8_t* take(size_t& n);

 private:
  struct Frame {
    uint64_t at;
    uint32_t bytes, content;
  };
  std::ifstream in_;
  std::vector<Frame> frames_;
  std::vector<uint8_t> plain_;
  size_t next_ = 0, pos_ = 0;
  uint64_t total_ = 0;
};

}  // namespace lizard
