// ai: The default client's compression (2026-11-14, header version 4: the transfer carries a file as it is and
// ai: compresses nothing; how a client compresses is the client's). The apps' rule, the web's twin `sim/pack.mjs` byte for
// ai: byte: a file goes as a zstd stream of independent frames, each of PACK_FRAME bytes of the file
// ai: (liblizard/zstd/shim.h, the level and parameters pinned, so every client packs a file to the same bytes and the
// ai: same root), named <name>.zst and typed `application/zstd; original="<type>"` (the original's `"` and `\` escaped
// ai: by a `\`), where it pays: a sample of the file (three slices of 1 MiB, its start, middle and end; the whole file
// ai: under 3 MiB) compressed to 90% or less, and the whole stream to 97% or less; else the file goes as it is. Any
// ai: zstd reads the stream (a receiver of another kind gets a .zst); a LIZARD receiver unpacks a file of that type
// ai: (unpack) and files it under its own name and type. The frames are made a worker a physical core, in order on the
// ai: disk, at most two a worker held in memory.
#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace lizard {

constexpr size_t PACK_FRAME = size_t{1} << 22;     // ai: 4 MiB of the file a frame
constexpr size_t PACK_FRAME_MAX = size_t{1} << 26; // ai: the most a frame may hold for unpack (another client's frames)

struct Packed {
  bool packed = false;            // ai: the stream at path, else the file goes as it is (path, name and type the caller's)
  std::string path, name, type;   // ai: what the transfer carries
  uint64_t size = 0;              // ai: its bytes
  uint64_t original = 0;          // ai: the file's own bytes
};

// ai: The media type a packed file carries for a file of `type`; "" where it would pass the header's 158 bytes (the
// ai: file then goes as it is).
std::string packType(const std::string& type);
// ai: true where `type` is a packed file's, its file's type into original.
bool packedType(const std::string& type, std::string& original);
// ai: A packed file's name back (<name>.zst less its ".zst"; another name as it is).
std::string unpackedName(const std::string& name);

// ai: The file at `path` (UTF-8), its name and media type as the caller would send it, packed into a new file in `dir`
// ai: where that pays (the caller removes it after the send), else returned as it is. Throws std::runtime_error where
// ai: the file cannot be read or the stream not written.
Packed pack(const std::string& path, const std::string& name, const std::string& type, const std::string& dir);

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
