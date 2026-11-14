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
#include <string>

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

}  // namespace lizard
