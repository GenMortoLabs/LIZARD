// ai: .lizard (2026-10-28): a replay as one file, the only form a replay takes (Save replays writes it, the phone's
// ai: Download is it, the PC's Upload camera plays it, the lab's tools read it; research/SPEC.md 15). A ZIP archive of
// ai: stored entries (method 0, no ZIP64, so under 4 GB and 65,535 entries), its comment "LIZARD replay 1": each frame
// ai: as NNNN.zst, in the run's order; then stats.jsonl where the run has rows; then meta.json, which ends it (frames,
// ai: sizes, ms, axes, layout, ...: replay.h finish). A reader finds the entries by the central directory. A frame's
// ai: entry is its w x h luma (meta.json's sizes[i], else w and h) as the residuals of JPEG-LS's median predictor
// ai: (LOCO-I's MED: of a, b, c, the neighbours left, above and above-left, min(a, b) where c >= max(a, b), max(a, b)
// ai: where c <= min(a, b), else a + b - c; a neighbour outside the frame 0), each the pixel less its prediction mod 256,
// ai: the w h of them in rows as one zstd frame, at the writer's level (2026-11-18: any valid zstd frame decodes; every
// ai: frame of one file packed alike, meta.json's "zstd" saying how: "level 1", the default, or "gpu", the GPU's coder).
// ai: Lossless; at level 1 on the S26's 2:1 frames 74.4% of the raw bytes (gzip -1 88.4%), about 1.4 ms a 960 x 960
// ai: frame to pack on its big core, 2.0 to unpack.
#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace lizard::lizfile {

// ai: What a .lizard holds, by its ZIP comment: a replay ("LIZARD replay 1": a camera's frames, the receivers' input) or
// ai: a stream ("LIZARD stream 1", 2026-11-18: a sender's frames painted ahead, for a player that shows them in turn at
// ai: meta.json's fps instead of painting; any receiver's Upload reads it as frames too).
enum class Kind { Replay, Stream };
const char* magicOf(Kind k);

// ai: ZIP's CRC-32 of n bytes (an entry's, as Writer::packed takes it)
uint32_t crc32(const uint8_t* p, size_t n);

// ai: Entries laid out by the caller (2026-11-18, the GPU export's batches: the cores put a batch's entries in place in
// ai: one buffer, each its local header then its bytes, and the writer writes the buffer at once): an entry's place in
// ai: the buffer (its header's), its bytes and its CRC-32; the header's bytes for frame `index` (its name NNNN.zst) and
// ai: the header itself
struct SpanEntry { uint64_t at = 0; uint32_t size = 0, crc = 0; };
size_t localHeaderBytes(size_t index);
void localHeader(size_t index, uint32_t size, uint32_t crc, uint8_t* out);

// ai: zstd's level the writers take unless told (1 packs these frames as small as 3, 74.4% both, in under half the time)
constexpr int LEVEL_DEFAULT = 1;

// ai: How every frame of one file is packed (meta.json's "zstd", name()): libzstd at `level`, or (gpu) frames the GPU
// ai: packed (tx/gpu_zstd.h: literals and runs, no match search), handed to Writer::packed as they are.
struct Packing {
  int level = LEVEL_DEFAULT;
  bool gpu = false;
  std::string name() const;
};

// ai: a frame's luma (rows `stride` apart) as its NNNN.zst entry, and back: false with why where the bytes are not one
// ai: zstd frame of w h residuals
std::vector<uint8_t> packFrame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, int level = LEVEL_DEFAULT);
bool unpackFrame(const uint8_t* z, size_t n, uint32_t w, uint32_t h, uint8_t* out, std::string& why);

// ai: The same codec with its zstd contexts and buffers kept, for a stream of frames: the phone's frames to a PC go as
// ai: .lizard frames (stream.h FRAME, 2026-10-28). One thread at a time.
class FrameCodec {
 public:
  explicit FrameCodec(int level = LEVEL_DEFAULT);
  ~FrameCodec();
  FrameCodec(const FrameCodec&) = delete;
  FrameCodec& operator=(const FrameCodec&) = delete;
  // ai: px's w x h luma (rows `stride` apart) as its packed bytes in out; false where zstd refused
  bool pack(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, std::vector<uint8_t>& out);
  // ai: n packed bytes back into out's w h bytes; false with why
  bool unpack(const uint8_t* z, size_t n, uint32_t w, uint32_t h, uint8_t* out, std::string& why);

 private:
  void* cctx_ = nullptr;
  void* dctx_ = nullptr;
  int level_;
  std::vector<uint8_t> res_;   // ai: the residuals of the frame in hand
};

// ai: A .lizard written as it goes: the frames in order, then finish's stats.jsonl and meta.json. Written through
// ai: path.part, renamed over path at finish; one destroyed unfinished removes its .part. One thread at a time.
class Writer {
 public:
  // ai: every frame packed by `packing` (frame() at its level; with gpu, packed() alone)
  static std::unique_ptr<Writer> create(const std::string& path, std::string& why, Kind kind = Kind::Replay, Packing packing = {});
  ~Writer();
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  // ai: false with why (a write the file system refused, ZIP's 4 GB): the archive keeps the frames before it, so a
  // ai: finish after it still makes a whole file of them
  bool frame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, std::string& why);
  // ai: a frame packed elsewhere (FrameCodec at this file's level on another thread, or the GPU's coder) as the next entry;
  // ai: its ZIP CRC-32 given where the packer worked it out (crc32 below: off the one writing thread), else worked out here
  bool packed(const uint8_t* z, size_t n, std::string& why);
  bool packed(const uint8_t* z, size_t n, uint32_t crc, std::string& why);
  // ai: entries laid out by the caller (SpanEntry: frames frames() on, in order) written with one call: the cores fill a
  // ai: batch's place in the file, one thread writes it in large writes (one thread reaches the disk's own rate)
  bool span(const uint8_t* buf, size_t n, const std::vector<SpanEntry>& es, std::string& why);
  // ai: how its frames are packed after all, before the first (the sender's export decides at its configure)
  void setPacking(const Packing& p) { if (!frames_) packing_ = p; }
  size_t frames() const { return frames_; }
  // ai: the file's bytes, -1 with why; stats "" none; meta.json takes "zstd" (Packing::name) where it has none
  int64_t finish(const std::string& meta, const std::string& stats, std::string& why);

 private:
  Writer() = default;
  std::string path_;
  std::FILE* f_ = nullptr;
  void* cctx_ = nullptr;
  void* zip_ = nullptr;
  Packing packing_;
  size_t frames_ = 0;
  bool done_ = false;
};

// ai: A .lizard as a folder of its parts (meta.json, stats.jsonl, NNNN.gray: the frames as raw luma; png: NNNN.png,
// ai: 8-bit grey PNGs any viewer opens), to look at them with other tools, and such a folder (of .gray frames) as a
// ai: .lizard. False / -1 with why; unpack refuses a folder that holds anything.
bool unpack(const std::string& path, const std::string& dir, std::string& why, bool png = false);
// ai: w x h grey bytes (rows `stride` apart) as an 8-bit grey PNG at path (its deflate stored, not compressed: a frame
// ai: of a code is noise to a compressor); false with why
bool writePng(const std::string& path, const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, std::string& why);
int64_t pack(const std::string& dir, const std::string& path, std::string& why);

// ai: A .lizard to read a frame at a time. One thread at a time.
struct FrameInfo {
  uint32_t w = 0, h = 0;
  double ms = 0;      // ai: its capture's time on the camera's clock; NaN where meta.json has none
  float ax = 0, ay = 0;   // ai: where the camera's optical axis fell in it; NaN its centre (meta.json's axes)
};
class Reader {
 public:
  static std::unique_ptr<Reader> open(const std::string& path, std::string& why);
  ~Reader();
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  const std::string& meta() const { return meta_; }   // ai: meta.json as the run wrote it
  const std::string& stats() const { return stats_; }   // ai: stats.jsonl, "" where the run has none
  const std::string& layout() const { return layout_; }   // ai: meta.json's layout, "1:1" or "2:1" ("1:1" where none)
  Kind kind() const { return kind_; }   // ai: a replay or a stream (its ZIP comment)
  const std::vector<FrameInfo>& frames() const { return frames_; }
  // ai: frame i's luma, w h bytes in rows of w; false with why (a CRC that does not hold, a short file)
  bool frame(size_t i, std::vector<uint8_t>& luma, std::string& why);

 private:
  Reader() = default;
  struct Entry { uint64_t at = 0, size = 0; uint32_t crc = 0; };
  std::string path_, meta_, stats_, layout_ = "1:1";
  Kind kind_ = Kind::Replay;
  std::FILE* f_ = nullptr;
  void* dctx_ = nullptr;         // ai: zstd's decompression context, made at the first frame
  std::vector<Entry> entries_;   // ai: a .lizard's frames' entries, in order
  std::vector<FrameInfo> frames_;
  std::vector<uint8_t> buf_;
  bool parseMeta(std::string& why);
  bool readEntry(const Entry& e, std::vector<uint8_t>& out, std::string& why);
};

}  // namespace lizard::lizfile
