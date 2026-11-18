// ai: Stream files' frames packed on the GPU (2026-11-18): each square of each frame in the sender's ring as one zstd
// ai: frame (RFC 8878) of its median-predictor residuals, as a .lizard's entry is (rx/lizfile.h), every block of it
// ai: its literals alone as zstd's four Huffman streams: no match search, so any zstd decoder reads it and the GPU does
// ai: the work. On the GPU (zstd_med.comp, zstd_hist.comp, zstd_huf.comp): the residuals, each 128 KiB block's
// ai: counts, then its four streams by the code built here; here, each block's code from its counts (zstd's own
// ai: Huffman builder, lengths at most 11 bits, and its table's description), and the frames put together from the
// ai: streams. A block of one value is an RLE block, a block the code would not shrink (or a short last one) a raw one.
// ai: The frames are larger than libzstd's at level 1 (literals only: 80.5% of the raw bytes against 73.3% on 2:1
// ai: LIZARD-480 frames, 2026-11-18), and every frame of a file packed alike (lizfile::Packing gpu). packCpu: the same
// ai: frames on the CPU, byte for byte, the check's twin.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace wg { struct Device; struct Buffer; struct Pipeline; struct BindGroupLayout; }

namespace lizard {

class GpuZstd {
 public:
  // ai: squares of side x side pixels, at most `squares` a batch (0: as many as a share of the device's memory holds for
  // ai: SETS batches, at most 128); throws where the kernels or the memory are not there
  static std::unique_ptr<GpuZstd> create(wg::Device& d, const std::string& assets, int side, int squares = 0);
  ~GpuZstd();
  int squares() const { return maxSq_; }
  int side() const { return side_; }
  // ai: A batch in three steps, so one batch's host work runs beside the next one's GPU work (SETS batches at once,
  // ai: each in a set of buffers of its own): begin (frames at ring slots `slots`, frameWords words a slot, rows rowWords
  // ai: words apart, each `codes` squares side by side `step` pixels apart: their residuals and counts submitted),
  // ai: codes (the counts read back and each block's code made on the cores, its streams submitted: the ring's slots
  // ai: are read by then, free for the painter), finish (the streams read back, a zstd frame a square put together on
  // ai: the cores, codes a frame in order; with crcs, each frame's ZIP CRC-32 too). devMu: the lock on recording onto the
  // ai: device, held while recording and submitting, not while the GPU works. begin returns the batch's set; throws on
  // ai: failure (and where no set is free: finish one first).
  static constexpr int SETS = 2;
  int begin(const wg::Buffer& ring, const std::vector<int>& slots, uint32_t frameWords, uint32_t rowWords, int codes, int step, std::mutex& devMu);
  void codes(int set, std::mutex& devMu);
  std::vector<std::vector<uint8_t>> finish(int set, std::vector<uint32_t>* crcs = nullptr);
  // ai: finish in two: sizes (the streams read back, each square's frame's bytes), then finishInto (each frame put
  // ai: together at its place dst[q], on the cores, with its CRC-32 where asked): a caller lays a batch's frames out in
  // ai: one buffer (a .lizard's entries: lizfile::SpanEntry) and nothing is copied twice
  std::vector<size_t> sizes(int set);
  void finishInto(int set, const std::vector<uint8_t*>& dst, std::vector<uint32_t>* crcs = nullptr);
  // ai: the three in turn
  std::vector<std::vector<uint8_t>> pack(const wg::Buffer& ring, const std::vector<int>& slots, uint32_t frameWords, uint32_t rowWords, int codes,
                                         int step, std::mutex& devMu);
  // ai: a square's frame on the CPU (px: side x side bytes, rows `stride` apart), byte for byte the GPU's
  static std::vector<uint8_t> packCpu(const uint8_t* px, int side, size_t stride);
  // ai: the last batch's ms: the residuals and counts on the GPU (submit to read back), the codes made here, the streams
  // ai: on the GPU (submit to read back), the frames put together here
  struct Times { double counts = 0, codes = 0, streams = 0, frames = 0; };
  const Times& last() const { return last_; }

 private:
  struct Set;
  explicit GpuZstd(wg::Device& d);
  void assemble(Set& x, size_t q, uint8_t* p, size_t& bytes, std::string& bad) const;
  wg::Device& d_;
  int side_ = 0, maxSq_ = 0, nextSet_ = 0;
  uint32_t n_ = 0, wps_ = 0, nb_ = 0;
  Times last_;
  std::shared_ptr<wg::BindGroupLayout> bglMed_, bglHist_, bglHuf_;
  std::shared_ptr<wg::Pipeline> med_, hist_, huf_;
  std::vector<std::unique_ptr<Set>> sets_;
};

}  // namespace lizard
