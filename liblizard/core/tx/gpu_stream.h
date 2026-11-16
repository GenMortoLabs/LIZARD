// ai: The sender's blocks made on the GPU (2026-11-16): the CPU prepares, the GPU streams. The transfer's workers solve
// ai: a file's chunks on the CPU's cores as ever (xfer_tx.h makeOnCpu, a chunk a core) and hand each chunk's L
// ai: intermediate symbols here (put), its encoder let go after, so nothing is held twice; the worker writes them into
// ai: SYM itself (a chunk's symbols 480 B apart, about 1.03 times the file): in place where the device shares the system's
// ai: memory, else through a slot and a copy on the device's second queue, beside the painting;
// ai: each frame's blocks are
// ai: then made on the device from their ids alone (blocks: gpu/wgsl/rq_enc.mjs BLOCKS, the tuple of RFC 6330 5.3.5.4
// ai: made there, the XOR of its intermediate symbols written into the painter's BLOCKS in the encode's own submission),
// ai: or read back for the CPU's painters (host). The host hands over 8 bytes a block.
// ai: Why the split: the solve is a chain (2,121 peel levels and 104,741 Gauss-Jordan steps in order at 16 MiB), which
// ai: the cores make as fast or faster than the GPU on every device measured, once, at the start; the blocks, every
// ai: frame, are independent XORs the GPU takes off the CPU (the S26 sending LIZARD-1024 x 2 at 60: 1.85 to 0.57 core-ms
// ai: a frame). STATUS "RaptorQ's encoder on the GPU in the sender" (a GPU solve built, measured and taken out).
// ai: Memory: the file's symbols must fit one buffer and a share of the device's memory (a quarter where the device
// ai: shares the system's, a half on a discrete one; create says why not), else the CPU's path (its spill past RAM).
// ai: Threads: create anywhere; put on any thread (a worker; on a discrete GPU it waits for a free slot and for its copy, on Device::queue2
// ai: and pool2 under this stream's lock); selfTest, blocks and host on the device's recording thread (wg.h) alone.
#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include "rq.h"
#include "xfer.h"
}

namespace wg { struct Device; struct Buffer; struct BindGroupLayout; struct Pipeline; struct BindGroup; struct Encoder; }

namespace lizard {

class GpuStream {
 public:
  // ai: the file as the transfer cuts it: chunks of chunkBytes but the last, `controls` control payloads (the header
  // ai: and the manifest blocks)
  struct File { size_t length = 0, chunkBytes = 0; uint32_t chunks = 0, controls = 0; };
  // ai: Throws std::runtime_error saying why the device cannot hold the file.
  static std::unique_ptr<GpuStream> create(wg::Device& dev, const std::string& assets, const File& file);
  ~GpuStream();
  GpuStream(const GpuStream&) = delete;
  GpuStream& operator=(const GpuStream&) = delete;

  // ai: The recording thread's, before any put: a small chunk solved on the CPU, its blocks made here (SELF_ESIS: the
  // ai: first source symbols, the last, K, K + 1 and 2^24 - 1) against rq_enc_symbol's: "" or why not (a device that
  // ai: makes them wrong sends from the CPU's encoders instead).
  std::string selfTest();
  // ai: Chunk c solved (a worker's): its intermediate symbols written into SYM in place where the device shares the
  // ai: system's memory, else into a free slot (waiting for one), copied onto the device on its second queue and waited
  // ai: for. False where the stream stopped (its destructor).
  bool put(uint32_t c, const rq_enc_t* enc);
  void stop();    // ai: a put waiting for a slot returns false, and every put after (the sender's teardown)
  bool done();    // ai: every chunk's symbols on the device
  // ai: The recording thread's, once done(): an encode's blocks (n, `stride` bytes apart, each its id's five bytes, a
  // ai: control block's payload after it) recorded into enc, written into out (the painter's BLOCKS, 480 B a block).
  void blocks(wg::Encoder& enc, const std::shared_ptr<wg::Buffer>& out, const uint8_t* blocks, uint32_t n, size_t stride);
  // ai: The recording thread's, once done(): data ids' payloads (471 B each) into out, a submission of its own.
  void host(const xfer_id_t* ids, uint32_t n, uint8_t* out);
  uint64_t symbolBytes() const { return symBytes_; }

 private:
  explicit GpuStream(wg::Device& dev) : dev_(dev) {}
  // ai: the blocks' dispatch over a set of SYM, TAB and CHUNKS (the file's, or the self-test's) into out
  void record(wg::Encoder& enc, const std::shared_ptr<wg::Buffer>& sym, const std::shared_ptr<wg::Buffer>& tab, const std::shared_ptr<wg::Buffer>& chunks,
              const std::shared_ptr<wg::Buffer>& out, uint32_t n, uint32_t ctl, uint32_t nchunks, uint32_t controls);
  void ids(const uint8_t* blocks, uint32_t n, size_t stride, bool controlsToo);

  wg::Device& dev_;
  File f_;
  std::vector<uint32_t> idw_;   // ai: the ids as written
  std::mutex poolMu_;           // ai: pool2's and queue2's (the workers' copies)
  bool uma_ = false;            // ai: the device shares the system's memory: SYM host-visible, written in place
  std::once_flag symOnce_;      // ai: SYM made by the first worker
  uint64_t symBytes_ = 0;
  uint32_t lMax_ = 0, ctl_ = 0;
  std::vector<uint32_t> base_, len_;   // ai: chunk c's first symbol in SYM, its L
  std::shared_ptr<wg::BindGroupLayout> bgl_;
  std::shared_ptr<wg::Pipeline> pipe_;
  std::shared_ptr<wg::Buffer> sym_, tab_, chunks_, ids_, par_, out_, outRd_, lastSym_, lastOut_;
  std::shared_ptr<wg::BindGroup> group_;
  std::vector<uint8_t> ctlUp_;   // ai: a control symbol written into SYM
  // ai: the upload slots (host-visible, lMax_ symbols of 480 B each) and their chunks (-1 free), the chunks on the
  // ai: device, under mu_
  std::vector<std::shared_ptr<wg::Buffer>> slot_;
  std::vector<int64_t> slotOf_;
  uint32_t onDevice_ = 0;
  bool stop_ = false;
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace lizard
