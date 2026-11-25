// ai: The sender's painter on the GPU (2026-10-02; the CPU's four painters already held 60 a second at LIZARD-1024 on
// ai: the S26, about 1.5 cores busy): liblizard/gpu/encoder.mjs natively, on the app's Vulkan host (core/wg). Its
// ai: kernels are the web's, compiled by liblizard/gen (gen/sender.mjs: out/spv/send_*.spv, setup/send.json); its
// ai: tables the C's (send_tables.h). An encode is up to frames() frames' blocks in, one submission (paint, irows,
// ai: ipic, rsv unless the taps are copies, rsh), their grey W x W pixels read back: on a phone's shared
// ai: memory a copy, not a transfer. The square is within a grey level of the C's (the transform's rounding), the
// ai: border the C's to the byte; check() holds a device to that, as the web's auto does before it takes the GPU
// ai: (2026-09-29: a sender's symbol need only decode). The anchors' squares inside the square (SPEC 6.13) are the C's
// ai: bytes too, so they are the C's to the byte as the border is.
#pragma once
#include <cstdint>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "send_tables.h"

namespace wg { struct Device; struct DeviceExtras; struct Buffer; struct BindGroupLayout; struct Pipeline; struct BindGroup; struct QuerySet; struct Encoder; struct Ticket; }

namespace lizard {

class GpuPainter {
 public:
  // ai: assets: liblizard/out's tree as the app ships it (filesDir/lizard: setup/send.json, blobs/, spv/send_*.spv). The
  // ai: device: the one named like `device` (a substring of its name), else LIZ_VK_DEVICE's, else the first discrete,
  // ai: else the first. Throws std::runtime_error saying why there is none.
  // ai: extras: what the host asks of the device beyond the kernels (the desktop presenter's surface and swapchain).
  static std::unique_ptr<GpuPainter> create(const std::string& assets, std::function<void(const std::string&)> log = nullptr, const std::string& device = "",
                                            const wg::DeviceExtras* extras = nullptr);
  ~GpuPainter();
  // ai: the device, for a host that presents from it (its handles are wg::Device's public fields)
  wg::Device* vk() const { return dev_.get(); }

  // ai: A format (sub-channels, the ring's span, the rate the word states; the picture is the format's, lizard_n_for), encodes of up to `frames` frames (halved
  // ai: while the device cannot hold them), each `codes` symbols side by side `gap` modules apart (send_tables.h), and
  // ai: the frame ring (2026-10-07): FRAME holds depth() frames, the most of `aheadFrames` and `aheadBytes` allow and at
  // ai: least 2 frames + margin + 1 (halved while the device cannot hold them, never under the floor), frame seq at slot
  // ai: seq mod depth. ringOnly: the ring alone (the CPU's painters fill it), no pipelines' buffers. "" or why not.
  // ai: Every picture with its 7 x 7 anchors as the C paints them (the border's bytes over their squares:
  // ai: send_tables.h anch, gpu/slang/src/send_rsh.slang).
  std::string configure(int subch, int span, int fps, int frames = 4, int codes = 1, int gap = GAP_MODULES,
                        int aheadFrames = 0, uint64_t aheadBytes = 0, int margin = 0, bool ringOnly = false);
  int frames() const { return frames_; }
  int depth() const { return depth_; }
  // ai: the ring: FRAME (grey, four pixels a u32; a frame frameWords() u32 from slot * frameWords(), its rows rowWords()
  // ai: u32 apart), shared so a host that reads it on another queue keeps it alive across a configure
  std::shared_ptr<wg::Buffer> ring() const { return FRAME_; }
  uint32_t frameWords() const { return static_cast<uint32_t>(t_.FS); }
  uint32_t rowWords() const { return static_cast<uint32_t>(t_.RW); }
  // ai: a frame painted elsewhere (the C), packed as the ring's rows (4 * rowWords() bytes a row, 4 * frameWords() in
  // ai: all), into its slot; returns once it is there. The CPU mode's uploader thread calls it, no other recorder.
  void upload(int slot, const uint8_t* packed);
  int side() const { return t_.W; }      // ai: a symbol's side, the frame's height
  int width() const { return t_.FW; }    // ai: the frame's width: codes symbols and the gaps between them
  int codes() const { return t_.codes; }
  int blocks() const { return t_.blocks; }   // ai: a symbol's blocks (the format's rate profile)
  const std::string& device() const { return name_; }

  // ai: count frames (1 to frames()), each codes x blocks x blockBytes (a block its id then its payload, as Sender makes
  // ai: them; a frame's first code's blocks, then its second's), into out[0 .. count) as width() x side() grey, the first
  // ai: painted as picture `parity` (its count mod 4, the pilots'; each frame after it one more, its codes alike). Returns the GPU's ms for the whole encode (timestamps), or the submission's wall ms where
  // ai: the device has none. Throws on a lost device.
  // ai: seq: the first frame's number: the frames go to slots seq mod depth() on (an encode never wraps the ring:
  // ai: count is at most depth() - seq mod depth()). out null: the frames stay on the device (the ring's readers take
  // ai: them there); else each is read back into out[f] as width() x side() grey.
  // ai: test: the blocks are the test stream's (src/xfer.h stream_fill), and only their ids (each block's first five bytes) are read and
  // ai: sent, 12 bytes a block packed (the bare id and the 48-bit TOI): PAINT makes each payload from its id on the device (BLAKE3, gpu/slang/src/send_paint.slang; 2026-11-15)
  // ai: fill (2026-11-16, the sender's RaptorQ blocks on the GPU, gpu_stream.h): records what writes BLOCKS (blocksBuffer) into
  // ai: the encode's own pass before PAINT, in place of the host's blocks (blocks then unread, may be null).
  double encode(const uint8_t* blocks, int count, int parity, std::vector<uint8_t>* out, uint64_t seq = 0, bool test = false,
                const std::function<void(wg::Encoder&)>& fill = {});
  // ai: the encode's blocks on the device (480 B a block: its id's twelve bytes, its 460-byte payload), remade at each
  // ai: configure
  std::shared_ptr<wg::Buffer> blocksBuffer() const { return BLOCKS_; }

  // ai: An encode in flight (2026-11-16): submit records and submits it and returns at once; done says whether the GPU
  // ai: finished it, finish waits for it (where it must), reads its frames back where `out` was given and returns its
  // ai: GPU ms (encode is submit then finish). At most FLIGHTS in flight (each its own timestamps); `out` wants the
  // ai: encode finished before the next is submitted (one readback buffer). The device runs a queue's submissions in turn
  // ai: (every command buffer begins with a barrier, wg.h), so encodes in flight share the painter's buffers safely and
  // ai: their queue writes (blocks, uniforms) land each at its own submission's head.
  static constexpr int FLIGHTS_MAX = 3, FLIGHTS = FLIGHTS_MAX;
  struct Pending { std::shared_ptr<wg::Ticket> ticket; int flight = 0, count = 0; std::vector<uint8_t>* out = nullptr; double t0 = 0; };
  // ai: The sender's clock the border's word states (src/lizard_phy.h lizard_clock: fps 0 to 240, delay 1 to 30, its
  // ai: places' largest): the border painted again and uploaded where it moved, the next encode's.
  void setClock(int fps, int delay);
  Pending submit(const uint8_t* blocks, int count, int parity, std::vector<uint8_t>* out, uint64_t seq = 0, bool test = false,
                 const std::function<void(wg::Encoder&)>& fill = {});
  bool done(const Pending& p);
  double finish(Pending& p);
  // ai: a ring slot's frame read back (width() x side() grey), a submission of its own, waited for (the checks')
  void readSlot(int slot, std::vector<uint8_t>& out);
  // ai: LIZ_SEND_STAGES=1 (lab, 2026-11-24): each stage its own timestamp pair (paint, irows, ipic, rsv, rsh); the
  // ai: GPU ms each stage took since the last call, summed over the encodes finished, and those encodes' count
  static constexpr int STAGES = 5;
  static constexpr const char* STAGE_NAMES[STAGES] = {"paint", "irows", "ipic", "rsv", "rsh"};
  bool staged() const { return staged_; }
  std::array<double, STAGES> takeStageMs(int* encodes = nullptr) {
    auto r = stageMs_;
    stageMs_.fill(0);
    if (encodes) *encodes = stageEncodes_;
    stageEncodes_ = 0;
    return r;
  }

  // ai: frames painted here and by the C (its own codec, told the same counts), the blocks the test stream's: here from
  // ai: their ids alone (encode's test), the C from their bytes, so the device's stream is held to the library's too: the border's pixels equal, the square's
  // ai: within one grey level (gpu/encoder.mjs checkNow's rule), the anchors' squares' pixels equal: anchorPx compared,
  // ai: anchorDark of them 0 here (their dark rings: the anchors are there), anchorDiff apart.
  struct Check { bool ok = false; long borderDiff = 0, squareDiff = 0, anchorPx = 0, anchorDark = 0, anchorDiff = 0; int squareMax = 0; };
  Check check(const uint8_t* blocks, int count);

 private:
  GpuPainter() = default;
  std::string pipes(int n);
  std::unique_ptr<wg::Device> dev_;
  std::string name_, assets_;
  std::function<void(const std::string&)> log_;
  SendConsts k_;
  lizard_t f_{};
  bool fReady_ = false;
  SendTables t_;
  int frames_ = 0, fps_ = 0, depth_ = 0;
  bool ringOnly_ = false;
  std::shared_ptr<wg::BindGroupLayout> bglPaint_, bglIrows_, bglIpic_, bglIpicFrame_, bglRsv_, bglRsh_;
  std::shared_ptr<wg::Pipeline> paint_, rsv_, rsh_, rshBorder_, irows_, ipic_, ipicFrame_;
  int pipesN_ = 0;
  std::shared_ptr<wg::Buffer> TAB_;   // ai: the paint's codes and whitening (send_tables.h SendConsts tab), made at create
  // ai: a configuration's buffers and groups
  std::shared_ptr<wg::Buffer> BLOCKS_, UV_, S_, PU_, TW_, ROWS_, SU_, Y_, TMP_, PIC_, TAPS_, BORDER_, GU_, FRAME_, READ_;
  std::shared_ptr<wg::BindGroup> gPaint_, gIrows_, gIpic_, gRsv_, gRsh_;
  // ai: each flight's timestamp queries and their readback (submit's, round robin)
  std::shared_ptr<wg::QuerySet> qs_[FLIGHTS_MAX];
  std::shared_ptr<wg::Buffer> time_[FLIGHTS_MAX];
  int flight_ = 0;
  std::vector<uint8_t> staging_;
  bool staged_ = false;
  std::array<double, STAGES> stageMs_{};
  int stageEncodes_ = 0;
};

}  // namespace lizard
