// ai: The app's sender (2026-10-01): lizard-web/send-worker.mjs natively, the C on the CPU (NEON). One thread makes each
// ai: frame's blocks in order (a file's ids by XferTx's schedule, or the test stream's from a random first id, BLAKE3 of
// ai: each id, as the page's); a few painters, each with its own codec (src/lizard_phy.c keeps its scratch in the codec),
// ai: encode and paint them ahead (lizard_encode, lizard_paint_grey: grey with the margin, the pilots' count the
// ai: frame's number mod 4) into the GPU painter's frame ring (2026-10-07), or the GPU paints them (painter, below);
// ai: the app takes them in order (take, takeSlot) and puts them on the screen at the display's pace.
// ai: A file's chunks (2026-11-16): solved by the transfer's CPU workers at the first configure; where the GPU paints and
// ai: holds the file, each chunk's symbols go up as it is made and the GPU makes every frame's blocks from their ids
// ai: (gpu_stream.h; read back where the CPU paints later), else the CPU's encoders; the stats' rq and rqWhy say which.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu_stream.h"
#include "gpu_painter.h"
#include "xfer_tx.h"

namespace lizard {

// ai: The symbol (lizard-web/send.mjs spec): subch the sub-channels (the version is subch / 16; 0 is no format, refused),
// ai: span the ring's (2 x its cells; 0 the default ring, src/lizard_phy.h LIZARD_RING_DEFAULT), fps the display rate the word
// ai: states; the picture is the format's (lizard_n_for(subch): a receiver takes it from the version); threads the painters. painter (2026-10-02, the
// ai: web's encoder switch): 0 the C on the CPU, 1 the GPU (gpu_painter.h; its absence an error), 2 auto: the GPU where
// ai: it builds and its first frames are the C's (the border to the byte, the square within a grey level), else the
// ai: CPU; assets the tree the GPU's kernels and tables come from (liblizard/out as the app ships it). codes (2026-10-03, the
// ai: desktop sender's; the web's #codes): 1, or 2 symbols side by side gap modules apart, one format, one word
// ai: and one pilot count, a frame's ids split between them (the first code the first blocks, as send-worker.mjs).
// ai: device (2026-10-03, the library): the GPU painter's Vulkan device, a substring of its name (empty: LIZ_VK_DEVICE's,
// ai: else the first discrete). gap (2026-10-04, the desktop sender's setting; the web's gap slider): the modules between
// ai: two codes, 0 to 64, GAP_MODULES unless set; one code has none. The frame ring (2026-10-07, every frame on the
// ai: GPU): aheadSecs and aheadBytes bound its depth (frames painted ahead of the screen), margin the slots a reader may
// ai: still hold when a frame is taken (the desktop presenter's swapchain images + 1); hostFrames keeps each painted
// ai: frame in host memory too, for a host that takes frames by take() (Android's present, the C API; the desktop
// ai: presenter takes slots and sets it off).
struct TxFormat {
  int subch = 0, span = 0, fps = 60, threads = 3, painter = 0, codes = 1, gap = GAP_MODULES;
  std::string assets, device;
  double aheadSecs = 1.0;
  uint64_t aheadBytes = 128ull << 20;
  int margin = 9;
  bool hostFrames = true;
};

class Sender {
 public:
  // ai: A file (data alive as long as this: the app maps it) or, with data null, the test stream.
  // ai: spillDir: where the transfer's spill goes (xfer_tx.h: no encoder kept, symbols in segments on disk); "" none.
  // ai: stopped: the transfer's (xfer_tx.h), its hash stopped where it says so
  Sender(const uint8_t* data, size_t length, const std::string& name, const std::string& type, const std::string& spillDir = "",
         const std::function<bool()>& stopped = {});
  ~Sender();
  Sender(const Sender&) = delete;
  Sender& operator=(const Sender&) = delete;

  // ai: (Re)starts the painters on a format; the transfer goes on where it was (a re-pick, as the page's respec). The
  // ai: frames painted ahead under the old format are dropped (their ids are lost to the fountain, as a frame never
  // ai: shown is). Returns an error, or "".
  std::string configure(const TxFormat& f);
  // ai: The GPU painter made ahead of a configure (its device and pipelines: off the thread that configures, which on
  // ai: the phone is the main one); "" or why there is none (configure then sends from the CPU, or refuses painter 1).
  // ai: extras: what a host asks of the device beyond the kernels (wg::DeviceExtras; the desktop presenter's surface
  // ai: and swapchain), kept for a configure that makes the painter later. prepareWhy(): the last prepare's refusal.
  std::string prepareGpu(const std::string& assets, const std::string& device = "", const wg::DeviceExtras* extras = nullptr);
  const std::string& prepareWhy() const { return prepareWhy_; }
  // ai: The painter's device (null before a prepare or a GPU configure): a host that presents from it (the desktop
  // ai: sender, 2026-10-07) shares its instance, device and queues; see wg::Device.
  wg::Device* device() const;
  // ai: The painted frame's pixels a side (a symbol and its margin: the frame's height), 0 before a configure; width(),
  // ai: the frame's width (codes symbols and the gaps between them; side() with one code).
  int side() const { return side_; }
  int width() const { return width_; }
  // ai: The format's picture, n samples a side (lizard_n_for(subch)), 0 before a configure.
  int n() const { return n_; }
  // ai: The next frame in order, RGBA width() x side(), into dst rows `stride` bytes apart, if it is painted: true, and
  // ai: it is the screen's now (its count advances); false and dst untouched if the painters are behind. From the host
  // ai: copy the painters keep under TxFormat.hostFrames (grey, made RGBA as it is copied); without it, false.
  bool take(uint8_t* dst, int stride);
  // ai: The next frame in order as its slot in the ring (ring(): frame seq at slot seq mod depth()), if it is painted:
  // ai: true, seq its number (its count mod 4 the pilots'), and it is the screen's now; false if the painters are
  // ai: behind. The desktop presenter's take (2026-10-07): no copy, the frame read on the device.
  bool takeSlot(uint64_t* seq, int* slot);
  // ai: The ring (gpu_painter.h ring(), frameWords(), rowWords()): null before a configure; a reader keeps the shared
  // ai: pointer for as long as its commands read the buffer, since a configure makes a new one.
  std::shared_ptr<wg::Buffer> ring() const;
  uint32_t ringFrameWords() const;
  uint32_t ringRowWords() const;
  int depth() const { return depth_; }
  const std::string& assets() const { return assets_; }
  // ai: Whether take would give a frame now (the one consumer asks before it locks a window buffer).
  bool ready();
  // ai: The stats JSON (Sender::stats in sender.cpp names the fields).
  std::string stats();
  // ai: Where the frames are painted: "gpu" or "cpu", and why not the GPU where it was asked for or auto and is not.
  std::string painter() const { return onGpu_ ? "gpu" : "cpu"; }
  const std::string& gpuWhy() const { return gpuWhy_; }
  const XferTx* xfer() const { return xfer_.get(); }
  // ai: Refresh (2026-11-17; Rate refresh of 2026-11-12, 1 to 4 a code rate, before): each block place of a symbol takes
  // ai: a new block every `every` pictures, its delay, a whole number (1, the default, every picture), so its rate is the
  // ai: format's fps over a whole number: picture k (the count) shows a new block there where k is a multiple of it, and
  // ai: repeats the picture before's block otherwise. The pilots stay the picture's count (painted with the picture, not
  // ai: kept with the block), so a receiver and its phase lock read every picture as ever: no format change, a repeated
  // ai: block one a receiver has. Only new places take ids. From the next frame made (frames already made ahead keep
  // ai: theirs). A client ties the two pilot rates (7/8 and 3/4) to one delay.
  static constexpr int REFRESH_MAX = 65535;
  // ai: per code rate (0 7/8, 1 3/4, 2 2/3, 3 1/2; the format's tiers from the inside out): every block of the rate takes
  // ai: its rate's; kept across configures
  void setRefresh(const int rate[4]);
  // ai: per block place of this format's symbol (every code alike); a configure to a format of another block count ends
  // ai: it, each place its rate's again. False where `place` is not this format's count.
  bool setRefreshBlocks(const std::vector<int>& place);
  // ai: whether picture k shows a new block at a place that takes one every `every` pictures
  static bool due(int every, uint64_t k) { return every <= 1 || k % static_cast<uint64_t>(every) == 0; }
  // ai: The animated header (2026-11-15, the apps' Header in Send's Refresh fold; src/fmt.h, SPEC 5.5): frames each
  // ai: carousel symbol is shown, 1 to CAROUSEL_MAX, 0 the static word (the default). The header is the format's own
  // ai: (version, fps, extension bytes 0 and 0: k 3). Code c of frame seq shows slot codes x (seq / frames) + c (two codes
  // ai: each its own, so a picture gives a receiver two), and slot t position carouselPosition(key, t): every pass of
  // ai: 255 slots shows each position once, each pass in an order of its own drawn from the sender's key (random a
  // ai: sender, seed's where one is given), so a capture that misses periodically (a camera reading every third frame:
  // ai: 255 = 3 x 5 x 17, which in a fixed order would see 85 positions forever) meets other positions each pass. On the
  // ai: GPU each encode's frames take theirs from every
  // ai: symbol's word made at each configure (gpu_painter.h wordsWhy: WORDS over the word cells after rsh); where that
  // ai: cannot be, a configure with it on paints on the CPU (gpuWhy says why). Live, as setRefresh: the next frame made
  // ai: takes it (those already made ahead keep theirs), no configure between; kept across configures.
  void setCarousel(int frames);
  static constexpr int CAROUSEL_MAX = 30;
  // ai: the carousel's slot t's position under key: pass t / 255 a Fisher-Yates order of the 255 positions from
  // ai: xorshift32 seeded by the key and the pass (the transfer's schedule's mix32), its (t mod 255)-th
  static int carouselPosition(uint32_t key, uint64_t slot);
  uint32_t carouselKey();
  // ai: A file's schedule drawn from seed in place of its random one (xfer_tx.h reseed): before its first frame only;
  // ai: false after, or for the test stream. The carousel's key drawn from it too, at any time, a file or the test
  // ai: stream, so a send repeats.
  bool seed(uint32_t seed);
  // ai: The code's colours (2026-11-16, Send's Colours): its ink and its paper, 0xRRGGBB, a frame's grey painted along
  // ai: the line between them where it is shown, each channel apart, ink + (paper - ink) v / 255 rounded (take here,
  // ai: the presenters' expand.comp on the GPU); black and white by default, the grey itself. The painters and the ring
  // ai: keep grey; any time, the next frame shown takes them.
  void setColours(uint32_t ink, uint32_t paper) { ink_ = ink & 0xffffffu; paper_ = paper & 0xffffffu; }
  uint32_t ink() const { return ink_; }
  uint32_t paper() const { return paper_; }

 private:
  std::atomic<uint32_t> ink_{0}, paper_{0xffffffu};
  // ai: car: setCarousel's when it was made; pos: each code's carousel position where car > 0
  struct Job { uint64_t seq; std::vector<uint8_t> blocks; int data; int car; uint8_t pos[2]; };
  void stopAll();
  void produce();
  void paint(int k);
  void paintGpu();
  void upload();

  std::unique_ptr<XferTx> xfer_;
  uint32_t nextId_;                 // ai: the test stream's next id (a random first one)
  TxFormat fmt_;
  int n_ = 0, side_ = 0, width_ = 0, gap_ = 0, blocks_ = 0, blockBytes_ = 0, gen_ = 0;
  std::string label_, error_, tiers_;   // ai: tiers_: the format's rate profile as text (the stats' tiers)
  int byRate_[4] = {1, 1, 1, 1};    // ai: setRefresh's, under mu_
  std::vector<int> byPlace_;        // ai: setRefreshBlocks's (empty: each place its rate's), under mu_
  std::vector<int> scheduleLocked() const;   // ai: each block place's delay now (mu_ held)
  int carousel_ = 0;                // ai: setCarousel's, under mu_
  uint32_t carKey_ = 0;             // ai: the carousel's order's key (carouselPosition), under mu_
  bool gpuWords_ = false;           // ai: the GPU paints and can paint an animated header (gpu_painter.h wordsWhy)
  std::vector<uint8_t> rateOf_;     // ai: a symbol's block b's code rate (0 7/8 ... 3 1/2), from the format's tiers
  std::mutex mu_;
  std::mutex xferMu_;               // ai: the transfer's state: the producer fills a frame under it, stats reads under it
  std::condition_variable cv_;
  bool stop_ = true;
  int depth_ = 0;                   // ai: the ring's frames (gpu_->depth())
  std::string assets_;
  uint64_t made_ = 0;               // ai: frames made (their blocks), the next to make; at most depth - margin past shown_
  uint64_t shown_ = 0;              // ai: frames taken, the next to take; its count mod 4 is the pilots'
  std::deque<Job> jobs_;            // ai: made, not painted
  std::map<uint64_t, int> ready_;   // ai: painted into the ring, not taken: seq -> slot
  std::map<uint64_t, std::vector<uint8_t>> hostOf_;    // ai: a painted frame's host copy (TxFormat.hostFrames), until taken
  std::map<uint64_t, std::vector<uint8_t>> toUpload_;  // ai: the CPU's painted frames, packed as the ring's rows, for the uploader
  std::map<uint64_t, int> dataOf_;  // ai: a frame's data blocks (its header and manifest not counted), for the lap
  std::vector<std::vector<uint8_t>> spare_;
  std::thread producer_, uploader_;
  std::vector<std::thread> painters_;
  std::unique_ptr<GpuPainter> gpu_;   // ai: kept across configures (its device made once a send)
  std::mutex devMu_;                  // ai: one recorder at a time on the painter's device (wg.h): paint, upload, read back
  bool onGpu_ = false;
  std::string gpuWhy_, gpuDevice_, prepareWhy_;
  std::shared_ptr<const wg::DeviceExtras> extras_;
  // ai: the stats' window: frames taken, data blocks shown, paint ms summed
  std::atomic<uint64_t> takenWin_{0}, dataWin_{0}, dataAll_{0}, paintedWin_{0}, encodesWin_{0};
  std::atomic<double> paintMsWin_{0};
  double winStart_ = 0, lastFps_ = 0, lastKBs_ = 0, lastPaintMs_ = 0, lastPaintBatch_ = 0;
  // ai: who makes a file's blocks (decided at the first configure that gets that far): the GPU from the CPU's solved
  // ai: chunks (stream_), or the CPU's encoders; why not the GPU. The destructor lets the transfer's workers go before
  // ai: the stream (a worker waits in its put)
  enum { RQ_NONE, RQ_CPU, RQ_GPU };
  std::atomic<int> holder_{RQ_NONE};
  std::string rqWhy_;
  std::unique_ptr<GpuStream> stream_;
};

}  // namespace lizard
