// ai: The app's sender (sender.h): lizard-web/send-worker.mjs and the page's paint ahead, natively.
#include "sender.h"

#include "wg.h"

#include <algorithm>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>

extern "C" {
#include "lizard_phy.h"
#include "xfer.h"
}
#include "priority.h"

namespace lizard {

namespace {
double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int PAYLOAD = XFER_PAYLOAD, ID_BYTES = XFER_ID_BYTES;
constexpr int TX_GPU_FRAMES = 32;   // ai: frames a GPU encode at most (configure)
bool control(xfer_id_t id) { return xfer_id_chunk(id) == XFER_CONTROL; }   // ai: sim/xfer.mjs isControlId
// ai: a string's text fit for the stats JSON (a driver's message may hold quotes or a backslash)
std::string esc(const std::string& s) {
  std::string o;
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (static_cast<unsigned char>(c) >= 0x20) o += c; }
  return o;
}
// ai: the transfer's schedule's draws (xfer_tx.cpp, sim/xfer.mjs): murmur3's finaliser; xorshift32, a state never 0
uint32_t mix32(uint32_t x) { x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16; return x; }
uint32_t xorshift(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
// ai: pass's order of the carousel's 255 positions under key (Sender::carouselPosition)
void carouselPass(uint32_t key, uint64_t pass, uint8_t order[OB_CAR_N]) {
  uint32_t s = mix32(key ^ mix32(static_cast<uint32_t>(pass) * 0x85ebca6bu + 0x27d4eb2fu));
  if (!s) s = 0x9e3779b9u;
  for (int i = 0; i < OB_CAR_N; i++) order[i] = static_cast<uint8_t>(i);
  for (int i = OB_CAR_N - 1; i > 0; i--) std::swap(order[i], order[xorshift(s) % static_cast<uint32_t>(i + 1)]);
}
uint32_t freshKey() {
  std::random_device rd;   // ai: the clock mixed in: some C++ libraries' random_device repeats from run to run
  return rd() ^ mix32(static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
}
}  // namespace

int Sender::carouselPosition(uint32_t key, uint64_t slot) {
  uint8_t order[OB_CAR_N];
  carouselPass(key, slot / OB_CAR_N, order);
  return order[slot % OB_CAR_N];
}

uint32_t Sender::carouselKey() {
  std::lock_guard<std::mutex> l(mu_);
  return carKey_;
}

Sender::Sender(const uint8_t* data, size_t length, const std::string& name, const std::string& type, const std::string& spillDir,
               const std::function<bool()>& stopped) {
  // ai: deferred: the first configure says who makes the chunks (sender.h)
  if (data) xfer_ = std::make_unique<XferTx>(data, length, name, type, -1, spillDir, stopped, true);
  // ai: the test stream starts at a random id each start, as the page's (lizard-web/send.mjs), so a receiver takes a restarted
  // ai: one as new with no signal
  std::random_device rd;
  nextId_ = (rd() & 0x3fffffffu) | 1u;
  carKey_ = freshKey();
  winStart_ = nowMs();
}

// ai: the threads first, then the transfer's workers (a worker in GpuStream::put let go by its stop, then joined), then the
// ai: stream: a worker hands its chunk to the stream and wakes the painter through mu_ and cv_, which outlive both
Sender::~Sender() {
  stopAll();
  if (stream_) stream_->stop();
  xfer_.reset();
  stream_.reset();
}

void Sender::stopAll() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (producer_.joinable()) producer_.join();
  for (auto& t : painters_) if (t.joinable()) t.join();
  painters_.clear();
  if (uploader_.joinable()) uploader_.join();
}

std::string Sender::configure(const TxFormat& f) {
  stopAll();
  int car;
  { std::lock_guard<std::mutex> l(mu_); car = carousel_; }
  if (f.codes < 1 || f.codes > 2) return "codes are 1 or 2, not " + std::to_string(f.codes);
  if (f.gap < 0 || f.gap > 64) return "the gap is 0 to 64 modules, not " + std::to_string(f.gap);
  // ai: one codec made here to say what the format is (and that it is one); each painter makes its own
  // ai: the format's picture (src/lizard_phy.h lizard_n_for: a receiver takes it from the word's version)
  lizard_t probe{};
  const int n = lizard_n_for(f.subch);
  if (lizard_init(&probe, n, f.subch, f.span)) return "the codec refused LIZARD-" + std::to_string(f.subch);
  // ai: The device and the painter are made here where no prepare did: the frame ring lives in the painter whatever
  // ai: paints (2026-10-07), so a sender needs a Vulkan device. The GPU paints where asked (or auto); auto keeps it only
  // ai: where two frames of the test stream's blocks come out as the C paints them (gpu/encoder.mjs checkNow's rule), so
  // ai: a driver that returns garbage sends from the CPU instead, into the same ring.
  bool gpu = false;
  gpuWhy_.clear();
  if (!gpu_) {
    try { gpu_ = GpuPainter::create(f.assets, nullptr, f.device, extras_.get()); }
    catch (const std::exception& e) { lizard_free(&probe); return std::string("no Vulkan device for the frames: ") + e.what(); }
  }
  assets_ = f.assets;
  const int aheadFrames = std::max(1, static_cast<int>(std::lround(f.fps * f.aheadSecs)));
  // ai: the GPU painter paints the anchors as the C does (their squares the C's bytes, gpu_painter.h), and auto's check
  // ai: holds its frames to the C's with them. Encodes of up to TX_GPU_FRAMES frames: the painter runs ahead of the
  // ai: screen, so a deep encode adds no latency and spreads each dispatch's fixed cost (halved where the device
  // ai: cannot hold them).
  if (f.painter != 0) {
    try {
      gpuWhy_ = gpu_->configure(f.subch, f.span, f.fps, TX_GPU_FRAMES, f.codes, f.gap, aheadFrames, f.aheadBytes, f.margin, false);
      if (gpuWhy_.empty() && f.painter == 2) {
        const int k = std::min(2, gpu_->frames()), T = probe.blocks * f.codes, B = probe.block_bytes;
        std::vector<uint8_t> blocks(static_cast<size_t>(k) * T * B);
        for (int t = 0; t < k * T; t++) {
          uint8_t* o = blocks.data() + static_cast<size_t>(t) * B;
          xfer_id_put(o, static_cast<xfer_id_t>(t));
          stream_fill(static_cast<xfer_id_t>(t), o + ID_BYTES, PAYLOAD);
        }
        const auto c = gpu_->check(blocks.data(), k);
        if (!c.ok) gpuWhy_ = "its frames are not the C's (border " + std::to_string(c.borderDiff) + " pixels apart, the square up to " + std::to_string(c.squareMax) + " levels" +
                             (c.anchorDiff ? ", the anchors " + std::to_string(c.anchorDiff) + " pixels apart" : "") + ")";
      }
      // ai: the animated header (setCarousel) on: each frame's word cells its own symbol, painted by WORDS after rsh (the
      // ai: check above holds the static word to the C's); where the GPU cannot, the CPU paints
      if (gpuWhy_.empty() && car > 0 && !gpu_->wordsWhy().empty()) gpuWhy_ = gpu_->wordsWhy();
    } catch (const std::exception& e) { gpuWhy_ = e.what(); }
    if (gpuWhy_.empty()) gpu = true;
    else if (f.painter == 1) { lizard_free(&probe); return "the GPU cannot paint here: " + gpuWhy_; }
  }
  if (!gpu) {
    std::string e;
    try { e = gpu_->configure(f.subch, f.span, f.fps, 1, f.codes, f.gap, aheadFrames, f.aheadBytes, f.margin, true); }
    catch (const std::exception& x) { e = x.what(); }
    if (!e.empty()) { lizard_free(&probe); return "no frame ring on the device: " + e; }
  }
  // ai: The file's chunks, decided here once (the first configure to get this far): the CPU's workers solve them either
  // ai: way; where the GPU paints, holds the file's symbols (gpu_stream.h's budget) and makes its self-test's blocks as
  // ai: the C does, each chunk's symbols go up as it is made and the GPU makes every frame's blocks from their ids (a
  // ai: later configure that paints on the CPU reads them back); else the CPU's encoders (the spill past RAM) as ever.
  // ai: LIZ_TX_RQ=cpu: the CPU's (the lab's A/B).
  if (xfer_ && xfer_->chunks() && holder_ == RQ_NONE) {
    const char* rqEnv = std::getenv("LIZ_TX_RQ");
    const bool cpuOnly = rqEnv && std::string(rqEnv) == "cpu";
    std::unique_ptr<GpuStream> st;
    if (gpu && !cpuOnly) {
      try {
        st = GpuStream::create(*gpu_->vk(), f.assets, {xfer_->length(), xfer_->chunkSize(), xfer_->chunks(), xfer_->controls()});
        rqWhy_ = st->selfTest();
        if (!rqWhy_.empty()) st.reset();
      } catch (const std::exception& e) { st.reset(); rqWhy_ = e.what(); }
    } else rqWhy_ = cpuOnly ? "LIZ_TX_RQ=cpu" : !gpuWhy_.empty() ? gpuWhy_ : "the CPU paints";
    {
      std::lock_guard<std::mutex> l(mu_);
      stream_ = std::move(st);
      holder_ = stream_ ? RQ_GPU : RQ_CPU;
    }
    std::lock_guard<std::mutex> x(xferMu_);
    if (GpuStream* sp = stream_.get()) xfer_->makeOnCpu([sp](uint32_t c, const rq_enc_t* e) { return sp->put(c, e); });
    else xfer_->makeOnCpu();
  }
  std::lock_guard<std::mutex> l(mu_);
  fmt_ = f;
  gpuWords_ = gpu && gpu_->wordsWhy().empty();
  n_ = n;
  fmt_.threads = gpu ? gpu_->frames() : std::max(1, f.threads);
  onGpu_ = gpu;
  gpuDevice_ = gpu_->device();
  depth_ = gpu_->depth();
  blocks_ = probe.blocks;
  blockBytes_ = probe.block_bytes;
  side_ = probe.px + 2 * LIZARD_QUIET * probe.pxm;
  gap_ = f.codes > 1 ? f.gap * probe.pxm : 0;
  width_ = f.codes * side_ + (f.codes - 1) * gap_;
  // ai: the profile as text: "7/8 x 34, 3/4 x 18, 2/3 x 10, 1/2 x 10" at LIZARD-592 (the stats' tiers), read before
  // ai: the probe is freed
  tiers_.clear();
  rateOf_.clear();
  for (int t = 0; t < probe.tiers; t++) {
    const int r = probe.tier[t].rate == 6 ? 0 : probe.tier[t].rate == 4 ? 1 : probe.tier[t].rate == 3 ? 2 : 3;
    tiers_ += (t ? ", " : "") + std::string(r == 0 ? "7/8" : r == 1 ? "3/4" : r == 2 ? "2/3" : "1/2") + " x " + std::to_string(probe.tier[t].blocks);
    rateOf_.insert(rateOf_.end(), static_cast<size_t>(probe.tier[t].blocks), static_cast<uint8_t>(r));
  }
  rateOf_.resize(static_cast<size_t>(probe.blocks), 3);   // ai: (the tiers lay every block: a guard, not a case)
  lizard_free(&probe);
  label_ = "LIZARD-" + std::to_string(f.subch) + (f.codes > 1 ? " x " + std::to_string(f.codes) : "");
  // ai: what was made or painted under the old format goes; the count goes on from the screen's
  jobs_.clear(); ready_.clear(); hostOf_.clear(); toUpload_.clear(); dataOf_.clear(); spare_.clear();
  made_ = shown_;
  stop_ = false;
  gen_++;
  // ai: each thread at the priority a user's process may give it (sys/priority.h)
  producer_ = std::thread([this] { sys::raiseThread(); produce(); });
  if (gpu) painters_.emplace_back([this] { sys::raiseThread(); paintGpu(); });
  else {
    for (int k = 0; k < fmt_.threads; k++) painters_.emplace_back([this, k] { sys::raiseThread(); paint(k); });
    uploader_ = std::thread([this] { sys::raiseThread(); upload(); });
  }
  return "";
}

std::string Sender::prepareGpu(const std::string& assets, const std::string& device, const wg::DeviceExtras* extras) {
  if (extras) extras_ = std::make_shared<wg::DeviceExtras>(*extras);
  if (gpu_) return "";
  try { gpu_ = GpuPainter::create(assets, nullptr, device, extras_.get()); prepareWhy_.clear(); return ""; }
  catch (const std::exception& e) { prepareWhy_ = e.what(); return prepareWhy_; }
}

wg::Device* Sender::device() const { return gpu_ ? gpu_->vk() : nullptr; }

// ai: Frames made ahead of the screen, up to the ring's depth less the margin (TxFormat): frame seq goes to slot seq
// ai: mod depth, so the slot's last frame (seq - depth) must be taken and past every reader's hold before it is made.
// ai: A frame's blocks are filled outside mu_ (under xferMu_, the transfer's), so a deep queue never holds the lock
// ai: against the screen's take.
void Sender::produce() {
  std::unique_lock<std::mutex> l(mu_);
  const int T = blocks_ * fmt_.codes, B = blockBytes_, room = std::max(1, depth_ - fmt_.margin);   // ai: a frame's blocks: its codes' one after the other
  std::vector<xfer_id_t> ids(T);
  // ai: Rate refresh (setRefresh): the frame before's blocks, which a held rate's blocks repeat; none at a configure's
  // ai: first frame (this thread is new), so it is new throughout
  std::vector<uint8_t> last;
  std::vector<uint8_t> held(T);
  std::vector<xfer_id_t> hostIds;   // ai: a frame's data ids whose payloads the GPU encoder reads back (the CPU paints)
  std::vector<int> hostAt;
  std::vector<uint8_t> hostOut;
  uint8_t order[OB_CAR_N];   // ai: the carousel's order for pass orderOf under orderKey (carouselPosition), made again as either moves
  uint64_t orderOf = ~0ull;
  uint32_t orderKey = 0;
  for (;;) {
    cv_.wait(l, [&] { return stop_ || made_ < shown_ + static_cast<uint64_t>(room); });
    if (stop_) return;
    Job j{made_++, std::vector<uint8_t>(static_cast<size_t>(T) * B), 0, carousel_, {0, 0}};
    for (int c = 0; j.car > 0 && c < fmt_.codes; c++) {
      const uint64_t slot = static_cast<uint64_t>(fmt_.codes) * (j.seq / static_cast<uint64_t>(j.car)) + static_cast<uint64_t>(c);
      if (slot / OB_CAR_N != orderOf || carKey_ != orderKey) { orderOf = slot / OB_CAR_N; orderKey = carKey_; carouselPass(orderKey, orderOf, order); }
      j.pos[c] = order[slot % OB_CAR_N];
    }
    int every[4];
    std::copy(refresh_, refresh_ + 4, every);
    l.unlock();
    int fresh = 0;
    for (int t = 0; t < T; t++) {
      const int r = every[rateOf_[static_cast<size_t>(t % blocks_)]];
      held[t] = !last.empty() && r > 1 && j.seq % static_cast<uint64_t>(r) != 0;
      fresh += !held[t];
    }
    std::string err;
    {
      std::lock_guard<std::mutex> x(xferMu_);
      if (fresh) {
        if (xfer_) xfer_->frameIds(ids.data(), fresh);
        else for (int k = 0; k < fresh; k++) ids[k] = nextId_++;
      }
      // ai: a file's data payloads: the transfer's (the CPU's encoders), none where the GPU makes them in the encode (the
      // ai: GPU paints), or read back from the GPU (the CPU paints a file whose blocks the GPU makes)
      const bool gpuFile = holder_.load() == RQ_GPU;
      hostIds.clear();
      hostAt.clear();
      for (int t = 0, k = 0; t < T; t++) {
        uint8_t* o = j.blocks.data() + static_cast<size_t>(t) * B;
        if (held[t]) { std::memcpy(o, last.data() + static_cast<size_t>(t) * B, static_cast<size_t>(B)); continue; }
        const xfer_id_t id = ids[k++];
        xfer_id_put(o, id);
        if (xfer_) {
          if (!gpuFile || control(id)) xfer_->block(id, o + ID_BYTES);
          else if (!onGpu_) { hostIds.push_back(id); hostAt.push_back(t); }
        } else if (!onGpu_) stream_fill(id, o + ID_BYTES, PAYLOAD);   // ai: the GPU makes the test stream from the id itself
        if (!xfer_ || !control(id)) j.data++;   // ai: new data blocks only: a repeat adds nothing to the lap
      }
      if (!hostIds.empty()) {
        hostOut.resize(hostIds.size() * PAYLOAD);
        try {
          std::lock_guard<std::mutex> g(devMu_);
          stream_->host(hostIds.data(), static_cast<uint32_t>(hostIds.size()), hostOut.data());
        } catch (const std::exception& e) { err = e.what(); }
        for (size_t i = 0; err.empty() && i < hostAt.size(); i++)
          std::memcpy(j.blocks.data() + static_cast<size_t>(hostAt[i]) * B + ID_BYTES, hostOut.data() + i * PAYLOAD, PAYLOAD);
      }
    }
    if (!err.empty()) {
      l.lock();
      error_ = "the GPU's RaptorQ stopped: " + err;
      cv_.notify_all();
      return;
    }
    last = j.blocks;
    l.lock();
    dataOf_[j.seq] = j.data;
    jobs_.push_back(std::move(j));
    cv_.notify_all();
  }
}

// ai: A painter: its own codec, the oldest frame made, the pilots' count its number mod 4 (src/lizard_phy.h lizard_parity).
// ai: The frame grey, packed as the ring's rows (4 x rowWords bytes a row; with two codes each symbol painted on its
// ai: own and copied into its place, the gap the margin's byte beside it, as lizard-web/send-worker.mjs does), handed
// ai: to the uploader.
void Sender::paint(int) {
  lizard_t f{};
  TxFormat fm;
  int n = 0;
  size_t rowBytes = 0, bytes = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    fm = fmt_;
    n = n_;
    rowBytes = 4ull * gpu_->rowWords();
    bytes = 4ull * gpu_->frameWords();
  }
  if (lizard_init(&f, n, fm.subch, fm.span)) {
    std::lock_guard<std::mutex> l(mu_);
    error_ = "a painter's codec failed";
    return;
  }
  lizard_fmt_fps(&f, fm.fps);
  // ai: the animated header (setCarousel): the format's own, version, fps and extension bytes 0 and 0 (k 3)
  uint8_t hdr[OB_CAR_SYM * OB_CAR_N];
  const uint8_t content[4] = {static_cast<uint8_t>(fm.subch / LIZARD_STEP), static_cast<uint8_t>(fm.fps), 0, 0};
  const int hk = ob_car_header(content, 4, hdr);
  std::vector<float> drive(static_cast<size_t>(f.px) * f.px);
  const int W = side_, codes = fm.codes, gap = gap_;
  const size_t per = static_cast<size_t>(blocks_) * blockBytes_;
  std::vector<uint8_t> part(static_cast<size_t>(W) * W);
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !jobs_.empty(); });
    if (stop_) break;
    Job j = std::move(jobs_.front());
    jobs_.pop_front();
    std::vector<uint8_t> out;
    if (!spare_.empty()) { out = std::move(spare_.back()); spare_.pop_back(); }
    l.unlock();
    out.resize(bytes);
    const double t0 = nowMs();
    lizard_parity(&f, static_cast<int>(j.seq & 3));
    if (j.car <= 0 && f.fmt.kind != OB_FMT_STATIC) {
      const uint8_t b[4] = {0, 0, 0, 0};   // ai: the carousel turned off: the static word again, extension bytes 0 and 0
      lizard_fmt_word(&f, OB_FMT_STATIC, b);
    }
    for (int k = 0; k < codes; k++) {
      if (j.car > 0 && hk > 0) {
        // ai: code k's symbol (setCarousel): the position the producer drew for it
        const int p = j.pos[k];
        uint8_t b[4] = {static_cast<uint8_t>(p), 0, 0, 0};
        ob_car_symbol(hdr, hk, p, b + 1);
        lizard_fmt_word(&f, OB_FMT_ANIM, b);
      }
      lizard_encode(&f, j.blocks.data() + k * per, drive.data());
      lizard_paint_grey(&f, drive.data(), part.data());
      const size_t x0 = static_cast<size_t>(k) * (W + gap);
      for (int y = 0; y < W; y++) {
        uint8_t* o = out.data() + y * rowBytes + x0;
        const uint8_t* src = part.data() + static_cast<size_t>(y) * W;
        std::memcpy(o, src, W);
        if (k + 1 < codes) std::memset(o + W, src[W - 1], gap);
      }
    }
    const double ms = nowMs() - t0;
    l.lock();
    paintMsWin_ = paintMsWin_ + ms;
    paintedWin_++;
    if (j.seq >= shown_) toUpload_[j.seq] = std::move(out);
    cv_.notify_all();
  }
  l.unlock();
  lizard_free(&f);
}

// ai: The uploader (the CPU painters' mode): each painted frame into its slot of the ring, in order, then ready; the
// ai: one thread recording on the device then (wg records on one thread).
void Sender::upload() {
  bool host = false;
  int depth = 1;
  {
    std::lock_guard<std::mutex> l(mu_);
    host = fmt_.hostFrames;
    depth = std::max(1, depth_);
  }
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !toUpload_.empty(); });
    if (stop_) break;
    auto it = toUpload_.begin();
    const uint64_t seq = it->first;
    std::vector<uint8_t> buf = std::move(it->second);
    toUpload_.erase(it);
    l.unlock();
    const int slot = static_cast<int>(seq % static_cast<uint64_t>(depth));
    std::string err;
    try {
      std::lock_guard<std::mutex> g(devMu_);
      gpu_->upload(slot, buf.data());
    } catch (const std::exception& e) { err = e.what(); }
    l.lock();
    if (!err.empty()) { error_ = "the frame upload stopped: " + err; cv_.notify_all(); break; }
    if (seq >= shown_) {
      ready_[seq] = slot;
      if (host) hostOf_[seq] = std::move(buf); else spare_.push_back(std::move(buf));
    }
    cv_.notify_all();
  }
}

// ai: The GPU's painter: the oldest frames made, up to an encode's worth (consecutive, so the first's count and one more
// ai: each is every frame's own; never past the ring's end), one submission into their slots; their host copies too
// ai: where the host takes frames by take().
// ai: Batched (2026-11-08; the web encoder's pump, lizard-web send.mjs): in steady state a frame is made each time one is
// ai: shown, so a painter taking what was made painted a frame a submission. It waits for `fit` frames: as many as the
// ai: newest encode's GPU ms a frame puts in half a frame interval (the asked rate's, 60's at most), so an encode never
// ai: holds the GPU from the presenter's next picture; at most an encode's worth, never past the ring's end. While fewer
// ai: than a quarter of the ring is painted ahead (the start, a stall), what is made goes at once, an encode's worth at
// ai: most, as before. LIZ_TX_BATCH=0 (the lab's A/B): a frame a submission as before.
void Sender::paintGpu() {
  const int R = gpu_->frames(), D = std::max(1, gpu_->depth());
  bool host = false;
  double budget = 0;
  int BB = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    host = fmt_.hostFrames;
    budget = 500.0 / std::max(60, fmt_.fps);
    BB = blockBytes_;
  }
  const char* tb = std::getenv("LIZ_TX_BATCH");
  const bool batched = !(tb && std::string(tb) == "0");
  double msPer = 0;   // ai: the newest encode's GPU ms a frame
  const auto fitOf = [&] { return batched && msPer > 0 ? std::clamp(static_cast<int>(budget / msPer), 1, R) : 1; };
  std::vector<std::vector<uint8_t>> outs(R);
  std::vector<uint8_t> blocks, pos;
  std::vector<Job> js;
  // ai: frames painted ahead: those ready and those in flight (counted as painted: a start that counted only the ready
  // ai: ones stayed in its catch-up with encodes in flight, an encode a frame, the S26's CPU a frame +40%)
  int inFlight = 0;
  const auto ahead = [&] { return static_cast<int>(ready_.size()) + inFlight; };
  const auto due = [&] {
    if (jobs_.empty()) return false;
    const int room = D - static_cast<int>(jobs_.front().seq % static_cast<uint64_t>(D));
    return !batched || ahead() < D / 4 || static_cast<int>(jobs_.size()) >= std::min(fitOf(), room);
  };
  // ai: Encodes in flight (2026-11-16): an encode submitted, the thread goes on to the next without waiting for the GPU;
  // ai: it waits for the oldest only when GpuPainter::FLIGHTS are in flight or it has nothing else to do, and a frame is
  // ai: ready (the screen's to take) once its encode is finished. A host that takes frames from host copies needs each
  // ai: encode's pixels (one readback buffer): one in flight there.
  struct Flight { GpuPainter::Pending p; std::vector<uint64_t> seqs; };
  std::deque<Flight> fl;
  const size_t flights = host ? 1 : static_cast<size_t>(GpuPainter::FLIGHTS);
  // ai: the oldest flight finished (waited for where `block`): its frames ready, the window's figures; under mu_ (let go
  // ai: meanwhile); false where the painter failed
  const auto retire = [&](std::unique_lock<std::mutex>& l, bool block) {
    Flight& f = fl.front();
    double ms = -1;
    std::string err;
    l.unlock();
    try { if (block || gpu_->done(f.p)) ms = gpu_->finish(f.p); } catch (const std::exception& e) { err = e.what(); }
    l.lock();
    if (!err.empty()) { error_ = "the GPU painter stopped: " + err; cv_.notify_all(); return false; }
    if (ms < 0) return true;
    const int k = static_cast<int>(f.seqs.size());
    paintMsWin_ = paintMsWin_ + ms;
    paintedWin_ += k;
    encodesWin_++;
    if (ms > 0) msPer = ms / k;
    for (int i = 0; i < k; i++) if (f.seqs[i] >= shown_) {
      ready_[f.seqs[i]] = static_cast<int>(f.seqs[i] % static_cast<uint64_t>(D));
      if (host) hostOf_[f.seqs[i]] = std::move(outs[i]);
    }
    inFlight -= k;
    fl.pop_front();
    cv_.notify_all();
    return true;
  };
  std::unique_lock<std::mutex> l(mu_);
  for (bool ok = true; ok;) {
    // ai: what the GPU finished, without waiting
    while (ok && !fl.empty()) {
      const size_t before = fl.size();
      ok = retire(l, false);
      if (fl.size() == before) break;
    }
    if (!ok || stop_) break;
    if (due() && fl.size() < flights) {
      const int room = D - static_cast<int>(jobs_.front().seq % static_cast<uint64_t>(D));
      const bool catchUp = !batched || ahead() < D / 4;
      const int k = std::min({catchUp ? R : fitOf(), static_cast<int>(jobs_.size()), room});
      js.clear();
      for (int i = 0; i < k; i++) { js.push_back(std::move(jobs_.front())); jobs_.pop_front(); }
      // ai: each frame's codes' carousel positions, OB_CAR_N the static word (a frame made with it off), none where no
      // ai: frame of the encode has it on or the GPU cannot paint it
      bool anyCar = false;
      pos.assign(static_cast<size_t>(k) * fmt_.codes, OB_CAR_N);
      for (int i = 0; gpuWords_ && i < k; i++) for (int c = 0; js[i].car > 0 && c < fmt_.codes; c++) { pos[static_cast<size_t>(i) * fmt_.codes + c] = js[i].pos[c]; anyCar = true; }
      if (host) for (int i = 0; i < k; i++) if (outs[i].empty() && !spare_.empty()) { outs[i] = std::move(spare_.back()); spare_.pop_back(); }
      l.unlock();
      const size_t per = js[0].blocks.size();
      blocks.resize(per * k);
      for (int i = 0; i < k; i++) std::memcpy(blocks.data() + per * i, js[i].blocks.data(), per);
      std::string err;
      // ai: a file whose blocks the GPU makes: an encode naming any data block takes every block from the stream in its
      // ai: own submission, the ids alone handed over (the producer left their payloads empty); the test stream's ids go
      // ai: packed (encode's test)
      std::function<void(wg::Encoder&)> fill;
      bool viaStream = false;
      if (xfer_ && holder_.load() == RQ_GPU)
        for (size_t t = 0; !viaStream && t < blocks.size(); t += static_cast<size_t>(BB)) viaStream = !control(xfer_id_get(blocks.data() + t));
      if (viaStream) fill = [&](wg::Encoder& e) { stream_->blocks(e, gpu_->blocksBuffer(), blocks.data(), static_cast<uint32_t>(blocks.size() / static_cast<size_t>(BB)), static_cast<size_t>(BB)); };
      Flight f;
      for (const Job& j : js) f.seqs.push_back(j.seq);
      try {
        std::lock_guard<std::mutex> g(devMu_);
        f.p = gpu_->submit(blocks.data(), k, static_cast<int>(js[0].seq & 3), host ? outs.data() : nullptr, js[0].seq, !xfer_, anyCar ? pos.data() : nullptr, fill);
      } catch (const std::exception& e) { err = e.what(); }
      l.lock();
      if (!err.empty()) { error_ = "the GPU painter stopped: " + err; cv_.notify_all(); break; }
      inFlight += k;
      fl.push_back(std::move(f));
      if (host) ok = retire(l, true);
      continue;
    }
    if (!fl.empty()) { ok = retire(l, true); continue; }
    cv_.wait(l, [&] { return stop_ || due(); });
  }
  // ai: the encodes still in flight finished before the painter is configured again or let go (their buffers in use)
  l.unlock();
  for (auto& f : fl) try { gpu_->finish(f.p); } catch (const std::exception&) {}
}

bool Sender::ready() {
  std::lock_guard<std::mutex> l(mu_);
  return ready_.count(shown_) != 0;
}

bool Sender::takeSlot(uint64_t* seq, int* slot) {
  std::lock_guard<std::mutex> l(mu_);
  auto it = ready_.find(shown_);
  if (it == ready_.end()) return false;
  *seq = shown_;
  *slot = it->second;
  ready_.erase(it);
  auto h = hostOf_.find(shown_);
  if (h != hostOf_.end()) { spare_.push_back(std::move(h->second)); hostOf_.erase(h); }
  auto d = dataOf_.find(shown_);
  int data = 0;
  if (d != dataOf_.end()) { data = d->second; dataOf_.erase(d); }
  shown_++;
  takenWin_++;
  dataWin_ += data;
  dataAll_ += data;
  cv_.notify_all();
  return true;
}

bool Sender::take(uint8_t* dst, int stride) {
  std::vector<uint8_t> buf;
  int data = 0;
  size_t srcRow = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    auto it = ready_.find(shown_);
    if (it == ready_.end()) return false;
    auto h = hostOf_.find(shown_);
    if (h == hostOf_.end()) { error_ = "take() needs TxFormat.hostFrames"; return false; }
    buf = std::move(h->second);
    hostOf_.erase(h);
    ready_.erase(it);
    auto d = dataOf_.find(shown_);
    if (d != dataOf_.end()) { data = d->second; dataOf_.erase(d); }
    shown_++;
    // ai: a GPU frame read back is width_ bytes a row; a CPU frame is packed as the ring's rows
    srcRow = buf.size() == static_cast<size_t>(width_) * side_ ? static_cast<size_t>(width_) : 4ull * gpu_->rowWords();
  }
  // ai: each grey level's RGBA (little-endian R G B A) on the line from the ink to the paper (setColours)
  uint32_t lut[256];
  const uint32_t ink = ink_, paper = paper_;
  for (uint32_t v = 0; v < 256; v++) {
    uint32_t px = 0xff000000u;
    for (int k = 0; k < 3; k++) {
      const uint32_t a = (ink >> (16 - 8 * k)) & 0xffu, b = (paper >> (16 - 8 * k)) & 0xffu;
      px |= ((a * (255 - v) + b * v + 127) / 255) << (8 * k);
    }
    lut[v] = px;
  }
  for (int y = 0; y < side_; y++) {
    uint32_t* d = reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(y) * stride);
    const uint8_t* g = buf.data() + static_cast<size_t>(y) * srcRow;
    for (int x = 0; x < width_; x++) d[x] = lut[g[x]];
  }
  takenWin_++;
  dataWin_ += data;
  dataAll_ += data;
  {
    std::lock_guard<std::mutex> l(mu_);
    spare_.push_back(std::move(buf));
  }
  cv_.notify_all();
  return true;
}

std::shared_ptr<wg::Buffer> Sender::ring() const { return gpu_ ? gpu_->ring() : nullptr; }
uint32_t Sender::ringFrameWords() const { return gpu_ ? gpu_->frameWords() : 0; }
uint32_t Sender::ringRowWords() const { return gpu_ ? gpu_->rowWords() : 0; }

// ai: Rate refresh (sender.h setRefresh), clamped to 1 .. REFRESH_MAX
void Sender::setRefresh(const int every[4]) {
  std::lock_guard<std::mutex> l(mu_);
  for (int r = 0; r < 4; r++) refresh_[r] = std::clamp(every[r], 1, REFRESH_MAX);
}

// ai: the animated header (sender.h setCarousel), clamped to 0 .. CAROUSEL_MAX; the producer's next frame takes it
void Sender::setCarousel(int frames) {
  std::lock_guard<std::mutex> l(mu_);
  carousel_ = std::clamp(frames, 0, CAROUSEL_MAX);
}

bool Sender::seed(uint32_t seed) {
  {
    std::lock_guard<std::mutex> l(mu_);
    carKey_ = mix32(seed ^ 0x3c6ef372u);
  }
  std::lock_guard<std::mutex> x(xferMu_);
  return xfer_ && xfer_->reseed(seed);
}

// ai: The stats a second (the app asks): label, n, subch, fps (asked), shownFps (frames on the screen a second over the
// ai: window), offeredKBs (data blocks shown a second x 471 B, 1000 B a KB, as the page's), paintMs (a frame's encode
// ai: and paint on one painter; on the GPU its ms a frame, an encode's timestamps over its frames), paintBatch (the GPU's
// ai: frames an encode over the window, 0 none), painters (the
// ai: GPU's: frames an encode), painter ("gpu" or "cpu"), device (the GPU's), gpuWhy (why not the GPU where it was
// ai: asked for), ahead (painted, not shown), side, test, and for a file chunks, made (the chunks made so far), rq
// ai: ("gpu": the GPU makes the blocks from the CPU's solved chunks, "cpu": the CPU's encoders; "" before the first
// ai: configure), rqWhy (why not the GPU), lap (data blocks a lap), pass (laps
// ai: shown, 1 the first), root, seed (the schedule's, 2026-11-14); refresh (Rate refresh's four, 7/8 to 1/2; offeredKBs counts new data blocks alone, a repeat none);
// ai: fresh (the share of a frame's blocks new in each frame under Rate refresh, by the format's blocks of each rate: the
// ai: sum of 1/every over them, over their count; 1 with none held), so a rate figured from blocks a frame counts new ones;
// ai: carousel (the animated header's frames a symbol, setCarousel, as frames made now paint it: 0 the static word, and
// ai: 0 where the GPU paints and cannot); error, the last failure.
std::string Sender::stats() {
  const double now = nowMs(), secs = std::max(1e-3, (now - winStart_) / 1000);
  if (secs >= 0.5) {
    lastFps_ = static_cast<double>(takenWin_.exchange(0)) / secs;
    lastKBs_ = static_cast<double>(dataWin_.exchange(0)) * PAYLOAD / 1000 / secs;
    const uint64_t p = paintedWin_.exchange(0), en = encodesWin_.exchange(0);
    const double ms = paintMsWin_.exchange(0);
    if (p) lastPaintMs_ = ms / static_cast<double>(p);
    lastPaintBatch_ = en ? static_cast<double>(p) / static_cast<double>(en) : 0;
    winStart_ = now;
  }
  std::lock_guard<std::mutex> l(mu_);
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(1);
  o << "{\"label\":\"" << label_ << "\",\"n\":" << n_ << ",\"subch\":" << fmt_.subch << ",\"fps\":" << fmt_.fps
    << ",\"shownFps\":" << lastFps_ << ",\"offeredKBs\":" << lastKBs_ << ",\"paintMs\":" << lastPaintMs_ << ",\"paintBatch\":" << lastPaintBatch_
    << ",\"painters\":" << fmt_.threads << ",\"painter\":\"" << (onGpu_ ? "gpu" : "cpu") << "\",\"device\":\"" << esc(gpuDevice_)
    << "\",\"gpuWhy\":\"" << esc(gpuWhy_) << "\",\"ahead\":" << ready_.size() << ",\"depth\":" << depth_ << ",\"side\":" << side_ << ",\"width\":" << width_
    << ",\"codes\":" << fmt_.codes << ",\"shown\":" << shown_ << ",\"blocks\":" << blocks_ << ",\"tiers\":\"" << tiers_ << "\""
    << ",\"refresh\":\"" << refresh_[0] << "," << refresh_[1] << "," << refresh_[2] << "," << refresh_[3] << "\""
    << ",\"carousel\":" << (onGpu_ && !gpuWords_ ? 0 : carousel_)
    << ",\"test\":" << (xfer_ ? "false" : "true");
  double fresh = 0;
  for (const uint8_t r : rateOf_) fresh += 1.0 / refresh_[r];
  o.precision(4);
  o << ",\"fresh\":" << (rateOf_.empty() ? 1.0 : fresh / static_cast<double>(rateOf_.size()));
  o.precision(1);
  if (xfer_) {
    std::lock_guard<std::mutex> x(xferMu_);
    const uint32_t lap = xfer_->lap();
    o.precision(2);
    o << ",\"chunks\":" << xfer_->chunks() << ",\"made\":" << xfer_->madeCount()
      << ",\"rq\":\"" << (holder_ == RQ_GPU ? "gpu" : holder_ == RQ_CPU ? "cpu" : "") << "\",\"rqWhy\":\"" << esc(rqWhy_) << "\""
      << ",\"lap\":" << lap << ",\"pass\":" << (lap ? 1.0 + static_cast<double>(dataAll_) / lap : 1.0)
      << ",\"root\":\"" << xfer_->rootHex() << "\",\"seed\":" << xfer_->seed();
  }
  // ai: a transfer's worker that failed (xfer_tx.h fault: its chunks cannot all be made) is the error where none other is
  std::string err = error_;
  if (err.empty() && xfer_) { std::lock_guard<std::mutex> x(xferMu_); err = xfer_->fault(); }
  o << ",\"error\":\"" << esc(err) << "\"}";
  return o.str();
}

}  // namespace lizard
