// ai: The app's sender (sender.h): lizard-web/send-worker.mjs and the page's paint ahead, natively.
#include "sender.h"

#include "lizfile.h"
#include "wg.h"
#include "json.hpp"

#include <algorithm>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdio>
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

// ai: An export's state (sender.h exportStart), under mu_: the file, how many frames go (target: 0 until the lap is
// ai: known) from which (from), the frames taken and not packed (raw), packed and not written (packed: by the writer's
// ai: order, a fused painter's every frame, the others' admitted ones alone), the writer's next frame, and its threads.
struct Sender::Sink {
  // ai: a frame's entries and their ZIP CRC-32s, worked out where they were packed (the writer's one thread spared them)
  struct Packed { bool in = false; std::vector<std::vector<uint8_t>> entries; std::vector<uint32_t> crcs; };
  ExportJob job;
  std::unique_ptr<lizfile::Writer> w;
  bool fused = false, drain = false, gpuPack = false, packDecided = false;
  // ai: GPU packing with the codes shown: the frames the screen took, in turn (takeSlot, take), and the first not yet
  // ai: read from its ring slot (produce makes no frame into a slot still to be read)
  std::deque<uint64_t> tapped;
  uint64_t readTo = 0;
  int target = 0, written = 0;
  uint64_t from = ~0ull, next = ~0ull, bytes = 0;
  std::deque<std::pair<uint64_t, std::vector<uint8_t>>> raw;
  std::map<uint64_t, Packed> packed;
  // ai: the GPU's batches laid out as the file's entries (lizfile::SpanEntry: each its local header, then its bytes),
  // ai: by their first frame: the drain fills them on the cores, the writer writes each with one call
  // ai: a span's buffer: its bytes uninitialised (the cores fill every one), kept for the next batch once written (spare)
  struct Buf { std::unique_ptr<uint8_t[]> p; size_t cap = 0, n = 0; };
  struct Span { uint64_t count = 0; int admitted = 0; Buf buf; std::vector<lizfile::SpanEntry> es; };
  std::map<uint64_t, Span> spans;
  std::vector<Buf> spare;
  bool done = false, stop = false;
  std::string error;
  std::vector<std::thread> packers;
  std::thread writer, drainer;
};

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
  // ai: the export's writer, then the GPU's packer, before the painter's device goes (zstd_ is declared before gpu_, so
  // ai: left to the members' order its buffers outlived the device: a sender freed after a GPU-packed export aborted
  // ai: in vkDestroyBuffer, 2026-11-18)
  sink_.reset();
  zstd_.reset();
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
  joinSink();
}

std::unique_ptr<Sender> Sender::play(const std::string& path, std::string& why) {
  auto r = lizfile::Reader::open(path, why);
  if (!r) return nullptr;
  if (r->kind() != lizfile::Kind::Stream) { why = path + " is a replay, not a stream: Upload decodes it"; return nullptr; }
  auto p = std::make_unique<StreamPlay>();
  p->path = path;
  p->meta = r->meta();
  try {
    const auto m = nlohmann::json::parse(p->meta);
    const auto& fm = m.at("format");
    p->blocks = fm.at("blocks").get<int>(); p->ring = fm.at("ring").get<int>(); p->codes = fm.at("codes").get<int>();
    p->frameBlocks = fm.value("frame_blocks", 0);
    p->fps = m.at("fps").get<int>(); p->gap = m.value("gap", 0); p->pictures = m.value("pictures", int(r->frames().size()) / std::max(1, p->codes)); p->side = m.at("h").get<int>();
    p->test = m.value("test", false);
    if (m.contains("file")) {
      const auto& fi = m["file"];
      p->name = fi.value("name", ""); p->type = fi.value("type", ""); p->root = fi.value("root", "");
      p->length = fi.value("length", uint64_t(0)); p->lap = fi.value("lap", uint64_t(0));
    }
  } catch (const std::exception& e) { why = path + ": its meta.json is not a stream's (" + e.what() + ")"; return nullptr; }
  if (p->codes < 1 || p->codes > 2 || p->ring < 0 || p->ring >= LIZARD_RINGS || p->fps < 1 || p->pictures < 1 ||
      r->frames().size() != size_t(p->pictures) * size_t(p->codes)) { why = path + ": its meta.json is not a stream's"; return nullptr; }
  // ai: its frames are its format's: a picture's side the symbol and its quiet zone at the ring's pixels a module, the
  // ai: gap whole modules
  lizard_t probe{};
  const int subch = LIZARD_STEP * p->blocks, span = 2 * LIZARD_RING[p->ring];
  if (lizard_init(&probe, lizard_n_for(subch), subch, span)) { why = path + ": LIZARD-" + std::to_string(subch) + " is no format"; return nullptr; }
  const int side = probe.px + 2 * LIZARD_QUIET * probe.pxm, pxm = probe.pxm;
  lizard_free(&probe);
  p->gapModules = p->codes > 1 ? p->gap / pxm : 0;
  for (const auto& fi : r->frames())
    if (int(fi.w) != side || int(fi.h) != side || p->side != side || (p->codes > 1 && p->gapModules * pxm != p->gap)) {
      why = path + ": its frames are not LIZARD-" + std::to_string(subch) + "'s"; return nullptr;
    }
  std::unique_ptr<Sender> s(new Sender());
  s->play_ = std::move(p);
  s->carKey_ = freshKey();
  s->winStart_ = nowMs();
  return s;
}

std::string Sender::configure(const TxFormat& fIn) {
  stopAll();
  // ai: a configure from outside with another format ends an export (a stream has one format); with the same (a phone's
  // ai: view resized, a re-pick that keeps the format) the export goes on, what was pending settled first; exportStart's
  // ai: and exportShow's own keep it
  {
    std::unique_lock<std::mutex> l(mu_);
    if (sink_ && !keepSink_) {
      const bool same = fIn.subch == asked_.subch && fIn.span == asked_.span && fIn.fps == asked_.fps && fIn.codes == asked_.codes &&
                        (fIn.codes < 2 || fIn.gap == asked_.gap) && fIn.painter == asked_.painter;
      if (same && !sink_->done && !sink_->stop) sinkSettle(l);
      else sinkEnd(l, "the format changed");
    }
  }
  asked_ = fIn;
  // ai: a stream's format is its own (play): unpacked on the CPU into the ring
  TxFormat f = fIn;
  if (play_) {
    f.subch = LIZARD_STEP * play_->blocks; f.span = 2 * LIZARD_RING[play_->ring]; f.fps = play_->fps; f.codes = play_->codes;
    f.gap = play_->codes > 1 ? play_->gapModules : f.gap; f.painter = 0;
  }
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
  int aheadFrames = std::max(1, static_cast<int>(std::lround(f.fps * f.aheadSecs)));
  uint64_t aheadBytes = f.aheadBytes;
  // ai: an export with no screen (exportStart): the ring as deep as an eighth of the device's memory allows, 1 GiB and
  // ai: 512 frames at most, so the painter fills one batch while the packer reads the last (sinkDrainGpu)
  {
    std::lock_guard<std::mutex> l(mu_);
    if (sink_ && !sink_->job.display) {
      aheadFrames = std::max(aheadFrames, 512);
      aheadBytes = std::max<uint64_t>(aheadBytes, std::min<uint64_t>(1ull << 30, gpu_->vk()->limits.deviceLocalBytes / 8));
    }
  }
  // ai: the GPU painter paints the anchors as the C does (their squares the C's bytes, gpu_painter.h), and auto's check
  // ai: holds its frames to the C's with them. Encodes of up to TX_GPU_FRAMES frames: the painter runs ahead of the
  // ai: screen, so a deep encode adds no latency and spreads each dispatch's fixed cost (halved where the device
  // ai: cannot hold them).
  if (f.painter != 0) {
    try {
      gpuWhy_ = gpu_->configure(f.subch, f.span, f.fps, TX_GPU_FRAMES, f.codes, f.gap, aheadFrames, aheadBytes, f.margin, false);
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
    try { e = gpu_->configure(f.subch, f.span, f.fps, 1, f.codes, f.gap, aheadFrames, aheadBytes, f.margin, true); }
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
  // ai: an export's frames: the host copies its taps and drain read (exportStart)
  if (sink_) {
    sink_->fused = !sink_->job.display && !gpu;
    sink_->drain = !sink_->job.display && gpu;
    fmt_.hostFrames = fmt_.hostFrames || (!sink_->fused && !sink_->gpuPack);
  }
  gpuWords_ = gpu && gpu_->wordsWhy().empty();
  n_ = n;
  // ai: an export with no screen on the CPU: a painter a core, each packing what it paints (exportStart)
  const bool fusedNow = sink_ && !sink_->job.display && !gpu;
  fmt_.threads = gpu ? gpu_->frames() : fusedNow ? static_cast<int>(std::max(1u, std::thread::hardware_concurrency())) : std::max(1, f.threads);
  onGpu_ = gpu;
  gpuDevice_ = gpu_->device();
  depth_ = gpu_->depth();
  blocks_ = probe.blocks;
  if (!byPlace_.empty() && byPlace_.size() != static_cast<size_t>(blocks_)) byPlace_.clear();   // ai: another format's places
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
  // ai: an export with no screen on the GPU: its frames packed there too (gpu_zstd.h), straight from the ring, where the
  // ai: packer is made; the painter then keeps no host copies (none is read). Else the CPU's packers, as asked.
  // ai: How an export packs, decided once, at its first configure, for every frame of its file (lizfile::Packing): on
  // ai: the GPU where it paints and the packer is made (its frames read straight from the ring, codes shown or hidden),
  // ai: else on the CPU at the job's level. With the codes hidden the GPU's frames are drained (no host copies); shown,
  // ai: the screen takes them and they are packed from their slots, which the painter reuses only once read (produce).
  if (sink_ && !sink_->packDecided) {
    sink_->packDecided = true;
    if (sink_->job.gpu && gpu) {
      std::string zwhy;
      try {
        if (!zstd_ || zstd_->side() != side_) zstd_ = GpuZstd::create(*gpu_->vk(), assets_, side_);
      } catch (const std::exception& e) { zstd_.reset(); zwhy = e.what(); }
      if (zstd_) {
        sink_->gpuPack = true;
        lizfile::Packing pk;
        pk.gpu = true;
        sink_->w->setPacking(pk);
      } else std::fprintf(stderr, "export: the CPU packs, not the GPU: %s\n", zwhy.c_str());
    }
  }
  if (sink_ && sink_->gpuPack) {
    if (!gpu || !zstd_) { sink_->error = "the GPU packs this export and paints no more"; sink_->stop = true; }
    else if (sink_->drain) fmt_.hostFrames = false;
  }
  // ai: what was made or painted under the old format goes; the count goes on from the screen's
  jobs_.clear(); ready_.clear(); hostOf_.clear(); toUpload_.clear(); dataOf_.clear(); spare_.clear();
  made_ = shown_;
  stop_ = false;
  gen_++;
  // ai: each thread at the priority a user's process may give it (sys/priority.h)
  playBase_ = made_;
  if (play_) label_ = "LIZARD-" + std::to_string(f.subch) + (f.codes > 1 ? " x " + std::to_string(f.codes) : "");
  producer_ = std::thread([this] { sys::raiseThread(); if (play_) playProduce(); else produce(); });
  if (gpu) painters_.emplace_back([this] { sys::raiseThread(); paintGpu(); });
  else {
    for (int k = 0; k < fmt_.threads; k++) painters_.emplace_back([this, k] { sys::raiseThread(); if (play_) playUnpack(k); else paint(k); });
    uploader_ = std::thread([this] { sys::raiseThread(); upload(); });
  }
  if (sink_) sinkStart();
  return "";
}

// ai: A stream's frames made (play): each the next picture of the loop, ahead of the screen as produce's; its blocks a
// ai: picture the stream's (the stats' rate and pass)
void Sender::playProduce() {
  std::unique_lock<std::mutex> l(mu_);
  const int room = std::max(1, depth_ - fmt_.margin);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || made_ < shown_ + static_cast<uint64_t>(room); });
    if (stop_) return;
    Job j{made_++, {}, play_->frameBlocks, 0, {0, 0}};
    dataOf_[j.seq] = j.data;
    jobs_.push_back(std::move(j));
    cv_.notify_all();
  }
}

// ai: An unpacker (play): its own reader; the oldest frame made, picture (seq - playBase_) mod pictures, its entries
// ai: (two at 2:1: the left square, then the right past the gap, the gap the margin's byte beside it, as paint lays two
// ai: codes) unpacked into the ring's rows for the uploader
void Sender::playUnpack(int) {
  std::string why;
  auto rd = lizfile::Reader::open(play_->path, why);
  size_t rowBytes = 0, bytes = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!rd) { error_ = "the stream does not read: " + why; cv_.notify_all(); return; }
    rowBytes = 4ull * gpu_->rowWords();
    bytes = 4ull * gpu_->frameWords();
  }
  const int W = side_, codes = fmt_.codes, gap = gap_;
  const uint64_t pictures = static_cast<uint64_t>(play_->pictures);
  std::vector<uint8_t> part;
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !jobs_.empty(); });
    if (stop_) break;
    Job j = std::move(jobs_.front());
    jobs_.pop_front();
    std::vector<uint8_t> out;
    if (!spare_.empty()) { out = std::move(spare_.back()); spare_.pop_back(); }
    const uint64_t pic = (j.seq - playBase_) % pictures;
    l.unlock();
    out.resize(bytes);
    const double t0 = nowMs();
    bool ok = true;
    for (int k = 0; ok && k < codes; k++) {
      ok = rd->frame(static_cast<size_t>(pic) * codes + k, part, why);
      const size_t x0 = static_cast<size_t>(k) * (W + gap);
      for (int y = 0; ok && y < W; y++) {
        uint8_t* o = out.data() + y * rowBytes + x0;
        const uint8_t* src = part.data() + static_cast<size_t>(y) * W;
        std::memcpy(o, src, W);
        if (k + 1 < codes) std::memset(o + W, src[W - 1], gap);
      }
    }
    const double ms = nowMs() - t0;
    l.lock();
    if (!ok) { error_ = "the stream's frame " + std::to_string(pic) + " does not read: " + why; cv_.notify_all(); break; }
    paintMsWin_ = paintMsWin_ + ms;
    paintedWin_++;
    if (j.seq >= shown_) toUpload_[j.seq] = std::move(out);
    cv_.notify_all();
  }
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
  // ai: a frame's blocks: its codes' one after the other. room: frames made ahead of the screen, the ring's less the
  // ai: margin; with painters packing for an export (no ring read), two a painter
  const int T = blocks_ * fmt_.codes, B = blockBytes_, room = sink_ && sink_->fused ? 2 * fmt_.threads : std::max(1, depth_ - fmt_.margin);
  std::vector<xfer_id_t> ids(T);
  // ai: Refresh (setRefresh, setRefreshBlocks): the frame before's blocks, which a held place repeats; none at a
  // ai: configure's first frame (this thread is new), so it is new throughout
  std::vector<uint8_t> last;
  std::vector<uint8_t> held(T);
  std::vector<xfer_id_t> hostIds;   // ai: a frame's data ids whose payloads the GPU encoder reads back (the CPU paints)
  std::vector<int> hostAt;
  std::vector<uint8_t> hostOut;
  uint8_t order[OB_CAR_N];   // ai: the carousel's order for pass orderOf under orderKey (carouselPosition), made again as either moves
  uint64_t orderOf = ~0ull;
  uint32_t orderKey = 0;
  // ai: a GPU-packed export with the codes shown: no frame into a slot whose frame the packer has yet to read
  const auto slotFree = [&] { return !sink_ || !sink_->gpuPack || !sink_->job.display || made_ < sink_->readTo + static_cast<uint64_t>(std::max(1, depth_)); };
  for (;;) {
    cv_.wait(l, [&] { return stop_ || (made_ < shown_ + static_cast<uint64_t>(room) && slotFree()); });
    if (stop_) return;
    // ai: an export with no screen makes nothing before every chunk is (its frames would carry the header alone, made as
    // ai: fast as the painter goes for nothing: a 50 MB file's chunks took 19,000 of them, 2026-11-18); the export's
    // ai: first frame is the first made after (sinkAdmit)
    if (sink_ && !sink_->job.display && xfer_ && sink_->target == 0) {
      uint64_t lap = 0;
      { std::lock_guard<std::mutex> x(xferMu_); lap = xfer_->lapWhenMade(); }
      if (lap == 0) { cv_.wait_for(l, std::chrono::milliseconds(5)); continue; }
      sinkAdmit(made_);
    }
    Job j{made_++, std::vector<uint8_t>(static_cast<size_t>(T) * B), 0, carousel_, {0, 0}};
    for (int c = 0; j.car > 0 && c < fmt_.codes; c++) {
      const uint64_t slot = static_cast<uint64_t>(fmt_.codes) * (j.seq / static_cast<uint64_t>(j.car)) + static_cast<uint64_t>(c);
      if (slot / OB_CAR_N != orderOf || carKey_ != orderKey) { orderOf = slot / OB_CAR_N; orderKey = carKey_; carouselPass(orderKey, orderOf, order); }
      j.pos[c] = order[slot % OB_CAR_N];
    }
    const std::vector<int> sched = scheduleLocked();
    l.unlock();
    int fresh = 0;
    for (int t = 0; t < T; t++) {
      held[t] = !last.empty() && !due(sched[static_cast<size_t>(t % blocks_)], j.seq);
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
  // ai: the animated header (setCarousel): the format's own configuration and the sender's rate (src/fmt.h
  // ai: ob_car_content); a static word states no rate
  uint8_t hdr[OB_CAR_SYM * OB_CAR_N], content[OB_CAR_CONTENT_MAX];
  ob_fmt_t cfg = f.fmt;
  cfg.fps = fm.fps;
  const int clen = ob_car_content(&cfg, content), hk = clen < 0 ? -1 : ob_car_header(content, clen, hdr);
  std::vector<float> drive(static_cast<size_t>(f.px) * f.px);
  const int W = side_, codes = fm.codes, gap = gap_;
  const size_t per = static_cast<size_t>(blocks_) * blockBytes_;
  std::vector<uint8_t> part(static_cast<size_t>(W) * W);
  std::unique_lock<std::mutex> l(mu_);
  // ai: an export with no screen (exportStart): this painter packs what it paints
  const bool fused = sink_ && sink_->fused;
  lizfile::FrameCodec codec(sink_ ? sink_->job.packing.level : lizfile::LEVEL_DEFAULT);
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
      const uint8_t b[4] = {0, 0, 0, 0};   // ai: the carousel turned off: the static word again
      lizard_fmt_word(&f, OB_FMT_STATIC, b);
    }
    for (int k = 0; k < codes; k++) {
      if (j.car > 0 && hk > 0) {
        // ai: code k's symbol (setCarousel): the id the producer drew for it
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
    if (fused) {
      l.lock();
      const bool in = sink_ && sinkAdmit(j.seq);
      l.unlock();
      Sink::Packed pk{in, {}, {}};
      if (in) { pk.entries = sinkEntries(codec, out); for (const auto& e : pk.entries) pk.crcs.push_back(lizfile::crc32(e.data(), e.size())); }
      l.lock();
      paintMsWin_ = paintMsWin_ + ms;
      paintedWin_++;
      if (sink_) sink_->packed[j.seq] = std::move(pk);
      spare_.push_back(std::move(out));
      cv_.notify_all();
      continue;
    }
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
      if (viaStream) fill = [&](wg::Encoder& e) {
        stream_->blocks(e, gpu_->blocksBuffer(), blocks.data(), static_cast<uint32_t>(blocks.size() / static_cast<size_t>(BB)), static_cast<size_t>(BB),
                        static_cast<uint32_t>(static_cast<size_t>(R) * per / static_cast<size_t>(BB)));
      };
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
  if (sink_ && sink_->gpuPack && sink_->job.display && !sink_->stop && !sink_->done) sink_->tapped.push_back(shown_);
  auto h = hostOf_.find(shown_);
  if (h != hostOf_.end()) { sinkTake(shown_, std::move(h->second)); hostOf_.erase(h); }
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
  uint64_t seq = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    seq = shown_;
    auto it = ready_.find(shown_);
    if (it == ready_.end()) return false;
    auto h = hostOf_.find(shown_);
    if (h == hostOf_.end()) { error_ = "take() needs TxFormat.hostFrames"; return false; }
    buf = std::move(h->second);
    hostOf_.erase(h);
    ready_.erase(it);
    if (sink_ && sink_->gpuPack && sink_->job.display && !sink_->stop && !sink_->done) sink_->tapped.push_back(shown_);
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
    sinkTake(seq, std::move(buf));
  }
  cv_.notify_all();
  return true;
}

void Sender::joinSink() {
  std::vector<std::thread> ts;
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!sink_) return;
    ts = std::move(sink_->packers);
    sink_->packers.clear();
    if (sink_->writer.joinable()) ts.push_back(std::move(sink_->writer));
    if (sink_->drainer.joinable()) ts.push_back(std::move(sink_->drainer));
  }
  cv_.notify_all();
  for (auto& t : ts) if (t.joinable()) t.join();
}

std::string Sender::exportStart(const ExportJob& job) {
  if (play_) return "a stream plays: its frames are a stream file already";
  if (job.packing.gpu) return "the GPU packs no frames in this build";
  {
    std::unique_lock<std::mutex> l(mu_);
    if (width_ <= 0) return "the sender is not configured";
    if (!xfer_ && job.frames <= 0) return "the test stream exports a count of frames";
    if (sink_) sinkEnd(l, "another export started");
  }
  std::string why;
  auto w = lizfile::Writer::create(job.path, why, lizfile::Kind::Stream, job.packing);
  if (!w) return why;
  auto k = std::make_unique<Sink>();
  k->job = job;
  k->w = std::move(w);
  {
    std::lock_guard<std::mutex> l(mu_);
    sink_ = std::move(k);
  }
  keepSink_ = true;
  const std::string e = configure(asked_);
  keepSink_ = false;
  if (!e.empty()) { std::unique_lock<std::mutex> l(mu_); if (sink_) sinkEnd(l, e); }
  return e;
}

Sender::ExportState Sender::exportState() {
  std::lock_guard<std::mutex> l(mu_);
  if (!sink_) return lastExport_;
  ExportState st;
  st.on = !sink_->done && sink_->error.empty();
  st.done = sink_->done;
  st.written = sink_->written;
  st.frames = sink_->target;
  st.bytes = sink_->bytes;
  st.path = sink_->job.path;
  st.error = sink_->error;
  return st;
}

void Sender::exportStop() {
  std::unique_lock<std::mutex> l(mu_);
  if (sink_) sinkEnd(l, sink_->done ? "" : "stopped");
}

void Sender::sinkEnd(std::unique_lock<std::mutex>& l, const std::string& error) {
  sink_->stop = true;
  if (sink_->error.empty() && !sink_->done) sink_->error = error;
  l.unlock();
  joinSink();
  l.lock();
  if (!sink_) return;
  lastExport_ = {false, sink_->done, sink_->written, sink_->target, sink_->bytes, sink_->job.path, sink_->error};
  for (auto& r : sink_->raw) spare_.push_back(std::move(r.second));
  sink_.reset();   // ai: an unfinished file's .part removed with its writer
  cv_.notify_all();
}

bool Sender::sinkAdmit(uint64_t seq) {
  Sink& k = *sink_;
  if (k.target == 0 && !k.stop) {
    uint64_t lap = 0;
    if (xfer_) { std::lock_guard<std::mutex> x(xferMu_); lap = xfer_->lapWhenMade(); }
    if (!xfer_ || lap > 0) {
      // ai: from the next frame made: those made ahead were scheduled before every chunk was in
      k.from = made_;
      const int per = k.job.info.frameBlocks > 0 ? k.job.info.frameBlocks : blocks_ * fmt_.codes;
      k.target = (k.job.frames > 0 ? k.job.frames : static_cast<int>(std::ceil(k.job.laps * static_cast<double>(lap) / std::max(1, per)))) | 1;
      if (!k.fused && !k.gpuPack) k.next = k.from;
      cv_.notify_all();
    }
  }
  return k.target > 0 && seq >= k.from && seq < k.from + static_cast<uint64_t>(k.target);
}

void Sender::sinkTake(uint64_t seq, std::vector<uint8_t>&& buf) {
  if (sink_ && sink_->gpuPack) { if (!buf.empty()) spare_.push_back(std::move(buf)); return; }   // ai: (packed from its slot: tapped)
  if (sink_ && !sink_->fused && !sink_->gpuPack && !sink_->stop && !sink_->done && sinkAdmit(seq)) { sink_->raw.emplace_back(seq, std::move(buf)); cv_.notify_all(); }
  else spare_.push_back(std::move(buf));
}

// ai: a frame's entries: its square, or at two codes its two (the left, then the right past the gap), each packed; a
// ai: frame's rows as the ring's (a CPU painter's) or width_ apart (the GPU's read back)
std::vector<std::vector<uint8_t>> Sender::sinkEntries(lizfile::FrameCodec& c, const std::vector<uint8_t>& buf) {
  const size_t stride = buf.size() == static_cast<size_t>(width_) * side_ ? static_cast<size_t>(width_) : 4ull * gpu_->rowWords();
  std::vector<std::vector<uint8_t>> e(static_cast<size_t>(fmt_.codes));
  for (int k = 0; k < fmt_.codes; k++)
    if (!c.pack(buf.data() + static_cast<size_t>(k) * (side_ + gap_), static_cast<uint32_t>(side_), static_cast<uint32_t>(side_), static_cast<uint32_t>(stride), e[k])) e[k].clear();
  return e;
}

void Sender::sinkStart() {
  Sink& k = *sink_;
  k.stop = false;
  if (!k.fused && !k.gpuPack) {
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    const int n = k.job.packers > 0 ? k.job.packers : k.job.display ? 2 : static_cast<int>(cores);
    for (int i = 0; i < n; i++) k.packers.emplace_back([this] { sinkPack(); });
  } else k.next = made_;   // ai: a fused painter's or the GPU's every frame passes the writer, which takes them in turn
  k.readTo = made_;
  k.tapped.clear();
  k.writer = std::thread([this] { sinkWrite(); });
  if (k.drain || k.gpuPack) k.drainer = std::thread([this] { sys::raiseThread(); sinkDrain(); });
}

// ai: a packer: the oldest frame taken and not packed, on its own codec at the file's level
void Sender::sinkPack() {
  std::unique_lock<std::mutex> l(mu_);
  lizfile::FrameCodec c(sink_->job.packing.level);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !sink_ || sink_->stop || sink_->done || !sink_->raw.empty(); });
    if (stop_ || !sink_ || sink_->stop || sink_->done) return;
    auto [seq, buf] = std::move(sink_->raw.front());
    sink_->raw.pop_front();
    l.unlock();
    Sink::Packed pk{true, sinkEntries(c, buf), {}};
    for (const auto& e : pk.entries) pk.crcs.push_back(lizfile::crc32(e.data(), e.size()));
    l.lock();
    spare_.push_back(std::move(buf));
    if (!sink_) return;
    sink_->packed[seq] = std::move(pk);
    cv_.notify_all();
  }
}

// ai: the writer: the frames in order into the file; a fused export's every frame passes here, its admitted ones
// ai: written, each then the screen's (shown_: the producer makes more). At the last, meta.json and the rename.
void Sender::sinkWrite() {
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !sink_ || sink_->stop || (sink_->next != ~0ull && (sink_->packed.count(sink_->next) || sink_->spans.count(sink_->next))); });
    if (stop_ || !sink_ || sink_->stop) return;
    if (sinkWriteOne(l) < 0) return;
  }
}

// ai: The writer's step (mu_ held through l, let go while writing): the next frame or GPU batch in order into the file
// ai: (a batch laid out as the file's entries, written with one call; a fused export's every frame passes here, its
// ai: admitted ones written, each then the screen's, shown_, so the producer makes more), and at the last frame
// ai: meta.json and the rename. 1: one written or passed; 0: none waiting; -1: the export ended (whole, or failed).
int Sender::sinkWriteOne(std::unique_lock<std::mutex>& l) {
  if (!sink_ || sink_->next == ~0ull) return 0;
  Sink& k = *sink_;
  std::string why;
  if (auto sp = k.spans.find(k.next); sp != k.spans.end()) {
    Sink::Span span = std::move(sp->second);
    k.spans.erase(sp);
    k.next += span.count;
    cv_.notify_all();
    if (!span.admitted) return 1;
    l.unlock();
    const bool ok = k.w->span(span.buf.p.get(), span.buf.n, span.es, why);
    l.lock();
    if (!sink_) return -1;
    if (!ok) { k.error = "the file: " + why; k.stop = true; cv_.notify_all(); return -1; }
    k.written += span.admitted;
    k.bytes += span.buf.n;
    k.spare.push_back(std::move(span.buf));
  } else {
    auto it = k.packed.find(k.next);
    if (it == k.packed.end()) return 0;
    Sink::Packed pk = std::move(it->second);
    k.packed.erase(it);
    const uint64_t seq = k.next++;
    cv_.notify_all();   // ai: (the GPU's drain waits on what waits for the writer)
    if (k.fused) {
      // ai: the frame is the screen's now (as a take): its blocks counted, the producer let on
      auto d = dataOf_.find(seq);
      const int data = d == dataOf_.end() ? 0 : d->second;
      if (d != dataOf_.end()) dataOf_.erase(d);
      shown_ = seq + 1;
      takenWin_++;
      dataWin_ += data;
      dataAll_ += data;
      cv_.notify_all();
    }
    if (!pk.in) return 1;
    bool ok = true;
    l.unlock();
    uint64_t bytes = 0;
    for (size_t i = 0; i < pk.entries.size(); i++) {
      const auto& e = pk.entries[i];
      if (e.empty()) { ok = false; why = "zstd refused a frame"; break; }
      bytes += e.size();
      if (!(i < pk.crcs.size() ? k.w->packed(e.data(), e.size(), pk.crcs[i], why) : k.w->packed(e.data(), e.size(), why))) { ok = false; break; }
    }
    l.lock();
    if (!sink_) return -1;
    if (!ok) { k.error = "the file: " + why; k.stop = true; cv_.notify_all(); return -1; }
    k.written++;
    k.bytes += bytes;
  }
  if (k.written < k.target) return 1;
  // ai: the last frame: meta.json (stream_export.h streamMeta reads the stats, so mu_ is let go) and the rename
  const ExportJob job = k.job;
  const int frames = k.target;
  l.unlock();
  const std::string meta = streamMeta(*this, job.info, frames);
  const int64_t n = k.w->finish(meta, "", why);
  l.lock();
  if (!sink_) return -1;
  if (n < 0) k.error = "the file: " + why;
  else { k.done = true; k.bytes = static_cast<uint64_t>(n); }
  cv_.notify_all();
  return -1;
}

// ai: The screen on or off mid-export (exportShow): the threads stopped (the GPU's drain finishing the batch it holds as
// ai: it stops), then what they left finished here in order: the frames taken and not packed packed, every waiting
// ai: frame written up to the screen's count (shown_); frames made ahead past it dropped (made again, as at any
// ai: configure: their ids lost to the fountain as a frame never shown is); then configured again for the new intake.
std::string Sender::exportShow(bool show) {
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!sink_) return "no export runs";
    // ai: an export ended (its frames all written, or stopped) as the switch came: nothing to switch, its end the app's
    // ai: to see (exportState)
    if (sink_->done || sink_->stop || sink_->job.display == show) return "";
  }
  stopAll();
  std::unique_lock<std::mutex> l(mu_);
  if (!sink_) return "no export runs";
  sinkSettle(l);
  if (!sink_ || sink_->done || sink_->stop) return sink_ && !sink_->error.empty() ? sink_->error : "";
  sink_->job.display = show;
  l.unlock();
  keepSink_ = true;
  const std::string e = configure(asked_);
  keepSink_ = false;
  if (!e.empty()) { std::unique_lock<std::mutex> g(mu_); if (sink_) sinkEnd(g, e); }
  return e;
}

// ai: An export's threads stopped (mu_ held through l): what they left finished in order, the frames taken and not packed
// ai: packed, every waiting frame written up to the screen's count (shown_), frames made ahead past it dropped (a
// ai: configure makes them again: their ids lost to the fountain as a frame never shown is)
void Sender::sinkSettle(std::unique_lock<std::mutex>& l) {
  {
    Sink& k = *sink_;
    if (!k.raw.empty()) {
      lizfile::FrameCodec c(k.job.packing.level);
      while (!k.raw.empty()) {
        auto [seq, buf] = std::move(k.raw.front());
        k.raw.pop_front();
        l.unlock();
        Sink::Packed pk{true, sinkEntries(c, buf), {}};
        for (const auto& e : pk.entries) pk.crcs.push_back(lizfile::crc32(e.data(), e.size()));
        l.lock();
        spare_.push_back(std::move(buf));
        k.packed[seq] = std::move(pk);
      }
    }
  }
  for (int r; sink_ && sink_->next < shown_ && (r = sinkWriteOne(l)) > 0;) {}
  if (!sink_ || sink_->done || sink_->stop) return;
  Sink& k = *sink_;
  for (auto it = k.packed.begin(); it != k.packed.end();) it = it->first >= shown_ ? k.packed.erase(it) : std::next(it);
  for (auto it = k.spans.begin(); it != k.spans.end();) {
    if (it->first >= shown_) { k.spare.push_back(std::move(it->second.buf)); it = k.spans.erase(it); } else ++it;
  }
}

// ai: the drain (an export with no screen, the GPU painting): each frame in turn as it is painted, as a screen takes it,
// ai: to the packers; no more than two a packer waiting, so the packers set the pace
void Sender::sinkDrain() {
  std::unique_lock<std::mutex> l(mu_);
  if (sink_ && sink_->gpuPack) { l.unlock(); sinkDrainGpu(); return; }
  for (;;) {
    cv_.wait(l, [&] {
      return stop_ || !sink_ || sink_->stop || sink_->done ||
             (ready_.count(shown_) && sink_->raw.size() < 2 * std::max<size_t>(1, sink_->packers.size()));
    });
    if (stop_ || !sink_ || sink_->stop || sink_->done) return;
    const uint64_t seq = shown_;
    ready_.erase(seq);
    auto h = hostOf_.find(seq);
    std::vector<uint8_t> buf;
    if (h != hostOf_.end()) { buf = std::move(h->second); hostOf_.erase(h); }
    auto d = dataOf_.find(seq);
    const int data = d == dataOf_.end() ? 0 : d->second;
    if (d != dataOf_.end()) dataOf_.erase(d);
    shown_++;
    takenWin_++;
    dataWin_ += data;
    dataAll_ += data;
    if (buf.empty()) { sink_->error = "the GPU's frames came back to no host copy"; sink_->stop = true; cv_.notify_all(); return; }
    sinkTake(seq, std::move(buf));
    cv_.notify_all();
  }
}

// ai: The GPU's packing of an export (gpu_zstd.h), its frames straight from their ring slots in GpuZstd's three steps,
// ai: two batches overlapped. Its frames: with the codes hidden, the frames painted in turn, as many as a batch holds
// ai: (or what is painted after 20 ms), drained (the screen's once read: shown_, so the painter reuses their slots);
// ai: shown, the frames the screen took (tapped), as many as wait (or after 50 ms), each slot the painter's again
// ai: once read (readTo). A batch begun and its codes made (its slots read then), then the batch before it finished
// ai: (laid out as the file's entries on the cores while the GPU works this one's streams) and handed to the writer;
// ai: the batch holding the export's last frame finished at once. Its entries numbered on from the file's own count
// ai: (the writer's frames: a switch of the screen may come between).
void Sender::sinkDrainGpu() {
  std::unique_lock<std::mutex> l(mu_);
  const int D = std::max(1, depth_), codes = fmt_.codes;
  const bool tap = sink_->job.display;
  // ai: a batch: what the packer holds, at most a third of what the painter may paint ahead (two batches in the packer,
  // ai: the painter filling the third meanwhile)
  const size_t K = static_cast<size_t>(std::max(1, std::min(zstd_->squares() / codes, (D - fmt_.margin) / 3)));
  const auto ring = gpu_->ring();
  const uint32_t fw = gpu_->frameWords(), rw = gpu_->rowWords();
  const auto quit = [&] { return stop_ || !sink_ || sink_->stop || sink_->done; };
  const auto run = [&] { size_t k = 0; while (k < K && ready_.count(shown_ + k)) k++; return k; };
  struct Batch { int set = -1; uint64_t first = 0; std::vector<bool> in; };
  Batch prev;
  size_t entry = sink_->w->frames();
  // ai: a batch finished and handed to the writer (mu_ let go meanwhile): its frames sized, laid out as the file's
  // ai: entries in one buffer (each its local header, then its bytes), put together there on the cores, the headers
  // ai: written with their CRC-32s; false where it failed (the error said)
  const auto finish = [&](Batch& b) {
    Sink::Span sp;
    sp.count = b.in.size();
    for (bool v : b.in) sp.admitted += v ? 1 : 0;
    std::string err;
    l.unlock();
    try {
      if (b.set >= 0) {
        const std::vector<size_t> sz = zstd_->sizes(b.set);
        uint64_t at = 0;
        std::vector<uint64_t> data;
        for (size_t q = 0; q < sz.size(); q++) {
          sp.es.push_back({at, static_cast<uint32_t>(sz[q]), 0});
          at += lizfile::localHeaderBytes(entry + q);
          data.push_back(at);
          at += sz[q];
        }
        {
          std::lock_guard<std::mutex> g(mu_);
          for (size_t i = 0; sink_ && i < sink_->spare.size(); i++)
            if (sink_->spare[i].cap >= at) { sp.buf = std::move(sink_->spare[i]); sink_->spare.erase(sink_->spare.begin() + static_cast<long>(i)); break; }
        }
        if (sp.buf.cap < at) { sp.buf.p.reset(new uint8_t[static_cast<size_t>(at) + (at >> 4)]); sp.buf.cap = static_cast<size_t>(at) + (at >> 4); }
        sp.buf.n = static_cast<size_t>(at);
        std::vector<uint8_t*> dst;
        for (uint64_t d : data) dst.push_back(sp.buf.p.get() + d);
        std::vector<uint32_t> crcs;
        zstd_->finishInto(b.set, dst, &crcs);
        for (size_t q = 0; q < sz.size(); q++) {
          sp.es[q].crc = crcs[q];
          lizfile::localHeader(entry + q, sp.es[q].size, crcs[q], sp.buf.p.get() + sp.es[q].at);
        }
        entry += sz.size();
      }
    } catch (const std::exception& e) { err = e.what(); }
    l.lock();
    if (!sink_) return false;
    if (!err.empty()) { sink_->error = "the GPU's packing: " + err; sink_->stop = true; cv_.notify_all(); return false; }
    sink_->spans[b.first] = std::move(sp);
    b = Batch{};
    cv_.notify_all();
    return true;
  };
  // ai: a batch of k frames from `first` begun and coded, then the frames let go (drained: the screen's; tapped: their
  // ai: slots read); the batch before it finished. false where it failed or the export ended.
  const auto step = [&](uint64_t first, size_t k) {
    Batch cur;
    cur.first = first;
    cur.in.resize(k);
    std::vector<int> slots;
    bool holdsLast = false;
    for (size_t i = 0; i < k; i++) {
      cur.in[i] = sinkAdmit(first + i);
      if (cur.in[i]) slots.push_back(static_cast<int>((first + i) % static_cast<uint64_t>(D)));
      if (sink_->target > 0 && first + i + 1 == sink_->from + static_cast<uint64_t>(sink_->target)) holdsLast = true;
    }
    l.unlock();
    std::string err;
    try {
      if (!slots.empty()) {
        cur.set = zstd_->begin(*ring, slots, fw, rw, codes, side_ + gap_, devMu_);
        zstd_->codes(cur.set, devMu_);
      }
    } catch (const std::exception& e) { err = e.what(); }
    l.lock();
    if (!sink_) return false;
    if (!err.empty()) { sink_->error = "the GPU's packing: " + err; sink_->stop = true; cv_.notify_all(); return false; }
    if (tap) sink_->readTo = first + k;
    else {
      for (size_t i = 0; i < k; i++) {
        const uint64_t seq = first + i;
        ready_.erase(seq);
        auto d = dataOf_.find(seq);
        const int data = d == dataOf_.end() ? 0 : d->second;
        if (d != dataOf_.end()) dataOf_.erase(d);
        takenWin_++;
        dataWin_ += data;
        dataAll_ += data;
      }
      shown_ = first + k;
    }
    cv_.notify_all();
    if (!prev.in.empty() && !finish(prev)) return false;
    if (holdsLast) return finish(cur);
    prev = std::move(cur);
    return true;
  };
  // ai: leaving: stopped (a configure, the screen switched: exportShow), the batch held finished first, and with the
  // ai: codes shown every frame the screen took and the packer did not read packed too (the painter is stopped by then:
  // ai: their slots hold them), so every frame before shown_ reaches the writer
  const auto leave = [&] {
    if (!stop_ || !sink_ || sink_->stop || sink_->done) return;
    while (tap && !sink_->tapped.empty() && sink_ && !sink_->stop && !sink_->done) {
      const uint64_t first = sink_->tapped.front();
      size_t k = 0;
      while (k < K && !sink_->tapped.empty()) { sink_->tapped.pop_front(); k++; }
      if (!step(first, k)) return;
    }
    if (sink_ && !prev.in.empty()) finish(prev);
  };
  for (;;) {
    // ai: no more than three batches waiting for the writer (held in memory)
    cv_.wait(l, [&] { return quit() || sink_->spans.size() < 3; });
    if (quit()) { leave(); return; }
    size_t k = 0;
    uint64_t first = 0;
    if (tap) {
      cv_.wait(l, [&] { return quit() || !sink_->tapped.empty(); });
      if (quit()) { leave(); return; }
      cv_.wait_for(l, std::chrono::milliseconds(50), [&] { return quit() || sink_->tapped.size() >= K; });
      if (quit()) { leave(); return; }
      first = sink_->tapped.front();
      while (k < K && !sink_->tapped.empty()) { sink_->tapped.pop_front(); k++; }
    } else {
      cv_.wait(l, [&] { return quit() || ready_.count(shown_); });
      if (quit()) { leave(); return; }
      cv_.wait_for(l, std::chrono::milliseconds(20), [&] { return quit() || run() >= K; });
      if (quit()) { leave(); return; }
      first = shown_;
      k = run();
    }
    if (!step(first, k)) return;
  }
}

std::shared_ptr<wg::Buffer> Sender::ring() const { return gpu_ ? gpu_->ring() : nullptr; }
uint32_t Sender::ringFrameWords() const { return gpu_ ? gpu_->frameWords() : 0; }
uint32_t Sender::ringRowWords() const { return gpu_ ? gpu_->rowWords() : 0; }

// ai: Refresh (sender.h setRefresh, setRefreshBlocks), each delay held to 1 .. REFRESH_MAX
void Sender::setRefresh(const int rate[4]) {
  std::lock_guard<std::mutex> l(mu_);
  for (int r = 0; r < 4; r++) byRate_[r] = std::clamp(rate[r], 1, REFRESH_MAX);
}
bool Sender::setRefreshBlocks(const std::vector<int>& place) {
  std::lock_guard<std::mutex> l(mu_);
  if (place.size() != static_cast<size_t>(blocks_)) return false;
  byPlace_.clear();
  for (const int e : place) byPlace_.push_back(std::clamp(e, 1, REFRESH_MAX));
  return true;
}
std::vector<int> Sender::scheduleLocked() const {
  if (byPlace_.size() == static_cast<size_t>(blocks_)) return byPlace_;
  std::vector<int> s(static_cast<size_t>(blocks_));
  for (size_t b = 0; b < s.size(); b++) s[b] = byRate_[b < rateOf_.size() ? rateOf_[b] : 3];
  return s;
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
// ai: shown, 1 the first), root, seed (the schedule's, 2026-11-14); refresh (setRefresh's four delays, 7/8 to 1/2) and
// ai: refreshPlaces (setRefreshBlocks's per place in force; offeredKBs counts new data blocks alone, a repeat none);
// ai: fresh (the share of a frame's blocks new in each frame: the sum of 1/delay over the places, over their count; 1
// ai: with none held), so a rate figured from blocks a frame counts new ones;
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
    << ",\"refresh\":\"" << byRate_[0] << "," << byRate_[1] << "," << byRate_[2] << "," << byRate_[3] << "\""
    << ",\"refreshPlaces\":" << (byPlace_.size() == static_cast<size_t>(blocks_) && blocks_ ? "true" : "false")
    << ",\"carousel\":" << (onGpu_ && !gpuWords_ ? 0 : carousel_)
    << ",\"test\":" << (play_ ? (play_->test ? "true" : "false") : xfer_ ? "false" : "true");
  double fresh = 0;
  const std::vector<int> sched = scheduleLocked();
  for (const int e : sched) fresh += 1.0 / e;
  o.precision(4);
  o << ",\"fresh\":" << (sched.empty() ? 1.0 : fresh / static_cast<double>(sched.size()));
  o.precision(1);
  // ai: a stream (play): its file and the laps of it shown, its pictures and the one on the screen
  if (play_) {
    const uint64_t lap = play_->lap;
    o.precision(2);
    o << ",\"stream\":true,\"pictures\":" << play_->pictures << ",\"picture\":" << (shown_ > playBase_ ? (shown_ - playBase_) % static_cast<uint64_t>(play_->pictures) : 0)
      << ",\"lap\":" << lap << ",\"pass\":" << (lap ? 1.0 + static_cast<double>(dataAll_) / static_cast<double>(lap) : 1.0) << ",\"root\":\"" << esc(play_->root) << "\"";
  }
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
