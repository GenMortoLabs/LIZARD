#include "gpu_painter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>

#include "json.hpp"
#include "codec.h"
#include "send_plan.h"
#include "xfer.h"
#include "wg.h"

namespace lizard {

namespace {
// ai: by its UTF-8 path (fs::u8path): Windows reads a narrow one in the ANSI code page, so the app's tree under a
// ai: %LOCALAPPDATA% with a letter past ASCII (a user folder with an accented name) was "missing" there (2026-10-25)
std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) throw std::runtime_error("the GPU sender's " + path + " is missing");
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}
double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

// ai: The sender's modules in the tree (setup/send.json, gen's index: the tree's version and each module's recipe as gen
// ai: compiled it) held to this host's (send_plan.h modules): a tree another host's gen wrote is refused, not run
void checkSendTree(const std::string& assets) {
  const std::string at = assets + "/setup/send.json";
  const auto b = readFile(at);
  nlohmann::json j;
  try { j = nlohmann::json::parse(b.begin(), b.end()); } catch (const std::exception& e) { throw std::runtime_error("the GPU sender's " + at + ": " + e.what()); }
  const int v = j.is_object() ? j.value("treeVersion", 0) : 0;
  if (v != plan::TREE_VERSION)
    throw std::runtime_error("the GPU sender's " + at + " is tree version " + std::to_string(v) + ", this host reads " + std::to_string(plan::TREE_VERSION) + ": regenerate liblizard/out");
  std::map<std::string, nlohmann::json> have;
  for (const auto& m : j.value("modules", nlohmann::json::array())) if (m.is_object()) have[m.value("name", std::string())] = m;
  for (const auto& m : send::modules()) {
    if (!m.native) continue;
    nlohmann::json defs = nlohmann::json::object();
    for (const auto& [k, x] : m.defs) defs[k] = x;
    const auto it = have.find(m.name);
    if (it == have.end() || it->second.value("src", std::string()) != m.src || it->second.value("defs", nlohmann::json::object()) != defs)
      throw std::runtime_error("the GPU sender's " + m.name + " in " + assets + " is not this host's (" + m.src + " and its constants): regenerate liblizard/out");
  }
}

std::unique_ptr<GpuPainter> GpuPainter::create(const std::string& assets, std::function<void(const std::string&)> log, const std::string& device,
                                               const wg::DeviceExtras* extras) {
  std::unique_ptr<GpuPainter> p(new GpuPainter());
  p->assets_ = assets;
  p->log_ = log ? log : [](const std::string&) {};
  checkSendTree(assets);
  const char* want = getenv("LIZ_VK_DEVICE");
  p->dev_ = wg::Device::create(!device.empty() ? device : want ? want : "", false, p->log_, extras ? *extras : wg::DeviceExtras());
  p->name_ = p->dev_->name;
  using wg::Bind;
  auto& d = *p->dev_;
  // ai: gpu/encoder.mjs buildShared's layouts, binding for binding
  // ai: paint: BLOCKS, TAB, SD, P; irows: SD, TW, Y, ROWS, SU, TAB, INV, P (it makes the coefficients from the data bits)
  p->bglPaint_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "send paint");
  p->bglIrows_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform, Bind::Uniform, Bind::RO, Bind::RO, Bind::Uniform}, "send irows");
  p->bglIpic_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "send ipic");
  // ai: ipic's frame form (the copy form's: the frame's bytes from workgroup memory): FRAME, SU, G, BORDER, TAPS
  p->bglIpicFrame_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform, Bind::Uniform, Bind::RO, Bind::RO}, "send ipic frame");
  p->bglRsh_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "send rsh");
  p->paint_ = d.createPipeline(wg::readModule(assets, "send_paint"), p->bglPaint_, "send paint");
  p->rsh_ = d.createPipeline(wg::readModule(assets, "send_rsh"), p->bglRsh_, "send rsh");
  p->rshBorder_ = d.createPipeline(wg::readModule(assets, "send_rshBorder"), p->bglRsh_, "send rsh border");
  const char* st = getenv("LIZ_SEND_STAGES");
  p->staged_ = st && *st && *st != '0';
  const int q = p->staged_ ? 2 * STAGES : 2;
  for (int f = 0; f < FLIGHTS; f++) {
    p->qs_[f] = d.createQuerySet(q, "send time");
    p->time_[f] = d.createBuffer(8ull * q, wg::MAP_READ | wg::COPY_DST, "send TIME");
  }
  // ai: TAB, the paint's codes and whitening: the same for every format, made once
  p->TAB_ = d.createBuffer(4ull * p->k_.tab.size(), wg::STORAGE | wg::COPY_DST, "send TAB", false);
  d.upload(*p->TAB_, 0, p->k_.tab.data(), 4ull * p->k_.tab.size());
  return p;
}

GpuPainter::~GpuPainter() {
  try { if (dev_) dev_->waitIdle(); } catch (const std::exception&) {}   // ai: a lost or hung device (wg::Lost): nothing to wait for
  gPaint_.reset(); gIrows_.reset(); gIpic_.reset(); gRsh_.reset();
  if (fReady_) lizard_free(&f_);
}

// ai: The two inverse transforms of a picture size (gpu/encoder.mjs pipesFor): ipic holds two exchange buffers of n words
// ai: (a complex value two halves a word) in workgroup memory, and n / 8 invocations a workgroup.
std::string GpuPainter::pipes(int n) {
  if (pipesN_ == n) return "";
  const auto& L = dev_->limits;
  // ai: ipic's two exchange buffers of n words (a complex value two halves a word)
  if (8u * n > L.maxComputeSharedMemorySize) return "the " + std::to_string(n) + " picture's inverse needs " + std::to_string(8 * n) + " B of workgroup memory, the device has " + std::to_string(L.maxComputeSharedMemorySize);
  if (static_cast<uint32_t>(n / 8) > L.maxComputeWorkGroupInvocations) return "the " + std::to_string(n) + " picture's inverse needs " + std::to_string(n / 8) + " invocations a workgroup";
  const std::string s = std::to_string(n);
  irows_ = dev_->createPipeline(wg::readModule(assets_, "send_irows" + s), bglIrows_, "send irows " + s);
  ipic_ = dev_->createPipeline(wg::readModule(assets_, "send_ipic" + s), bglIpic_, "send ipic " + s);
  ipicFrame_ = dev_->createPipeline(wg::readModule(assets_, "send_ipicFrame" + s), bglIpicFrame_, "send ipic frame " + s);
  pipesN_ = n;
  return "";
}

void GpuPainter::setClock(int fps, int delay) {
  if (!fReady_ || !BORDER_ || (f_.fmt.fps == fps && f_.fmt.delay == delay) || lizard_clock(&f_, fps, delay)) return;
  sendBorder(f_, t_.W, t_.border);
  dev_->upload(*BORDER_, 0, t_.border.data(), t_.border.size());
}

std::string GpuPainter::configure(int subch, int span, int fps, int frames, int codes, int gap, int aheadFrames, uint64_t aheadBytes, int margin, bool ringOnly) {
  if (fReady_) { lizard_free(&f_); fReady_ = false; }
  // ai: the format's picture (src/lizard_phy.h lizard_n_for)
  const int n = lizard_n_for(subch);
  if (lizard_init(&f_, n, subch, span)) return "the codec refused LIZARD-" + std::to_string(subch);
  fReady_ = true;
  fps_ = fps;
  try {
    if (!ringOnly) {
      const std::string pe = pipes(n);
      if (!pe.empty()) return pe;
    }
    ringOnly_ = ringOnly;
    using namespace wg;
    auto& d = *dev_;
    for (int R = std::max(1, frames); R >= 1; R /= 2) {
      const std::string te = sendTables(f_, k_, R, t_, codes, gap);
      if (!te.empty()) return te;
      const SendTables& t = t_;
      // ai: Z: the symbols an encode (frames x codes): every stage before rsh works a symbol a workgroup z, rsh a frame
      const uint64_t V = t.blocks, np = t.npos, Z = uint64_t(R) * t.codes;
      // ai: the ring's depth: the budget's, floored at 2R + margin + 1 (an encode's frames twice over and the slots a
      // ai: reader may still hold); halved on refusal down to that floor before R is
      const int floorDepth = 2 * R + std::max(0, margin) + 1;
      // ai: the ring is bound whole (rsh, words, the GPU packer), so it stays inside the device's largest binding
      // ai: (lavapipe's 128 MiB, some phones' too): past it a kernel's reads and writes are undefined
      const uint64_t range = std::min<uint64_t>(d.limits.maxStorageBufferRange, d.limits.maxBufferSize ? d.limits.maxBufferSize : UINT64_MAX);
      const uint64_t byRange = range / (4ull * t.FS);
      if (uint64_t(floorDepth) > byRange) continue;
      int depth = floorDepth;
      if (aheadFrames > 0 && aheadBytes > 0) {
        const uint64_t byBytes = std::min<uint64_t>(aheadBytes / (4ull * t.FS), byRange);
        depth = static_cast<int>(std::min<uint64_t>(static_cast<uint64_t>(aheadFrames), byBytes));
        depth = std::max(depth, floorDepth);
      }
      for (; depth >= floorDepth; depth = depth / 2 >= floorDepth ? depth / 2 : (depth > floorDepth ? floorDepth : 0)) {
      try {
        gPaint_.reset(); gIrows_.reset(); gIpic_.reset(); gRsh_.reset();
        BLOCKS_.reset(); SD_.reset(); Y_.reset(); PIC_.reset(); FRAME_.reset(); READ_.reset();
        FRAME_ = d.createBuffer(uint64_t(depth) * 4ull * t.FS, STORAGE | COPY_SRC | COPY_DST, "send FRAME");
        READ_ = d.createBuffer(R * 4ull * t.FS, MAP_READ | COPY_DST, "send READ");
        if (ringOnly) {
          d.flush();
          frames_ = R;
          depth_ = depth;
          log_("gpu sender: LIZARD-" + std::to_string(subch) + (t.codes > 1 ? " x " + std::to_string(t.codes) : "") + " at " + std::to_string(n) + ", " + std::to_string(t.FW) + " x " + std::to_string(t.W) + " px, a ring of " +
               std::to_string(depth) + " frames for the CPU's painters, " + std::to_string(d.allocated.load() >> 20) + " MB on " + name_);
          return "";
        }
        BLOCKS_ = d.createBuffer(Z * V * send::BLOCK_BYTES, STORAGE | COPY_DST, "send BLOCKS");
        // ai: INV, a disc entry's frame coefficient (bit 31 conjugated): UV (the frame coefficient's entry) turned round
        std::vector<uint32_t> inv(np);
        for (uint64_t i = 0; i < np; i++) inv[t.uv[i] & 0x7fffffffu] = static_cast<uint32_t>(i) | (t.uv[i] & 0x80000000u);
        INV_ = d.createBuffer(4 * np, STORAGE | COPY_DST, "send INV", false);
        d.upload(*INV_, 0, inv.data(), 4 * np);
        // ai: SD, the data coefficients' bits (paint's, 16 a word, sdWords a symbol)
        SD_ = d.createBuffer(Z * 4ull * t.sdWords, STORAGE, "send SD");
        PU_ = d.createBuffer(4 * t.pu.size(), UNIFORM | COPY_DST, "send PU", false);
        d.writeBuffer(*PU_, 0, t.pu.data(), 4 * t.pu.size());
        TW_ = d.createBuffer(8ull * n, STORAGE | COPY_DST, "send TW", false);
        d.upload(*TW_, 0, t.tw.data(), 8ull * n);
        ROWS_ = d.createBuffer(4 * t.rows.size(), UNIFORM | COPY_DST, "send ROWS", false);
        d.writeBuffer(*ROWS_, 0, t.rows.data(), 4 * t.rows.size());
        SU_ = d.createBuffer(48, UNIFORM | COPY_DST, "send SU", false);
        d.writeBuffer(*SU_, 0, t.su.data(), 48);
        // ai: Y between the inverses, a complex value two halves a word (irows, ipic)
        Y_ = d.createBuffer(Z * uint64_t(t.Vr) * n * 4, STORAGE, "send Y");
        // ai: PIC, ipic's picture, where the taps are not copies (the copy form's ipic writes the frame; rsh's border form
        // ai: binds it unread)
        PIC_ = d.createBuffer(t.copy ? 16 : Z * 4ull * n * n, STORAGE, "send PIC");
        TAPS_ = d.createBuffer(4 * t.taps.size(), STORAGE | COPY_DST, "send TAPS", false);
        d.upload(*TAPS_, 0, t.taps.data(), 4 * t.taps.size());
        BORDER_ = d.createBuffer(t.border.size(), STORAGE | COPY_DST, "send BORDER", false);
        d.upload(*BORDER_, 0, t.border.data(), t.border.size());
        // ai: G and the anchors' 8 words after it (rsh reads 80 B, the other stages 48)
        GU_ = d.createBuffer(48 + 4 * t.anch.size(), UNIFORM | COPY_DST, "send GU", false);
        d.writeBuffer(*GU_, 0, t.g.data(), 48);
        d.writeBuffer(*GU_, 48, t.anch.data(), 4 * t.anch.size());
        gPaint_ = d.createBindGroup(bglPaint_, {BLOCKS_, TAB_, SD_, PU_});
        gIrows_ = d.createBindGroup(bglIrows_, {SD_, TW_, Y_, ROWS_, SU_, TAB_, INV_, PU_});
        gIpic_ = t.copy ? d.createBindGroup(bglIpicFrame_, {Y_, TW_, FRAME_, SU_, GU_, BORDER_, TAPS_}) : d.createBindGroup(bglIpic_, {Y_, TW_, PIC_, SU_});
        gRsh_ = d.createBindGroup(bglRsh_, {PIC_, TAPS_, BORDER_, FRAME_, GU_});
        d.flush();
        frames_ = R;
        depth_ = depth;
        staging_.assign(Z * V * send::BLOCK_BYTES, 0);
        log_("gpu sender: LIZARD-" + std::to_string(subch) + (t.codes > 1 ? " x " + std::to_string(t.codes) : "") + " at " + std::to_string(n) + ", " + std::to_string(t.FW) + " x " + std::to_string(t.W) + " px, encodes of " + std::to_string(R) +
             (t.copy ? ", the copy form" : "") + ", a ring of " + std::to_string(depth) + " frames, " + std::to_string(d.allocated.load() >> 20) + " MB on " + name_);
        return "";
      } catch (const OutOfMemory&) {
        log_("gpu sender: encodes of " + std::to_string(R) + " with a ring of " + std::to_string(depth) + " do not fit; fewer");
      }
      }
    }
    return "no encode of LIZARD-" + std::to_string(subch) + " fits this device's memory";
  } catch (const std::exception& e) {
    return std::string("the GPU sender: ") + e.what();
  }
}

double GpuPainter::encode(const uint8_t* blocks, int count, int parity, std::vector<uint8_t>* out, uint64_t seq, bool test,
                          const std::function<void(wg::Encoder&)>& fill) {
  Pending p = submit(blocks, count, parity, out, seq, test, fill);
  return finish(p);
}

GpuPainter::Pending GpuPainter::submit(const uint8_t* blocks, int count, int parity, std::vector<uint8_t>* out, uint64_t seq, bool test,
                                       const std::function<void(wg::Encoder&)>& fill) {
  const SendTables& t = t_;
  const int V = t.blocks, B = t.blockBytes, BB = send::BLOCK_BYTES;
  if (ringOnly_) throw std::runtime_error("the GPU painter holds the ring alone here (the CPU paints)");
  if (count < 1 || count > frames_) throw std::runtime_error("an encode is 1 to " + std::to_string(frames_) + " frames");
  const uint32_t first = static_cast<uint32_t>(seq % static_cast<uint64_t>(depth_));
  if (static_cast<int>(first) + count > depth_) throw std::runtime_error("an encode of " + std::to_string(count) + " frames from slot " + std::to_string(first) + " would wrap the ring of " + std::to_string(depth_));
  auto& d = *dev_;
  const int Z = count * t.codes;   // ai: the symbols, a frame's codes one after the other
  // ai: the test stream's ids alone, packed XFER_ID_BYTES (12) a block (PAINT reads the id's three words and makes the payload), else
  // ai: the blocks at BB bytes, unless `fill` writes them on the device
  if (!fill) {
    const size_t idb = XFER_ID_BYTES, at = test ? idb : static_cast<size_t>(BB), take = test ? idb : static_cast<size_t>(B);
    for (int i = 0; i < Z * V; i++) std::memcpy(staging_.data() + static_cast<size_t>(i) * at, blocks + static_cast<size_t>(i) * B, take);
    d.writeBuffer(*BLOCKS_, 0, staging_.data(), static_cast<uint64_t>(Z) * V * at);
  }
  const uint32_t pc = static_cast<uint32_t>(parity & 3) | (static_cast<uint32_t>(t.codes) << 2) | (test ? 0x80000000u : 0u);
  d.writeBuffer(*GU_, 36, &first, 4);
  d.writeBuffer(*PU_, 4ull * (send::P_DIMS + 3), &pc, 4);
  const uint32_t R = static_cast<uint32_t>(count), RZ = static_cast<uint32_t>(Z), n = static_cast<uint32_t>(t.n);
  // ai: its flight's timestamps (a few encodes in flight, each reading back its own)
  const int fl = flight_++ % FLIGHTS;
  auto enc = d.encoder();
  // ai: staged (LIZ_SEND_STAGES): a pass a stage, its own pair; otherwise one pass, one pair
  wg::QuerySet* qs = qs_[fl].get();
  const auto stage = [&](int k) { if (staged_) { if (k) enc.endPass(); enc.beginComputePass(qs, 2 * k, 2 * k + 1); } };
  enc.beginComputePass(qs, 0, 1);
  if (fill) fill(enc);
  enc.setPipeline(*paint_); enc.setBindGroup(*gPaint_); enc.dispatch(V, 1, RZ);
  stage(1);
  enc.setPipeline(*irows_); enc.setBindGroup(*gIrows_); enc.dispatch(t.Vr, 1, RZ);
  stage(2);
  enc.setPipeline(t.copy ? *ipicFrame_ : *ipic_); enc.setBindGroup(*gIpic_); enc.dispatch(n / 4, 1, RZ);
  stage(3);
  // ai: the copy form's border words, a lane a word; else the resample, a workgroup a strip of words by a band of rows
  const uint32_t strips = (t.RW + send::RSH_THREADS - 1) / send::RSH_THREADS;
  if (t.copy) { enc.setPipeline(*rshBorder_); enc.setBindGroup(*gRsh_); enc.dispatch(strips, t.W, R); }
  else { enc.setPipeline(*rsh_); enc.setBindGroup(*gRsh_); enc.dispatch(strips, (t.W + send::RSH_BAND - 1) / send::RSH_BAND, R); }
  enc.endPass();
  enc.resolveQuerySet(*qs, 0, staged_ ? 2 * STAGES : 2, *time_[fl], 0);
  if (out) enc.copyBufferToBuffer(*FRAME_, uint64_t(first) * 4 * t.FS, *READ_, 0, uint64_t(R) * 4 * t.FS);
  Pending p;
  p.flight = fl;
  p.count = count;
  p.out = out;
  p.t0 = nowMs();
  p.ticket = d.submit(enc);
  return p;
}

bool GpuPainter::done(const Pending& p) { return !p.ticket || p.ticket->done(); }

double GpuPainter::finish(Pending& p) {
  const SendTables& t = t_;
  auto& d = *dev_;
  p.ticket->wait();
  const double wall = nowMs() - p.t0;
  if (p.out) {
    const uint8_t* px = d.read(*READ_, 0, uint64_t(p.count) * 4 * t.FS);
    for (int f = 0; f < p.count; f++) {
      p.out[f].resize(static_cast<size_t>(t.FW) * t.W);
      for (int y = 0; y < t.W; y++) std::memcpy(p.out[f].data() + static_cast<size_t>(y) * t.FW, px + 4 * (static_cast<size_t>(f) * t.FS + static_cast<size_t>(y) * t.RW), t.FW);
    }
  }
  p.ticket.reset();
  if (!d.features.timestamps) return wall;
  if (!staged_) {
    uint64_t ts[2];
    std::memcpy(ts, d.read(*time_[p.flight], 0, 16), 16);
    return double((ts[1] - ts[0]) & d.timestampMask) * d.timestampPeriod / 1e6;
  }
  uint64_t ts[2 * STAGES];
  std::memcpy(ts, d.read(*time_[p.flight], 0, sizeof ts), sizeof ts);
  double all = 0;
  for (int k = 0; k < STAGES; k++) {
    const double ms = double((ts[2 * k + 1] - ts[2 * k]) & d.timestampMask) * d.timestampPeriod / 1e6;
    stageMs_[k] += ms;
    all += ms;
  }
  stageEncodes_++;
  return all;
}

void GpuPainter::readSlot(int slot, std::vector<uint8_t>& out) {
  const SendTables& t = t_;
  auto& d = *dev_;
  auto enc = d.encoder();
  enc.copyBufferToBuffer(*FRAME_, uint64_t(slot) * 4 * t.FS, *READ_, 0, 4ull * t.FS);
  d.submit(enc)->wait();
  const uint8_t* px = d.read(*READ_, 0, 4ull * t.FS);
  out.resize(static_cast<size_t>(t.FW) * t.W);
  for (int y = 0; y < t.W; y++) std::memcpy(out.data() + static_cast<size_t>(y) * t.FW, px + 4 * static_cast<size_t>(y) * t.RW, t.FW);
}

void GpuPainter::upload(int slot, const uint8_t* packed) {
  auto& d = *dev_;
  const uint64_t bytes = 4ull * t_.FS;
  d.upload(*FRAME_, static_cast<uint64_t>(slot) * bytes, packed, bytes);
  auto enc = d.encoder();
  auto ticket = d.submit(enc);
  ticket->wait();
}

GpuPainter::Check GpuPainter::check(const uint8_t* blocks, int count) {
  Check c;
  const SendTables& t = t_;
  const size_t per = static_cast<size_t>(t.blocks) * t.blockBytes, perFrame = per * t.codes;
  std::vector<std::vector<uint8_t>> got(frames_);
  std::vector<float> drive(static_cast<size_t>(f_.px) * f_.px);
  std::vector<uint8_t> rgba(static_cast<size_t>(t.W) * t.W * 4), luma(static_cast<size_t>(t.W) * t.W);
  // ai: the C's blind receiver (cpu/codec.h), every symbol read from the GPU's pixels alone
  cpu_dec_t* dec = cpu_dec_new(t.n);
  if (!dec) return c;
  const int top = cpu_dec_top(dec), BB = cpu_dec_block_bytes(dec);
  std::vector<uint8_t> rb(static_cast<size_t>(top) * BB), rok(static_cast<size_t>(top));
  double sq2 = 0;
  // ai: a symbol's row or column on one of the anchors' runs (send_tables.h anch): a pixel on both is an anchor's
  auto onRun = [&](int v) { for (int i = 0; i < 7; i++) if (static_cast<uint32_t>(v) - t.anch[i] < t.anch[7]) return true; return false; };
  for (int f0 = 0; f0 < count; f0 += frames_) {
    const int k = std::min(frames_, count - f0);
    encode(blocks + f0 * perFrame, k, f0 & 3, got.data(), 0, true);
    for (int i = 0; i < k; i++) {
      // ai: each code against the C's paint of its own blocks, the frame's count; the gap's pixels the margin's byte
      lizard_parity(&f_, (f0 + i) & 3);
      for (int cd = 0; cd < t.codes; cd++) {
        lizard_encode(&f_, blocks + (f0 + i) * perFrame + cd * per, drive.data());
        lizard_paint_rgba(&f_, drive.data(), rgba.data());
        const int x0 = cd * (t.W + t.gap);
        for (int y = 0; y < t.W; y++) {
          const bool ra = onRun(y);
          for (int x = 0; x < t.W; x++) {
            const uint8_t g = got[i][static_cast<size_t>(y) * t.FW + x0 + x];
            const int dv = std::abs(int(g) - int(rgba[4 * (static_cast<size_t>(y) * t.W + x)]));
            luma[static_cast<size_t>(y) * t.W + x] = g;
            if (ra && onRun(x)) { c.anchorPx++; c.anchorDark += g == 0; c.anchorDiff += dv != 0; continue; }
            if (x >= t.sq && x < t.sq + t.q && y >= t.sq && y < t.sq + t.q) {
              c.squarePx++; sq2 += double(dv) * dv;
              if (dv) { c.squareDiff++; c.squareMax = std::max(c.squareMax, dv); }
            } else if (dv) c.borderDiff++;
          }
        }
        // ai: read blind: every block of the symbol verified and the block given
        cpu_frame_t fr{};
        cpu_dec_frame(dec, luma.data(), t.W, t.W, rb.data(), rok.data(), &fr);
        const uint8_t* want = blocks + (f0 + i) * perFrame + cd * per;
        for (int b = 0; b < t.blocks; b++) {
          c.readBlocks++;
          if (b >= fr.total || !rok[static_cast<size_t>(b)] || std::memcmp(rb.data() + static_cast<size_t>(b) * BB, want + static_cast<size_t>(b) * t.blockBytes, static_cast<size_t>(t.blockBytes))) c.readBad++;
        }
        if (cd + 1 < t.codes) for (int y = 0; y < t.W; y++) for (int x = 0; x < t.gap; x++)
          if (got[i][static_cast<size_t>(y) * t.FW + x0 + t.W + x] != rgba[4 * (static_cast<size_t>(y) * t.W + t.W - 1)]) c.borderDiff++;
      }
    }
  }
  lizard_parity(&f_, 0);
  cpu_dec_free(dec);
  c.squareRms = c.squarePx ? std::sqrt(sq2 / c.squarePx) : 0;
  c.ok = c.borderDiff == 0 && c.anchorDiff == 0 && c.readBlocks > 0 && c.readBad == 0;
  return c;
}

}  // namespace lizard
