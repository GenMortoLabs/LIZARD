// ai: The sender's blocks made on the GPU (gpu_stream.h), its kernel gpu/slang/src/send_rq_blocks.slang.
#include "gpu_stream.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>

#include "wg.h"

extern "C" {
#include "rq_plan.h"
#include "rq_tables.h"
}

namespace lizard {

namespace {
constexpr uint32_t T = XFER_PAYLOAD, SB = 480;
// ai: the upload slots: a source block's symbols each, the workers waiting for one while the recording thread copies
constexpr uint32_t SLOTS = 4;
// ai: TAB's layout (gpu/plan/send.mjs TAB_DEG, TAB_PLANS, TAB_PLAN)
constexpr uint32_t TAB_DEG = 1024, TAB_PLANS = 1056, TAB_PLAN = 8;
// ai: the self-test's source block (K 218 at 460 B) and its ESIs past the first source symbols
constexpr size_t SELF_BYTES = 100000;
std::string mb(uint64_t b) { return std::to_string((b + (1u << 19)) >> 20); }
std::vector<uint32_t> spirv(const std::string& path) {
  std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) throw std::runtime_error("the tree has no " + path + ": regenerate liblizard/out");
  const std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<uint32_t> w(b.size() / 4);
  std::memcpy(w.data(), b.data(), w.size() * 4);
  return w;
}
// ai: TAB: RFC 6330's tables and each plan's K, K', J, W, P, P1
std::vector<uint32_t> tabOf(const std::vector<rq_par_t>& plans) {
  std::vector<uint32_t> tab(TAB_PLANS + TAB_PLAN * plans.size(), 0);
  std::copy(RQ_V0, RQ_V0 + 256, tab.begin());
  std::copy(RQ_V1, RQ_V1 + 256, tab.begin() + 256);
  std::copy(RQ_V2, RQ_V2 + 256, tab.begin() + 512);
  std::copy(RQ_V3, RQ_V3 + 256, tab.begin() + 768);
  std::copy(RQ_DEG_F, RQ_DEG_F + 31, tab.begin() + TAB_DEG);
  for (size_t i = 0; i < plans.size(); i++) {
    const rq_par_t& q = plans[i];
    const uint32_t v[6] = {q.K, q.Kp, q.J, q.W, q.P, q.P1};
    std::copy(v, v + 6, tab.begin() + TAB_PLANS + TAB_PLAN * i);
  }
  return tab;
}
// ai: a solved source block's symbols at SYM's stride (480 B, the 460 zero padded)
void spread(const uint8_t* C, uint32_t L, uint8_t* out) {
  for (uint32_t i = 0; i < L; i++) {
    std::memcpy(out + size_t{i} * SB, C + size_t{i} * T, T);
    std::memset(out + size_t{i} * SB + T, 0, SB - T);
  }
}
}  // namespace

std::unique_ptr<GpuStream> GpuStream::create(wg::Device& d, const std::string& assets, const File& f) {
  if (!f.sourceBlocks || !f.sbBytes) throw std::runtime_error("no source blocks");
  std::unique_ptr<GpuStream> s(new GpuStream(d));
  s->f_ = f;
  // ai: each source block's L from its K (a file has at most two lengths: the full source blocks', a short last one's)
  std::map<size_t, uint32_t> planOf;
  std::vector<rq_par_t> plans;
  std::vector<uint32_t> ch;
  uint64_t nsym = 0;
  s->total_ = f.sourceBlocks;
  for (uint32_t c = 0; c < s->total_; c++) {
    const size_t cl = std::min(f.sbBytes, f.length - size_t{c} * f.sbBytes);
    auto it = planOf.find(cl);
    if (it == planOf.end()) {
      rq_par_t q{};
      if (rq_par_for(rq_symbols(cl, T), &q)) throw std::runtime_error("no RaptorQ block of " + std::to_string(cl) + " B");
      it = planOf.emplace(cl, static_cast<uint32_t>(plans.size())).first;
      plans.push_back(q);
    }
    const uint32_t L = plans[it->second].L;
    s->base_.push_back(static_cast<uint32_t>(nsym));
    s->len_.push_back(L);
    s->lMax_ = std::max(s->lMax_, L);
    ch.insert(ch.end(), {static_cast<uint32_t>(nsym), it->second});
    nsym += L;
  }
  s->ctl_ = static_cast<uint32_t>(nsym);
  nsym += f.controls;
  s->symBytes_ = nsym * SB;
  const uint64_t slots = uint64_t{SLOTS} * s->lMax_ * SB;
  const wg::Limits& Lm = d.limits;
  const uint64_t range = std::min<uint64_t>(Lm.maxStorageBufferRange, Lm.maxBufferSize ? Lm.maxBufferSize : UINT64_MAX);
  // ai: the kernel indexes SYM's vec4u in 32 bits: 30 a symbol
  if (nsym * 30 >= (uint64_t{1} << 32) || s->symBytes_ > range)
    throw std::runtime_error("the file's symbols (" + mb(s->symBytes_) + " MB) pass the device's largest buffer (" + mb(range) + " MB)");
  // ai: a quarter of the memory where the device shares the system's (an integrated GPU, a phone's), a half on its own
  const uint64_t share = Lm.deviceLocalBytes / (d.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 4);
  if (s->symBytes_ + slots > share) throw std::runtime_error("the file's symbols (" + mb(s->symBytes_) + " MB) pass the GPU's share of its memory (" + mb(share) + " MB)");
  s->bgl_ = d.createBindGroupLayout({wg::Bind::RO, wg::Bind::RO, wg::Bind::RO, wg::Bind::RO, wg::Bind::RW, wg::Bind::RO}, "rq blocks");
  s->pipe_ = d.createPipeline(spirv(assets + "/spv/send_rqBlocks.spv"), s->bgl_, "", "rq blocks");
  // ai: SYM is made by the first worker (put): host-visible, zeroed by wg as a host buffer is, where the device shares
  // ai: the system's memory, which takes a while for a large file and is not the configuring thread's (a phone's main one)
  s->uma_ = d.type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  const auto up = [&](const std::vector<uint32_t>& v, const char* id) {
    auto b = d.createBuffer(std::max<size_t>(16, 4 * v.size()), wg::STORAGE | wg::COPY_DST, id, v.empty());
    d.upload(*b, 0, v.data(), 4 * v.size());
    return b;
  };
  s->tab_ = up(tabOf(plans), "rq TAB");
  s->sourceBlocks_ = up(ch, "rq CHUNKS");
  s->par_ = d.createBuffer(32, wg::STORAGE | wg::COPY_DST, "rq PAR", false);
  for (uint32_t i = 0; !s->uma_ && i < SLOTS; i++) s->slot_.push_back(d.createBuffer(uint64_t{s->lMax_} * SB, wg::STORAGE | wg::MAP_WRITE, "rq SLOT", false));
  s->slotOf_.assign(s->slot_.size(), -1);
  s->ctlUp_.assign(f.controls, 0);
  return s;
}

GpuStream::~GpuStream() { stop(); }

void GpuStream::stop() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  cv_.notify_all();
}

// ai: Source block c (a worker's). Where the device shares the system's memory (a phone, an integrated GPU) SYM is host-visible
// ai: and the worker writes the source block's symbols straight into it: no copy and no wait (the GPU reads them in the encodes
// ai: after every source block was handed over; a host write is visible to a submission after it). On a discrete GPU, its
// ai: symbols into a free slot, then copied into SYM on the device's second queue (Device::queue2
// ai: and pool2: the receiver's camera ingest takes them on its own device, a sender's are free), the copy waited for
// ai: here and the slot freed: the painting thread never touches the upload, and the copy runs beside the painting on the
// ai: GPU (where the family has one queue, queue2 is the queue and the copy goes in turn, under its lock as the ingest's
// ai: does). Same family, so SYM needs no ownership transfer; a data encode comes after every source block's copy finished (the
// ai: transfer names no data block before every sink returned), and wg begins every command buffer with a barrier.
bool GpuStream::put(uint32_t c, const rq_enc_t* enc) {
  rq_par_t q{};
  const uint8_t* C = rq_enc_intermediate(enc, &q);
  if (c >= total_ || q.L != len_[c]) throw std::runtime_error("source block " + std::to_string(c) + " is not the stream's");
  auto& d = dev_;
  std::call_once(symOnce_, [&] { sym_ = d.createBuffer(symBytes_, wg::STORAGE | (uma_ ? wg::MAP_WRITE : wg::COPY_DST), "rq SYM", false); });
  if (!sym_) throw std::runtime_error("no SYM");
  if (uma_) {
    {
      std::lock_guard<std::mutex> l(mu_);
      if (stop_) return false;
    }
    spread(C, q.L, sym_->mapped + uint64_t{base_[c]} * SB);
    std::lock_guard<std::mutex> l(mu_);
    onDevice_++;
    return true;
  }
  size_t s;
  {
    std::unique_lock<std::mutex> l(mu_);
    cv_.wait(l, [&] { return stop_ || !failed_.empty() || std::find(slotOf_.begin(), slotOf_.end(), -1) != slotOf_.end(); });
    if (stop_) return false;
    if (!failed_.empty()) throw std::runtime_error(failed_);
    s = static_cast<size_t>(std::find(slotOf_.begin(), slotOf_.end(), -1) - slotOf_.begin());
    slotOf_[s] = c;
  }
  spread(C, q.L, slot_[s]->mapped);
  VkCommandBuffer cb = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  bool submitted = false;
  const auto release = [&] {
    std::lock_guard<std::mutex> p(poolMu_);
    if (fence) vkDestroyFence(d.dev, fence, nullptr);
    if (cb) vkFreeCommandBuffers(d.dev, d.pool2, 1, &cb);
  };
  try {
    {
      std::lock_guard<std::mutex> p(poolMu_);
      VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      cai.commandPool = d.pool2;
      cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      cai.commandBufferCount = 1;
      wg::check(vkAllocateCommandBuffers(d.dev, &cai, &cb), "vkAllocateCommandBuffers (rq copy)");
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cb, &bi);
      const VkBufferCopy bc{0, uint64_t{base_[c]} * SB, uint64_t{len_[c]} * SB};
      vkCmdCopyBuffer(cb, slot_[s]->buf, sym_->buf, 1, &bc);
      wg::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer (rq copy)");
    }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    wg::check(vkCreateFence(d.dev, &fci, nullptr, &fence), "vkCreateFence (rq copy)");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    {
      // ai: a queue takes one submitting thread at a time (Vulkan's external synchronisation): queue2's lock, which the
      // ai: desktop's presenter takes around its submits and presents on the same queue (wg.h queue2Mu)
      std::lock_guard<std::mutex> p(poolMu_);
      std::lock_guard<std::mutex> q2(d.queue2Mu());
      wg::check(vkQueueSubmit(d.queue2, 1, &si, fence), "vkQueueSubmit (rq copy)");
      submitted = true;
    }
    const VkResult r = vkWaitForFences(d.dev, 1, &fence, VK_TRUE, d.hangMs ? uint64_t{d.hangMs} * 1000000 : UINT64_MAX);
    if (r == VK_TIMEOUT) throw std::runtime_error("a source block's copy to the device did not finish");
    wg::check(r, "vkWaitForFences (rq copy)");
  } catch (const std::exception& e) {
    // ai: a copy that failed fails the stream: every put waiting for a slot, and every put after, throws its reason, so
    // ai: the transfer's fault reaches the sender (until 2026-11-17 the slot was lost and nothing else knew: four lost,
    // ai: the other workers waited for one for ever and the send hung). A copy submitted and not finished keeps its
    // ai: slot, command buffer and fence, which the device may still use (wg's rule for a hung submit).
    {
      std::lock_guard<std::mutex> l(mu_);
      if (failed_.empty()) failed_ = e.what();
    }
    cv_.notify_all();
    if (!submitted) release();
    throw;
  }
  release();
  {
    std::lock_guard<std::mutex> l(mu_);
    slotOf_[s] = -1;
    onDevice_++;
  }
  cv_.notify_all();
  return true;
}

bool GpuStream::done() {
  std::lock_guard<std::mutex> l(mu_);
  return onDevice_ == total_;
}

// ai: The ids of n blocks into IDS, made to hold `room` (at least n) when it must grow; a control block's payload into
// ai: its symbol the first time it is named (a control block named is final: the session header, the one
// ai: kind since 2026-11-21). IDS and its bind group are replaced only to grow, which frees what a submission still in
// ai: flight names (its write and its dispatch): the painter's room is the most blocks any of its encodes names, so
// ai: IDS grows only when the painter is configured again, nothing in flight; host and the self-test wait for theirs.
// ai: (Until 2026-11-18 IDS grew with the encodes: a larger one after a smaller, the smaller still in flight, crashed
// ai: lavapipe in its write to the freed IDS.)
void GpuStream::ids(const uint8_t* bl, uint32_t n, size_t stride, bool controlsToo, uint32_t room) {
  auto& d = dev_;
  if (!ids_ || ids_->size < 8ull * n) {
    ids_ = d.createBuffer(8ull * std::max(room, n), wg::STORAGE | wg::COPY_DST, "rq IDS", false);
    group_.reset();
  }
  // ai: queue writes (landing at the head of the encode's submission), not a mapped buffer: the painter keeps several
  // ai: encodes in flight, each reading its own ids
  idw_.resize(2 * size_t{n});
  uint32_t* ip = idw_.data();
  for (uint32_t i = 0; i < n; i++) {
    const uint8_t* p = bl + i * stride;
    const xfer_id_t id = xfer_id_get(p);
    const xfer_toi_t toi = xfer_toi_get(p);
    // ai: the id's first two words as BLOCKS writes them (the bare id and the TOI's low 24 bits, bytes 0 to 7); its TOI's
    // ai: high 24 bits PAR's, the file's (a data block's) and the session's (the session header's): a send's blocks
    // ai: share the two TOIs
    const uint64_t w = id | (toi & 0xffffffu) << 40;
    if (xfer_id_sbn(id) == XFER_CONTROL) toiHi_[1] = uint32_t(toi >> 24); else toiHi_[0] = uint32_t(toi >> 24);
    ip[2 * i] = static_cast<uint32_t>(w);
    ip[2 * i + 1] = static_cast<uint32_t>(w >> 32);
    if (!controlsToo || xfer_id_sbn(id) != XFER_CONTROL) continue;
    const uint32_t s = xfer_id_esi(id);
    if (s < f_.controls && !ctlUp_[s]) {
      uint8_t sb[SB] = {};
      std::memcpy(sb, p + XFER_ID_BYTES, T);
      d.writeBuffer(*sym_, uint64_t{ctl_ + s} * SB, sb, SB);
      ctlUp_[s] = 1;
    }
  }
  d.writeBuffer(*ids_, 0, idw_.data(), 8ull * n);
}

void GpuStream::record(wg::Encoder& enc, const std::shared_ptr<wg::Buffer>& sym, const std::shared_ptr<wg::Buffer>& tab, const std::shared_ptr<wg::Buffer>& sourceBlocks,
                       const std::shared_ptr<wg::Buffer>& out, uint32_t n, uint32_t ctl, uint32_t nsb, uint32_t controls) {
  if (!group_ || out != lastOut_ || sym != lastSym_) {
    group_ = dev_.createBindGroup(bgl_, {sym, ids_, tab, sourceBlocks, out, par_});
    lastOut_ = out;
    lastSym_ = sym;
  }
  const uint32_t pb[8] = {n, ctl, nsb, controls, toiHi_[0], toiHi_[1], 0, 0};
  dev_.writeBuffer(*par_, 0, pb, sizeof pb);
  enc.setPipeline(*pipe_);
  enc.setBindGroup(*group_);
  enc.dispatch((n + 1) / 2);
}

void GpuStream::blocks(wg::Encoder& enc, const std::shared_ptr<wg::Buffer>& out, const uint8_t* bl, uint32_t n, size_t stride, uint32_t room) {
  if (!done()) throw std::runtime_error("a data block named before every source block's symbols were on the device");
  ids(bl, n, stride, true, room);
  record(enc, sym_, tab_, sourceBlocks_, out, n, ctl_, total_, f_.controls);
}

void GpuStream::host(const xfer_id_t* idv, uint32_t n, uint8_t* out) {
  if (!n) return;
  if (!done()) throw std::runtime_error("a data block named before every source block's symbols were on the device");
  auto& d = dev_;
  if (!out_ || out_->size < uint64_t{SB} * n) {
    out_ = d.createBuffer(uint64_t{SB} * n, wg::STORAGE | wg::COPY_SRC, "rq OUT", false);
    outRd_ = d.createBuffer(uint64_t{SB} * n, wg::MAP_READ | wg::COPY_DST, "rq OUT read");
  }
  std::vector<uint8_t> bl(size_t{n} * XFER_ID_BYTES);
  for (uint32_t i = 0; i < n; i++) {
    if (xfer_id_sbn(idv[i]) == XFER_CONTROL) throw std::runtime_error("host() makes data blocks alone");
    xfer_id_put(bl.data() + size_t{i} * XFER_ID_BYTES, idv[i]);
  }
  ids(bl.data(), n, XFER_ID_BYTES, false);
  auto e = d.encoder();
  e.beginComputePass();
  record(e, sym_, tab_, sourceBlocks_, out_, n, ctl_, total_, f_.controls);
  e.endPass();
  e.copyBufferToBuffer(*out_, 0, *outRd_, 0, uint64_t{SB} * n);
  d.submit(e)->wait();
  const uint8_t* r = d.read(*outRd_, 0, uint64_t{SB} * n);
  for (uint32_t i = 0; i < n; i++) std::memcpy(out + size_t{i} * T, r + size_t{i} * SB + XFER_ID_BYTES, T);
}

std::string GpuStream::selfTest() {
  try {
    auto& d = dev_;
    std::vector<uint8_t> data(SELF_BYTES);
    std::mt19937 g(7);
    for (auto& b : data) b = static_cast<uint8_t>(g());
    std::unique_ptr<rq_enc_t, void (*)(rq_enc_t*)> enc(rq_enc_new(data.data(), data.size(), T), rq_enc_free);
    if (!enc) return "the self-test's encoder";
    rq_par_t q{};
    const uint8_t* C = rq_enc_intermediate(enc.get(), &q);
    std::vector<uint8_t> sym(size_t{q.L} * SB);
    spread(C, q.L, sym.data());
    auto symB = d.createBuffer(sym.size(), wg::STORAGE | wg::COPY_DST, "rq SELF SYM", false);
    d.upload(*symB, 0, sym.data(), sym.size());
    const std::vector<uint32_t> tab = tabOf({q}), ch = {0, 0};
    auto tabB = d.createBuffer(4 * tab.size(), wg::STORAGE | wg::COPY_DST, "rq SELF TAB", false);
    d.upload(*tabB, 0, tab.data(), 4 * tab.size());
    auto chB = d.createBuffer(8, wg::STORAGE | wg::COPY_DST, "rq SELF CHUNKS", false);
    d.upload(*chB, 0, ch.data(), 8);
    // ai: the first source symbols, the last, K, K + 1 and on, and 2^24 - 1
    std::vector<xfer_id_t> esi;
    for (uint32_t k = 0; k < 64; k++) esi.push_back(xfer_id(0, k));
    for (uint32_t k : {q.K - 1, q.K, q.K + 1, q.K + 1000, (1u << 24) - 1}) esi.push_back(xfer_id(0, k));
    const uint32_t n = static_cast<uint32_t>(esi.size());
    std::vector<uint8_t> bl(size_t{n} * XFER_ID_BYTES);
    for (uint32_t i = 0; i < n; i++) xfer_id_put(bl.data() + size_t{i} * XFER_ID_BYTES, esi[i]);
    ids(bl.data(), n, XFER_ID_BYTES, false);
    auto outB = d.createBuffer(uint64_t{SB} * n, wg::STORAGE | wg::COPY_SRC, "rq SELF OUT", false);
    auto rd = d.createBuffer(uint64_t{SB} * n, wg::MAP_READ | wg::COPY_DST, "rq SELF read");
    auto e = d.encoder();
    e.beginComputePass();
    record(e, symB, tabB, chB, outB, n, 0, 1, 0);
    e.endPass();
    e.copyBufferToBuffer(*outB, 0, *rd, 0, uint64_t{SB} * n);
    d.submit(e)->wait();
    group_.reset();   // ai: its group named the self-test's buffers
    const uint8_t* r = d.read(*rd, 0, uint64_t{SB} * n);
    uint8_t want[T];
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i++) {
      rq_enc_symbol(enc.get(), xfer_id_esi(esi[i]), want);
      bad += std::memcmp(want, r + size_t{i} * SB + XFER_ID_BYTES, T) != 0 || std::memcmp(bl.data() + size_t{i} * XFER_ID_BYTES, r + size_t{i} * SB, XFER_ID_BYTES) != 0;
    }
    return bad ? "its RaptorQ blocks are not the C's (" + std::to_string(bad) + " of " + std::to_string(n) + ")" : "";
  } catch (const std::exception& x) {
    return std::string("the GPU's RaptorQ blocks: ") + x.what();
  }
}

}  // namespace lizard
