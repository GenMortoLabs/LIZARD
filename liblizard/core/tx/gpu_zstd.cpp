// ai: Stream files' frames packed on the GPU (gpu_zstd.h).
#include "gpu_zstd.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>

#include "lizfile.h"
#include "wg.h"
#include "zstd.h"

// ai: zstd's own Huffman builder and table writer (lib/compress/huf_compress.c, in the one-file build lizard_xfer_core
// ai: compiles): a code of lengths at most maxNbBits from the counts, each entry its length in the low byte and its
// ai: bits left-aligned in the word above (HUF_setValue); the description zstd's decoder reads (RFC 8878 4.2.1)
extern "C" {
typedef size_t HUF_CElt;
size_t HUF_buildCTable_wksp(HUF_CElt* tree, const unsigned* count, unsigned maxSymbolValue, unsigned maxNbBits, void* workSpace, size_t wkspSize);
size_t HUF_writeCTable_wksp(void* dst, size_t maxDstSize, const HUF_CElt* CTable, unsigned maxSymbolValue, unsigned huffLog, void* workspace,
                            size_t workspaceSize);
}

namespace lizard {

namespace {

constexpr uint32_t BLOCK = 131072;   // ai: zstd's largest block (Block_Maximum_Size, 128 KiB)
constexpr uint32_t MAX_BITS = 11;    // ai: zstd's longest Huffman code
// ai: the CPU twin's scratch for a stream: a quarter block's symbols at MAX_BITS each and the end mark, in words
constexpr uint32_t CAPW = 12288;
constexpr uint32_t MODE_HUF = 0, MODE_RAW = 1, MODE_RLE = 2;
// ai: a block below this goes raw (zstd's four streams want a few literals; a short last block gains nothing)
constexpr uint32_t RAW_BELOW = 256;

std::vector<uint32_t> spirv(const std::string& path) {
  std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) throw std::runtime_error("the tree has no " + path + ": regenerate liblizard/out");
  const std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<uint32_t> w(b.size() / 4);
  std::memcpy(w.data(), b.data(), w.size() * 4);
  return w;
}

// ai: JPEG-LS's median predictor (rx/lizfile.h), as zstd_med.slang
inline int med(int a, int b, int c) {
  const int mx = a > b ? a : b, mn = a < b ? a : b;
  return c >= mx ? mn : c <= mn ? mx : a + b - c;
}

// ai: A block's plan from its counts: RLE (one value), raw, or Huffman with its code (an entry a value: its bits | its
// ai: length << 16, as zstd_huf.slang takes it) and the table's description
struct Plan {
  uint32_t mode = MODE_RAW, rle = 0;
  uint32_t code[256] = {};
  uint64_t bits = 0;   // ai: the block's code bits in all (its four streams' sum)
  std::vector<uint8_t> desc;
};
// ai: a block's place in the output, bytes, a multiple of 4: none for RLE, its residuals for raw, its streams' bound
// ai: for Huffman (the four streams' bytes, each its bits and the end mark rounded up: at most (bits + 4) / 8 + 4)
uint32_t boundOf(const Plan& p, uint32_t bn) {
  const uint64_t b = p.mode == MODE_RLE ? 0 : p.mode == MODE_RAW ? bn : (p.bits + 4) / 8 + 5;
  return uint32_t((b + 3) & ~uint64_t{3});
}
Plan planOf(const uint32_t count[256], uint32_t bn) {
  Plan p;
  unsigned c[256], maxSym = 0, distinct = 0;
  for (int v = 0; v < 256; v++) { c[v] = count[v]; if (c[v]) { maxSym = unsigned(v); distinct++; } }
  if (distinct == 1) { p.mode = MODE_RLE; p.rle = maxSym; return p; }
  if (bn < RAW_BELOW) return p;
  HUF_CElt ct[258];
  alignas(8) uint8_t wk[8192];
  const size_t nb = HUF_buildCTable_wksp(ct, c, maxSym, MAX_BITS, wk, sizeof wk);
  if (ZSTD_isError(nb)) return p;
  uint8_t desc[256];
  const size_t dl = HUF_writeCTable_wksp(desc, sizeof desc, ct, maxSym, unsigned(nb), wk, sizeof wk);
  if (ZSTD_isError(dl) || dl == 0) return p;
  uint64_t bits = 0;
  for (unsigned v = 0; v <= maxSym; v++) {
    const HUF_CElt e = ct[1 + v];
    const uint32_t len = uint32_t(e & 0xff), val = len ? uint32_t(e >> (sizeof(HUF_CElt) * 8 - len)) : 0;
    p.code[v] = val | (len << 16);
    bits += uint64_t(len) * c[v];
  }
  // ai: the block as zstd would take it: its header, the description, the jump table, the four streams and their end
  // ai: marks, and the sequences' byte; raw where that is not smaller
  if (5 + dl + 6 + bits / 8 + 4 + 1 >= bn) return p;
  p.mode = MODE_HUF;
  p.bits = bits;
  p.desc.assign(desc, desc + dl);
  return p;
}

// ai: f(i) for i in [0, n) on the machine's cores, each a run of i in turn (the codes and the frames: a square's work
// ai: its own)
template <typename F> void each(size_t n, const F& f) {
  const size_t T = std::min<size_t>(n, std::max(1u, std::thread::hardware_concurrency()));
  if (T <= 1) { for (size_t i = 0; i < n; i++) f(i); return; }
  std::vector<std::thread> ts;
  for (size_t t = 0; t < T; t++) ts.emplace_back([&, t] { for (size_t i = n * t / T; i < n * (t + 1) / T; i++) f(i); });
  for (auto& th : ts) th.join();
}

// ai: A square's zstd frame: the magic, the header (single segment, its content size in 4 bytes, no checksum), then each
// ai: block: RLE, raw, or compressed (its literals' header in 5 bytes, size format 3: four streams, 18-bit sizes; the
// ai: table's description; the jump table; the streams; no sequences). Written at p, or with p null its bytes counted
// ai: (at): a batch's frames are counted, given their places, then written there.
struct Frame {
  uint8_t* p;
  size_t at = 0;
  Frame(uint8_t* out, uint32_t n) : p(out) { le(0xFD2FB528u, 4); byte(0xA0); le(n, 4); }
  void byte(uint8_t v) { if (p) p[at] = v; at++; }
  void bytes(const uint8_t* b, size_t k) { if (p && k) std::memcpy(p + at, b, k); at += k; }
  void le(uint64_t v, int k) { for (int i = 0; i < k; i++) byte(uint8_t(v >> (8 * i))); }
  void header(uint32_t size, uint32_t type, bool last) { le((uint64_t(size) << 3) | (type << 1) | (last ? 1 : 0), 3); }
  void rle(uint32_t bn, uint8_t v, bool last) { header(bn, 1, last); byte(v); }
  void raw(const uint8_t* b, uint32_t bn, bool last) { header(bn, 0, last); bytes(b, bn); }
  void huf(uint32_t bn, const std::vector<uint8_t>& desc, const uint8_t* const s[4], const uint32_t sz[4], bool last) {
    const uint32_t comp = uint32_t(desc.size()) + 6 + sz[0] + sz[1] + sz[2] + sz[3];
    header(5 + comp + 1, 2, last);
    le(2u | (3u << 2) | (uint64_t(bn) << 4) | (uint64_t(comp) << 22), 5);
    bytes(desc.data(), desc.size());
    for (int k = 0; k < 3; k++) le(sz[k], 2);
    for (int k = 0; k < 4; k++) bytes(s[k], sz[k]);
    byte(0);   // ai: no sequences
  }
};
// ai: a square's frame's bound: its residuals, each block's header and its literals' header (raw never past it)
size_t frameBound(uint32_t n, uint32_t nb) { return size_t(n) + 16ull * nb + 64; }

// ai: a stream's bits as zstd_huf.slang writes them: its symbols from the last to the first, each code least significant
// ai: bit first, then the end mark; its bytes
uint32_t streamCpu(const uint8_t* sym, uint32_t len, const uint32_t code[256], uint8_t* out) {
  uint64_t pos = 0;
  std::memset(out, 0, CAPW * 4);
  for (uint32_t i = len; i-- > 0;) {
    const uint32_t e = code[sym[i]], l = e >> 16, c = e & 0xffff;
    for (uint32_t b = 0; b < l; b++) if ((c >> b) & 1) out[(pos + b) >> 3] |= uint8_t(1u << ((pos + b) & 7));
    pos += l;
  }
  out[pos >> 3] |= uint8_t(1u << (pos & 7));
  return uint32_t(pos >> 3) + 1;
}

}  // namespace

std::vector<uint8_t> GpuZstd::packCpu(const uint8_t* px, int side, size_t stride) {
  const uint32_t n = uint32_t(side) * uint32_t(side);
  std::vector<uint8_t> r(n);
  for (int y = 0; y < side; y++)
    for (int x = 0; x < side; x++) {
      const int a = x ? px[size_t(y) * stride + x - 1] : 0, b = y ? px[size_t(y - 1) * stride + x] : 0, c = x && y ? px[size_t(y - 1) * stride + x - 1] : 0;
      r[size_t(y) * side + x] = uint8_t(px[size_t(y) * stride + x] - med(a, b, c));
    }
  std::vector<uint8_t> o(frameBound(n, (n + BLOCK - 1) / BLOCK));
  Frame f(o.data(), n);
  std::vector<uint8_t> st(4 * CAPW * 4);
  for (uint32_t start = 0; start < n; start += BLOCK) {
    const uint32_t bn = std::min(BLOCK, n - start);
    const bool last = start + bn >= n;
    uint32_t count[256] = {};
    for (uint32_t i = 0; i < bn; i++) count[r[start + i]]++;
    const Plan p = planOf(count, bn);
    if (p.mode == MODE_RLE) { f.rle(bn, uint8_t(p.rle), last); continue; }
    if (p.mode == MODE_RAW) { f.raw(r.data() + start, bn, last); continue; }
    const uint32_t seg = (bn + 3) / 4;
    const uint8_t* s[4];
    uint32_t sz[4];
    for (uint32_t k = 0; k < 4; k++) {
      const uint32_t len = k < 3 ? seg : bn - 3 * seg;
      sz[k] = streamCpu(r.data() + start + k * seg, len, p.code, st.data() + k * CAPW * 4);
      s[k] = st.data() + k * CAPW * 4;
    }
    f.huf(bn, p.desc, s, sz, last);
  }
  o.resize(f.at);
  return o;
}

// ai: A batch's buffers and state (GpuZstd::begin, codes, finish): its slots and parameters, the residuals, the
// ai: counts, the codes and the blocks' places, the streams and their sizes, their readbacks, its bind groups (kept
// ai: while their commands run), its plans and its tickets
struct GpuZstd::Set {
  std::shared_ptr<wg::Buffer> slots, res, hist, tab, mode, out, size, pMed, pHist, pHuf, readHist, readOut, readSize;
  std::shared_ptr<wg::BindGroup> bMed, bHist, bHuf;
  std::shared_ptr<wg::Ticket> t1, t2;
  std::vector<Plan> plans;
  std::vector<uint32_t> place;   // ai: each block's place in the output, bytes
  uint64_t total = 0;            // ai: the output's bytes
  uint32_t S = 0;
  int stage = 0;                 // ai: 0 free, 1 begun, 2 coded, 3 sized (its streams read back, its frames counted)
  const uint8_t* out_host = nullptr;        // ai: the streams read back (stage 3)
  const uint32_t* sizes_host = nullptr;     // ai: their sizes
  std::vector<size_t> frameBytes;           // ai: each square's frame's bytes
  std::chrono::steady_clock::time_point t;
};

GpuZstd::GpuZstd(wg::Device& d) : d_(d) {}

std::unique_ptr<GpuZstd> GpuZstd::create(wg::Device& d, const std::string& assets, int side, int squares) {
  using namespace wg;
  if (side <= 0) throw std::runtime_error("no squares to pack");
  std::unique_ptr<GpuZstd> z(new GpuZstd(d));
  z->side_ = side;
  z->n_ = uint32_t(side) * uint32_t(side);
  z->wps_ = (z->n_ + 3) / 4;
  z->nb_ = (z->n_ + BLOCK - 1) / BLOCK;
  // ai: a square's bytes on the device: its residuals, its counts, codes and places, its output twice (the device's and
  // ai: the readback's, each at most its residuals' bytes and a little: a block packs to no more than it is); SETS
  // ai: batches in a share of the memory (a quarter on a discrete GPU, an eighth where the GPU shares the system's), at
  // ai: most 1 GiB, 128 squares a batch
  const uint64_t outSq = uint64_t(z->nb_) * BLOCK + 64ull * z->nb_;
  const uint64_t per = SETS * (4ull * z->wps_ + 2ull * z->nb_ * 1024 + 2 * outSq + z->nb_ * 40);
  const uint64_t share = std::min<uint64_t>(1ull << 30, d.limits.deviceLocalBytes / (d.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 4 : 8));
  const uint64_t range = std::min<uint64_t>(d.limits.maxStorageBufferRange, d.limits.maxBufferSize ? d.limits.maxBufferSize : UINT64_MAX);
  int sq = squares > 0 ? squares : int(std::min<uint64_t>(128, share / per));
  sq = int(std::min<uint64_t>(uint64_t(std::max(1, sq)), range / outSq));
  if (sq < 1) throw std::runtime_error("a square's output passes the device's largest buffer");
  z->maxSq_ = sq;
  const std::string spv = assets + "/spv/send_zstd_";
  z->bglMed_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "zstd med");
  z->bglHist_ = d.createBindGroupLayout({Bind::RO, Bind::RW, Bind::Uniform}, "zstd hist");
  z->bglHuf_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RO, Bind::RW, Bind::RW, Bind::Uniform}, "zstd huf");
  z->med_ = d.createPipeline(spirv(spv + "med.spv"), z->bglMed_, "", "zstd med");
  z->hist_ = d.createPipeline(spirv(spv + "hist.spv"), z->bglHist_, "", "zstd hist");
  z->huf_ = d.createPipeline(spirv(spv + "huf.spv"), z->bglHuf_, "", "zstd huf");
  const uint64_t S = uint64_t(sq), blocks = S * z->nb_;
  for (int k = 0; k < SETS; k++) {
    auto x = std::make_unique<Set>();
    x->slots = d.createBuffer(16 + 4 * S, STORAGE | COPY_DST, "zstd SLOTS");
    x->res = d.createBuffer(4ull * z->wps_ * S, STORAGE, "zstd RES", false);
    x->hist = d.createBuffer(1024 * blocks, STORAGE | COPY_SRC, "zstd HIST", false);
    x->tab = d.createBuffer(1024 * blocks, STORAGE | COPY_DST, "zstd TAB", false);
    x->mode = d.createBuffer(8 * blocks, STORAGE | COPY_DST, "zstd MODE", false);
    x->out = d.createBuffer(outSq * S, STORAGE | COPY_SRC | COPY_DST, "zstd OUT", false);
    x->size = d.createBuffer(16 * blocks, STORAGE | COPY_SRC, "zstd SIZE", false);
    x->pMed = d.createBuffer(32, UNIFORM | COPY_DST, "zstd P med");
    x->pHist = d.createBuffer(16, UNIFORM | COPY_DST, "zstd P hist");
    x->pHuf = d.createBuffer(16, UNIFORM | COPY_DST, "zstd P huf");
    x->readHist = d.createBuffer(1024 * blocks, MAP_READ | COPY_DST, "zstd read HIST", false);
    x->readOut = d.createBuffer(outSq * S, MAP_READ | COPY_DST, "zstd read OUT", false);
    x->readSize = d.createBuffer(16 * blocks, MAP_READ | COPY_DST, "zstd read SIZE", false);
    z->sets_.push_back(std::move(x));
  }
  return z;
}

GpuZstd::~GpuZstd() {
  // ai: the commands still in flight finished before their buffers go
  for (auto& x : sets_) {
    try { if (x->t1) x->t1->wait(); if (x->t2) x->t2->wait(); } catch (const std::exception&) {}
  }
}

int GpuZstd::begin(const wg::Buffer& ring, const std::vector<int>& slots, uint32_t frameWords, uint32_t rowWords, int codes, int step, std::mutex& devMu) {
  using namespace wg;
  const uint32_t S = uint32_t(slots.size()) * uint32_t(codes), nb = nb_;
  if (S == 0 || S > uint32_t(maxSq_)) throw std::runtime_error("a batch of " + std::to_string(S) + " squares, not 1 to " + std::to_string(maxSq_));
  int k = -1;
  for (int i = 0; i < SETS && k < 0; i++) if (sets_[(nextSet_ + i) % SETS]->stage == 0) k = (nextSet_ + i) % SETS;
  if (k < 0) throw std::runtime_error("no free set: a batch finished first");
  nextSet_ = (k + 1) % SETS;
  Set& x = *sets_[k];
  x.S = S;
  x.t = std::chrono::steady_clock::now();
  const uint64_t blocks = uint64_t(S) * nb;
  std::lock_guard<std::mutex> g(devMu);
  std::vector<uint32_t> sl(slots.begin(), slots.end());
  d_.writeBuffer(*x.slots, 0, sl.data(), 4 * sl.size());
  const uint32_t pm[8] = {S, uint32_t(side_), rowWords, frameWords, uint32_t(codes), uint32_t(step), n_, wps_};
  d_.writeBuffer(*x.pMed, 0, pm, sizeof pm);
  const uint32_t ph[4] = {S, n_, wps_, nb};
  d_.writeBuffer(*x.pHist, 0, ph, sizeof ph);
  x.bMed = d_.createBindGroup(bglMed_, {const_cast<Buffer*>(&ring), x.slots, x.res, x.pMed}, "zstd med");
  x.bHist = d_.createBindGroup(bglHist_, {x.res, x.hist, x.pHist}, "zstd hist");
  Encoder e = d_.encoder();
  e.beginComputePass();
  e.setPipeline(*med_);
  e.setBindGroup(*x.bMed);
  e.dispatch((wps_ + 255) / 256, S);
  e.setPipeline(*hist_);
  e.setBindGroup(*x.bHist);
  e.dispatch(nb, S);
  e.endPass();
  e.copyBufferToBuffer(*x.hist, 0, *x.readHist, 0, 1024 * blocks);
  x.t1 = d_.submit(e);
  x.stage = 1;
  return k;
}

void GpuZstd::codes(int set, std::mutex& devMu) {
  using namespace wg;
  Set& x = *sets_.at(size_t(set));
  if (x.stage != 1) throw std::runtime_error("codes of a batch not begun");
  const uint32_t nb = nb_;
  const uint64_t blocks = uint64_t(x.S) * nb;
  using clk = std::chrono::steady_clock;
  const auto msSince = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
  x.t1->wait();
  x.t1.reset();
  last_.counts = msSince(x.t);
  auto tm = clk::now();
  // ai: each block's plan from its counts (zstd's Huffman builder) on the cores; its place in the output, then the
  // ai: codes, modes and places up
  x.plans.assign(blocks, Plan{});
  std::vector<uint32_t> tab(256 * blocks), mode(2 * blocks);
  const uint32_t* h = reinterpret_cast<const uint32_t*>(d_.read(*x.readHist, 0, 1024 * blocks));
  each(size_t(blocks), [&](size_t b) {
    const uint32_t bi = uint32_t(b % nb), bn = std::min(BLOCK, n_ - bi * BLOCK);
    x.plans[b] = planOf(h + 256 * b, bn);
    std::memcpy(tab.data() + 256 * b, x.plans[b].code, 1024);
  });
  x.place.assign(blocks, 0);
  uint64_t at = 0;
  for (uint64_t b = 0; b < blocks; b++) {
    const uint32_t bi = uint32_t(b % nb), bn = std::min(BLOCK, n_ - bi * BLOCK);
    x.place[b] = uint32_t(at);
    mode[2 * b] = x.plans[b].mode;
    mode[2 * b + 1] = uint32_t(at);
    at += boundOf(x.plans[b], bn);
  }
  x.total = std::max<uint64_t>(4, at);
  last_.codes = msSince(tm);
  x.t = clk::now();
  std::lock_guard<std::mutex> g(devMu);
  d_.writeBuffer(*x.tab, 0, tab.data(), 4 * tab.size());
  d_.writeBuffer(*x.mode, 0, mode.data(), 4 * mode.size());
  const uint32_t pf[4] = {x.S, n_, wps_, nb};
  d_.writeBuffer(*x.pHuf, 0, pf, sizeof pf);
  x.bHuf = d_.createBindGroup(bglHuf_, {x.res, x.tab, x.mode, x.out, x.size, x.pHuf}, "zstd huf");
  Encoder e = d_.encoder();
  e.clearBuffer(*x.out, 0, x.total);
  e.beginComputePass();
  e.setPipeline(*huf_);
  e.setBindGroup(*x.bHuf);
  e.dispatch(nb, x.S);
  e.endPass();
  e.copyBufferToBuffer(*x.out, 0, *x.readOut, 0, x.total);
  e.copyBufferToBuffer(*x.size, 0, *x.readSize, 0, 16 * blocks);
  x.t2 = d_.submit(e);
  x.stage = 2;
}

// ai: a square's frame from its blocks' streams in the readback (p null: counted), as zstd_huf.slang placed them
void GpuZstd::assemble(Set& x, size_t q, uint8_t* p, size_t& bytes, std::string& bad) const {
  const uint32_t nb = nb_;
  const uint8_t* out = x.out_host;
  Frame f(p, n_);
  for (uint32_t bi = 0; bi < nb; bi++) {
    const uint64_t b = uint64_t(q) * nb + bi;
    const uint32_t bn = std::min(BLOCK, n_ - bi * BLOCK);
    const bool last = bi + 1 == nb;
    const uint8_t* region = out + x.place[b];
    const Plan& pl = x.plans[b];
    if (pl.mode == MODE_RLE) { f.rle(bn, uint8_t(pl.rle), last); continue; }
    if (pl.mode == MODE_RAW) { f.raw(region, bn, last); continue; }
    const uint8_t* st[4];
    uint32_t sz[4];
    uint32_t off = 0;
    for (int k = 0; k < 4; k++) {
      sz[k] = x.sizes_host[b * 4 + k];
      st[k] = region + off;
      off += sz[k];
      if (sz[k] == 0 || sz[k] > 0xffff) { bad = "a stream of " + std::to_string(sz[k]) + " B"; return; }
    }
    if (off > boundOf(pl, bn)) { bad = "a block's streams past their bound"; return; }
    f.huf(bn, pl.desc, st, sz, last);
  }
  bytes = f.at;
}

std::vector<size_t> GpuZstd::sizes(int set) {
  Set& x = *sets_.at(size_t(set));
  if (x.stage == 3) return x.frameBytes;
  if (x.stage != 2) throw std::runtime_error("sizes of a batch not coded");
  using clk = std::chrono::steady_clock;
  x.t2->wait();
  x.t2.reset();
  x.bMed.reset(); x.bHist.reset(); x.bHuf.reset();
  last_.streams = std::chrono::duration<double, std::milli>(clk::now() - x.t).count();
  x.t = clk::now();
  const uint64_t blocks = uint64_t(x.S) * nb_;
  x.out_host = d_.read(*x.readOut, 0, x.total);
  x.sizes_host = reinterpret_cast<const uint32_t*>(d_.read(*x.readSize, 0, 16 * blocks));
  x.frameBytes.assign(x.S, 0);
  std::vector<std::string> bad(x.S);
  for (size_t q = 0; q < x.S; q++) assemble(x, q, nullptr, x.frameBytes[q], bad[q]);
  for (const auto& b : bad) if (!b.empty()) { x.stage = 0; throw std::runtime_error(b); }
  x.stage = 3;
  return x.frameBytes;
}

void GpuZstd::finishInto(int set, const std::vector<uint8_t*>& dst, std::vector<uint32_t>* crcs) {
  Set& x = *sets_.at(size_t(set));
  if (x.stage == 2) sizes(set);
  if (x.stage != 3) throw std::runtime_error("finish of a batch not coded");
  if (dst.size() != x.S) throw std::runtime_error("a place a square");
  if (crcs) crcs->assign(x.S, 0);
  std::vector<std::string> bad(x.S);
  each(x.S, [&](size_t q) {
    size_t n = 0;
    assemble(x, q, dst[q], n, bad[q]);
    if (crcs && bad[q].empty()) (*crcs)[q] = lizfile::crc32(dst[q], n);
  });
  x.stage = 0;
  for (const auto& b : bad) if (!b.empty()) throw std::runtime_error(b);
  last_.frames = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - x.t).count();
}

std::vector<std::vector<uint8_t>> GpuZstd::finish(int set, std::vector<uint32_t>* crcs) {
  const std::vector<size_t> sz = sizes(set);
  std::vector<std::vector<uint8_t>> frames(sz.size());
  std::vector<uint8_t*> dst;
  for (size_t q = 0; q < sz.size(); q++) { frames[q].resize(sz[q]); dst.push_back(frames[q].data()); }
  finishInto(set, dst, crcs);
  return frames;
}

std::vector<std::vector<uint8_t>> GpuZstd::pack(const wg::Buffer& ring, const std::vector<int>& slots, uint32_t frameWords, uint32_t rowWords,
                                                 int codes, int step, std::mutex& devMu) {
  if (slots.empty()) return {};
  const int k = begin(ring, slots, frameWords, rowWords, codes, step, devMu);
  this->codes(k, devMu);
  return finish(k);
}

}  // namespace lizard
