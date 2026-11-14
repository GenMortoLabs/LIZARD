// ai: The transfer's sending end (xfer_tx.h), liblizard/sim/xfer.mjs XferSender and Schedule line for line.
#include "xfer_tx.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <set>
#include <unistd.h>
#include <fstream>
#endif
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#endif

extern "C" {
#include "xfer.h"
int lizard_wh_init(void);
int lizard_wh_encoder_create(const void* message, uint32_t message_bytes, uint32_t block_bytes, void** codec_out);
int lizard_wh_encode(void* codec, uint32_t block_id, void* out, uint32_t capacity);
void lizard_wh_free(void* codec);
}

namespace lizard {

namespace {
constexpr uint32_t PAYLOAD = XFER_PAYLOAD, SYMBOL_BITS = XFER_SYMBOL_BITS, SYMBOL_MASK = (1u << SYMBOL_BITS) - 1;
constexpr uint32_t CONTROL = XFER_CONTROL, ID_HEADER = XFER_ID_HEADER, CONTROL_EVERY = 63;
uint32_t idOf(uint32_t c, uint32_t s) { return c << SYMBOL_BITS | s; }
// ai: the schedule's draws (sim/xfer.mjs mix32, xorshift): murmur3's finaliser, a word's bits mixed; xorshift32, a
// ai: state that must not be 0
uint32_t mix32(uint32_t x) { x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16; return x; }
uint32_t xorshift(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
uint32_t nonzero(uint32_t s) { return s ? s : 0x9e3779b9u; }

// ai: sim/xfer.mjs nameBytes: UTF-8, cut to 255 bytes at a character's start
std::string nameBytes(const std::string& name) {
  size_t n = std::min<size_t>(name.size(), XFER_NAME_MAX);
  while (n < name.size() && n > 0 && (static_cast<uint8_t>(name[n]) & 0xc0) == 0x80) n--;
  return name.substr(0, n);
}
// ai: The prepare's workers: one a physical core, not a hardware thread (2026-11-06, 1 GiB on the 9950X, 16 cores of
// ai: 2 threads: 16 workers 0.46 s random and 1.09 s compressible, 32 workers 0.62 and 1.18 s at twice the CPU time;
// ai: two chunks' Wirehair solves on one core's caches go slower than one). The hardware threads where the system
// ai: does not say; one under emscripten (no threads in the wasm build).
unsigned countCores() {
  unsigned n = 0;
#if defined(__EMSCRIPTEN__)
  n = 1;
#elif defined(_WIN32)
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  std::vector<uint8_t> b(bytes);
  if (bytes && GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(b.data()), &bytes))
    for (DWORD o = 0; o < bytes; o += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(b.data() + o)->Size) n++;
#elif defined(__APPLE__)
  int c = 0;
  size_t len = sizeof c;
  if (!sysctlbyname("hw.physicalcpu", &c, &len, nullptr, 0) && c > 0) n = static_cast<unsigned>(c);
#elif defined(__linux__)
  // ai: the online CPUs' distinct sibling sets (an offline CPU has no topology)
  std::set<std::string> cores;
  const long conf = sysconf(_SC_NPROCESSORS_CONF);
  for (long i = 0; i < conf; i++) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
    std::string list;
    if (f >> list) cores.insert(list);
  }
  n = static_cast<unsigned>(cores.size());
#endif
  return n ? n : std::max(1u, std::thread::hardware_concurrency());
}
// ai: the chunk size's log2: the caller's, else xfer_chunk_log2_for's for the length (XFER_LOG2_MAX past what it holds,
// ai: which the header then refuses)
int pickLog2(size_t length, int k) {
  if (k >= 0) return k;
  const int f = xfer_chunk_log2_for(length);
  return f < 0 ? XFER_LOG2_MAX : f;
}
// ai: Wirehair's seed attempt for a message of `len` bytes, which it picks from the block count alone (the equations'
// ai: structure, not the data, src/xfer.h): an encoder made once over zeros of that length, kept by length; 0 for one
// ai: block. The header states it for every chunk but the last and for the last before any chunk is made.
int seedFor(size_t len) {
  if (xfer_blocks(len) < 2) return 0;
  static std::mutex mu;
  static std::map<size_t, int> memo;
  std::lock_guard<std::mutex> l(mu);
  const auto it = memo.find(len);
  if (it != memo.end()) return it->second;
  std::vector<uint8_t> zeros(len);
  void* e = nullptr;
  const int s = lizard_wh_encoder_create(zeros.data(), static_cast<uint32_t>(len), PAYLOAD, &e);
  if (s < 0) throw std::runtime_error("wirehair encoder: " + std::to_string(s));
  lizard_wh_free(e);
  return memo[len] = s;
}
// ai: The spill's file (xfer_tx.h): made in dir and unlinked at once (POSIX), deleted on close (Windows), `bytes` long;
// ai: -1 where it cannot be made. Reads and writes at an offset, whole or false.
intptr_t spillOpen(const std::string& dir, uint64_t bytes) {
#if defined(__EMSCRIPTEN__)
  (void)dir; (void)bytes;
  return -1;
#elif defined(_WIN32)
  static std::atomic<uint32_t> made{0};
  const std::string path = dir + "\\lizard-spill-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(made++) + ".tmp";
  const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  std::wstring w(wn > 0 ? wn : 1, L'\0');
  if (wn > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wn);
  HANDLE h = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  if (h == INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER at;
  at.QuadPart = static_cast<LONGLONG>(bytes);
  if (!SetFilePointerEx(h, at, nullptr, FILE_BEGIN) || !SetEndOfFile(h)) { CloseHandle(h); return -1; }
  return reinterpret_cast<intptr_t>(h);
#else
  std::string path = dir + "/lizard-spill-XXXXXX";
  const int fd = mkstemp(path.data());
  if (fd < 0) return -1;
  unlink(path.c_str());
  if (ftruncate(fd, static_cast<off_t>(bytes))) { close(fd); return -1; }
  return fd;
#endif
}
bool spillIo(intptr_t f, uint64_t at, uint8_t* p, size_t n, bool write) {
#if defined(_WIN32)
  while (n) {
    OVERLAPPED o{};
    o.Offset = static_cast<DWORD>(at);
    o.OffsetHigh = static_cast<DWORD>(at >> 32);
    DWORD done = 0;
    const DWORD want = static_cast<DWORD>(std::min<size_t>(n, 1u << 30));
    const BOOL ok = write ? WriteFile(reinterpret_cast<HANDLE>(f), p, want, &done, &o) : ReadFile(reinterpret_cast<HANDLE>(f), p, want, &done, &o);
    if (!ok || !done) return false;
    p += done; at += done; n -= done;
  }
  return true;
#else
  while (n) {
    const ssize_t k = write ? pwrite(static_cast<int>(f), p, n, static_cast<off_t>(at)) : pread(static_cast<int>(f), p, n, static_cast<off_t>(at));
    if (k <= 0) return false;
    p += k; at += static_cast<uint64_t>(k); n -= static_cast<size_t>(k);
  }
  return true;
#endif
}
void spillClose(intptr_t f) {
#if defined(_WIN32)
  CloseHandle(reinterpret_cast<HANDLE>(f));
#else
  close(static_cast<int>(f));
#endif
}
// ai: typeBytes: printable ASCII of at most 158, else none
std::string typeBytes(const std::string& t) {
  if (t.size() > XFER_TYPE_MAX) return "";
  for (char c : t) if (c < 0x20 || c > 0x7e) return "";
  return t;
}
}  // namespace

unsigned physicalCores() {
  static const unsigned n = countCores();
  return n;
}

XferTx::XferTx(const uint8_t* data, size_t length, const std::string& name, const std::string& type, int chunkLog2,
               const std::string& spillDir)
    : size_(size_t{1} << pickLog2(length, chunkLog2)), data_(data) {
  static std::once_flag wh;
  std::call_once(wh, [] { lizard_wh_init(); });
  chunkLog2 = pickLog2(length, chunkLog2);
  xfer_header_t h{};
  if (xfer_header_init(&h, length, chunkLog2)) throw std::runtime_error("the file is too large for 2^" + std::to_string(chunkLog2) + " chunks");
  const uint32_t C = static_cast<uint32_t>((length + size_ - 1) / size_), M = static_cast<uint32_t>(xfer_manifest_blocks(C));
  chunks_ = C;
  len_.resize(C); K_.resize(C); enc_.resize(C); one_.resize(C); sym_.assign(C, 0);
  for (uint32_t c = 0; c < C; c++) { len_[c] = std::min(size_, length - c * size_); K_[c] = xfer_blocks(len_[c]); }
  if (!spillDir.empty() && C) {
    // ai: the spill's slots: three segments a chunk of a quarter of its blocks, at least 64 (a chunk of one block none)
    seg_.resize(C); off_.resize(C); base_.resize(C); segBase_.resize(C); taken_.assign(C, 0);
    have_.reset(new std::atomic<int64_t>[C]);
    uint64_t at = 0;
    for (uint32_t c = 0; c < C; c++) {
      seg_[c] = K_[c] > 1 ? std::max<uint32_t>(64, (K_[c] + 3) / 4) : 0;
      off_[c] = at;
      at += 3ull * seg_[c] * PAYLOAD;
      have_[c] = -1;
    }
    spill_ = spillOpen(spillDir, at);
    if (spill_ < 0) throw std::runtime_error("the spill's file could not be made in " + spillDir);
  }
  {
    // ai: a random seed (the clock mixed in: some C++ libraries' random_device repeats from run to run)
    std::random_device rd;
    seedWith(rd() ^ mix32(static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count())));
  }
  // ai: the seed attempts the header states (src/xfer.h, version 4): every chunk's but the last, and the last's
  seedFull_ = static_cast<uint8_t>(C >= 2 ? seedFor(size_) : 0);
  seedLast_ = static_cast<uint8_t>(C ? seedFor(len_[C - 1]) : 0);
  h.seed = seedFull_; h.seed_last = seedLast_;
  ready_.reset(new std::atomic<uint8_t>[std::max<uint32_t>(C, 1)]());
  mready_.reset(new std::atomic<uint8_t>[std::max<uint32_t>(M, 1)]());
  left_.reset(new std::atomic<uint32_t>[std::max<uint32_t>(M, 1)]());
  for (uint32_t m = 0; m < M; m++) left_[m] = std::min<uint32_t>(XFER_PER_BLOCK, C - m * XFER_PER_BLOCK);
  manifest_.assign(M, std::vector<uint8_t>(PAYLOAD));
  const uint32_t W = std::min<uint32_t>(std::max<uint32_t>(C, 1), physicalCores());
  if (C >= 2) {
    // ai: every chunk's chaining value first (its own bytes, BLAKE3's wide hash), a thread a physical core, then the
    // ai: root: all the header needs with the two seeds, so the first frame waits on a hash of the file and not on its
    // ai: fountains
    cvs_.resize(32 * size_t{C});
    std::atomic<uint32_t> nc{0}, bad{UINT32_MAX};
    auto hash = [&] {
      for (uint32_t c; (c = nc.fetch_add(1, std::memory_order_relaxed)) < C;)
        if (xfer_chunk_cv(data + c * size_, len_[c], c, chunkLog2, cvs_.data() + 32 * c)) { uint32_t none = UINT32_MAX; bad.compare_exchange_strong(none, c); }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < W; w++) try { pool.emplace_back(hash); } catch (const std::system_error&) { break; }   // ai: a thread refused: fewer do the work
    hash();
    for (auto& t : pool) t.join();
    if (bad != UINT32_MAX) throw std::runtime_error("chaining value of chunk " + std::to_string(bad.load()));
    if (xfer_root(cvs_.data(), C, root_)) throw std::runtime_error("root");
  } else {
    if (C == 1) make(0);
    xfer_b3_hash(data, length, root_);
  }
  std::memcpy(h.root, root_, 32);
  const std::string nm = nameBytes(name), ty = typeBytes(type);
  h.name_len = static_cast<uint8_t>(nm.size()); std::memcpy(h.name, nm.data(), nm.size());
  h.type_len = static_cast<uint8_t>(ty.size()); std::memcpy(h.type, ty.data(), ty.size());
  if (xfer_header_write(&h, header_)) throw std::runtime_error("header refused");
#ifndef __EMSCRIPTEN__
  if (spill_ >= 0) spillThread_ = std::thread([this] { spiller(); });
#endif
  if (C < 2) { schedule(); return; }   // ai: made already: its lap known at once
  // ai: the chunks made behind the constructor, a worker a physical core; on this thread where no other can start
  // ai: (emscripten has no threads), every chunk then before it returns
#ifndef __EMSCRIPTEN__
  for (uint32_t w = 0; w < W; w++) try { workers_.emplace_back([this] { work(); }); } catch (const std::system_error&) { break; }
#endif
  if (workers_.empty()) { work(); const std::string f = fault(); if (!f.empty()) throw std::runtime_error(f); schedule(); }
}

XferTx::~XferTx() {
  {
    std::lock_guard<std::mutex> l(spillMu_);
    stop_ = true;
  }
  spillCv_.notify_all();
  for (auto& t : workers_) if (t.joinable()) t.join();
  if (spillThread_.joinable()) spillThread_.join();
  if (spill_ >= 0) spillClose(spill_);
}

bool XferTx::reseed(uint32_t seed) {
  if (frame_) return false;
  std::lock_guard<std::mutex> l(spillMu_);
  seedWith(seed);
  // ai: the spill's segments made from the old seed's first symbols: made again from the new
  if (spill_ >= 0)
    for (uint32_t c = 0; c < chunks_; c++)
      if (seg_[c] && have_[c] >= 0 && segBase_[c] != base_[c]) {
        have_[c] = -1;
        asks_.push_back({c, 0}); asks_.push_back({c, 1});
      }
  spillCv_.notify_all();
  return true;
}

// ai: Chunk c's first symbol mix32(seed ^ (c + 1) 0x9e3779b1) in the symbols' 18 bits, a chunk of one block's 0 always;
// ai: the rounds' generator from mix32(seed ^ 0x5bd1e995); each frame's shuffle from mix32((frame + 1) 0x9e3779b1 ^ seed)
void XferTx::seedWith(uint32_t seed) {
  seed_s_ = seed;
  for (uint32_t c = 0; c < chunks_; c++) sym_[c] = mix32(seed ^ (c + 1) * 0x9e3779b1u) & SYMBOL_MASK;
  if (!base_.empty()) std::copy(sym_.begin(), sym_.end(), base_.begin());   // ai: the spill's: a segment's symbols count from it
  rs_ = nonzero(mix32(seed ^ 0x5bd1e995u));
  if (scheduled_) { i_ = 0; newRound(); }
}

// ai: The round's order: every chunk once (each takes the round's visit by its weight, next), Fisher-Yates from the
// ai: rounds' generator, so a round's places hold no chunk from one round to the next
void XferTx::newRound() {
  perm_.resize(chunks_);
  std::iota(perm_.begin(), perm_.end(), 0u);
  for (uint32_t i = chunks_; i > 1; i--) std::swap(perm_[i - 1], perm_[xorshift(rs_) % i]);
}

// ai: Chunk c: a fountain over its bytes where they are 2 blocks or more, else the one block kept (one_). Its slots
// ai: written, then ready_; the last of a manifest group writes the group's block, then mready_.
void XferTx::make(uint32_t c) {
  const uint8_t* plain = data_ + c * size_;
  const uint32_t n = static_cast<uint32_t>(len_[c]);
  if (K_[c] > 1) {
    void* e = nullptr;
    const int seed = lizard_wh_encoder_create(plain, n, PAYLOAD, &e);
    if (seed < 0) throw std::runtime_error("wirehair encoder: " + std::to_string(seed));
    std::unique_ptr<void, WhFree> enc(e);
    // ai: Wirehair's attempt is the block count's: a chunk that took another than the header states would not decode
    const int want = c + 1 == chunks_ ? seedLast_ : seedFull_;
    if (seed != want) throw std::runtime_error("Wirehair's seed attempt " + std::to_string(seed) + ", the header's " + std::to_string(want));
    if (spill_ >= 0) {
      // ai: the spill: segments 0 and 1 written, the encoder let go; a seed given meanwhile (reseed) makes them again
      uint32_t base;
      { std::lock_guard<std::mutex> l(spillMu_); base = base_[c]; }
      writeSegment(c, e, 0, base);
      writeSegment(c, e, 1, base);
      std::lock_guard<std::mutex> l(spillMu_);
      segBase_[c] = base;
      if (base == base_[c]) have_[c] = 1;
      else { asks_.push_back({c, 0}); asks_.push_back({c, 1}); }
      spillCv_.notify_all();
    } else enc_[c] = std::move(enc);
  } else one_[c].assign(plain, plain + n);
  ready_[c].store(1, std::memory_order_release);
  if (chunks_ >= 2) {
    const uint32_t m = c / XFER_PER_BLOCK;
    if (left_[m].fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (xfer_manifest_write(cvs_.data(), chunks_, root_, m, manifest_[m].data())) throw std::runtime_error("manifest block " + std::to_string(m));
      mready_[m].store(1, std::memory_order_release);
    }
  }
  made_.fetch_add(1, std::memory_order_acq_rel);
}

// ai: A worker: the next chunk until none is left; a failure is kept (the first) and stops every worker
void XferTx::work() {
  for (uint32_t c; !stop_.load(std::memory_order_relaxed) && (c = next_.fetch_add(1, std::memory_order_relaxed)) < chunks_;) {
    try { make(c); } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(faultMu_);
      if (fault_.empty()) fault_ = "chunk " + std::to_string(c) + ": " + e.what();
      stop_ = true;
    }
  }
}

void XferTx::waitMade() {
  for (auto& t : workers_) if (t.joinable()) t.join();
  const std::string f = fault();
  if (!f.empty()) throw std::runtime_error(f);
}

std::string XferTx::fault() const {
  std::lock_guard<std::mutex> l(faultMu_);
  return fault_;
}

// ai: the schedule (Schedule's constructor), once every chunk is made: a round visits every chunk of the most blocks
// ai: once; a chunk of fewer in proportion to its blocks, plus 2 sqrt(K) visits a lap and at least 64. The symbols a
// ai: chunk sent while the others were made go on (sym_), so none is sent twice. Each round visits the chunks in an order
// ai: of its own (newRound).
void XferTx::schedule() {
  scheduled_ = true;
  const uint32_t C = chunks_;
  acc_.assign(C, 0);
  w_.clear();
  max_ = C ? *std::max_element(K_.begin(), K_.end()) : 0;
  uint32_t lap = 0;
  for (uint32_t c = 0; c < C; c++) {
    const uint32_t k = K_[c];
    w_.push_back(k >= max_ ? max_ : std::min(max_, std::max<uint32_t>(64, k + static_cast<uint32_t>(std::ceil(2 * std::sqrt(static_cast<double>(k)))))));
    lap += w_[c];
  }
  // ai: the whole control cycle (the header every other block, the manifest between) within an eighth of a lap
  every_ = std::max<uint32_t>(1, std::min<uint32_t>(CONTROL_EVERY, lap / (8 * std::max<uint32_t>(1, static_cast<uint32_t>(manifest_.size())))));
  i_ = 0; since_ = UINT32_MAX;
  newRound();
  lap_ = lap;
}

// ai: A frame before every chunk is made: a control block (where it has more than one slot, or nothing is made yet, or
// ai: every other frame of one), the header every other time and the manifest blocks written so far between; the rest
// ai: the chunks made so far in turn, each its next symbol (a chunk of one block its one); with none made, every slot a
// ai: control block.
void XferTx::warm(uint32_t* out, int n) {
  std::vector<uint32_t> ready, ms;
  for (uint32_t c = 0; c < chunks_; c++) if (ready_[c].load(std::memory_order_acquire)) ready.push_back(c);
  for (uint32_t m = 0; m < manifest_.size(); m++) if (mready_[m].load(std::memory_order_acquire)) ms.push_back(m);
  auto ctl = [&] { const uint32_t j = ctl_++; return ms.empty() || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ms[(j >> 1) % ms.size()]; };
  int k = 0;
  if (n > 1 || ready.empty() || !(frame_ & 1)) out[k++] = ctl();
  for (; k < n; k++) {
    if (ready.empty()) { out[k] = ctl(); continue; }
    const uint32_t c = ready[turn_++ % ready.size()];
    if (K_[c] == 1) { out[k] = idOf(c, 0); continue; }
    out[k] = idOf(c, sym_[c]);
    sym_[c] = (sym_[c] + 1) & SYMBOL_MASK;
    took(c);
  }
}

void XferTx::WhFree::operator()(void* e) const { lizard_wh_free(e); }

// ai: Segment n of chunk c (its symbols base + v for v in [n seg, (n + 1) seg), the symbols' 18 bits) into slot n mod 3
void XferTx::writeSegment(uint32_t c, void* enc, uint64_t n, uint32_t base) {
  const uint32_t L = seg_[c];
  std::vector<uint8_t> buf(static_cast<size_t>(L) * PAYLOAD);
  for (uint32_t k = 0; k < L; k++) {
    const uint32_t sym = static_cast<uint32_t>((base + n * L + k) & SYMBOL_MASK);
    const int got = lizard_wh_encode(enc, sym, buf.data() + static_cast<size_t>(k) * PAYLOAD, PAYLOAD);
    if (got <= 0) throw std::runtime_error("wirehair encode " + std::to_string(got) + " on chunk " + std::to_string(c));
  }
  if (!spillIo(spill_, off_[c] + (n % 3) * L * uint64_t{PAYLOAD}, buf.data(), buf.size(), true))
    throw std::runtime_error("the spill's file could not be written");
}

// ai: The spill's thread: each segment asked for, in order: the chunk's encoder made again from the mapped file, the
// ai: segment written, the encoder let go; a failure is the transfer's (fault) and stops it
void XferTx::spiller() {
  for (;;) {
    std::pair<uint32_t, uint64_t> a;
    uint32_t base;
    {
      std::unique_lock<std::mutex> l(spillMu_);
      spillCv_.wait(l, [&] { return stop_.load() || !asks_.empty(); });
      if (stop_) return;
      a = asks_.front();
      asks_.pop_front();
      base = base_[a.first];
    }
    const uint32_t c = a.first;
    try {
      void* e = nullptr;
      const int seed = lizard_wh_encoder_create(data_ + c * size_, static_cast<uint32_t>(len_[c]), PAYLOAD, &e);
      if (seed < 0) throw std::runtime_error("wirehair encoder: " + std::to_string(seed));
      std::unique_ptr<void, WhFree> enc(e);
      writeSegment(c, e, a.second, base);
    } catch (const std::exception& x) {
      {
        std::lock_guard<std::mutex> l(faultMu_);
        if (fault_.empty()) fault_ = "chunk " + std::to_string(c) + ": " + x.what();
      }
      std::lock_guard<std::mutex> l(spillMu_);
      stop_ = true;
      spillCv_.notify_all();
      return;
    }
    std::lock_guard<std::mutex> l(spillMu_);
    segBase_[c] = base;
    if (base == base_[c] && static_cast<int64_t>(a.second) > have_[c]) have_[c] = static_cast<int64_t>(a.second);
    spillCv_.notify_all();
  }
}

// ai: A symbol of chunk c named: the spill's count, and entering segment n (n >= 1) the segment after asked for (a
// ai: segment's airtime ahead; segments 0 and 1 are make's)
void XferTx::took(uint32_t c) {
  if (spill_ < 0) return;
  const uint64_t v = taken_[c]++;
  if (v && v % seg_[c] == 0) {
    std::lock_guard<std::mutex> l(spillMu_);
    asks_.push_back({c, v / seg_[c] + 1});
    spillCv_.notify_all();
  }
}

uint32_t XferTx::next() {
  uint32_t c;
  for (;;) {
    if (i_ >= chunks_) { i_ = 0; newRound(); }
    c = perm_[i_++];
    if ((acc_[c] += w_[c]) >= max_) { acc_[c] -= max_; break; }
  }
  if (K_[c] == 1) return idOf(c, 0);   // ai: unfountained: the one block, again
  const uint32_t s = sym_[c];
  sym_[c] = (s + 1) & SYMBOL_MASK;     // ai: past 2^18 ids a chunk the sender repeats (src/xfer.h)
  took(c);
  return idOf(c, s);
}

void XferTx::frameIds(uint32_t* out, int n) {
  if (!chunks_) { std::fill(out, out + n, ID_HEADER); frame_++; return; }   // ai: an empty file is its header
  if (!scheduled_ && made()) schedule();
  if (!scheduled_) warm(out, n);
  else {
    int first = 0;
    if (since_ >= every_ && n > 1) {
      const uint32_t m = static_cast<uint32_t>(manifest_.size()), j = ctl_++;
      out[first++] = !m || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ((j >> 1) % m);
      since_ = 0;
    }
    for (int k = first; k < n; k++) out[k] = next();
    if (since_ != UINT32_MAX) since_ += static_cast<uint32_t>(n - first);
  }
  // ai: Fisher-Yates over every slot, the control block's too, from a generator seeded by the frame count and the seed
  // ai: (xorshift32).
  // ai: Until 2026-10-10 the control block kept slot 0 and only the data slots were shuffled: slot 0 is block 0, on
  // ai: the innermost sub-channel, which since the rate profile carries the 7/8 code and decodes for some contents
  // ai: never (the header among them), so a file's header stopped arriving and its transfer starved (STATUS
  // ai: "A file's header pinned to slot 0").
  uint32_t s = nonzero(mix32(((frame_ + 1) * 0x9e3779b1u) ^ seed_s_));
  for (int i = n - 1; i > 0; i--) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const int j = static_cast<int>(s % static_cast<uint32_t>(i + 1));
    std::swap(out[i], out[j]);
  }
  frame_++;
}

void XferTx::block(uint32_t id, uint8_t* out) const {
  const uint32_t c = id >> SYMBOL_BITS, s = id & SYMBOL_MASK;
  if (c == CONTROL) {
    if (s == 0) std::memcpy(out, header_, PAYLOAD);
    else if (s - 1 < manifest_.size() && mready_[s - 1].load(std::memory_order_acquire)) std::memcpy(out, manifest_[s - 1].data(), PAYLOAD);
    else std::memset(out, 0, PAYLOAD);
    return;
  }
  std::memset(out, 0, PAYLOAD);
  if (c >= chunks_ || !ready_[c].load(std::memory_order_acquire)) return;   // ai: frameIds names only chunks made
  if (spill_ >= 0 && K_[c] > 1) {
    // ai: the spill: the symbol's count v (the newest named with s's place in the symbols' 18 bits; a frame's ids are
    // ai: all of its last segment or the one before), its segment n, read from slot n mod 3 once written
    const uint64_t last = taken_[c] - 1;
    const uint32_t off = (s - base_[c]) & SYMBOL_MASK;
    const uint64_t v = last - ((static_cast<uint32_t>(last) - off) & SYMBOL_MASK);
    const uint64_t n = v / seg_[c];
    {
      std::unique_lock<std::mutex> l(spillMu_);
      spillCv_.wait(l, [&] { return stop_.load() || have_[c] >= static_cast<int64_t>(n); });
      if (have_[c] < static_cast<int64_t>(n)) return;   // ai: stopped (a failure is fault()'s): zeros
    }
    if (!spillIo(spill_, off_[c] + ((n % 3) * seg_[c] + v % seg_[c]) * uint64_t{PAYLOAD}, out, PAYLOAD, false)) std::memset(out, 0, PAYLOAD);
    return;
  }
  if (enc_[c]) {
    uint8_t buf[PAYLOAD];
    const int n = lizard_wh_encode(enc_[c].get(), s, buf, PAYLOAD);
    if (n > 0) std::memcpy(out, buf, static_cast<size_t>(n));
    return;
  }
  std::memcpy(out, one_[c].data(), one_[c].size());   // ai: a chunk of one block, zero padded
}

std::string XferTx::rootHex() const {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : root_) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}

}  // namespace lizard
