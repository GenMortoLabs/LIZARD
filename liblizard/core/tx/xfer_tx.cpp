// ai: The transfer's sending end (xfer_tx.h): every sender's, the web's too since 2026-11-25 (its JS twin, sim/xfer.mjs
// ai: XferSender and Carousel, line for line until then, deleted).
#include "xfer_tx.h"

#include <algorithm>
#include <set>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <cstring>
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
}

// ai: Threads: natively, and in a wasm build made with them (-pthread: the web's threaded module, Emscripten's pthreads
// ai: over one shared heap, 2026-11-25); the plain wasm module has none, and makes every source block on the caller's thread
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#define LIZ_TX_THREADS 1
#endif

namespace lizard {

namespace {
constexpr uint32_t PAYLOAD = XFER_PAYLOAD;
constexpr uint32_t CONTROL = XFER_CONTROL, CONTROL_EVERY = 189;   // ai: (63 until 2026-11-21, with the manifest)
constexpr xfer_id_t ID_HEADER = XFER_ID_HEADER;
xfer_id_t idOf(uint32_t c, uint32_t s) { return xfer_id(c, s); }
// ai: the carousel's draws (mix32, xorshift): murmur3's finaliser, a word's bits mixed; xorshift32, a
// ai: state that must not be 0
uint32_t mix32(uint32_t x) { x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16; return x; }
// ai: A source block's symbols, pass by pass (2026-11-15): its v-th is the (v mod 2^24)-th of pass v >> 24 under a keyed
// ai: shuffle of the 2^24 ids (four Feistel rounds over two 12-bit halves, the round function mix32), the pass's key
// ai: drawn from the source block's (the seed's) and the pass. Every id once a pass and each pass in an order of its own, so a
// ai: receiver never meets the same ids at the same times twice: with consecutive ids from a start (until 2026-11-15)
// ai: each pass repeated the last, and someone back a whole number of passes later (a kiosk looping one file, a pass of
// ai: a one-source block file about 19 minutes at the 2:1 sender's rate), watching the stretch they had watched, held every
// ai: id it showed.
uint32_t passKey(uint32_t sbKey, uint64_t pass) { return mix32(sbKey ^ mix32(static_cast<uint32_t>(pass) * 0x85ebca6bu + 0x27d4eb2fu)); }
uint32_t shuffle24(uint32_t key, uint32_t k) {
  uint32_t l = k >> 12 & 0xfff, r = k & 0xfff;
  for (uint32_t i = 0; i < 4; i++) {
    const uint32_t t = r;
    r = l ^ (mix32(key ^ i * 0x9e3779b9u ^ r) & 0xfff);
    l = t;
  }
  return l << 12 | r;
}
uint32_t unshuffle24(uint32_t key, uint32_t e) {
  uint32_t l = e >> 12 & 0xfff, r = e & 0xfff;
  for (uint32_t i = 4; i-- > 0;) {
    const uint32_t t = l;
    l = r ^ (mix32(key ^ i * 0x9e3779b9u ^ l) & 0xfff);
    r = t;
  }
  return l << 12 | r;
}
uint32_t esiOf(uint32_t sbKey, uint64_t v) { return shuffle24(passKey(sbKey, v >> 24), static_cast<uint32_t>(v) & 0xffffffu); }
uint32_t xorshift(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
uint32_t nonzero(uint32_t s) { return s ? s : 0x9e3779b9u; }

// ai: The prepare's workers: one a physical core, not a hardware thread (2026-11-06, 1 GiB on the 9950X, 16 cores of
// ai: 2 threads: 16 workers 0.46 s random and 1.09 s compressible, 32 workers 0.62 and 1.18 s at twice the CPU time;
// ai: two source blocks' solves on one core's caches go slower than one). The hardware threads where the system
// ai: does not say; one in the plain wasm module. The threaded one: LIZ_CORES where its page sets it (a browser tells
// ai: hardwareConcurrency alone, its hardware threads; the page knows the architecture: lizard-web/send-worker.mjs), else
// ai: the hardware threads.
unsigned countCores() {
  unsigned n = 0;
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  n = 1;
#elif defined(__EMSCRIPTEN__)
  const char* e = getenv("LIZ_CORES");
  n = e && *e ? static_cast<unsigned>(atoi(e)) : 0;
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
// ai: the source block size's log2: the caller's, else xfer_sb_log2_for's for the length (XFER_LOG2_MAX past what it holds,
// ai: which the header then refuses)
int pickLog2(uint64_t length, int k) {
  if (k >= 0) return k;
  const int f = xfer_sb_log2_for(length);
  return f < 0 ? XFER_LOG2_MAX : f;
}
// ai: source block c's RaptorQ encoder over its own bytes (src/rq.h)
rq_enc_t* encoderOf(const uint8_t* plain, size_t len, uint32_t T = PAYLOAD) {
  rq_enc_t* e = rq_enc_new(plain, len, T);
  if (!e) throw std::runtime_error("RaptorQ encoder: out of memory");
  return e;
}
// ai: The spill's file (xfer_tx.h): made in dir and unlinked at once (POSIX), deleted on close (Windows), `bytes` long;
// ai: -1 where it cannot be made. Reads and writes at an offset, whole or false. It grows 256 MiB at a time, `stopped`
// ai: asked between (std::runtime_error "cancelled", the file closed): a file system without holes writes the zeros
// ai: (2026-11-16: ecryptfs, about 0.9 s a GiB of the file sent, one call that could not be stopped).
void spillClose(intptr_t f);
intptr_t spillOpen(const std::string& dir, uint64_t bytes, const std::function<bool()>& stopped) {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  (void)dir; (void)bytes; (void)stopped;
  return -1;
#else
#if defined(_WIN32)
  static std::atomic<uint32_t> made{0};
  const std::string path = dir + "\\lizard-spill-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(made++) + ".tmp";
  const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  std::wstring w(wn > 0 ? wn : 1, L'\0');
  if (wn > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wn);
  HANDLE h = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  if (h == INVALID_HANDLE_VALUE) return -1;
  const intptr_t f = reinterpret_cast<intptr_t>(h);
  const auto grow = [h](uint64_t n) {
    LARGE_INTEGER at;
    at.QuadPart = static_cast<LONGLONG>(n);
    return SetFilePointerEx(h, at, nullptr, FILE_BEGIN) && SetEndOfFile(h);
  };
#else
  std::string path = dir + "/lizard-spill-XXXXXX";
  const int fd = mkstemp(path.data());
  if (fd < 0) return -1;
  unlink(path.c_str());
  const intptr_t f = fd;
  const auto grow = [fd](uint64_t n) { return ftruncate(fd, static_cast<off_t>(n)) == 0; };
#endif
  constexpr uint64_t STEP = uint64_t{1} << 28;
  for (uint64_t at = 0; at < bytes;) {
    if (stopped && stopped()) { spillClose(f); throw std::runtime_error("cancelled"); }
    at = std::min(bytes, at + STEP);
    if (!grow(at)) { spillClose(f); return -1; }
  }
  return f;
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
}  // namespace

unsigned physicalCores() {
  static const unsigned n = countCores();
  return n;
}

XferTx::XferTx(const uint8_t* data, size_t length, const std::string& name, int sbLog2, const std::string& spillDir,
               const std::function<bool()>& stopped, bool defer, uint8_t compression)
    : XferTx(data, Reader{}, length, name, sbLog2, spillDir, stopped, defer, compression) {}

XferTx::XferTx(Reader read, uint64_t length, const std::string& name, int sbLog2, const std::string& spillDir,
               const std::function<bool()>& stopped, bool defer, uint8_t compression)
    : XferTx(nullptr, std::move(read), length, name, sbLog2, spillDir, stopped, defer, compression) {}

const uint8_t* XferTx::bytesOf(uint32_t c, std::vector<uint8_t>& buf) const {
  const uint64_t at = uint64_t(c) * size_;
  if (data_) return data_ + at;
  buf.resize(len_[c]);
  if (!read_ || !read_(at, len_[c], buf.data())) throw std::runtime_error("the file could not be read at chunk " + std::to_string(c));
  return buf.data();
}

XferTx::XferTx(const uint8_t* data, Reader read, uint64_t length, const std::string& name, int sbLog2, const std::string& spillDir,
               const std::function<bool()>& stopped, bool defer, uint8_t compression)
    : size_(size_t{1} << pickLog2(length, sbLog2)), length_(length), data_(data), read_(std::move(read)) {
  sbLog2 = pickLog2(length, sbLog2);
  xfer_header_t h{};
  if (xfer_header_init(&h, length, sbLog2)) throw std::runtime_error("the file is too large for source blocks of 2^" + std::to_string(sbLog2) + " bytes");
  const uint32_t C = static_cast<uint32_t>((length + size_ - 1) / size_);
  sourceBlocks_ = C;
  len_.resize(C); K_.resize(C); enc_.resize(C); key_.assign(C, 0); cnt_.assign(C, 0);
  for (uint32_t c = 0; c < C; c++) { len_[c] = size_t(std::min<uint64_t>(size_, length - uint64_t(c) * size_)); K_[c] = uint32_t((len_[c] + PAYLOAD - 1) / PAYLOAD); }
  if (!spillDir.empty() && C) {
    // ai: the spill's slots: three segments a source block of a quarter of its blocks, at least 64
    seg_.resize(C); off_.resize(C); base_.resize(C); segBase_.resize(C); taken_.assign(C, 0);
    have_.reset(new std::atomic<int64_t>[C]);
    uint64_t at = 0;
    for (uint32_t c = 0; c < C; c++) {
      seg_[c] = std::max<uint32_t>(64, (K_[c] + 3) / 4);
      off_[c] = at;
      at += 3ull * seg_[c] * PAYLOAD;
      have_[c] = -1;
    }
    spillDir_ = spillDir;
    spillBytes_ = at;
    if (!defer) {
      spill_ = spillOpen(spillDir, at, stopped);
      if (spill_ < 0) throw std::runtime_error("the spill's file could not be made in " + spillDir);
    }
  }
  {
    // ai: a random seed (the clock mixed in: some C++ libraries' random_device repeats from run to run)
    std::random_device rd;
    seedWith(rd() ^ mix32(static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count())));
  }
  ready_.reset(new std::atomic<uint8_t>[std::max<uint32_t>(C, 1)]());
  const uint32_t W = std::min<uint32_t>(std::max<uint32_t>(C, 1), physicalCores());
  if (C >= 2) {
    // ai: every source block's chaining value first (its own bytes, BLAKE3's wide hash), a thread a physical core, then the
    // ai: root: all the header needs, so the first frame waits on a hash of the file and not on its fountains
    cvs_.resize(32 * size_t{C});
    std::atomic<uint32_t> nc{0}, bad{UINT32_MAX};
    std::atomic<bool> halt{false};
    auto hash = [&] {
      std::vector<uint8_t> buf;   // ai: a source block read (read_), this thread's
      for (uint32_t c; !halt.load(std::memory_order_relaxed) && (c = nc.fetch_add(1, std::memory_order_relaxed)) < C;) {
        if (stopped && stopped()) { halt = true; break; }
        const uint8_t* b = nullptr;
        try { b = bytesOf(c, buf); } catch (const std::exception&) {}
        if (!b || xfer_sb_cv(b, len_[c], c, sbLog2, cvs_.data() + 32 * c)) { uint32_t none = UINT32_MAX; bad.compare_exchange_strong(none, c); }
      }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < W; w++) try { pool.emplace_back(hash); } catch (const std::system_error&) { break; }   // ai: a thread refused: fewer do the work
    hash();
    for (auto& t : pool) t.join();
    // ai: a constructor that throws runs no destructor: the spill closed first
    const auto fail = [&](const std::string& why) {
      if (spill_ >= 0) { spillClose(spill_); spill_ = -1; }
      throw std::runtime_error(why);
    };
    if (halt) fail("cancelled");
    if (bad != UINT32_MAX) fail("chaining value of chunk " + std::to_string(bad.load()));
    if (xfer_root(cvs_.data(), C, root_)) fail("root");
  } else {
    if (C == 1 && !defer) make(0);
    std::vector<uint8_t> buf;
    xfer_b3_hash(C ? bytesOf(0, buf) : data, size_t(length), root_);
  }
  std::memcpy(h.root, root_, 32);
  // ai: the name a description carries: one past 255 bytes cut in its stem, its extension kept (src/xfer.h xfer_name_cut;
  // ai: a file alone's description always fits its session header
  h.name_len = static_cast<uint8_t>(xfer_name_cut(reinterpret_cast<const uint8_t*>(name.data()), name.size(), h.name));
  h.compression = compression;
  hdr_ = h;
  headed_ = true;
  drawTois();
  writeHeader();
#ifdef LIZ_TX_THREADS
  if (spill_ >= 0) spillThread_ = std::thread([this] { spiller(); });
#endif
  if (defer && C) return;   // ai: makeOnCpu makes them
  making_ = true;
  if (C < 2) { buildCarousel(); return; }   // ai: made already: its cycle known at once
  // ai: the source blocks made behind the constructor, a worker a physical core; on this thread where no other can start
  // ai: (the plain wasm module has no threads), every source block then before it returns
#ifdef LIZ_TX_THREADS
  for (uint32_t w = 0; w < W; w++) try { workers_.emplace_back([this] { work(); }); } catch (const std::system_error&) { break; }
#endif
  if (workers_.empty()) { work(); const std::string f = fault(); if (!f.empty()) throw std::runtime_error(f); buildCarousel(); }
}

XferTx::~XferTx() {
  {
    std::lock_guard<std::mutex> l(spillMu_);
    stop_ = true;
  }
  spillCv_.notify_all();
  for (auto& t : workers_) if (t.joinable()) t.join();
  if (maker_.joinable()) maker_.join();   // ai: its spill thread set before it ends, or never
  if (spillThread_.joinable()) spillThread_.join();
  if (spill_ >= 0) spillClose(spill_);
}

void XferTx::makeOnCpu(Sink sink) {
  if (!sourceBlocks_ || making_.exchange(true)) return;
  sink_ = std::move(sink);
#ifdef LIZ_TX_THREADS
  try { maker_ = std::thread([this] { makeAll(); }); return; } catch (const std::system_error&) {}
#endif
  makeAll();   // ai: no thread (the plain wasm module, or one refused): every source block here
}

// ai: makeOnCpu's: the spill's file first where a folder was given (grown here, off the configuring thread; the
// ai: destructor's stop ends its growth), then a worker a physical core, this thread one of them
void XferTx::makeAll() {
  if (!sink_ && !spillDir_.empty() && spill_ < 0) {
    intptr_t f = -1;
    try { f = spillOpen(spillDir_, spillBytes_, [this] { return stop_.load(); }); } catch (const std::exception&) { return; }   // ai: stopped
    if (f < 0) {
      std::lock_guard<std::mutex> l(faultMu_);
      if (fault_.empty()) fault_ = "the spill's file could not be made in " + spillDir_;
      return;
    }
    std::lock_guard<std::mutex> l(spillMu_);
    spill_ = f;
#ifdef LIZ_TX_THREADS
    spillThread_ = std::thread([this] { spiller(); });
#endif
  }
  [[maybe_unused]] const uint32_t W = std::min<uint32_t>(sourceBlocks_, physicalCores());   // ai: (the plain wasm module has no pool)
  std::vector<std::thread> pool;
#ifdef LIZ_TX_THREADS
  for (uint32_t w = 1; w < W; w++) try { pool.emplace_back([this] { work(); }); } catch (const std::system_error&) { break; }
#endif
  work();
  for (auto& t : pool) t.join();
}



bool XferTx::reseed(uint32_t seed) {
  if (frame_) return false;
  std::lock_guard<std::mutex> l(spillMu_);
  seedWith(seed);
  // ai: the spill's segments made from the old seed's first symbols: made again from the new
  if (spill_ >= 0)
    for (uint32_t c = 0; c < sourceBlocks_; c++)
      if (have_[c] >= 0 && segBase_[c] != base_[c]) {
        have_[c] = -1;
        asks_.push_back({c, 0}); asks_.push_back({c, 1});
      }
  spillCv_.notify_all();
  return true;
}

// ai: Source block c's first symbol mix32(seed ^ (c + 1) 0x9e3779b1) in the symbols' 24 bits;
// ai: the rounds' generator from mix32(seed ^ 0x5bd1e995); each frame's shuffle from mix32((frame + 1) 0x9e3779b1 ^ seed);
// ai: the session's TOI from it (writeHeader, sessionToi); the file's TOI its content's (drawTois, once the root is known)
void XferTx::seedWith(uint32_t seed) {
  seed_s_ = seed;
  for (uint32_t c = 0; c < sourceBlocks_; c++) key_[c] = mix32(seed ^ (c + 1) * 0x9e3779b1u);
  if (!base_.empty()) std::copy(key_.begin(), key_.end(), base_.begin());   // ai: the spill's: a segment's symbols drawn by it
  rs_ = nonzero(mix32(seed ^ 0x5bd1e995u));
  if (carouselBuilt_) { i_ = 0; newRound(); }
  if (headed_) { drawTois(); writeHeader(); }
}

// ai: The session's TOI as every sender draws it (sessionToi): from the send's seed,
// ai: mix32(seed ^ 0x3c6ef372) in its high 24 bits and mix32(seed ^ 0x165667b1) in its low 24, 0 taken as 1, and one past
// ai: the file's where it is that.
static xfer_toi_t sessionToi(uint32_t seed, xfer_toi_t avoid) {
  xfer_toi_t t = xfer_toi_t(mix32(seed ^ 0x3c6ef372u) & 0xffffffu) << 24 | (mix32(seed ^ 0x165667b1u) & 0xffffffu);
  if (!t) t = 1;
  if (t == avoid) t = t % XFER_TOI_MASK + 1;
  return t;
}

void XferTx::drawTois() {
  toi_ = xfer_toi_of(root_, hdr_.length, hdr_.sb_log2);
}

// ai: The session header (src/xfer.h): the file's description (its own map), under the
// ai: session's TOI (the seed's). A new seed's rewrites it.
void XferTx::writeHeader() {
  uint8_t e[XFER_DESC_LONGEST];
  const int n = xfer_desc_write(&hdr_, e);
  if (n < 0) throw std::runtime_error("a file makes no description");
  desc_ = std::make_unique<XferDesc>(std::vector<uint8_t>(e, e + n), sessionToi(seed_s_, toi_));
  std::memcpy(header_, desc_->header, PAYLOAD);
}

xfer_toi_t XferTx::headerToi() const { return desc_ ? desc_->toi : toi_; }

XferDesc::XferDesc(std::vector<uint8_t> desc, xfer_toi_t t) : toi(t), bytes(std::move(desc)) {
  if (xfer_session_write(header, bytes.data(), bytes.size())) throw std::runtime_error("session header refused");
}

void XferDesc::block(xfer_id_t id, uint8_t* out) const {
  if (xfer_id_sbn(id) == CONTROL && !xfer_id_esi(id)) std::memcpy(out, header, PAYLOAD); else std::memset(out, 0, PAYLOAD);
}

// ai: The round's order: every source block once (each takes the round's visit by its weight, next), Fisher-Yates from the
// ai: rounds' generator, so a round's places hold no source block from one round to the next
void XferTx::newRound() {
  perm_.resize(sourceBlocks_);
  std::iota(perm_.begin(), perm_.end(), 0u);
  for (uint32_t i = sourceBlocks_; i > 1; i--) std::swap(perm_[i - 1], perm_[xorshift(rs_) % i]);
}

// ai: Source block c: a RaptorQ encoder over its bytes (a source block of one block too: K = 1). Its slots written, then ready_.
void XferTx::make(uint32_t c) {
  std::unique_ptr<rq_enc_t, EncFree> enc;
  {
    std::vector<uint8_t> buf;   // ai: the source block read (read_), let go once its encoder is made
    enc.reset(encoderOf(bytesOf(c, buf), len_[c]));
  }
  if (sink_) {
    // ai: the GPU streams its blocks (gpu_stream.h): the encoder handed over, then let go
    if (!sink_(c, enc.get())) throw std::runtime_error("stopped");
  } else if (spill_ >= 0) {
    // ai: the spill: segments 0 and 1 written, the encoder let go; a seed given meanwhile (reseed) makes them again
    uint32_t base;
    { std::lock_guard<std::mutex> l(spillMu_); base = base_[c]; }
    writeSegment(c, enc.get(), 0, base);
    writeSegment(c, enc.get(), 1, base);
    std::lock_guard<std::mutex> l(spillMu_);
    segBase_[c] = base;
    if (base == base_[c]) have_[c] = 1;
    else { asks_.push_back({c, 0}); asks_.push_back({c, 1}); }
    spillCv_.notify_all();
  } else enc_[c] = std::move(enc);
  ready_[c].store(1, std::memory_order_release);
  made_.fetch_add(1, std::memory_order_acq_rel);
}

// ai: A worker: the next source block until none is left; a failure is kept (the first) and stops every worker
void XferTx::work() {
  for (uint32_t c; !stop_.load(std::memory_order_relaxed) && (c = next_.fetch_add(1, std::memory_order_relaxed)) < sourceBlocks_;) {
    try { make(c); } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(faultMu_);
      if (fault_.empty()) fault_ = "chunk " + std::to_string(c) + ": " + e.what();
      stop_ = true;
    }
  }
}

void XferTx::waitMade() {
  for (auto& t : workers_) if (t.joinable()) t.join();
  if (maker_.joinable()) maker_.join();
  const std::string f = fault();
  if (!f.empty()) throw std::runtime_error(f);
}

std::string XferTx::fault() const {
  std::lock_guard<std::mutex> l(faultMu_);
  return fault_;
}

// ai: the carousel (Carousel's constructor), once every source block is made: a round visits every source block of the most blocks
// ai: once; a source block of fewer in proportion to its blocks, plus 2 sqrt(K) visits a cycle and at least 64. The symbols a
// ai: source block sent while the others were made go on (cnt_), so none is sent twice. Each round visits the source blocks in an order
// ai: of its own (newRound).
std::vector<uint32_t> XferTx::weights() const {
  std::vector<uint32_t> w;
  const uint32_t mx = sourceBlocks_ ? *std::max_element(K_.begin(), K_.end()) : 0;
  for (uint32_t c = 0; c < sourceBlocks_; c++) {
    const uint32_t k = K_[c];
    w.push_back(k >= mx ? mx : std::min(mx, std::max<uint32_t>(64, k + static_cast<uint32_t>(std::ceil(2 * std::sqrt(static_cast<double>(k)))))));
  }
  return w;
}

uint32_t XferTx::cycleWhenMade() const {
  if (!made()) return 0;   // ai: (made_'s acquire: K_ written before it; scheduled_ and cycle_ are the painters' thread's)
  uint32_t cycle = 0;
  for (uint32_t v : weights()) cycle += v;
  return cycle;
}

void XferTx::buildCarousel() {
  carouselBuilt_ = true;
  const uint32_t C = sourceBlocks_;
  acc_.assign(C, 0);
  max_ = C ? *std::max_element(K_.begin(), K_.end()) : 0;
  w_ = weights();
  uint32_t cycle = 0;
  for (uint32_t v : w_) cycle += v;
  // ai: the session header on a control slot about every eighth of a cycle (frameIds), CONTROL_EVERY data blocks apart
  // ai: at most
  every_ = std::max<uint32_t>(1, std::min<uint32_t>(CONTROL_EVERY, cycle / 8));
  i_ = 0; since_ = UINT32_MAX;
  newRound();
  cycle_ = cycle;
}

// ai: A frame before every source block is made: the header alone. The start-up is taken whole before any data block goes (2026-11-16):
// ai: every source block then starts the carousel level. Frames that carried the source blocks made so far, in turn, left the source blocks made first ahead
// ai: for the whole transfer while the file waited on the last made (a 64 GiB run with frames at 430 a second: about 3%
// ai: of its symbols past the file's, the source blocks spread about fifteen times wider than its losses alone spread them).
// ai: The source blocks are made a worker a physical core, so the wait is short.
void XferTx::warm(xfer_id_t* out, int n) { std::fill(out, out + n, ID_HEADER); }

void XferTx::EncFree::operator()(rq_enc_t* e) const { rq_enc_free(e); }

// ai: Segment n of source block c (its symbols esiOf(base, v) for v in [n seg, (n + 1) seg)) into slot n mod 3
void XferTx::writeSegment(uint32_t c, const rq_enc_t* enc, uint64_t n, uint32_t base) {
  const uint32_t L = seg_[c];
  std::vector<uint8_t> buf(static_cast<size_t>(L) * PAYLOAD);
  for (uint32_t k = 0; k < L; k++)
    rq_enc_symbol(enc, esiOf(base, n * L + k), buf.data() + static_cast<size_t>(k) * PAYLOAD);
  if (!spillIo(spill_, off_[c] + (n % 3) * L * uint64_t{PAYLOAD}, buf.data(), buf.size(), true))
    throw std::runtime_error("the spill's file could not be written");
}

// ai: The spill's thread: each segment asked for, in order: the source block's encoder made again from the mapped file, the
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
      std::vector<uint8_t> buf;
      std::unique_ptr<rq_enc_t, EncFree> enc(encoderOf(bytesOf(c, buf), len_[c]));
      buf = {};
      writeSegment(c, enc.get(), a.second, base);
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

// ai: A symbol of source block c named: the spill's count, and entering segment n (n >= 1) the segment after asked for (a
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

xfer_id_t XferTx::next() {
  uint32_t c;
  for (;;) {
    if (i_ >= sourceBlocks_) { i_ = 0; newRound(); }
    c = perm_[i_++];
    if ((acc_[c] += w_[c]) >= max_) { acc_[c] -= max_; break; }
  }
  const uint32_t s = esiOf(key_[c], cnt_[c]++);   // ai: past 2^24 a source block, a pass in a new order (esiOf)
  took(c);
  return idOf(c, s);
}

void XferTx::frameIds(xfer_id_t* out, int n) {
  if (!sourceBlocks_) { std::fill(out, out + n, ID_HEADER); frame_++; return; }   // ai: an empty file is its header
  if (!carouselBuilt_ && made()) buildCarousel();
  if (!carouselBuilt_) warm(out, n);
  else {
    // ai: A control block every P frames, P odd (2026-11-18): the frames that carry every_ data blocks between controls,
    // ai: rounded up to an odd count, every control block the header (no manifest since 2026-11-21), so the header's
    // ai: frames are an odd count apart. A camera taking every other frame, in either
    // ai: phase, then meets the header. Before, a control block every frame (116 blocks a frame at 2:1, every_ at most
    // ai: 63) alternated header and manifest, so the header rode every other frame alone and a camera in step with the
    // ai: other half never met it; with one code (58), a control every second frame, so that half met no control block
    // ai: at all (a stream file's odd frames alone, 2026-11-18, research/build/stream-ui-2026-11-18).
    int first = 0;
    const uint32_t per = static_cast<uint32_t>(std::max(1, n - 1));
    const uint32_t P = ((every_ + per - 1) / per) | 1u;
    if ((since_ == UINT32_MAX || ++since_ >= P) && n > 1) {
      out[first++] = ID_HEADER;
      since_ = 0;
    }
    for (int k = first; k < n; k++) out[k] = next();
  }
  // ai: Fisher-Yates over every slot, the control block's too, from a generator seeded by the frame count and the seed
  // ai: (xorshift32).
  // ai: Until 2026-10-10 the control block kept slot 0 and only the data slots were shuffled: slot 0 is block 0, on
  // ai: the innermost sub-channel, which since the rate profile carries the 6/7 code and decodes for some contents
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

void XferTx::block(xfer_id_t id, uint8_t* out) const {
  const uint32_t c = xfer_id_sbn(id), s = xfer_id_esi(id);
  if (c == CONTROL) {
    if (s != 0) { std::memset(out, 0, PAYLOAD); return; }
    std::memcpy(out, header_, PAYLOAD);   // ai: the file's
    return;
  }
  std::memset(out, 0, PAYLOAD);
  if (c >= sourceBlocks_ || !ready_[c].load(std::memory_order_acquire)) return;   // ai: frameIds names only source blocks made
  if (spill_ >= 0) {
    // ai: the spill: the symbol's count v (the newest named with id s: its place in the last named's pass, or in the
    // ai: pass before where that comes after the last named; a frame's ids are all of its last segment or the one
    // ai: before), its segment n, read from slot n mod 3 once written. An id named twice in those segments (once either
    // ai: side of a pass's end: each pass names every id) maps to its later naming, the same ESI and so the same bytes,
    // ai: in a segment still kept (research/build/idorder-2026-11-15/spill_map_check.cpp).
    const uint64_t last = taken_[c] - 1, p = last >> 24;
    uint64_t v = p << 24 | unshuffle24(passKey(base_[c], p), s);
    if (v > last && p) v = (p - 1) << 24 | unshuffle24(passKey(base_[c], p - 1), s);
    const uint64_t n = v / seg_[c];
    {
      std::unique_lock<std::mutex> l(spillMu_);
      spillCv_.wait(l, [&] { return stop_.load() || have_[c] >= static_cast<int64_t>(n); });
      if (have_[c] < static_cast<int64_t>(n)) return;   // ai: stopped (a failure is fault()'s): zeros
    }
    if (!spillIo(spill_, off_[c] + ((n % 3) * seg_[c] + v % seg_[c]) * uint64_t{PAYLOAD}, out, PAYLOAD, false)) std::memset(out, 0, PAYLOAD);
    return;
  }
  if (enc_[c]) rq_enc_symbol(enc_[c].get(), s, out);
}


std::string XferTx::rootHex() const {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : root_) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}

uint32_t XferTx::sourceSymbols() const {
  uint32_t n = 0;
  for (uint32_t k : K_) n += k;
  return n;
}

}  // namespace lizard
