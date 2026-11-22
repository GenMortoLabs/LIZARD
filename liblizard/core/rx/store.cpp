// ai: The receiving end's stores (store.h). FileStore is lizard-web/fountain-worker.mjs OpfsStore on plain files, as
// ai: xfer_rx.cpp held it until 2026-10-03, now on stdio alone (FILE*, 64-bit seeks, std::filesystem) so it builds on
// ai: Windows as on Linux and Android, and since 2026-10-22 a folder a transfer, kept across receivers (store.h);
// ai: MemoryStore is sim/xfer.mjs MemoryStore, one transfer at a time.
#include "store.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <new>
#include <unordered_map>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

extern "C" {
#include "xfer.h"
}

namespace fs = std::filesystem;

namespace lizard {

static constexpr uint32_t PAYLOAD = XFER_PAYLOAD, REC = XFER_BLOCK, IDB = XFER_ID_BYTES, VREC = 4 + XFER_CV;
// ai: a verified source block's number in the folder's list: 4 bytes, little-endian, then its 32-byte chaining value
static void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = uint8_t(v >> 8 * i); }
static uint32_t get32(const uint8_t* p) { return p[0] | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }

// ai: paths are UTF-8 everywhere; Windows' narrow stdio would read them in the ANSI code page
static fs::path pathOf(const std::string& utf8) { return fs::u8path(utf8); }
static FILE* openFile(const std::string& utf8, const char* mode) {
#ifdef _WIN32
  std::wstring m(mode, mode + strlen(mode));
  return _wfopen(pathOf(utf8).c_str(), m.c_str());
#else
  return fopen(utf8.c_str(), mode);
#endif
}
static bool seek64(FILE* f, uint64_t off) {
#ifdef _WIN32
  return _fseeki64(f, int64_t(off), SEEK_SET) == 0;
#else
  return fseeko(f, off_t(off), SEEK_SET) == 0;
#endif
}
// ai: false where the bytes may not have reached the disk (a write deferred until now that failed)
static bool syncFile(FILE* f) {
  if (fflush(f) != 0) return false;
#ifdef _WIN32
  return _commit(_fileno(f)) == 0;
#else
  return fsync(fileno(f)) == 0 || errno == EINVAL || errno == EROFS;
#endif
}

// ai: the header's name as one path component: no separators, no control bytes, never empty, . or ..; on Windows also
// ai: none of <>:"|?* nor a trailing dot or space, and no device name (CON, NUL, COM1, ...) as its stem
static std::string safeName(const std::string& s) {
  std::string r = s;
  for (char& ch : r) if (ch == '/' || ch == '\\' || (unsigned char)ch < 0x20 || ch == 0x7f) ch = '_';
#ifdef _WIN32
  for (char& ch : r) if (strchr("<>:\"|?*", ch)) ch = '_';
  while (!r.empty() && (r.back() == '.' || r.back() == ' ')) r.back() = '_';
  std::string stem = r.substr(0, r.find('.'));
  for (char& ch : stem) ch = char(toupper((unsigned char)ch));
  static const char* dev[] = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
                              "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};
  for (const char* d : dev) if (stem == d) { r = "_" + r; break; }
#endif
  return r.empty() || r == "." || r == ".." ? "file" : r;
}

// ai: A folder a transfer under dir (store.h fileStore). Source block files are kept open through a small LRU of FILE*s, since
// ai: a file of many source blocks has every one in flight until the first cycle ends and a phone's descriptors are few.
class FileStore final : public Store {
 public:
  // ai: the budget the stores made from one share: how many source blocks may be open at once (openBudget), and the folders
  // ai: open among them (each takes its share, at least two source blocks)
  struct Shared { size_t cap = 32, live = 0; };
  explicit FileStore(std::string d) : dir(std::move(d)), shared(std::make_shared<Shared>()) {
    shared->cap = openBudget();
    std::error_code e;
    fs::create_directories(pathOf(dir), e);
    if (e) { fail("store: " + dir + ": " + e.message()); return; }
    // ai: a parked transfer (a folder with its header) stays for its header to come again, and a finished file not yet
    // ai: taken out of it (finished) for its taker; anything else is nobody's
    for (auto& p : fs::directory_iterator(pathOf(dir), e))
      if (!p.is_directory() || !(fs::exists(p.path() / "header") || finished(p.path()))) { std::error_code r; fs::remove_all(p.path(), r); }
  }
  // ai: another transfer's: the same directory, not cleaned again, the same budget
  FileStore(std::string d, std::shared_ptr<Shared> sh) : dir(std::move(d)), shared(std::move(sh)) {}
  ~FileStore() override { closeAll(); }
  std::unique_ptr<Store> make() override { return std::make_unique<FileStore>(dir, shared); }
  bool keeps() const override { return true; }

  // ai: the transfer in hand let go whole, its folder (a finished file not yet moved out of it too)
  void clear() override {
    const std::string was = cur;
    closeAll();
    if (!was.empty()) { std::error_code e; fs::remove_all(pathOf(was), e); }
  }
  // ai: the receiver's Clear: the transfer in hand (its finished file too) and what is nobody's; parked transfers stay,
  // ai: an app's list's to delete, and so does a finished file of another transfer not yet taken out
  void removeAll() override {
    clear();
    error.clear();
    std::error_code e;
    for (auto& p : fs::directory_iterator(pathOf(dir), e))
      if (!fs::exists(p.path() / "header") && !finished(p.path())) { std::error_code r; fs::remove_all(p.path(), r); }
  }
  void park() override { closeAll(); }
  // ai: a folder by its id (28 hex digits: the root's 16 and the TOI's 12; a second of the same file's beside a received
  // ai: one not yet taken out, -2 on, open), or every folder of a root (its hex, or its first 16 digits; a folder named
  // ai: before 2026-11-20 by the 16 alone among them)
  void forget(const std::string& id) override {
    if (id.size() < 16 || id.find('/') != std::string::npos || id.find('\\') != std::string::npos || id.find('.') != std::string::npos) return;
    const bool folder = id.size() != 16 && id.size() != 64;
    const std::string want = folder ? id : id.substr(0, 16);
    std::error_code e;
    std::vector<std::string> names;
    for (auto& d : fs::directory_iterator(pathOf(dir), e)) {
      const std::string n = d.path().filename().u8string();
      if (folder ? n == want : n.compare(0, 16, want) == 0) names.push_back(n);
    }
    for (const auto& n : names) {
      if (cur == dir + "/" + n) { clear(); continue; }
      std::error_code r;
      fs::remove_all(pathOf(dir + "/" + n), r);
    }
  }

  bool open(const std::string& id, const uint8_t* desc, size_t descLen, uint64_t length, Before& before) override {
    before = Before{};
    error.clear();
    closeAll();
    if (!within(length)) return false;
    buffersFor(length);
    cur = dir + "/" + id;
    // ai: a folder holding a received file not yet taken out (a take that failed) is never opened again: the same file
    // ai: received again (its TOIs let go, xfer_rx_core.h releaseDone) goes to a folder of its own beside it, id-2 on
    for (int n = 2; finished(pathOf(cur)) && n < 1000; n++) cur = dir + "/" + id + "-" + std::to_string(n);
    shared->live++;
    std::error_code e;
    // ai: the folder resumed where it holds this file's description (xfer_same_file: its name and compression may be another
    // ai: sender's; the folder keeps the first), else made afresh; a folder of a header before 2026-11-20 (464 bytes, no
    // ai: description) is made afresh
    const std::vector<uint8_t> had = slurp(cur + "/header");
    xfer_header_t a, b;
    const bool same = desc && !had.empty() && !xfer_desc_read(had.data(), had.size(), 0, &a, nullptr) &&
                      !xfer_desc_read(desc, descLen, 0, &b, nullptr) && xfer_same_file(&a, &b) && fs::exists(pathOf(cur + "/out"));
    if (!same) {
      // ai: room: the free space where the store is (a resumed folder is not asked again: what it holds is there)
      if (room > 0) {
        std::error_code se;
        const fs::space_info sp = fs::space(pathOf(dir), se);
        if (!se && double(sp.available) < room * double(length)) {
          refusal = "room";
          free = sp.available;
          cur.clear();
          fail("store: a file of " + std::to_string(length) + " B wants " + std::to_string(uint64_t(room * double(length))) + " B free, " + std::to_string(sp.available) + " B are");
          return false;
        }
      }
      fs::remove_all(pathOf(cur), e);
      fs::create_directories(pathOf(cur), e);
      if (e) { fail("store: " + cur + ": " + e.message()); return false; }
      out = openFile(cur + "/out", "w+b");
      // ai: the file at its full length (zeros), as ftruncate made it: a source block is written at its own offset
      if (!out || (length && (!seek64(out, length - 1) || fputc(0, out) == EOF))) { fail("store: " + cur + "/out: " + strerror(errno)); return false; }
      // ai: the header last, so a folder with one holds the output it describes
      if (desc && !writeAll(cur + "/header", desc, descLen)) { fail("store: " + cur + "/header: " + strerror(errno)); return false; }
      return true;
    }
    out = openFile(cur + "/out", "r+b");
    if (!out) { fail("store: " + cur + "/out: " + strerror(errno)); return false; }
    before.found = true;
    const std::vector<uint8_t> v = slurp(cur + "/verified");
    if (v.size() % VREC == 0)
      for (size_t o = 0; o + VREC <= v.size(); o += VREC) before.verified.push_back({get32(v.data() + o), std::vector<uint8_t>(v.begin() + o + 4, v.begin() + o + VREC)});
    for (auto& p : fs::directory_iterator(pathOf(cur), e)) {
      const std::string f = p.path().filename().string();
      if (f.size() < 2 || f[0] != 'c' || f.find_first_not_of("0123456789", 1) != std::string::npos) continue;
      const uint32_t c = uint32_t(std::stoul(f.substr(1)));
      const uint64_t size = p.file_size(e);
      // ai: a record cut short by a stop goes, so the next one appended lands on a record's boundary
      if (size % REC) fs::resize_file(p.path(), size - size % REC, e);
      grow(c);
      sizes[c] = int64_t(size - size % REC);
      before.sourceBlocks.push_back(c);
    }
    return true;
  }
  bool append(uint32_t c, uint32_t sym, const uint8_t* payload) override {
    Open* h = handle(c);
    uint8_t r[REC] = {}, i[4];   // ai: zeroed: the id's bytes 8 to 11 (its TSI and spare) are no record's
    xfer_id_put(r, sym);
    memcpy(r + IDB, payload, PAYLOAD);
    put32(i, sym);
    if (!h || fwrite(r, 1, REC, h->f) != REC || fwrite(i, 1, 4, h->g) != 4) { fail("store: a block of chunk " + std::to_string(c) + " not written"); return false; }
    sizes[c] += REC;
    return true;
  }
  // ai: the ids file (i<k>), one 4-byte symbol a record; where it does not hold exactly one a record (a folder from
  // ai: before it, or a stop between a record's two writes) it is made again from the records
  std::vector<uint32_t> ids(uint32_t c) override {
    if (!has(c)) return {};
    const size_t n = size_t(sizes[c] / REC);
    if (files[c].g) fflush(files[c].g);
    std::vector<uint8_t> b(4 * n + 4);
    FILE* f = openFile(idsPath(c), "rb");
    const size_t got = f ? fread(b.data(), 1, b.size(), f) : 0;
    if (f) fclose(f);
    std::vector<uint32_t> v(n);
    if (got == 4 * n) {
      for (size_t k = 0; k < n; k++) v[k] = get32(b.data() + 4 * k);
      return v;
    }
    const auto recs = read(c);
    v.resize(recs.size() / REC);
    b.resize(4 * v.size());
    for (size_t k = 0; k < v.size(); k++) { v[k] = uint32_t(xfer_id_get(recs.data() + k * REC)); put32(b.data() + 4 * k, v[k]); }
    closeSb(c);
    writeAll(idsPath(c), b.data(), b.size());
    return v;
  }
  std::vector<uint8_t> read(uint32_t c) override {
    if (!has(c)) return {};
    if (files[c].f) fflush(files[c].f);
    std::vector<uint8_t> b(static_cast<size_t>(sizes[c]));
    FILE* f = openFile(sbPath(c), "rb");
    size_t got = f ? fread(b.data(), 1, b.size(), f) : 0;
    if (f) fclose(f);
    b.resize(got - got % REC);
    return b;
  }
  void drop(uint32_t c) override {
    closeSb(c);
    if (has(c)) { sizes[c] = -1; std::error_code e; fs::remove(pathOf(sbPath(c)), e); fs::remove(pathOf(idsPath(c)), e); }
  }
  bool writeOut(uint64_t off, const uint8_t* p, size_t len) override {
    if (!out || !seek64(out, off) || fwrite(p, 1, len, out) != len) { fail("store: " + cur + "/out not written: " + strerror(errno)); return false; }
    return true;
  }
  // ai: the output's bytes on the disk before the source block is noted: a source block noted and lost to a power cut would come back
  // ai: verified and wrong; a source block on the disk and not noted is only collected again
  void noteVerified(uint32_t c, const uint8_t* cv) override {
    if (!out || !syncFile(out)) return;
    uint8_t r[VREC] = {};
    put32(r, c);
    if (cv) memcpy(r + 4, cv, XFER_CV);
    appendTo(cur + "/verified", r, VREC);
  }
  void forgetVerified() override {
    if (cur.empty()) return;
    std::error_code e;
    fs::remove(pathOf(cur + "/verified"), e);
  }
  void noteMeta(const std::string& json) override {
    if (cur.empty()) return;
    const std::string tmp = cur + "/meta.json.tmp";
    std::error_code e;
    if (writeAll(tmp, json.data(), json.size())) fs::rename(pathOf(tmp), pathOf(cur + "/meta.json"), e);
  }
  // ai: OpfsStore.finish, and the file given its own name in a folder of its own, so the app shares it as sent; the
  // ai: transfer whole, nothing of it is kept to resume but meta.json, which names it until it is taken out (store.h)
  std::string finish(const std::string& name) override {
    if (!out) return "";
    const bool synced = syncFile(out);
    const bool closed = fclose(out) == 0;
    out = nullptr;
    const std::string from = cur + "/out";
    if (!synced || !closed) { fail("store: " + from + " not written to the end: " + strerror(errno)); return from; }
    std::string d = cur + "/done", path = d + "/" + safeName(name);
    std::error_code e;
    fs::create_directories(pathOf(d), e);
    fs::rename(pathOf(from), pathOf(path), e);
    if (e) { fail("store: " + path + ": " + e.message()); return from; }
    for (const char* f : {"header", "verified"}) fs::remove(pathOf(cur + "/" + f), e);
    return path;
  }

 private:
  // ai: A source block's files stay open (2026-11-15): blocks come interleaved over every source block (a round visits each once), so
  // ai: with more source blocks than open files nearly every append closed one source block's two streams and opened another's (32
  // ai: source blocks open before: the S26 5.0 us a block at 64 source blocks and 9.3 at 4,096, against 0.4 at 32;
  // ai: research/build/perf-scan-2026-11-15/store_bench.cpp). Now as many source blocks open as the process's descriptors
  // ai: allow (openBudget), every table flat by the source block's number; past the budget one is closed in turn (any order
  // ai: does: the source blocks come uniformly), found in O(1).
  struct Open { FILE* f = nullptr; FILE* g = nullptr; };   // ai: a source block's records and its ids
  static constexpr uint32_t NONE = UINT32_MAX;
  std::string dir, cur;   // ai: cur: the transfer in hand's folder, "" none
  std::shared_ptr<Shared> shared;
  FILE* out = nullptr;
  std::vector<int64_t> sizes;           // ai: by source block: its records' bytes, -1 no file
  std::vector<Open> files;              // ai: by source block: its streams where open
  std::vector<uint32_t> openList, at;   // ai: the source blocks open, and each source block's place in that list (NONE closed)
  size_t hand = 0;
  // ai: this folder's share of the open source blocks: the budget over the folders open, two at least
  size_t cap() const { return std::max<size_t>(2, shared->cap / std::max<size_t>(1, shared->live)); }
  // ai: each open source block's stdio buffers: a page of records (stdio's own size, so its writes land on page boundaries)
  // ai: and a kilobyte of ids up to 4,096 source blocks (64 GiB), less past it so a file's buffers stay near 20 MB
  size_t recBuf = 4096, idsBuf = 1024;
  bool has(uint32_t c) const { return c < sizes.size() && sizes[c] >= 0; }
  void grow(uint32_t c) {
    if (c < sizes.size()) return;
    const size_t n = std::max<size_t>(size_t(c) + 1, sizes.size() * 2);
    sizes.resize(n, -1); files.resize(n); at.resize(n, NONE);
  }
  void buffersFor(uint64_t length) {
    const uint64_t chunks = (length >> 24) + 1;
    recBuf = chunks <= 4096 ? 4096 : size_t(std::max<uint64_t>(1024, (uint64_t(4096) * 4096 / chunks) & ~uint64_t(511)));
    idsBuf = chunks <= 4096 ? 1024 : 256;
  }
  // ai: how many source blocks may be open at once, two streams each: the process's descriptor limit less room for the rest
  // ai: of the process (Android's apps have 32,768; a desktop JVM raises its own soft limit to the hard one at start);
  // ai: Windows' stdio 8,192 streams at most
  static size_t openBudget() {
#ifdef _WIN32
    int m = _setmaxstdio(8192);
    if (m < 0) m = _getmaxstdio();
    return std::max<size_t>(32, (size_t(m) - 64) / 2);
#else
    struct rlimit r;
    if (getrlimit(RLIMIT_NOFILE, &r)) return 32;
    const uint64_t n = r.rlim_cur == RLIM_INFINITY ? uint64_t(1) << 20 : uint64_t(r.rlim_cur);
    return n >= 2048 ? size_t(std::min<uint64_t>(n, 2 * 65535 + 1024) - 512) / 2 : std::max<size_t>(32, size_t(n) / 4);
#endif
  }
  std::string sbPath(uint32_t c) const { return cur + "/c" + std::to_string(c); }
  std::string idsPath(uint32_t c) const { return cur + "/i" + std::to_string(c); }
  void closeSb(uint32_t c) {
    if (c >= files.size() || !files[c].f) return;
    // ai: a stream's last bytes go at its close: one that cannot (the disk full) is the store's failure
    const bool ok = fclose(files[c].f) == 0;
    if ((fclose(files[c].g) != 0 || !ok) && error.empty()) fail("store: chunk " + std::to_string(c) + " not written: " + strerror(errno));
    files[c] = Open{};
    const uint32_t i = at[c], last = openList.back();
    openList[i] = last;
    at[last] = i;
    openList.pop_back();
    at[c] = NONE;
  }
  // ai: a transfer's folder holding its finished file not yet taken out (store.h): no header, meta.json, a file in done/
  static bool finished(const fs::path& d) {
    std::error_code e;
    if (fs::exists(d / "header", e) || !fs::is_regular_file(d / "meta.json", e)) return false;
    for (auto& f : fs::directory_iterator(d / "done", e))
      if (f.is_regular_file(e)) return true;
    return false;
  }
  void closeAll() {
    while (!openList.empty()) closeSb(openList.back());
    sizes.clear(); files.clear(); at.clear();
    if (out) { fclose(out); out = nullptr; }
    if (!cur.empty() && shared->live) shared->live--;
    cur.clear();
  }
  static std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> b;
    FILE* f = openFile(path, "rb");
    if (!f) return b;
    uint8_t buf[1 << 16];
    for (size_t k; (k = fread(buf, 1, sizeof buf, f)) > 0;) b.insert(b.end(), buf, buf + k);
    fclose(f);
    return b;
  }
  static bool writeAll(const std::string& path, const void* p, size_t len) {
    FILE* f = openFile(path, "wb");
    if (!f) return false;
    const bool ok = fwrite(p, 1, len, f) == len;
    return fclose(f) == 0 && ok;
  }
  static void appendTo(const std::string& path, const void* p, size_t len) {
    FILE* f = openFile(path, "ab");
    if (!f) return;
    fwrite(p, 1, len, f);
    fclose(f);
  }
  Open* handle(uint32_t c) {
    grow(c);
    Open& o = files[c];
    if (o.f) return &o;
    // ai: OpfsStore.handle truncates a source block's file when it first opens it; a source block closed for room, or one resumed,
    // ai: reopens to append (its ids file the same). The process out of descriptors all the same (others took them):
    // ai: the budget comes down to what is open and one is closed for it.
    const bool fresh = sizes[c] < 0;
    const auto both = [&] {
      FILE* f = openFile(sbPath(c), fresh ? "wb" : "ab");
      if (!f) return false;
      FILE* g = openFile(idsPath(c), fresh ? "wb" : "ab");
      if (!g) { const int e = errno; fclose(f); errno = e; return false; }
      o.f = f; o.g = g;
      return true;
    };
    while (openList.size() >= cap() && !openList.empty()) closeSb(openList[hand++ % openList.size()]);
    while (!both()) {
      if ((errno != EMFILE && errno != ENFILE) || openList.empty()) return nullptr;
      shared->cap = std::max<size_t>(2, (openList.size() - 1) * std::max<size_t>(1, shared->live));
      closeSb(openList[hand++ % openList.size()]);
    }
    setvbuf(o.f, nullptr, _IOFBF, recBuf);
    setvbuf(o.g, nullptr, _IOFBF, idsBuf);
    if (fresh) sizes[c] = 0;
    at[c] = uint32_t(openList.size());
    openList.push_back(c);
    return &o;
  }
};

class MemoryStore final : public Store {
 public:
  // ai: the cap the stores made from one share, and the bytes all of them hold
  struct Pool { uint64_t cap = 0, used = 0; };
  explicit MemoryStore(uint64_t cap) : pool(std::make_shared<Pool>()) { pool->cap = cap; }
  explicit MemoryStore(std::shared_ptr<Pool> p) : pool(std::move(p)) {}
  ~MemoryStore() override { clear(); }
  std::unique_ptr<Store> make() override { return std::make_unique<MemoryStore>(pool); }
  // ai: a new transfer's: the finished file goes too (XferRxCore::data no longer reaches it, and it counted against the
  // ai: cap; the web's MemoryStore.clear lets its output go)
  void clear() override {
    pool->used -= std::min(pool->used, held + out.size() + done.size());
    chunks.clear(); held = 0; out.clear(); out.shrink_to_fit(); done.clear(); done.shrink_to_fit();
  }
  void removeAll() override { clear(); error.clear(); }
  bool open(const std::string&, const uint8_t*, size_t, uint64_t length, Before& before) override {
    before = Before{};
    error.clear();
    clear();
    if (!within(length)) return false;
    if (length > uint64_t(SIZE_MAX) || !fits(length)) {
      fail("store: a file of " + std::to_string(length) + " B is over the memory store's " + (length > uint64_t(SIZE_MAX) ? std::string("address space") : std::to_string(pool->cap) + " B"));
      return false;
    }
    try { out.assign(size_t(length), 0); }
    catch (const std::bad_alloc&) { fail("store: no memory for a file of " + std::to_string(length) + " B"); return false; }
    pool->used += length;
    return true;
  }
  bool append(uint32_t c, uint32_t sym, const uint8_t* payload) override {
    if (!fits(REC)) { fail("store: the memory store is full (" + std::to_string(pool->cap) + " B)"); return false; }
    auto& v = chunks[c];
    const size_t at = v.size();
    v.resize(at + REC);   // ai: zeroed by resize: the id's bytes 8 to 11 too
    xfer_id_put(v.data() + at, sym);
    memcpy(v.data() + at + IDB, payload, PAYLOAD);
    held += REC;
    pool->used += REC;
    return true;
  }
  std::vector<uint8_t> read(uint32_t c) override {
    auto it = chunks.find(c);
    return it == chunks.end() ? std::vector<uint8_t>{} : it->second;
  }
  std::vector<uint32_t> ids(uint32_t c) override {
    std::vector<uint32_t> v;
    auto it = chunks.find(c);
    if (it == chunks.end()) return v;
    v.reserve(it->second.size() / REC);
    for (size_t o = 0; o + REC <= it->second.size(); o += REC) v.push_back(uint32_t(xfer_id_get(it->second.data() + o)));
    return v;
  }
  void drop(uint32_t c) override {
    auto it = chunks.find(c);
    if (it == chunks.end()) return;
    held -= it->second.size();
    pool->used -= std::min<uint64_t>(pool->used, it->second.size());
    chunks.erase(it);
  }
  bool writeOut(uint64_t off, const uint8_t* p, size_t len) override {
    if (off + len > out.size()) { fail("store: a chunk past the file's end"); return false; }
    memcpy(out.data() + off, p, len);
    return true;
  }
  std::string finish(const std::string&) override { done = std::move(out); out.clear(); return ""; }
  const std::vector<uint8_t>* data() const override { return &done; }

 private:
  std::shared_ptr<Pool> pool;
  uint64_t held = 0;
  std::unordered_map<uint32_t, std::vector<uint8_t>> chunks;
  std::vector<uint8_t> out, done;
  // ai: every store's stored blocks, outputs and finished files of the pool count (extra is what is about to be added)
  bool fits(uint64_t extra) const { return !pool->cap || pool->used + extra <= pool->cap; }
};

std::vector<uint32_t> Store::ids(uint32_t c) {
  const auto recs = read(c);
  std::vector<uint32_t> v(recs.size() / REC);
  for (size_t k = 0; k < v.size(); k++) v[k] = uint32_t(xfer_id_get(recs.data() + k * REC));
  return v;
}

std::unique_ptr<Store> fileStore(const std::string& dir) { return std::make_unique<FileStore>(dir); }
std::unique_ptr<Store> memoryStore(uint64_t maxBytes) { return std::make_unique<MemoryStore>(maxBytes); }

}  // namespace lizard
