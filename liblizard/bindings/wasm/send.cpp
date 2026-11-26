// ai: The web sender's transfer (2026-11-25), in the threaded module alone (lizard-mt.mjs): a picked file made into the C's
// ai: sending end (lizard.h liz_tx) without ever being whole in memory. The file is read where it lies, a range at a time,
// ai: by a reader of its own (lizard-web/blob-reader.mjs: a worker holding the file, reading a slice with FileReaderSync
// ai: into the shared heap as a thread asks through the prep's slot, a request at a time, and back to its event loop
// ai: between requests, so the collector takes each slice's buffer; no module thread waits on the sender's own worker,
// ai: which may itself be waiting on those threads). It is packed where that pays (core/tx/pack.h, the
// ai: default client's: the stream written to a folder of its own under /opfs/lizard-send through WasmFS's OPFS backend,
// ai: and read back from there), and the transfer's spill (core/tx/xfer_tx.h: no source block's encoder kept, its
// ai: symbols in one file) goes in the same folder: memory is a source block a thread, whatever the file's size. Where the
// ai: origin keeps no files the folder is WasmFS's memory (/tmp/lizard-send) and nothing spills (every encoder kept).
// ai: All of it runs on a thread of its own (lizw_prepare), so the sender's worker goes on taking messages: a cancel
// ai: (lizw_prep_cancel) stops the pack at its next round of frames and the hash at its next source block.
// ai: The web's in-tree interface, as src/wasm.c is the codec's: no stable interface.
#include <emscripten.h>
#include <emscripten/threading.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include "lizard.h"
#include "pack.h"
#include "xfer_tx.h"

namespace fs = std::filesystem;

namespace {

// ai: The reader's slot (blob-reader.mjs reads it at these offsets): req, the requests made (the C's count); done, the
// ai: last answered (the reader's); then a request, the offset (a double, exact to 2^53), the bytes and where they go,
// ai: and its answer, the bytes read (-1 a failure). One request at a time (mu).
struct Slot {
  std::atomic<int32_t> req{0}, done{0};
  double off = 0;
  int32_t n = 0;
  uint32_t into = 0;
  int32_t got = 0;
  std::mutex mu;
};
static_assert(offsetof(Slot, off) == 8 && offsetof(Slot, n) == 16 && offsetof(Slot, into) == 20 && offsetof(Slot, got) == 24, "blob-reader.mjs's offsets");

struct Prep {
  Slot slot;
  std::string name, dir, packed, why;
  uint64_t length = 0, sent = 0;
  int sbLog2 = -1, compression = 0;
  bool pack = true;
  std::atomic<int> state{0};   // ai: 0 under way, 1 made, -1 failed or cancelled
  std::atomic<bool> cancel{false};
  liz_tx* tx = nullptr;
  std::thread t;
};

// ai: the file read through its reader (liz_read_fn: 1 whole, 0 not): the request in the slot, the reader woken, its
// ai: answer waited for (a thread of the module's, never the page's)
int readBlob(void* user, uint64_t off, size_t n, uint8_t* into) {
  Slot& s = static_cast<Prep*>(user)->slot;
  std::lock_guard<std::mutex> l(s.mu);
  s.off = double(off); s.n = int32_t(n); s.into = uint32_t(reinterpret_cast<uintptr_t>(into)); s.got = -1;
  const int32_t r = s.req.load() + 1;
  s.req.store(r);
  emscripten_futex_wake(&s.req, 1);
  for (int32_t d; (d = s.done.load()) != r;) emscripten_futex_wait(&s.done, uint32_t(d), 1000);
  return s.got == int32_t(n) ? 1 : 0;
}
// ai: the packed stream read back from its folder
int readPacked(void* user, uint64_t off, size_t n, uint8_t* into) {
  const Prep* p = static_cast<const Prep*>(user);
  FILE* f = fopen(p->packed.c_str(), "rb");
  if (!f) return 0;
  const bool ok = fseeko(f, off_t(off), SEEK_SET) == 0 && fread(into, 1, n, f) == n;
  fclose(f);
  return ok ? 1 : 0;
}

void run(Prep* p) {
  try {
    const auto stopped = [p] { return p->cancel.load(); };
    p->sent = p->length;
    if (p->pack) {
      const lizard::Packed k = lizard::pack([p](uint64_t off, size_t n, uint8_t* into) { return readBlob(p, off, n, into) == 1; }, p->length, p->name, p->dir, stopped);
      if (k.packed) { p->packed = k.path; p->name = k.name; p->sent = k.size; p->compression = 1; }
    }
    if (p->cancel) throw std::runtime_error("cancelled");
    const bool spill = p->dir.rfind("/opfs/", 0) == 0;
    p->tx = liz_tx_new_read(p->packed.empty() ? readBlob : readPacked, p, p->sent, p->name.c_str(), p->compression ? LIZ_TX_ZSTD : 0, p->sbLog2,
                            spill ? p->dir.c_str() : nullptr);
    if (!p->tx) throw std::runtime_error(liz_last_error());
    p->state = 1;
  } catch (const std::exception& e) {
    p->why = p->cancel ? "cancelled" : e.what();
    p->state = -1;
  }
}

}  // namespace

extern "C" {

// ai: A picked file made into a transfer on a thread of its own: length its bytes, name the name its description carries,
// ai: sbLog2 the source block size's power of two (-1 the format's choice), pack 1 to pack it where that pays. store
// ai: "/opfs" (the origin's files: the packed stream and the spill there) or "" (memory). Its folder is
// ai: store/lizard-send/<a fresh id>; folders there no thread holds (an earlier page's) are removed first. Its reader is
// ai: the caller's to start, on lizw_prep_slot (the thread waits on it meanwhile).
EMSCRIPTEN_KEEPALIVE Prep* lizw_prepare(double length, const char* name, int sbLog2, int pack, const char* store) {
  auto p = std::make_unique<Prep>();
  p->length = uint64_t(length); p->name = name ? name : ""; p->sbLog2 = sbLog2; p->pack = pack != 0;
  const std::string base = std::string(store && *store ? store : "/tmp") + "/lizard-send";
  std::error_code e;
  fs::create_directories(base, e);
  // ai: an earlier page's folders: a file another page still holds open (its spill) refuses removal on OPFS, so a page
  // ai: sending meanwhile keeps its own
  for (auto& d : fs::directory_iterator(base, e)) { std::error_code r; fs::remove_all(d.path(), r); }
  p->dir = base + "/" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  fs::create_directories(p->dir, e);
  Prep* raw = p.release();
  raw->t = std::thread(run, raw);
  return raw;
}
// ai: the reader's slot (blob-reader.mjs)
EMSCRIPTEN_KEEPALIVE Slot* lizw_prep_slot(Prep* p) { return &p->slot; }
// ai: 0 under way, 1 made (lizw_prep_tx), -1 failed or cancelled (lizw_prep_why)
EMSCRIPTEN_KEEPALIVE int lizw_prep_state(Prep* p) { return p->state.load(); }
EMSCRIPTEN_KEEPALIVE const char* lizw_prep_why(Prep* p) { return p->why.c_str(); }
// ai: the transfer once made, the prep's to free with it (lizw_prep_free frees it)
EMSCRIPTEN_KEEPALIVE liz_tx* lizw_prep_tx(Prep* p) { return p->state == 1 ? p->tx : nullptr; }
// ai: the bytes the transfer carries (the packed stream's where packed), its name and compression (1: zstd, pack.h)
EMSCRIPTEN_KEEPALIVE double lizw_prep_sent(Prep* p) { return double(p->sent); }
EMSCRIPTEN_KEEPALIVE const char* lizw_prep_name(Prep* p) { return p->name.c_str(); }
EMSCRIPTEN_KEEPALIVE int lizw_prep_compression(Prep* p) { return p->compression; }
EMSCRIPTEN_KEEPALIVE void lizw_prep_cancel(Prep* p) { p->cancel = true; }
// ai: the transfer let go (its threads joined), then its folder (the packed stream, the spill); waits for a prepare
// ai: under way to stop (cancel it first)
EMSCRIPTEN_KEEPALIVE void lizw_prep_free(Prep* p) {
  if (!p) return;
  if (p->t.joinable()) p->t.join();
  if (p->tx) liz_tx_free(p->tx);
  std::error_code e;
  fs::remove_all(p->dir, e);
  delete p;
}

}  // extern "C"
