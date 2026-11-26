// ai: The web receiver's engine (2026-11-25), in the threaded module alone (lizard-mt.mjs, CMakeLists.txt
// ai: LIZ_WASM_THREADS): the whole receiving end of one engine worker (lizard-web/engine-worker.mjs) over the module's one
// ai: shared heap. The decoders are core/cpu/pool.h's CpuPool (a thread a worker, each its own codec, the policy the web's
// ai: pool had: one more while frames are lost to busy workers, one parked after ten slack seconds); each frame's verified
// ai: blocks go from its decode thread into the transfer's threaded receiving end (core/rx/xfer_rx.h XferRx: the judge
// ai: and the dedupe on that thread, the core and its solvers behind), by pointer, no copy through JS. The store is
// ai: core/rx/store.h's file store under /opfs, WasmFS's OPFS backend (the native layout, lizard-xfer/<id>/: header,
// ai: c<k>, i<k>, out, verified, meta.json, done/<name>), or memory where the origin keeps no files. Back to the page go
// ai: each frame's summary (its tracker and stats row are the page's), the transfer's progress as JSON, and each
// ai: finished file filed into the received files (lizard-files/<id>/data and meta.json, the layout lizard-web/library.mjs
// ai: reads), a packed one unpacked as it is filed (core/tx/pack.h). The web's in-tree interface, as src/wasm.c is the
// ai: codec's: no stable interface.
#include <emscripten.h>
#include <emscripten/console.h>
#include <emscripten/wasmfs.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "pool.h"
#include "pack.h"
#include "xfer_rx.h"

extern "C" {
#include "internal.h"   // ai: ob_luma, the codec's RGBA to luma
}

namespace fs = std::filesystem;
using namespace lizard;

namespace {

// ai: every picture of the ladder (core/rx/receiver.cpp CPU_NMAX)
constexpr int NMAX = 1536;
// ai: a frame's summary, RES doubles (lizw_rx_results): tag, decode ms, found, ring, the picture n, word read, version,
// ai: the four counts, fps, delay, the quad's eight coordinates, the pilots' sub-channels read, then the judge's verdict
// ai: (seen, bad, judged, fresh, test) and the blocks verified
enum { R_TAG, R_MS, R_FOUND, R_RING, R_N, R_WORD, R_VERSION, R_COUNT, R_FPS = R_COUNT + 4, R_DELAY, R_QUAD, R_PILOT = R_QUAD + 8,
       R_SEEN, R_BAD, R_JUDGED, R_FRESH, R_TEST, R_BLOCKS, RES = 32 };
static_assert(R_BLOCKS < RES, "a frame's summary fits RES doubles");

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += char(c);
  }
  return o + "\"";
}

}  // namespace

struct lizw_rx {
  std::unique_ptr<XferRx> xfer;
  std::unique_ptr<CpuPool> pool;
  std::mutex mu;                  // ai: the results not yet taken
  std::vector<double> results;
  std::atomic<bool> woken{false};   // ai: a wake posted to the engine's thread and not yet answered by a take
  std::atomic<int> inflight{0};     // ai: frames offered and not yet answered: a worker is idle while fewer than the pool's size
  std::string json, store;
  std::vector<uint8_t> luma;        // ai: lizw_rx_luma's
};

extern "C" {

// ai: /opfs, WasmFS's OPFS backend (the origin private file system, through sync access handles on a thread of its own),
// ai: mounted once: 0, or -1 where the browser gives none (a private window, an old engine, node)
EMSCRIPTEN_KEEPALIVE int lizw_opfs(void) {
  static int state = 0;   // ai: 0 not tried, 1 mounted, -1 refused
  if (state) return state > 0 ? 0 : -1;
  const int has = EM_ASM_INT({ return typeof navigator == "object" && navigator.storage && typeof navigator.storage.getDirectory == "function" && typeof FileSystemFileHandle == "function" && "createSyncAccessHandle" in FileSystemFileHandle.prototype ? 1 : 0; });
  if (!has) { state = -1; return -1; }
  backend_t b = wasmfs_create_opfs_backend();
  state = b && wasmfs_create_directory("/opfs", 0777, b) == 0 ? 1 : -1;
  // ai: proved, not assumed: a file made, written and removed
  if (state > 0) {
    FILE* f = fopen("/opfs/.probe", "wb");
    state = f && fwrite("x", 1, 1, f) == 1 && fclose(f) == 0 && remove("/opfs/.probe") == 0 ? 1 : -1;
  }
  return state > 0 ? 0 : -1;
}

// ai: A receiver: its store under storeDir ("/opfs": the origin's files; "": memory), its pool fixed at `fixed` workers
// ai: (0: the policy's, from one) and at most `ceiling` (0: the machine's hardware threads). NULL where it cannot be made.
EMSCRIPTEN_KEEPALIVE lizw_rx* lizw_rx_new(const char* storeDir, int fixed, int ceiling) {
  try {
    auto r = std::make_unique<lizw_rx>();
    r->store = storeDir ? storeDir : "";
    // ai: the transfer's and the pool's lines (a store's failure, a decoder added or parked) to the console, from any thread
    const auto log = [](const std::string& line) { emscripten_console_log(line.c_str()); };
    r->xfer = std::make_unique<XferRx>(r->store, log);
    lizw_rx* p = r.get();
    r->pool = std::make_unique<CpuPool>(NMAX, ceiling, fixed, [p](CpuFrameOut&& o) {
      const int nb = o.blockBytes ? int(o.blocks.size() / size_t(o.blockBytes)) : 0;
      std::vector<Block> blocks(static_cast<size_t>(nb));
      for (int b = 0; b < nb; b++) memcpy(blocks[size_t(b)].bytes.data(), o.blocks.data() + size_t(b) * size_t(o.blockBytes), XFER_BLOCK);
      const FrameVerdict v = p->xfer->frame(blocks);
      double s[RES] = {};
      s[R_TAG] = double(o.tag); s[R_MS] = o.ms; s[R_FOUND] = o.found; s[R_RING] = o.ring; s[R_N] = o.n; s[R_WORD] = o.hasWord;
      s[R_VERSION] = o.version;
      for (int k = 0; k < 4; k++) s[R_COUNT + k] = o.count[k];
      s[R_FPS] = o.fps; s[R_DELAY] = o.delay;
      for (int k = 0; k < 8; k++) s[R_QUAD + k] = o.quad[k];
      s[R_PILOT] = o.pilotSubch;
      s[R_SEEN] = v.seen; s[R_BAD] = v.bad; s[R_JUDGED] = v.judged; s[R_FRESH] = v.fresh; s[R_TEST] = v.test; s[R_BLOCKS] = nb;
      {
        std::lock_guard<std::mutex> l(p->mu);
        p->results.insert(p->results.end(), s, s + RES);
      }
      p->inflight--;
      // ai: the engine's thread told once until it takes what is waiting (lizw_rx_results)
      if (!p->woken.exchange(true)) MAIN_THREAD_ASYNC_EM_ASM({ if (globalThis.lizwWake) globalThis.lizwWake(); });
    }, log);
    return r.release();
  } catch (const std::exception&) {
    return nullptr;
  }
}

// ai: the decoders stopped and joined (each frame in hand answered) before the transfer they feed goes
EMSCRIPTEN_KEEPALIVE void lizw_rx_free(lizw_rx* r) {
  if (!r) return;
  r->pool->stop();
  r->pool.reset();
  delete r;
}

// ai: A frame's crop (w x h at (x, y) of a luma plane whose rows are `stride` bytes apart) to an idle decoder, copied
// ai: before the call returns: 1 taken, 0 every decoder busy (the frame lost: lizw_rx_lost). ax, ay: the camera's optical
// ai: axis in the crop (NaN: its centre). tag: the page's (a number exact to 2^53).
EMSCRIPTEN_KEEPALIVE int lizw_rx_offer(lizw_rx* r, const uint8_t* plane, int stride, int x, int y, int w, int h, double tag, float ax, float ay) {
  r->inflight++;
  if (r->pool->offer(plane, uint32_t(stride), uint32_t(x), uint32_t(y), uint32_t(w), uint32_t(h), uint64_t(tag), 0, ax, ay)) return 1;
  r->inflight--;
  return 0;
}
// ai: 1 where a decoder is idle now (a frame offered would be taken), so the caller copies no frame for nothing
EMSCRIPTEN_KEEPALIVE int lizw_rx_room(lizw_rx* r) { return r->inflight.load() < r->pool->size(); }
// ai: a luma buffer of n bytes the engine keeps (a frame's copy before an offer), its address; grown, never shrunk
EMSCRIPTEN_KEEPALIVE uint8_t* lizw_rx_luma(lizw_rx* r, int n) { if (size_t(n) > r->luma.size()) r->luma.resize(size_t(n)); return r->luma.data(); }
// ai: n RGBA pixels at rgba as luma at out, the codec's formula (src/acquire.c ob_luma: a canvas grab's frame)
EMSCRIPTEN_KEEPALIVE void lizw_luma_of_rgba(const uint8_t* rgba, int n, uint8_t* out) { ob_luma(rgba, n, out); }

// ai: The frames answered since the last call, RES doubles each into out (at most cap frames): their count
EMSCRIPTEN_KEEPALIVE int lizw_rx_results(lizw_rx* r, double* out, int cap) {
  r->woken = false;
  std::lock_guard<std::mutex> l(r->mu);
  const int n = std::min<int>(cap, int(r->results.size() / RES));
  memcpy(out, r->results.data(), size_t(n) * RES * sizeof(double));
  r->results.erase(r->results.begin(), r->results.begin() + ptrdiff_t(n) * RES);
  if (!r->results.empty()) r->woken = true;   // ai: the rest at the next call, which the caller makes at once
  return n;
}
EMSCRIPTEN_KEEPALIVE int lizw_rx_res(void) { return RES; }
EMSCRIPTEN_KEEPALIVE int lizw_rx_size(lizw_rx* r) { return r->pool->size(); }
EMSCRIPTEN_KEEPALIVE int lizw_rx_ceiling(lizw_rx* r) { return r->pool->ceiling(); }
// ai: the pool's policy (cpu/pool.h): a frame lost to busy decoders (repeatShare: the last second's decoded frames that
// ai: carried nothing new), and a second passed with busySkips frames so lost
EMSCRIPTEN_KEEPALIVE void lizw_rx_lost(lizw_rx* r, double repeatShare) { r->pool->lost(repeatShare); }
EMSCRIPTEN_KEEPALIVE void lizw_rx_tick(lizw_rx* r, int busySkips) { r->pool->tick(busySkips); }

// ai: The transfer's progress as JSON, the web's names (the fountain worker's before): the file in focus's header
// ai: { name, length, sourceBlocks, root, compression }, lightSeen (a header read), verified, rejected, refused, done,
// ai: inHand, solving, bytesIn, fraction, secs, solveMs, failed and error, per (each source block's share in hand, 0 to
// ai: 99, 255 verified) where asked, and files: every file held, in the order their transfers opened, with its path once
// ai: done (a folder store's).
EMSCRIPTEN_KEEPALIVE const char* lizw_rx_progress(lizw_rx* r, int perSb) {
  r->xfer->keepSbs(perSb != 0);
  const XferProgress p = r->xfer->progress();
  std::string j = "{\"lightSeen\":" + std::string(p.header ? "true" : "false");
  if (p.header)
    j += ",\"header\":{\"name\":" + jstr(p.name) + ",\"length\":" + std::to_string(p.length) + ",\"sourceBlocks\":" + std::to_string(p.sourceBlocks) +
         ",\"root\":\"" + p.root + "\",\"compression\":" + std::to_string(p.compression) + "}";
  else j += ",\"header\":null";
  char f[64];
  snprintf(f, sizeof f, ",\"fraction\":%.6f,\"secs\":%.3f,\"solveMs\":%.1f", p.fraction, p.secs, p.solveMs);
  j += f;
  j += ",\"verified\":" + std::to_string(p.verified) + ",\"rejected\":" + std::to_string(p.rejected) + ",\"refused\":" + std::to_string(p.refused) +
       ",\"solving\":" + std::to_string(p.solving) + ",\"inHand\":" + (p.inHand ? "true" : "false") + ",\"done\":" + (p.done ? "true" : "false") +
       ",\"bytesIn\":" + std::to_string(p.bytesIn) + ",\"descs\":" + std::to_string(p.descs) + ",\"descsRefused\":" + std::to_string(p.descsRefused) +
       ",\"failed\":" + (p.failed ? "true" : "false") + ",\"error\":" + jstr(p.error) + ",\"queued\":" + std::to_string(p.queued) + ",\"lost\":" + std::to_string(p.lost) +
       ",\"store\":\"" + (r->store.empty() ? "memory" : "opfs") + "\"";
  if (perSb) {
    j += ",\"per\":[";
    for (size_t k = 0; k < p.per.size(); k++) j += (k ? "," : "") + std::to_string(p.per[k]);
    j += "]";
  }
  j += ",\"files\":[";
  for (size_t k = 0; k < p.files.size(); k++) {
    const XferFileProgress& x = p.files[k];
    j += (k ? ",{" : "{") + std::string("\"toi\":") + std::to_string(x.toi) + ",\"name\":" + jstr(x.name) + ",\"root\":\"" + x.root + "\",\"path\":" + jstr(x.path) +
         ",\"compression\":" + std::to_string(x.compression) + ",\"length\":" + std::to_string(x.length) + ",\"done\":" + (x.done ? "true" : "false") +
         ",\"failed\":" + (x.failed ? "true" : "false") + ",\"secs\":" + std::to_string(x.secs) + "}";
  }
  j += "]}";
  r->json = std::move(j);
  return r->json.c_str();
}

// ai: a memory store's finished file (the file in focus), its bytes and their length into *len; NULL before it is done
EMSCRIPTEN_KEEPALIVE const uint8_t* lizw_rx_data(lizw_rx* r, double* len) {
  const std::vector<uint8_t>* d = r->xfer->data();
  *len = d ? double(d->size()) : 0;
  return d ? d->data() : nullptr;
}
// ai: the finished transfers' TOIs let go (the camera stopped for a file in: the same file at a later Start is received
// ai: again); one transfer forgotten by its root or folder id (Home's Delete of an unfinished one); every transfer let go
// ai: (a new receive); the transfer's clock paused while the camera is off. Each done when it returns.
EMSCRIPTEN_KEEPALIVE void lizw_rx_release(lizw_rx* r) { r->xfer->releaseDone(); }
EMSCRIPTEN_KEEPALIVE void lizw_rx_forget(lizw_rx* r, const char* root) { r->xfer->forget(root ? root : ""); }
EMSCRIPTEN_KEEPALIVE void lizw_rx_clear(lizw_rx* r) { r->xfer->clear(); }
EMSCRIPTEN_KEEPALIVE void lizw_rx_pause(lizw_rx* r, int on) { r->xfer->pause(on != 0); }

// ai: A finished file filed into the received files (lizard-web/library.mjs): lib/<id>/data and lib/<id>/meta.json
// ai: ({ name, size, root, at }), the id the root's first 16 hex digits, -2 on where one is taken (a second download of a
// ai: file a second entry); a packed file (compression 1, core/tx/pack.h) unpacked into data under its name less the
// ai: .zst, one that does not unpack moved as it came. The file's transfer folder (<id>/done/<name>'s <id>) removed.
// ai: Returns JSON { id, name, size } or { error }.
EMSCRIPTEN_KEEPALIVE const char* lizw_file_away(const char* path, const char* name, const char* root, int compression, const char* lib, double at) {
  static std::string out;
  std::error_code e;
  const fs::path src = fs::u8path(path), libDir = fs::u8path(lib);
  const std::string r = root ? root : "", r16 = r.substr(0, 16);
  std::string fileName = name && *name ? name : "received.bin";
  fs::create_directories(libDir, e);
  std::string id = r16;
  for (int n = 2; fs::exists(libDir / id, e); n++) id = r16 + "-" + std::to_string(n);
  const fs::path dir = libDir / id, data = dir / "data";
  fs::create_directory(dir, e);
  if (e) { out = "{\"error\":" + jstr("the received files' folder: " + e.message()) + "}"; return out.c_str(); }
  bool moved = false;
  if (compression == 1) {
    const std::string why = unpack(src.u8string(), data.u8string());
    if (why.empty()) {
      moved = true;
      if (fileName.size() > 4 && fileName.compare(fileName.size() - 4, 4, ".zst") == 0) fileName.resize(fileName.size() - 4);
      fs::remove(src, e);
    }
  }
  if (!moved) {
    fs::rename(src, data, e);
    if (e) { e.clear(); fs::copy_file(src, data, fs::copy_options::overwrite_existing, e); if (!e) fs::remove(src, e); }
    if (e) { out = "{\"error\":" + jstr("the file was not filed: " + e.message()) + "}"; return out.c_str(); }
  }
  const uint64_t size = fs::file_size(data, e);
  // ai: the transfer's folder, its file gone: <folder>/done/<name>
  if (src.has_parent_path() && src.parent_path().filename() == "done") fs::remove_all(src.parent_path().parent_path(), e);
  char t[32];
  snprintf(t, sizeof t, "%.0f", at);
  const std::string meta = "{\"name\":" + jstr(fileName) + ",\"size\":" + std::to_string(size) + ",\"root\":\"" + r + "\",\"at\":" + t + "}";
  std::ofstream m(dir / "meta.json", std::ios::binary | std::ios::trunc);
  m << meta;
  m.close();
  out = "{\"id\":" + jstr(id) + ",\"name\":" + jstr(fileName) + ",\"size\":" + std::to_string(size) + "}";
  return out.c_str();
}

// ai: A packed file's bytes (a memory store's, compression 1) unpacked by core/tx/pack.h through WasmFS's memory (/tmp):
// ai: the file's bytes, their length into *len, the caller's to free; NULL where they do not unpack
EMSCRIPTEN_KEEPALIVE uint8_t* lizw_unpack(const uint8_t* p, double n, double* len) {
  const std::string from = "/tmp/lizw-unpack.zst", to = "/tmp/lizw-unpack";
  std::error_code e;
  {
    std::ofstream f(from, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(p), std::streamsize(n));
    if (!f) { fs::remove(from, e); return nullptr; }
  }
  const std::string why = unpack(from, to);
  fs::remove(from, e);
  if (!why.empty()) return nullptr;
  const uint64_t size = fs::file_size(to, e);
  uint8_t* out = e ? nullptr : static_cast<uint8_t*>(malloc(size ? size : 1));
  std::ifstream f(to, std::ios::binary);
  if (out && size && !f.read(reinterpret_cast<char*>(out), std::streamsize(size))) { free(out); out = nullptr; }
  f.close();
  fs::remove(to, e);
  *len = out ? double(size) : 0;
  return out;
}

}  // extern "C"
