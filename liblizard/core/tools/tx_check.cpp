// ai: The app's sender off the phone (2026-10-01):
// ai:   tx_check ids <file> <frames> <blocks a frame> <out> [seed]   the sending end's ids and blocks, frame after frame
// ai:       (the ids little-endian, then each block's 472 bytes) under the schedule's seed (1 unless given), for
// ai:       gen/xfer_tx_ref.mjs to hold against the JS sender's
// ai:   tx_check paint <file | -> <subch> [frames] [threads] [store dir]   frames painted by the Sender (core/tx), each
// ai:       decoded blind by the C (cpu/codec.h) from its luma: the word, every block verified; the test stream's ("-")
// ai:       blocks held to SHAKE256 of their ids, a file's through the receiving end (rx/xfer_rx.h) until it is whole and
// ai:       its root the sender's. Exit 1 on a fault.
// ai:   tx_check bench <subch> [frames]   one painter's ms a frame (encode and paint) at that format
// ai:   tx_check decode <luma> <w> <h> [held]   one frame read blind (a screenshot of the phone's Send screen, grey)
// ai: The GPU's painter (2026-10-02, tx/gpu_painter.h), its kernels and manifest from LIZ_ASSETS (out, liblizard/out run from liblizard/, by default):
// ai:   tx_check gputables <dir>   tx/send_tables.cpp's tables against gen/send_tables_ref.mjs's, every format in its
// ai:       index, section by section (exact; the twiddles to a float's last bit, the border's picture, which the GPU
// ai:       paints again, within a grey level)
// ai:   tx_check gpucheck <subch> [ring] [frames]   the GPU's frames of the test stream's blocks against
// ai:       the C's paint at the same counts (the border to the byte, the square within a grey level: the web's auto
// ai:       rule), then its GPU ms a frame over encodes of its most; ring an index into src/lizard_phy.h LIZARD_RING, the
// ai:       default ring (LIZARD_RING_DEFAULT, the 128) unset
// ai:   LIZ_PAINTER=gpu|auto tx_check paint ...   the Sender painting on the GPU (or auto), every frame read blind as before
// ai:   LIZ_CODES=2 [LIZ_GAP=<modules>] tx_check paint ...   two codes a frame (2026-10-03), the gap between them set
// ai:       (2026-10-04, TxFormat.gap; GAP_MODULES unset), each read blind at its own offset
// ai:   LIZ_RUN_OUT=<file> tx_check paint ...   every symbol read written as a recording too (a .lizard, rx/lizfile.h:
// ai:       W x W luma, a code a frame, and its meta.json), which lizard_gpu_check replay reads (2026-10-07)
// ai:   LIZ_SPAN=<2 B> tx_check paint ...   the symbols in ring B (2026-10-10; 128, the 64 ring's, unset): 512 the 256 ring
// ai:   LIZ_TYPE=<media type> tx_check paint <file> ...   the header carries that type, as the apps' files do (2026-10-10)
// ai:   LIZ_SPILL=<dir> tx_check ids|paint <file> ...   the transfer's spill there (2026-11-14, xfer_tx.h: no encoder kept,
// ai:   its symbols in segments on disk), the apps' path: the ids and bytes the same
// ai:   LIZ_REFRESH=<7/8>,<3/4>,<2/3>,<1/2> tx_check paint ...   Rate refresh (2026-11-12, Sender::setRefresh) set before the
// ai:       first frame: every read block's id held against the frame before's at its place, the same exactly where its
// ai:       rate holds (the frame's count not a multiple of its refresh), a fault otherwise; the repeats a rate counted
// ai:   tx_check gpucheck ...   the GPU's frames against the C's: the anchors' squares (SPEC 6.13) to the byte besides
// ai:   the border, their pixels and the dark ones counted
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "codec.h"
#include "gpu_painter.h"
#include "json.hpp"
#include "lizfile.h"
#include "send_tables.h"
#include "sender.h"
#include "xfer_rx.h"
#include "xfer_tx.h"

extern "C" {
#include "lizard_phy.h"
#include "shake.h"
}

using namespace lizard;

namespace {
struct Mapped {
  const uint8_t* p = nullptr;
  size_t n = 0;
  explicit Mapped(const char* path) {
    int fd = open(path, O_RDONLY);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st)) { perror(path); exit(2); }
    n = static_cast<size_t>(st.st_size);
    if (n) p = static_cast<const uint8_t*>(mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0));
    close(fd);
  }
  ~Mapped() { if (p) munmap(const_cast<uint8_t*>(p), n); }
};
double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::string assets() { const char* a = getenv("LIZ_ASSETS"); return a ? a : "out"; }
// ai: LIZ_SPILL=<dir>: the transfer's spill there (xfer_tx.h: its symbols on disk, no encoder kept), the apps' path
std::string spill() { const char* a = getenv("LIZ_SPILL"); return a ? a : ""; }
std::vector<uint8_t> slurp(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { perror(path.c_str()); exit(2); }
  std::vector<uint8_t> b;
  uint8_t buf[1 << 16];
  for (size_t k; (k = fread(buf, 1, sizeof buf, f)) > 0;) b.insert(b.end(), buf, buf + k);
  fclose(f);
  return b;
}
// ai: count frames of the test stream's blocks from id 0, blocks x 476 B each (id then SHAKE256 of it)
std::vector<uint8_t> testBlocks(int count, int blocks) {
  std::vector<uint8_t> b(static_cast<size_t>(count) * blocks * XFER_BLOCK);
  for (int t = 0; t < count * blocks; t++) {
    uint8_t* o = b.data() + static_cast<size_t>(t) * XFER_BLOCK;
    for (int i = 0; i < 4; i++) o[i] = static_cast<uint8_t>(t >> 8 * i);
    stream_fill(static_cast<uint32_t>(t), o + 4, XFER_PAYLOAD);
  }
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "ids" && argc >= 6) {
    Mapped m(argv[2]);
    XferTx x(m.p, m.n, std::filesystem::path(argv[2]).filename().string(), "", -1, spill());
    x.reseed(argc >= 7 ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 1u);   // ai: the schedule's seed, gen/xfer_tx_ref.mjs's
    x.waitMade();   // ai: every chunk made before the first frame, so the frames are the serial JS sender's (no warm frames)
    const int frames = atoi(argv[3]), T = atoi(argv[4]);
    FILE* out = fopen(argv[5], "wb");
    std::vector<uint32_t> ids(T);
    uint8_t b[XFER_PAYLOAD];
    for (int f = 0; f < frames; f++) {
      x.frameIds(ids.data(), T);
      fwrite(ids.data(), 4, T, out);
      for (uint32_t id : ids) { x.block(id, b); fwrite(b, 1, XFER_PAYLOAD, out); }
    }
    fclose(out);
    printf("ids: %d frames of %d, chunks %u, lap %u, root %s, seed %u\n", frames, T, x.chunks(), x.lap(), x.rootHex().c_str(), x.seed());
    return 0;
  }
  // ai: repick <subch A> <subch B> [painter 0|1|2] [codes] (2026-10-04, the desktop sender's stall at a re-pick): the
  // ai: test stream at A in the 128 ring, a few frames taken, then configured to B as a re-pick does: the configure's ms
  // ai: and the ms until B's first frame is ready, A to B and back, three times
  if (mode == "repick" && argc >= 4) {
    Sender s(nullptr, 0, "", "");
    TxFormat f;
    f.span = 256; f.fps = 60; f.threads = 3; f.assets = assets();
    f.painter = argc > 4 ? atoi(argv[4]) : 2;
    f.codes = argc > 5 ? atoi(argv[5]) : 1;
    std::vector<uint8_t> buf;
    auto to = [&](int subch) {
      f.subch = subch;
      const double t0 = nowMs();
      const std::string e = s.configure(f);
      const double t1 = nowMs();
      if (!e.empty()) { printf("FAIL configure LIZARD-%d: %s\n", subch, e.c_str()); exit(1); }
      buf.resize(static_cast<size_t>(s.width()) * s.side() * 4);
      while (!s.ready()) std::this_thread::sleep_for(std::chrono::microseconds(100));
      const double t2 = nowMs();
      for (int k = 0; k < 8; k++) while (!s.take(buf.data(), s.width() * 4)) std::this_thread::sleep_for(std::chrono::microseconds(100));
      printf("LIZARD-%d (n %d) on the %s: configure %.1f ms, first frame %.1f ms after it\n", subch, s.n(), s.painter().c_str(), t1 - t0, t2 - t1);
    };
    const int a = atoi(argv[2]), b = atoi(argv[3]);
    to(a);
    for (int r = 0; r < 3; r++) { to(b); to(a); }
    return 0;
  }
  if (mode == "paint" && argc >= 4) {
    const bool test = std::string(argv[2]) == "-";
    std::unique_ptr<Mapped> m;
    if (!test) m = std::make_unique<Mapped>(argv[2]);
    const int subch = atoi(argv[3]), frames = argc > 4 ? atoi(argv[4]) : 64, threads = argc > 5 ? atoi(argv[5]) : 3;
    const std::string store = argc > 6 ? argv[6] : "/tmp/tx_check_store";
    // ai: LIZ_TYPE=<media type>: the header's type, as the apps send a file's (2026-10-10; "" before, so no check painted one)
    Sender s(test ? nullptr : m->p, test ? 0 : m->n, test ? "" : std::filesystem::path(argv[2]).filename().string(), getenv("LIZ_TYPE") ? getenv("LIZ_TYPE") : "", test ? "" : spill());
    TxFormat f;
    f.subch = subch; f.span = getenv("LIZ_SPAN") ? atoi(getenv("LIZ_SPAN")) : 128; f.fps = 60; f.threads = threads;
    // ai: LIZ_CODES=2: two codes a frame, each read blind on its own (the desktop sender's 2:1, 2026-10-03)
    f.codes = getenv("LIZ_CODES") ? atoi(getenv("LIZ_CODES")) : 1;
    f.gap = getenv("LIZ_GAP") ? atoi(getenv("LIZ_GAP")) : GAP_MODULES;
    const char* pt = getenv("LIZ_PAINTER");
    f.painter = !pt ? 0 : std::string(pt) == "gpu" ? 1 : std::string(pt) == "auto" ? 2 : 0;
    f.assets = assets();
    // ai: the blocks a frame carries: the format's rate profile's (src/lizard_phy.h lizard_blocks_for)
    const int wantBlocks = lizard_blocks_for(subch), wantSubch = subch;
    int every[4] = {1, 1, 1, 1};
    const bool refresh = getenv("LIZ_REFRESH") && sscanf(getenv("LIZ_REFRESH"), "%d,%d,%d,%d", &every[0], &every[1], &every[2], &every[3]) == 4;
    if (refresh) s.setRefresh(every);
    const std::string err = s.configure(f);
    if (!err.empty()) { printf("FAIL configure: %s\n", err.c_str()); return 1; }
    printf("painter: %s%s%s\n", s.painter().c_str(), s.gpuWhy().empty() ? "" : ", not the GPU: ", s.gpuWhy().c_str());
    const int W = s.side(), FW = s.width(), codes = f.codes, gap = codes > 1 ? (FW - codes * W) / (codes - 1) : 0;
    std::filesystem::create_directories(store);
    std::unique_ptr<XferRx> rx;
    if (!test) rx = std::make_unique<XferRx>(store);
    cpu_dec_t* d = cpu_dec_new(1536);
    const int top = cpu_dec_top(d), BB = cpu_dec_block_bytes(d);
    std::vector<uint8_t> rgba(static_cast<size_t>(FW) * W * 4), luma(static_cast<size_t>(W) * W), blocks(static_cast<size_t>(top) * BB), ok(top), want(XFER_PAYLOAD);
    int held = 0, words = 0, faults = 0;
    long verified = 0, total = 0;
    // ai: LIZ_REFRESH: each block's code rate (0 7/8 ... 3 1/2) in the format's order, and the ids read a symbol before
    std::vector<int> rateAt;
    std::vector<std::vector<int64_t>> before(static_cast<size_t>(codes));
    long same[4] = {0, 0, 0, 0}, seen[4] = {0, 0, 0, 0};
    if (refresh) {
      lizard_t p{};
      if (lizard_init(&p, s.n(), subch, f.span)) { printf("FAIL: the codec refused the format\n"); return 1; }
      for (int t = 0; t < p.tiers; t++) rateAt.insert(rateAt.end(), static_cast<size_t>(p.tier[t].blocks), p.tier[t].rate == 6 ? 0 : p.tier[t].rate == 4 ? 1 : p.tier[t].rate == 3 ? 2 : 3);
      lizard_free(&p);
    }
    std::unique_ptr<lizard::lizfile::Writer> runOut;
    std::string runWhy;
    if (const char* ro = getenv("LIZ_RUN_OUT")) {
      runOut = lizard::lizfile::Writer::create(ro, runWhy);
      if (!runOut) printf("LIZ_RUN_OUT: %s\n", runWhy.c_str());
    }
    const double t0 = nowMs();
    for (int k = 0; k < frames * codes; k++) {
      if (k % codes == 0) while (!s.take(rgba.data(), FW * 4)) std::this_thread::sleep_for(std::chrono::microseconds(200));
      const int x0 = (k % codes) * (W + gap);
      for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) luma[static_cast<size_t>(y) * W + x] = rgba[4 * (static_cast<size_t>(y) * FW + x0 + x)];
      if (runOut && !runOut->frame(luma.data(), W, W, W, runWhy)) { printf("LIZ_RUN_OUT: %s\n", runWhy.c_str()); runOut.reset(); }
      cpu_frame_t fr{};
      const int got = cpu_dec_frame(d, luma.data(), W, W, held, blocks.data(), ok.data(), &fr);
      if (fr.word) { words++; held = fr.version; }
      verified += got; total += fr.total;
      if (fr.version != wantSubch / LIZARD_STEP || got != wantBlocks) { faults++; if (faults < 5) printf("frame %d: word %d version %d, %d of %d blocks\n", k, fr.word, fr.version, got, fr.total); }
      std::vector<int64_t> now(static_cast<size_t>(fr.total), -1);
      for (int b = 0; b < fr.total; b++) {
        if (!ok[b]) continue;
        const uint8_t* blk = blocks.data() + static_cast<size_t>(b) * BB;
        const uint32_t id = blk[0] | blk[1] << 8 | blk[2] << 16 | static_cast<uint32_t>(blk[3]) << 24;
        now[static_cast<size_t>(b)] = id;
        if (test) { stream_fill(id, want.data(), XFER_PAYLOAD); if (memcmp(want.data(), blk + 4, XFER_PAYLOAD)) { faults++; if (faults < 5) printf("frame %d block %d: id %u not its stream\n", k, b, id); } }
        else rx->take(blk);
      }
      if (refresh) {
        auto& was = before[static_cast<size_t>(k % codes)];
        const uint64_t seq = static_cast<uint64_t>(k / codes);
        for (int b = 0; b < fr.total && b < static_cast<int>(was.size()) && b < static_cast<int>(rateAt.size()); b++) {
          if (now[static_cast<size_t>(b)] < 0 || was[static_cast<size_t>(b)] < 0) continue;
          const int r = rateAt[static_cast<size_t>(b)], e = std::clamp(every[r], 1, Sender::REFRESH_MAX);
          const bool hold = e > 1 && seq % static_cast<uint64_t>(e) != 0, rep = now[static_cast<size_t>(b)] == was[static_cast<size_t>(b)];
          seen[r]++; same[r] += rep;
          if (hold != rep) { faults++; if (faults < 5) printf("frame %llu block %d (rate %d, every %d): %s\n", static_cast<unsigned long long>(seq), b, r, e, rep ? "repeated" : "new where held"); }
        }
        was = now;
      }
    }
    const double ms = nowMs() - t0;
    if (runOut) {
      nlohmann::json meta = {{"w", W}, {"h", W}, {"frames", runOut->frames()}, {"source", "tx_check paint"},
                             {"config", {{"spec", {{"n", s.n()}, {"subch", wantSubch}, {"span", f.span}}}}}};
      if (runOut->finish(meta.dump(), "", runWhy) < 0) printf("LIZ_RUN_OUT: %s\n", runWhy.c_str());
    }
    printf("paint: LIZARD-%d x %d n %d, %d x %d px (gap %d px), %d painters: %d frames, %ld of %ld blocks verified, words %d, %.1f frames a second taken\n",
           wantSubch, codes, s.n(), FW, W, gap, threads, frames, verified, total, words, 1000.0 * frames / ms);
    printf("stats: %s\n", s.stats().c_str());
    if (refresh) printf("refresh %d,%d,%d,%d: repeated 7/8 %ld of %ld, 3/4 %ld of %ld, 2/3 %ld of %ld, 1/2 %ld of %ld\n", every[0], every[1], every[2], every[3],
                        same[0], seen[0], same[1], seen[1], same[2], seen[2], same[3], seen[3]);
    if (!test) {
      rx->drain();
      const XferProgress p = rx->progress();
      printf("file: %s, done %d, root %s (the sender's %s)\n", p.name.c_str(), p.done ? 1 : 0, p.root.c_str(), s.xfer()->rootHex().c_str());
      if (!p.done) { printf("FAIL: the file did not finish in %d frames\n", frames); faults++; }
      else if (p.root != s.xfer()->rootHex()) { printf("FAIL: the root differs\n"); faults++; }
    }
    cpu_dec_free(d);
    printf(faults ? "FAILED %d\n" : "ok\n", faults);
    return faults ? 1 : 0;
  }
  // ai: decode <luma file> <width> <height> [held]: one frame of 8-bit luma (a screenshot of the phone's Send screen, made
  // ai: grey by the caller), read blind by the C: the word, the blocks verified, and those that are the test stream's
  if (mode == "gputables" && argc >= 3) {
    const std::string dir = argv[2], man = assets() + "/setup/send.json";
    const auto mb = slurp(man);
    const std::string text(mb.begin(), mb.end());
    SendConsts k;
    const std::string le = k.load(text, slurp(assets() + "/" + nlohmann::json::parse(text)["tab"]["blob"].get<std::string>()));
    if (!le.empty()) { printf("FAIL %s\n", le.c_str()); return 1; }
    const auto ib = slurp(dir + "/index.json");
    const auto index = nlohmann::json::parse(std::string(ib.begin(), ib.end()));
    int bad = 0, done = 0;
    for (const auto& e : index) {
      const int subch = e["subch"], ring = e["ring"], n = e["n"];
      const std::string name = e["name"];
      lizard_t f{};
      if (ring < 0 || ring >= LIZARD_RINGS || lizard_init(&f, n, subch, 2 * LIZARD_RING[ring])) { printf("FAIL %s: the codec refused it\n", name.c_str()); bad++; continue; }
      lizard_fmt_fps(&f, 60);
      SendTables t;
      const std::string te = sendTables(f, k, 4, t, e.value("codes", 1));
      if (!te.empty()) { printf("FAIL %s: %s\n", name.c_str(), te.c_str()); bad++; lizard_free(&f); continue; }
      std::vector<std::string> off;
      auto same = [&](const char* sec, const void* p, size_t bytes, bool ulp = false) {
        const auto ref = slurp(dir + "/" + name + "." + sec + ".bin");
        if (ref.size() != bytes) { off.push_back(std::string(sec) + " " + std::to_string(bytes) + " B against " + std::to_string(ref.size())); return; }
        size_t diff = 0, far = 0;
        for (size_t i = 0; i + 3 < bytes || (i < bytes && !ulp); i += ulp ? 4 : 1) {
          if (ulp) {
            int32_t a, b; std::memcpy(&a, static_cast<const uint8_t*>(p) + i, 4); std::memcpy(&b, ref.data() + i, 4);
            if (a != b) { diff++; if (std::abs(static_cast<long>(a) - b) > 1) far++; }
          } else if (static_cast<const uint8_t*>(p)[i] != ref[i]) diff++;
        }
        if (far || (!ulp && diff)) off.push_back(std::string(sec) + " " + std::to_string(ulp ? far : diff) + (ulp ? " floats past a last bit" : " bytes"));
        else if (diff) off.push_back(std::string("(") + sec + " " + std::to_string(diff) + " floats a last bit apart)");
      };
      same("rows", t.rows.data(), 4 * t.rows.size()); same("uv", t.uv.data(), 4 * t.uv.size()); same("tw", t.tw.data(), 4 * t.tw.size(), true);
      same("su", t.su.data(), 48); same("g", t.g.data(), 48); same("taps", t.taps.data(), 4 * t.taps.size());
      same("pu", t.pu.data(), 4 * t.pu.size());
      // ai: the border as the painter reads it: every pixel outside the square and in the anchors' squares to the byte;
      // ai: the picture between them (painted again on the GPU, never read from here) within a grey level, as gpucheck
      // ai: holds the GPU's square: x86-64's vector kernels round it apart from the scalar C and the wasm (2026-11-05)
      {
        const auto ref = slurp(dir + "/" + name + ".border.bin");
        if (ref.size() != t.border.size()) off.push_back("border " + std::to_string(t.border.size()) + " B against " + std::to_string(ref.size()));
        else {
          auto inAnchor = [&](int v) { for (int i = 0; i < 7; i++) if (v >= static_cast<int>(t.anch[i]) && v < static_cast<int>(t.anch[i] + t.anch[7])) return true; return false; };
          size_t exact = 0, far = 0, near = 0;
          for (int y = 0; y < t.W; y++) for (int x = 0; x < t.W; x++) {
            const size_t i = static_cast<size_t>(y) * t.W + x;
            const int d = std::abs(static_cast<int>(t.border[i]) - ref[i]);
            if (!d) continue;
            const bool square = x >= t.sq && x < t.sq + t.q && y >= t.sq && y < t.sq + t.q;
            if (!square || (inAnchor(x) && inAnchor(y))) exact++; else if (d > 1) far++; else near++;
          }
          if (exact) off.push_back("border " + std::to_string(exact) + " bytes outside the picture");
          if (far) off.push_back("border " + std::to_string(far) + " picture bytes past a grey level");
          if (near) off.push_back("(border " + std::to_string(near) + " picture bytes a grey level apart)");
        }
      }
      const bool fail = std::any_of(off.begin(), off.end(), [](const std::string& x) { return x[0] != '('; });
      if (fail || (done < 3 && !off.empty())) { printf("%s %s:", fail ? "FAIL" : "note", name.c_str()); for (auto& x : off) printf(" %s;", x.c_str()); printf("\n"); }
      bad += fail;
      done++;
      lizard_free(&f);
    }
    printf("%d formats: %d differ\n%s\n", done, bad, bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
  }
  if (mode == "gpucheck" && argc >= 3) {
    const int subch = atoi(argv[2]), ring = argc > 3 ? atoi(argv[3]) : LIZARD_RING_DEFAULT, frames = argc > 4 ? atoi(argv[4]) : 8, codes = argc > 5 ? atoi(argv[5]) : 1;
    if (ring < 0 || ring >= LIZARD_RINGS) { printf("FAIL ring %d: an index of LIZARD_RING, 0 to %d\n", ring, LIZARD_RINGS - 1); return 1; }
    std::unique_ptr<GpuPainter> g;
    try { g = GpuPainter::create(assets(), [](const std::string& m) { printf("%s\n", m.c_str()); }); }
    catch (const std::exception& e) { printf("FAIL no GPU painter: %s\n", e.what()); return 1; }
    const std::string ce = g->configure(subch, 2 * LIZARD_RING[ring], 60, 4, codes, GAP_MODULES, 0, 0, 0, false);
    if (!ce.empty()) { printf("FAIL configure: %s\n", ce.c_str()); return 1; }
    const int V = subch / 8 * codes, R = g->frames();
    const auto blocks = testBlocks(std::max(frames, R), V);
    const auto c = g->check(blocks.data(), frames);
    char an[160] = "";
    snprintf(an, sizeof an, ", the anchors %ld pixels (%ld dark) %ld apart", c.anchorPx, c.anchorDark, c.anchorDiff);
    printf("check, %d frames of LIZARD-%d x %d in the %d ring on %s: border %ld pixels apart, the square %ld pixels apart, at most %d levels%s: %s\n",
           frames, subch, codes, LIZARD_RING[ring], g->device().c_str(), c.borderDiff, c.squareDiff, c.squareMax, an, c.ok ? "ok" : "FAILED");
    std::vector<std::vector<uint8_t>> out(R);
    double gpu = 0, wall = 0;
    const int encodes = 16;
    for (int i = 0; i < encodes; i++) { const double t0 = nowMs(); gpu += g->encode(blocks.data(), R, i & 3, out.data()); wall += nowMs() - t0; }
    printf("time: %.2f ms of GPU a frame, %.2f ms an encode of %d frames end to end (readback included), %d px a side\n", gpu / (encodes * R), wall / encodes, R, g->side());
    return c.ok ? 0 : 1;
  }
  if (mode == "decode" && argc >= 5) {
    const int iw = atoi(argv[3]), ih = atoi(argv[4]), held0 = argc > 5 ? atoi(argv[5]) : 0;
    std::vector<uint8_t> img(static_cast<size_t>(iw) * ih);
    FILE* in = fopen(argv[2], "rb");
    if (!in || fread(img.data(), 1, img.size(), in) != img.size()) { printf("FAIL: %s is not %d x %d bytes\n", argv[2], iw, ih); return 2; }
    fclose(in);
    cpu_dec_t* d = cpu_dec_new(1536);
    const int top = cpu_dec_top(d), BB = cpu_dec_block_bytes(d);
    std::vector<uint8_t> blocks(static_cast<size_t>(top) * BB), ok(top), want(XFER_PAYLOAD);
    cpu_frame_t fr{};
    const int got = cpu_dec_frame(d, img.data(), iw, ih, held0, blocks.data(), ok.data(), &fr);
    int stream = 0;
    for (int b = 0; b < fr.total; b++) {
      if (!ok[b]) continue;
      const uint8_t* blk = blocks.data() + static_cast<size_t>(b) * BB;
      const uint32_t id = blk[0] | blk[1] << 8 | blk[2] << 16 | static_cast<uint32_t>(blk[3]) << 24;
      stream_fill(id, want.data(), XFER_PAYLOAD);
      if (!memcmp(want.data(), blk + 4, XFER_PAYLOAD)) stream++;
    }
    printf("decode: found %d, ring %d, n %d, word %d (version %d = LIZARD-%d, %d a second), %d blocks verified, %d of them the test stream's, pilots %d blocks r %.2f %.2f\n",
           fr.found, fr.ring, fr.n, fr.word, fr.version, LIZARD_STEP * fr.version, fr.fps, got, stream, fr.pilot_blocks, fr.pilot_r[0], fr.pilot_r[1]);
    cpu_dec_free(d);
    return got > 0 ? 0 : 1;
  }
  if (mode == "bench" && argc >= 3) {
    const int subch = atoi(argv[2]), frames = argc > 3 ? atoi(argv[3]) : 32;
    Sender s(nullptr, 0, "", "");
    TxFormat f;
    f.subch = subch; f.span = 128; f.fps = 60; f.threads = 1; f.assets = assets();
    if (const std::string e = s.configure(f); !e.empty()) { fprintf(stderr, "bench: %s\n", e.c_str()); return 1; }
    std::vector<uint8_t> rgba(static_cast<size_t>(s.side()) * s.side() * 4);
    const double t0 = nowMs();
    for (int k = 0; k < frames; k++) while (!s.take(rgba.data(), s.side() * 4)) std::this_thread::sleep_for(std::chrono::microseconds(100));
    const double ms = nowMs() - t0;
    printf("bench: LIZARD-%d n %d, %d px a side: %.2f ms a frame on one painter (%s)\n", subch, s.n(), s.side(), ms / frames, s.stats().c_str());
    return 0;
  }
  // ai: pace <subch> [secs] [fps] [codes] (2026-11-08): the GPU painter (painter 1, frames left on the device) taken as a
  // ai: presenter takes them, a frame each 1/fps by takeSlot on a steady schedule; each second the sender's stats line
  // ai: (paintMs: GPU ms a frame; paintBatch: frames an encode), then "pace:" with the frames late (not painted when
  // ai: due). The sender's batching A/B: LIZ_TX_BATCH=0 a frame a submission.
  if (mode == "pace" && argc >= 3) {
    const int subch = atoi(argv[2]), secs = argc > 3 ? atoi(argv[3]) : 10, fps = argc > 4 ? atoi(argv[4]) : 60, codes = argc > 5 ? atoi(argv[5]) : 1;
    Sender s(nullptr, 0, "", "");
    TxFormat f;
    f.subch = subch; f.span = 128; f.fps = fps; f.codes = codes; f.painter = 1; f.hostFrames = false; f.assets = assets();
    if (const std::string e = s.configure(f); !e.empty()) { fprintf(stderr, "pace: %s\n", e.c_str()); return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    const auto per = std::chrono::nanoseconds(1000000000LL / fps);
    long late = 0, taken = 0;
    for (long k = 0; k < (long)secs * fps; k++) {
      std::this_thread::sleep_until(t0 + per * (k + 1));
      uint64_t seq = 0;
      int slot = 0;
      if (s.takeSlot(&seq, &slot)) taken++;
      else late++;
      if ((k + 1) % fps == 0) printf("%s\n", s.stats().c_str());
    }
    printf("pace: LIZARD-%d x %d at %d a second, %d s: %ld taken, %ld late\n", subch, codes, fps, secs, taken, late);
    return late ? 1 : 0;
  }
  fprintf(stderr, "usage: tx_check ids <file> <frames> <T> <out> | paint <file|-> <subch> [frames] [threads] [store] | bench <subch> [frames]\n"
                  "       | pace <subch> [secs] [fps] [codes]\n");
  return 2;
}
