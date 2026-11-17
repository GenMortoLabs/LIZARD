// ai: The app's sender off the phone (2026-10-01):
// ai:   tx_check ids <file> <frames> <blocks a frame> <out> [seed]   the sending end's ids and blocks, frame after frame
// ai:       (the ids, 5 bytes little-endian, then each block's 471 bytes) under the schedule's seed (1 unless given), for
// ai:       gen/xfer_tx_ref.mjs to hold against the JS sender's
// ai:   tx_check paint <file | -> <subch> [frames] [threads] [store dir]   frames painted by the Sender (core/tx), each
// ai:       decoded blind by the C (cpu/codec.h) from its luma: the word, every block verified; the test stream's ("-")
// ai:       blocks held to BLAKE3 of their ids, a file's through the receiving end (rx/xfer_rx.h) until it is whole and
// ai:       its root the sender's. Exit 1 on a fault.
// ai:   tx_check bench <subch> [frames]   one painter's ms a frame (encode and paint) at that format
// ai:   tx_check colours <subch> <ink> <paper>   the code's colours (2026-11-16, Sender::setColours; RRGGBB each): a frame
// ai:       of the test stream taken (take, the CPU's RGBA) with them set: its corner the paper, its pixel of least luma
// ai:       the ink (of most where the ink is the lighter: an inverted code), every pixel on the line between them (each
// ai:       channel within a level of the grey's place on it). Exit 1 on a fault
// ai:   tx_check decode <luma> <w> <h> [held]   one frame read blind (a screenshot of the phone's Send screen, grey)
// ai: The GPU's painter (2026-10-02, tx/gpu_painter.h), its kernels and manifest from LIZ_ASSETS (out, liblizard/out run from liblizard/, by default):
// ai:   tx_check gputables <dir>   tx/send_tables.cpp's tables against gen/send_tables_ref.mjs's, every format in its
// ai:       index, section by section (exact; the twiddles to a float's last bit, the border's picture, which the GPU
// ai:       paints again, within a grey level)
// ai:   tx_check gpucheck <subch> [ring] [frames]   the GPU's frames of the test stream's blocks against
// ai:       the C's paint at the same counts (the border to the byte, the square within a grey level: the web's auto
// ai:       rule), then its GPU ms a frame over encodes of its most, the test stream made on the device and the blocks
// ai:       given whole alternately (their difference the device's BLAKE3); ring an index into src/lizard_phy.h
// ai:       LIZARD_RING, the default ring (LIZARD_RING_DEFAULT, the 128) unset
// ai:   LIZ_PAINTER=gpu|auto tx_check paint ...   the Sender painting on the GPU (or auto), every frame read blind as before
// ai:   LIZ_CODES=2 [LIZ_GAP=<modules>] tx_check paint ...   two codes a frame (2026-10-03), the gap between them set
// ai:       (2026-10-04, TxFormat.gap; GAP_MODULES unset), each read blind at its own offset
// ai:   LIZ_RUN_OUT=<file> tx_check paint ...   every symbol read written as a recording too (a .lizard, rx/lizfile.h:
// ai:       W x W luma, a code a frame, and its meta.json), which lizard_gpu_check replay reads (2026-10-07)
// ai:   LIZ_SPAN=<2 B> tx_check paint ...   the symbols in ring B (2026-10-10; 128, the 64 ring's, unset): 512 the 256 ring
// ai:   LIZ_TYPE=<media type> tx_check paint <file> ...   the header carries that type, as the apps' files do (2026-10-10)
// ai:   LIZ_SPILL=<dir> tx_check ids|paint <file> ...   the transfer's spill there (2026-11-14, xfer_tx.h: no encoder kept,
// ai:   its symbols in segments on disk), the apps' path: the ids and bytes the same
// ai:   LIZ_REFRESH=<7/8>,<3/4>,<2/3>,<1/2> tx_check paint ...   Refresh (2026-11-17, Sender::setRefresh; Rate refresh's
// ai:       1-to-4 holds of 2026-11-12 before) set before the first frame, each rate's delay, a whole number of pictures:
// ai:       every read block's id held against the frame before's at its place, the same exactly where its delay holds
// ai:       (Sender::due false: the count no multiple of it), a fault otherwise; the repeats a rate counted.
// ai:       LIZ_REFRESH_BLOCKS=<delay>,... in its place (Sender::setRefreshBlocks after the configure): place b takes entry
// ai:       b mod their count
// ai:   LIZ_CAROUSEL=<frames> tx_check paint ...   the animated header (2026-11-15, Sender::setCarousel): every word a
// ai:       carousel symbol held that many frames (with two codes each its own: code c's slot codes x (frame / frames) + c,
// ai:       its position the sender's order, Sender::carouselPosition, under the sender's key), read blind and held as the receivers hold it (codec.h cpu_hold_*): a
// ai:       frame before the header stands decodes nothing, after it every block; the symbols read, the frame the header
// ai:       stood at and a wrong symbol (its position not the frame's) counted. <frames>@<on>[,<off>]: changed live, as
// ai:       the apps' slider does: static until frame <on> is taken, then setCarousel(<frames>), and setCarousel(0) at
// ai:       <off>; every block read throughout (the static word held first), the words static, then animated from a frame
// ai:       at or after <on> (the frames made ahead keep theirs), then static again from one at or after <off>, each
// ai:       switch once; the frames between the ask and the first word to show it printed
// ai:   tx_check carousel [keys]   the carousel's order against a periodic capture (2026-11-15): a receiver reading every
// ai:       n-th slot from a random one, the reads until it holds k (3, 10, 30, 85) distinct positions, the fixed order
// ai:       (slot t at t mod 255: at most 255 / gcd(n, 255) ever) beside the sender's (Sender::carouselPosition)
// ai:   tx_check gpucheck ...   the GPU's frames against the C's: the anchors' squares (SPEC 6.13) to the byte besides
// ai:   the border, their pixels and the dark ones counted
// ai:   LIZ_REPAINT=<frame> tx_check paint ...   at that frame the Sender configured again on the other painter (the GPU's
// ai:       becomes the CPU's, the CPU's the GPU's; 2026-11-16): a file the GPU made read back for the CPU's painters
// ai:       (tx/gpu_encoder.h host), the rest of the run read blind as before
// ai:   tx_check flights <subch> [encodes] [frames an encode]   the GPU painter's encodes in flight (2026-11-16,
// ai:       GpuPainter::submit): the test stream's frames painted GpuPainter::FLIGHTS at a time, nothing waited for
// ai:       between, then painted again one encode at a time and waited for: every ring frame byte-equal (6 encodes of 8
// ai:       unless given; LIZ_SPAN, LIZ_CODES as paint's). Exit 1 on a fault
// ai:   tx_check rqcheck <file | <MiB>m> [frames] [blocks a frame]   the sender's blocks on the GPU (2026-11-16,
// ai:       tx/gpu_stream.h) held to the CPU's: the stream's self-test, a file (or that many MiB of seeded random bytes)
// ai:       solved by the CPU's workers and its chunks handed up (timed), then frames of the schedule's ids (60 of 248
// ai:       unless given) and chosen ones (each chunk's first 300 source symbols, its last, K, K + 1, 2^24 - 1) made on
// ai:       the device and read back, every payload byte-equal to XferTx::block's (a reference transfer's encoders); frames
// ai:       0 the making alone, no reference. LIZ_VK_DEVICE the device. Exit 1 on a fault
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <functional>
#include <random>
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
#include "gpu_stream.h"
#include "gpu_painter.h"
#include "json.hpp"
#include "lizfile.h"
#include "send_tables.h"
#include "sender.h"
#include "xfer_rx.h"
#include "xfer_tx.h"
#include "wg.h"

extern "C" {
#include "lizard_phy.h"
#include "xfer.h"
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
// ai: count frames of the test stream's blocks from id 0, blocks x 476 B each (id then BLAKE3 of it)
std::vector<uint8_t> testBlocks(int count, int blocks) {
  std::vector<uint8_t> b(static_cast<size_t>(count) * blocks * XFER_BLOCK);
  for (int t = 0; t < count * blocks; t++) {
    uint8_t* o = b.data() + static_cast<size_t>(t) * XFER_BLOCK;
    xfer_id_put(o, static_cast<xfer_id_t>(t));
    stream_fill(static_cast<xfer_id_t>(t), o + XFER_ID_BYTES, XFER_PAYLOAD);
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
    std::vector<xfer_id_t> ids(T);
    uint8_t b[XFER_PAYLOAD], idb[XFER_ID_BYTES];
    for (int f = 0; f < frames; f++) {
      x.frameIds(ids.data(), T);
      for (xfer_id_t id : ids) { xfer_id_put(idb, id); fwrite(idb, 1, XFER_ID_BYTES, out); }
      for (xfer_id_t id : ids) { x.block(id, b); fwrite(b, 1, XFER_PAYLOAD, out); }
    }
    fclose(out);
    printf("ids: %d frames of %d, chunks %u, lap %u, root %s, seed %u\n", frames, T, x.chunks(), x.lap(), x.rootHex().c_str(), x.seed());
    return 0;
  }
  if (mode == "flights" && argc >= 3) {
    const int subch = atoi(argv[2]), E = argc > 3 ? atoi(argv[3]) : 6, R = argc > 4 ? atoi(argv[4]) : 8;
    const int span = getenv("LIZ_SPAN") ? atoi(getenv("LIZ_SPAN")) : 128, codes = getenv("LIZ_CODES") ? atoi(getenv("LIZ_CODES")) : 1;
    auto gp = GpuPainter::create(assets());
    if (const std::string e = gp->configure(subch, span, 60, R, codes, GAP_MODULES, E * R, 1ull << 30, 0); !e.empty()) { printf("FAIL configure: %s\n", e.c_str()); return 1; }
    if (gp->frames() != R || gp->depth() < E * R) { printf("FAIL: encodes of %d, a ring of %d, for %d x %d\n", gp->frames(), gp->depth(), E, R); return 1; }
    const int per = codes * gp->blocks(), N = E * R;
    const std::vector<uint8_t> blocks = testBlocks(N, per);
    const size_t fb = static_cast<size_t>(per) * XFER_BLOCK;
    // ai: in flight: FLIGHTS at a time, the oldest finished only when that many are out
    std::deque<GpuPainter::Pending> fl;
    double t0 = nowMs();
    for (int e = 0; e < E; e++) {
      if (static_cast<int>(fl.size()) == GpuPainter::FLIGHTS) { gp->finish(fl.front()); fl.pop_front(); }
      fl.push_back(gp->submit(blocks.data() + e * R * fb, R, (e * R) & 3, nullptr, static_cast<uint64_t>(e) * R, true));
    }
    for (auto& p : fl) gp->finish(p);
    const double inFlight = nowMs() - t0;
    std::vector<std::vector<uint8_t>> a(N);
    for (int f = 0; f < N; f++) gp->readSlot(f, a[f]);
    // ai: one at a time, waited for
    t0 = nowMs();
    for (int e = 0; e < E; e++) gp->encode(blocks.data() + e * R * fb, R, (e * R) & 3, nullptr, static_cast<uint64_t>(e) * R, true);
    const double oneByOne = nowMs() - t0;
    long apart = 0;
    std::vector<uint8_t> b;
    for (int f = 0; f < N; f++) { gp->readSlot(f, b); apart += b != a[f]; }
    printf("flights: LIZARD-%d x %d, %d encodes of %d frames: in flight %.1f ms, one at a time %.1f ms (wall); %ld of %d frames apart\n", subch, codes, E, R, inFlight, oneByOne, apart, N);
    printf(apart ? "FAIL\n" : "ok\n");
    return apart ? 1 : 0;
  }
  if (mode == "rqcheck" && argc >= 3) {
    const std::string src = argv[2];
    std::vector<uint8_t> rnd;
    std::unique_ptr<Mapped> m;
    const uint8_t* data = nullptr;
    size_t len = 0;
    if (!src.empty() && src.back() == 'm' && !std::filesystem::exists(src)) {
      len = static_cast<size_t>(std::atof(src.c_str()) * (1 << 20));
      rnd.resize(len);
      std::mt19937_64 g(5);
      for (size_t i = 0; i < len; i += 8) { const uint64_t v = g(); std::memcpy(&rnd[i], &v, std::min<size_t>(8, len - i)); }
      data = rnd.data();
    } else {
      m = std::make_unique<Mapped>(src.c_str());
      data = m->p;
      len = m->n;
    }
    const int frames = argc >= 4 ? atoi(argv[3]) : 60, BPF = argc >= 5 ? atoi(argv[4]) : 248;
    XferTx x(data, len, "f.bin", "application/octet-stream", -1, "", {}, true);
    const uint32_t C = x.chunks();
    printf("rqcheck: %zu B, %u chunks of %zu B, %u controls\n", len, C, x.chunkSize(), x.controls());
    const char* want = getenv("LIZ_VK_DEVICE");
    auto dev = wg::Device::create(want ? want : "", false, [](const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); });
    printf("device: %s\n", dev->name.c_str());
    std::unique_ptr<GpuStream> st;
    try { st = GpuStream::create(*dev, assets(), {len, x.chunkSize(), C, x.controls()}); }
    catch (const std::exception& e) { printf("FAIL: no GPU stream: %s\n", e.what()); return 1; }
    if (const std::string why = st->selfTest(); !why.empty()) { printf("FAIL: the self-test: %s\n", why.c_str()); return 1; }
    printf("self-test ok; %.0f MB of symbols on the device\n", st->symbolBytes() / 1e6);
    // ai: the CPU's workers solving, each chunk copied up by its worker on the device's second queue
    double t0 = nowMs();
    GpuStream* sp = st.get();
    x.makeOnCpu([sp](uint32_t c, const rq_enc_t* e) { return sp->put(c, e); });
    x.waitMade();
    if (!x.fault().empty() || !st->done()) { printf("FAIL: %s\n", x.fault().empty() ? "chunks missing on the device" : x.fault().c_str()); return 1; }
    printf("every chunk solved and on the device in %.0f ms\n", nowMs() - t0);
    if (!frames) { printf("timed alone (frames 0: no reference, nothing compared)\n"); return 0; }
    t0 = nowMs();
    XferTx ref(data, len, "f.bin", "application/octet-stream", -1, "");
    ref.waitMade();
    printf("the CPU's encoders made in %.0f ms (the reference)\n", nowMs() - t0);
    // ai: the schedule's frames, then the chosen ids
    std::vector<xfer_id_t> chosen;
    for (uint32_t c = 0; c < C; c++) {
      const uint32_t K = rq_symbols(x.chunkBytes(c), XFER_PAYLOAD);
      for (uint32_t k = 0; k < std::min<uint32_t>(300, K); k++) chosen.push_back(xfer_id(c, k));
      for (uint32_t s : {K - 1, K, K + 1, (1u << 24) - 1}) chosen.push_back(xfer_id(c, s));
    }
    std::vector<xfer_id_t> ids(BPF), data_;
    std::vector<uint8_t> got, refb(XFER_PAYLOAD);
    long blocks = 0, wrong = 0, ctl = 0;
    double hostMs = 0;
    const int extra = static_cast<int>((chosen.size() + BPF - 1) / BPF);
    for (int f = 0; f < frames + extra; f++) {
      if (f < frames) x.frameIds(ids.data(), BPF);
      else for (int i = 0; i < BPF; i++) ids[i] = chosen[(static_cast<size_t>(f - frames) * BPF + i) % chosen.size()];
      data_.clear();
      for (xfer_id_t id : ids) { if (xfer_id_chunk(id) == XFER_CONTROL) ctl++; else data_.push_back(id); }
      got.resize(data_.size() * XFER_PAYLOAD);
      const double th = nowMs();
      st->host(data_.data(), static_cast<uint32_t>(data_.size()), got.data());
      hostMs += nowMs() - th;
      for (size_t i = 0; i < data_.size(); i++) {
        ref.block(data_[i], refb.data());
        if (std::memcmp(refb.data(), got.data() + i * XFER_PAYLOAD, XFER_PAYLOAD)) {
          if (wrong < 5) printf("block: chunk %u symbol %u differs\n", xfer_id_chunk(data_[i]), xfer_id_symbol(data_[i]));
          wrong++;
        }
      }
      blocks += static_cast<long>(data_.size());
    }
    printf("blocks: %ld data blocks in %d frames of %d (%ld control left out): %ld differ from XferTx::block; %.3f ms a frame made and read back\n",
           blocks, frames + extra, BPF, ctl, wrong, hostMs / (frames + extra));
    printf(wrong ? "FAIL\n" : "ok\n");
    return wrong ? 1 : 0;
  }
  // ai: carousel [keys]: a receiver reading every n-th slot from a random one, the reads until it holds k distinct
  // ai: positions, the fixed order (slot t at t mod 255) beside the sender's (the mean and worst over keys keys, 100
  // ai: unless given); "-" none within 100 passes' reads
  if (mode == "carousel") {
    const int keys = argc > 2 ? atoi(argv[2]) : 100;
    const int strides[] = {1, 2, 3, 4, 5, 7, 15, 17, 51, 85}, ks[] = {3, 10, 30, 85};
    const uint64_t cap = 100 * 255;
    std::mt19937 rng(12345);
    // ai: reads until k distinct, for each k (0 never) of the positions at slots start + i n
    const auto readsTo = [&](const std::function<int(uint64_t)>& at, uint64_t start, int n, uint64_t out[4]) {
      std::vector<uint8_t> seen(255);
      int have = 0, next = 0;
      for (int q = 0; q < 4; q++) out[q] = 0;
      for (uint64_t i = 0; i * n < cap && next < 4; i++) {
        const int p = at(start + i * static_cast<uint64_t>(n));
        if (!seen[static_cast<size_t>(p)]) { seen[static_cast<size_t>(p)] = 1; have++; }
        while (next < 4 && have >= ks[next]) out[next++] = i + 1;
      }
    };
    printf("every n-th slot: reads until k distinct positions, the fixed order / the shuffled (mean, worst of %d keys)\n", keys);
    for (const int n : strides) {
      printf("n %2d:", n);
      uint64_t fixed[4], one[4];
      readsTo([](uint64_t t) { return static_cast<int>(t % 255); }, rng() % 255, n, fixed);
      double sum[4] = {0, 0, 0, 0};
      uint64_t worst[4] = {0, 0, 0, 0};
      int never[4] = {0, 0, 0, 0};
      for (int r = 0; r < keys; r++) {
        const uint32_t key = static_cast<uint32_t>(rng());
        readsTo([key](uint64_t t) { return Sender::carouselPosition(key, t); }, rng() % 255, n, one);
        for (int q = 0; q < 4; q++) { if (!one[q]) never[q]++; sum[q] += static_cast<double>(one[q]); worst[q] = std::max(worst[q], one[q]); }
      }
      for (int q = 0; q < 4; q++) {
        if (fixed[q]) printf("  k %2d %3llu /", ks[q], static_cast<unsigned long long>(fixed[q]));
        else printf("  k %2d   - /", ks[q]);
        if (never[q]) printf(" %d never", never[q]);
        else printf(" %5.1f, %3llu", sum[q] / keys, static_cast<unsigned long long>(worst[q]));
      }
      printf("\n");
    }
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
    // ai: LIZ_REFRESH and LIZ_REFRESH_BLOCKS: whole delays, comma-separated
    const auto refreshes = [](const char* v) {
      std::vector<int> r;
      std::string all = v ? v : "", e;
      for (size_t at = 0; at <= all.size(); at++) {
        if (at < all.size() && all[at] != ',') { e += all[at]; continue; }
        if (!e.empty()) r.push_back(std::max(1, std::atoi(e.c_str())));
        e.clear();
      }
      return r;
    };
    const std::vector<int> byRate = refreshes(getenv("LIZ_REFRESH")), byEntry = refreshes(getenv("LIZ_REFRESH_BLOCKS"));
    const bool refresh = byRate.size() == 4 || !byEntry.empty();
    if (byRate.size() == 4) s.setRefresh(byRate.data());
    int car = 0, carOn = 0, carOff = -1;
    if (const char* c = getenv("LIZ_CAROUSEL")) sscanf(c, "%d@%d,%d", &car, &carOn, &carOff);
    const bool live = car > 0 && (carOn > 0 || carOff >= 0);
    const int repaint = getenv("LIZ_REPAINT") ? atoi(getenv("LIZ_REPAINT")) : -1;
    if (car > 0 && carOn == 0) s.setCarousel(car);
    int animFrom = -1, staticFrom = -1, lastKind = -1, switches = 0;
    const uint32_t carKey = s.carouselKey();
    const std::string err = s.configure(f);
    if (!err.empty()) { printf("FAIL configure: %s\n", err.c_str()); return 1; }
    printf("painter: %s%s%s\n", s.painter().c_str(), s.gpuWhy().empty() ? "" : ", not the GPU: ", s.gpuWhy().c_str());
    // ai: each block place's schedule as the sender holds it: its rate's, or LIZ_REFRESH_BLOCKS's
    std::vector<int> sched(static_cast<size_t>(wantBlocks), 1);
    if (!byEntry.empty()) {
      for (size_t b = 0; b < sched.size(); b++) sched[b] = byEntry[b % byEntry.size()];
      if (!s.setRefreshBlocks(sched)) { printf("FAIL: the sender refused a schedule a place\n"); return 1; }
    }
    const int W = s.side(), FW = s.width(), codes = f.codes, gap = codes > 1 ? (FW - codes * W) / (codes - 1) : 0;
    std::filesystem::create_directories(store);
    std::unique_ptr<XferRx> rx;
    if (!test) rx = std::make_unique<XferRx>(store);
    cpu_dec_t* d = cpu_dec_new(1536);
    const int top = cpu_dec_top(d), BB = cpu_dec_block_bytes(d);
    std::vector<uint8_t> rgba(static_cast<size_t>(FW) * W * 4), luma(static_cast<size_t>(W) * W), blocks(static_cast<size_t>(top) * BB), ok(top), want(XFER_PAYLOAD);
    int held = 0, words = 0, faults = 0, symbols = 0, stood = -1;
    cpu_hold_t* hold = cpu_hold_new();
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
      if (byEntry.empty()) for (size_t b = 0; b < sched.size() && b < rateAt.size(); b++) sched[b] = byRate[static_cast<size_t>(rateAt[b])];
    }
    std::unique_ptr<lizard::lizfile::Writer> runOut;
    std::string runWhy;
    if (const char* ro = getenv("LIZ_RUN_OUT")) {
      runOut = lizard::lizfile::Writer::create(ro, runWhy);
      if (!runOut) printf("LIZ_RUN_OUT: %s\n", runWhy.c_str());
    }
    const double t0 = nowMs();
    for (int k = 0; k < frames * codes; k++) {
      if (k % codes == 0) {
        if (live && k / codes == carOn && carOn > 0) s.setCarousel(car);
        if (live && k / codes == carOff) s.setCarousel(0);
        if (k / codes == repaint) {
          f.painter = f.painter ? 0 : 1;
          const std::string e = s.configure(f);
          if (!e.empty()) { printf("FAIL configure at frame %d: %s\n", repaint, e.c_str()); return 1; }
          printf("repaint at frame %d: %s\n", repaint, s.stats().c_str());
        }
        while (!s.take(rgba.data(), FW * 4)) std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
      const int x0 = (k % codes) * (W + gap);
      for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) luma[static_cast<size_t>(y) * W + x] = rgba[4 * (static_cast<size_t>(y) * FW + x0 + x)];
      if (runOut && !runOut->frame(luma.data(), W, W, W, runWhy)) { printf("LIZ_RUN_OUT: %s\n", runWhy.c_str()); runOut.reset(); }
      cpu_frame_t fr{};
      const int got = cpu_dec_frame(d, luma.data(), W, W, held, blocks.data(), ok.data(), &fr);
      if (fr.word) { words++; const int v = cpu_hold_take(hold, fr.kind, fr.version, fr.pos, fr.sym); if (v) held = v; }
      verified += got; total += fr.total;
      if (live) {
        // ai: changed live: every block read (the static word held from the first frame); a word static until the ask at
        // ai: carOn shows, then the frame's symbol until the ask at carOff shows, each kind change once
        const int seq = k / codes, pos = Sender::carouselPosition(carKey, static_cast<uint64_t>(codes) * static_cast<uint64_t>(seq / car) + static_cast<uint64_t>(k % codes));
        if (fr.word) {
          symbols += fr.kind == 1;
          if (lastKind >= 0 && fr.kind != lastKind) {
            switches++;
            if (fr.kind == 1 && animFrom < 0) animFrom = seq;
            if (fr.kind == 0 && staticFrom < 0) staticFrom = seq;
          }
          lastKind = fr.kind;
          const bool animOk = seq >= carOn && (carOff < 0 || staticFrom < 0);
          if (fr.kind == 1 && (!animOk || fr.pos != pos)) { faults++; if (faults < 5) printf("frame %d: animated word position %d, the frame's %d, asked at %d\n", k, fr.pos, pos, carOn); }
          if (fr.kind == 0 && animFrom >= 0 && (carOff < 0 || seq < carOff)) { faults++; if (faults < 5) printf("frame %d: a static word after the animated began, none asked\n", k); }
        }
        if (got != wantBlocks) { faults++; if (faults < 5) printf("frame %d: %d of %d blocks\n", k, got, fr.total); }
      } else if (car > 0) {
        // ai: the animated header: each word the frame's symbol (its position by the frame's number), every block once
        // ai: the header stands (held at the format's version), none before
        const int seq = k / codes, pos = Sender::carouselPosition(carKey, static_cast<uint64_t>(codes) * static_cast<uint64_t>(seq / car) + static_cast<uint64_t>(k % codes));
        if (fr.word) symbols++;
        if (fr.word && (fr.kind != 1 || fr.pos != pos)) { faults++; if (faults < 5) printf("frame %d: word kind %d position %d, the frame's %d\n", k, fr.kind, fr.pos, pos); }
        if (stood < 0 && got > 0) stood = seq;
        if (got > 0 && got != wantBlocks) { faults++; if (faults < 5) printf("frame %d: %d of %d blocks\n", k, got, fr.total); }
        if (stood >= 0 && got == 0) { faults++; if (faults < 5) printf("frame %d: no blocks after the header stood\n", k); }
      } else if (fr.version != wantSubch / LIZARD_STEP || got != wantBlocks) { faults++; if (faults < 5) printf("frame %d: word %d version %d, %d of %d blocks\n", k, fr.word, fr.version, got, fr.total); }
      std::vector<int64_t> now(static_cast<size_t>(fr.total), -1);
      for (int b = 0; b < fr.total; b++) {
        if (!ok[b]) continue;
        const uint8_t* blk = blocks.data() + static_cast<size_t>(b) * BB;
        const xfer_id_t id = xfer_id_get(blk);
        now[static_cast<size_t>(b)] = static_cast<int64_t>(id);
        if (test) { stream_fill(id, want.data(), XFER_PAYLOAD); if (memcmp(want.data(), blk + XFER_ID_BYTES, XFER_PAYLOAD)) { faults++; if (faults < 5) printf("frame %d block %d: id %llu not its stream\n", k, b, static_cast<unsigned long long>(id)); } }
        else rx->take(blk);
      }
      if (refresh) {
        auto& was = before[static_cast<size_t>(k % codes)];
        const uint64_t seq = static_cast<uint64_t>(k / codes);
        for (int b = 0; b < fr.total && b < static_cast<int>(was.size()) && b < static_cast<int>(rateAt.size()); b++) {
          if (now[static_cast<size_t>(b)] < 0 || was[static_cast<size_t>(b)] < 0) continue;
          const int r = rateAt[static_cast<size_t>(b)];
          const int e = sched[static_cast<size_t>(b)];
          const bool hold = !Sender::due(e, seq), rep = now[static_cast<size_t>(b)] == was[static_cast<size_t>(b)];
          // ai: a fresh block may name the control block the frame before named there (the header, every other control
          // ai: slot; the warm frames, xfer_tx.h, carry nothing else): no repeat
          if (rep && !hold && xfer_id_chunk(static_cast<xfer_id_t>(now[static_cast<size_t>(b)])) == XFER_CONTROL) continue;
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
    if (live) {
      printf("carousel live: a symbol every %d frames asked at frame %d, the first animated word at frame %d (%d frames after), %d symbols read",
             car, carOn, animFrom, animFrom - carOn, symbols);
      if (carOff >= 0) printf("; off asked at frame %d, the first static word after at frame %d (%d frames after)", carOff, staticFrom, staticFrom - carOff);
      printf("; %d switches\n", switches);
      const int want = carOff >= 0 ? 2 : 1;
      if (switches != want || animFrom < 0 || (carOff >= 0 && staticFrom < 0)) { faults++; printf("FAIL: %d switches of the word's kind, %d expected\n", switches, want); }
    } else if (car > 0) printf("carousel: a symbol every %d frames, %d symbols read, the header stood at frame %d (k %d)\n", car, symbols, stood, 3);
    printf("stats: %s\n", s.stats().c_str());
    if (refresh) printf("refresh %s%s: repeated 7/8 %ld of %ld, 3/4 %ld of %ld, 2/3 %ld of %ld, 1/2 %ld of %ld\n", byEntry.empty() ? "by rate " : "by place ",
                        byEntry.empty() ? getenv("LIZ_REFRESH") : getenv("LIZ_REFRESH_BLOCKS"),
                        same[0], seen[0], same[1], seen[1], same[2], seen[2], same[3], seen[3]);
    if (!test) {
      rx->drain();
      const XferProgress p = rx->progress();
      printf("file: %s, done %d, root %s (the sender's %s)\n", p.name.c_str(), p.done ? 1 : 0, p.root.c_str(), s.xfer()->rootHex().c_str());
      if (!p.done) { printf("FAIL: the file did not finish in %d frames\n", frames); faults++; }
      else if (p.root != s.xfer()->rootHex()) { printf("FAIL: the root differs\n"); faults++; }
    }
    cpu_hold_free(hold);
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
    // ai: the encodes alternate the test stream made on the device from the ids (the sender's test path) and the same
    // ai: blocks given whole (a file's path): the difference a frame is the device's BLAKE3 (2026-11-15)
    double gpu[2] = {0, 0}, wall[2] = {0, 0};
    const int encodes = 16;
    for (int i = 0; i < 2 * encodes; i++) {
      const int k = i & 1;
      const double t0 = nowMs();
      gpu[k] += g->encode(blocks.data(), R, (i >> 1) & 3, out.data(), 0, k == 0);
      wall[k] += nowMs() - t0;
    }
    const double perFrame = encodes * R, nb = static_cast<double>(g->blocks()) * g->codes();
    printf("time: %.3f ms of GPU a frame making the test stream, %.3f with the blocks given: BLAKE3 %+.3f ms a frame, %+.2f us a block "
           "(%g blocks); %.2f and %.2f ms an encode of %d frames end to end (readback included), %d px a side\n",
           gpu[0] / perFrame, gpu[1] / perFrame, (gpu[0] - gpu[1]) / perFrame, 1000 * (gpu[0] - gpu[1]) / (perFrame * nb), nb,
           wall[0] / encodes, wall[1] / encodes, R, g->side());
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
      stream_fill(xfer_id_get(blk), want.data(), XFER_PAYLOAD);
      if (!memcmp(want.data(), blk + XFER_ID_BYTES, XFER_PAYLOAD)) stream++;
    }
    printf("decode: found %d, ring %d, n %d, word %d (version %d = LIZARD-%d, %d a second), %d blocks verified, %d of them the test stream's, pilots %d blocks r %.2f %.2f\n",
           fr.found, fr.ring, fr.n, fr.word, fr.version, LIZARD_STEP * fr.version, fr.fps, got, stream, fr.pilot_blocks, fr.pilot_r[0], fr.pilot_r[1]);
    cpu_dec_free(d);
    return got > 0 ? 0 : 1;
  }
  if (mode == "colours" && argc >= 5) {
    const int subch = atoi(argv[2]);
    const uint32_t ink = static_cast<uint32_t>(strtoul(argv[3], nullptr, 16)) & 0xffffffu, paper = static_cast<uint32_t>(strtoul(argv[4], nullptr, 16)) & 0xffffffu;
    Sender s(nullptr, 0, "", "");
    TxFormat f;
    f.subch = subch; f.span = 128; f.fps = 60; f.threads = 1; f.assets = assets();
    if (const std::string e = s.configure(f); !e.empty()) { fprintf(stderr, "colours: %s\n", e.c_str()); return 1; }
    s.setColours(ink, paper);
    const int W = s.side();
    std::vector<uint8_t> rgba(static_cast<size_t>(W) * W * 4);
    while (!s.take(rgba.data(), W * 4)) std::this_thread::sleep_for(std::chrono::microseconds(100));
    const auto ch = [](uint32_t c, int k) { return static_cast<int>((c >> (16 - 8 * k)) & 0xffu); };
    const auto px = [&](size_t i) { return (uint32_t{rgba[4 * i]} << 16) | (uint32_t{rgba[4 * i + 1]} << 8) | rgba[4 * i + 2]; };
    // ai: each pixel's place on the line (its channel of most swing), every channel then within a level of ink + (paper - ink) t
    int kmax = 0;
    for (int k = 1; k < 3; k++) if (abs(ch(paper, k) - ch(ink, k)) > abs(ch(paper, kmax) - ch(ink, kmax))) kmax = k;
    const int span = ch(paper, kmax) - ch(ink, kmax);
    const auto lumaOf = [&](uint32_t c) { return 77 * ch(c, 0) + 150 * ch(c, 1) + 29 * ch(c, 2); };
    const bool inverted = lumaOf(ink) > lumaOf(paper);
    size_t off = 0, dark = 0;
    int darkL = 1 << 30;
    for (size_t i = 0; i < static_cast<size_t>(W) * W; i++) {
      const uint32_t c = px(i);
      const int l = inverted ? -lumaOf(c) : lumaOf(c);
      if (l < darkL) { darkL = l; dark = i; }
      if (rgba[4 * i + 3] != 255) { off++; continue; }
      if (!span) { if (c != ink) off++; continue; }
      const double t = static_cast<double>(ch(c, kmax) - ch(ink, kmax)) / span;
      for (int k = 0; k < 3; k++) if (fabs(ch(c, k) - (ch(ink, k) + t * (ch(paper, k) - ch(ink, k)))) > 1.01 + fabs(ch(paper, k) - ch(ink, k)) / (2.0 * abs(span))) { off++; break; }
    }
    printf("colours: LIZARD-%d %d px a side, ink %06X paper %06X: corner %06X, %s %06X, %zu pixels off the line\n", subch, W, ink, paper, px(0),
           inverted ? "lightest" : "darkest", px(dark), off);
    return px(0) == paper && px(dark) == ink && off == 0 ? 0 : 1;
  }
  // ai: devinfo (2026-11-16): the device's memory types (DL device-local, HV host-visible, HC host-coherent, CA
  // ai: host-cached, LZ lazy) and its cooperative matrix shapes (VK_KHR_cooperative_matrix), where it has them
  if (mode == "devinfo") {
    const char* want = getenv("LIZ_VK_DEVICE");
    auto dev = wg::Device::create(want ? want : "", false, [](const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); });
    printf("device: %s\n", dev->name.c_str());
    for (uint32_t i = 0; i < dev->mem.memoryTypeCount; i++) {
      const auto fl = dev->mem.memoryTypes[i].propertyFlags;
      printf("memory type %u (heap %u):%s%s%s%s%s\n", i, dev->mem.memoryTypes[i].heapIndex, fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " DL" : "",
             fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " HV" : "", fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " HC" : "",
             fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " CA" : "", fl & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT ? " LZ" : "");
    }
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(dev->phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ex(ne);
    vkEnumerateDeviceExtensionProperties(dev->phys, nullptr, &ne, ex.data());
    bool coop = false;
    for (const auto& e : ex) if (strstr(e.extensionName, "cooperative_matrix") || strstr(e.extensionName, "COOPERATIVE_MATRIX")) { printf("extension: %s\n", e.extensionName); coop = coop || !strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME); }
    if (!coop) { printf("cooperative matrix: none\n"); return 0; }
    uint32_t nc = 0;
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev->phys, &nc, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> cp(nc, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev->phys, &nc, cp.data());
    const auto ty = [](VkComponentTypeKHR t) { switch (t) { case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16"; case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
      case VK_COMPONENT_TYPE_SINT8_KHR: return "s8"; case VK_COMPONENT_TYPE_UINT8_KHR: return "u8"; case VK_COMPONENT_TYPE_SINT32_KHR: return "s32"; case VK_COMPONENT_TYPE_UINT32_KHR: return "u32"; default: return "?"; } };
    for (const auto& c : cp) printf("coop %ux%ux%u A %s B %s C %s R %s scope %d\n", c.MSize, c.NSize, c.KSize, ty(c.AType), ty(c.BType), ty(c.CType), ty(c.ResultType), static_cast<int>(c.scope));
    return 0;
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
    // ai: LIZ_PACE_FILE=<file>: a file sent in place of the test stream (2026-11-16), the process's CPU said: until every
    // ai: chunk is made (from the configure), then a frame (LIZ_TX_RQ=cpu the CPU's encoders, else the GPU's); the test
// ai: stream's a frame from 2 s on (the painting alone)
    const char* pf = getenv("LIZ_PACE_FILE");
    std::unique_ptr<Mapped> pm;
    if (pf) pm = std::make_unique<Mapped>(pf);
    const auto cpuMs = [] { rusage r{}; getrusage(RUSAGE_SELF, &r); return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1e3 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e3; };
    Sender s(pf ? pm->p : nullptr, pf ? pm->n : 0, pf ? std::filesystem::path(pf).filename().string() : "", "");
    TxFormat f;
    f.subch = subch; f.span = 128; f.fps = fps; f.codes = codes; f.painter = 1; f.hostFrames = false; f.assets = assets();
    const double c0 = cpuMs(), w0 = nowMs();
    if (const std::string e = s.configure(f); !e.empty()) { fprintf(stderr, "pace: %s\n", e.c_str()); return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    const auto per = std::chrono::nanoseconds(1000000000LL / fps);
    long late = 0, taken = 0, madeAt = -1;
    double cMade = 0, wMade = 0;
    for (long k = 0; k < (long)secs * fps; k++) {
      std::this_thread::sleep_until(t0 + per * (k + 1));
      uint64_t seq = 0;
      int slot = 0;
      if (s.takeSlot(&seq, &slot)) taken++;
      else late++;
      if (pf && madeAt < 0 && s.xfer()->made()) { madeAt = k; cMade = cpuMs(); wMade = nowMs(); }
      if (!pf && k == 2 * fps) { madeAt = k; cMade = cpuMs(); wMade = nowMs(); }   // ai: the test stream's steady state from 2 s on
      if ((k + 1) % fps == 0) printf("%s\n", s.stats().c_str());
    }
    printf("pace: LIZARD-%d x %d at %d a second, %d s: %ld taken, %ld late\n", subch, codes, fps, secs, taken, late);
    if (pf && madeAt >= 0) {
      const long after = (long)secs * fps - madeAt;
      printf("cpu: every chunk made %.0f ms after the configure for %.0f core-ms; then %.3f core-ms a frame over %ld frames (%.1f%% of a core)\n",
             wMade - w0, cMade - c0, (cpuMs() - cMade) / after, after, (cpuMs() - cMade) / (nowMs() - wMade) * 100);
    } else if (pf) printf("cpu: not every chunk made in the run\n");
    else if (madeAt >= 0) {
      const long after = (long)secs * fps - madeAt;
      printf("cpu: %.3f core-ms a frame from 2 s on over %ld frames (%.1f%% of a core; the test stream: the painting alone)\n", (cpuMs() - cMade) / after, after,
             (cpuMs() - cMade) / (nowMs() - wMade) * 100);
    }
    return late ? 1 : 0;
  }
  fprintf(stderr, "usage: tx_check ids <file> <frames> <T> <out> | paint <file|-> <subch> [frames] [threads] [store] | bench <subch> [frames]\n"
                  "       | pace <subch> [secs] [fps] [codes]\n");
  return 2;
}
