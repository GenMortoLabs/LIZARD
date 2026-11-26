// ai: The app's sender off the phone (2026-10-01):
// ai:   tx_check ids <file> <frames> <blocks a frame> <out> [seed]   the sending end's ids and blocks, frame after frame
// ai:       (the ids, 12 bytes: the bare id and the 48-bit TOI, little-endian, then a reserved byte; then each block's 460
// ai:       bytes) under the carousel's seed (1 unless given), held byte for byte to gen/xfer_tx_ref.mjs's (the wasm build
// ai:       of the same C)
// ai:   tx_check paint <file | -> <subch> [frames] [threads] [store dir]   frames painted by the Sender (core/tx), each
// ai:       decoded blind by the C (cpu/codec.h) from its luma: the word, every block verified; the test stream's ("-")
// ai:       blocks held to BLAKE3 of their ids, a file's through the receiving end (rx/xfer_rx.h) until it is whole and
// ai:       its root the sender's. Exit 1 on a fault.
// ai:   tx_check broadcast <file> <store dir> [key=value ...]   the broadcast test (2026-11-20): the transfer protocol, no
// ai:       picture. The sender's pictures as the Sender assembles them (sender.cpp: the format's block places, inner rates
// ai:       first; each place's rate's Refresh delay, a held place repeating its block; the fresh ids in place order; one
// ai:       code) through a channel straight into the receiving end (rx/xfer_rx.h, its file store in <store dir>, emptied
// ai:       first), each picture taken in whole and its solves back before the next (drain), until the file is whole
// ai:       (each source block and the root verified by the receiver) or `frames`; then the file byte for byte against the
// ai:       source. Keys (defaults): subch=480 the format; refresh=1,1,1,1 each rate's delay (6/7, 3/4, 2/3, 1/2);
// ai:       block=0 each block lost i.i.d. (erasures); frame=0 and burst=1 each picture lost, in runs of `burst` pictures
// ai:       on average (two states: a loss run ends with probability 1 / burst); tier=0,0,0,0 each rate's blocks lost
// ai:       i.i.d. (1: never read); tierframe=0,0,0,0 a rate's blocks lost together in a picture; join=0 the receiver's
// ai:       first picture after that many cycles of the carousel (the carousel's end at 0.9 to 1); chunk the source block
// ai:       size's log2 (xfer_sb_log2_for's); seed=1 the carousel's and the channel's; frames a cap on pictures after the
// ai:       join; decoder=none (blocks straight into the transfer) or gpu or cpu (the receive path, rx/receiver.h: each
// ai:       picture painted by the C, a lost block's place a reserved control id, a lost picture never pushed; batch=32
// ai:       pictures pushed, full batches, then drained; LIZ_ASSETS the generated tree; the decode's misses reported and a
// ai:       fault). LIZ_SPILL the sender's spill. A line each 5 s; at the end the pictures from the join to the last source
// ai:       block complete, the airtime against a lossless receiver's and against the measured loss's 1 / (1 - p), the
// ai:       fresh data blocks taken by then against the file's source symbols (the reception overhead),
// ai:       the solves past K, the times and the anonymous memory's peak (sampled at the lines; the mapped file's pages
// ai:       left out). Exit 1 on a fault, 2 on a bad argument.
// ai:   tx_check bench <subch> [frames]   one painter's ms a frame (encode and paint) at that format
// ai:   tx_check colours <subch> <ink> <paper>   the code's colours (2026-11-16, Sender::setColours; RRGGBB each): a frame
// ai:       of the test stream taken (take, the CPU's RGBA) with them set: its corner the paper, its pixel of least luma
// ai:       the ink (of most where the ink is the lighter: an inverted code), every pixel on the line between them (each
// ai:       channel within a level of the grey's place on it). Exit 1 on a fault
// ai:   tx_check decode <luma> <w> <h>   one frame read blind (a screenshot of the phone's Send screen, grey)
// ai:   tx_check noise <subch> [ring] [codes] [frames] [sigma,...]   the GPU painter's margin against the C's paint: the
// ai:       same blocks, the same Gaussian noise, both read blind by the C (blocks verified, LDPC iterations), by sigma
// ai: The GPU's painter (2026-10-02, tx/gpu_painter.h), its kernels and manifest from LIZ_ASSETS (out, liblizard/out run from liblizard/, by default):
// ai:   tx_check gputables [all]   tx/send_tables.cpp's tables of every format held to the C through the kernels that read
// ai:       them: each format configured on the GPU painter and GpuPainter::check (the GPU's frames of the test stream
// ai:       against the C's paint of the same blocks: the border and the anchors to the byte, the square within the
// ai:       painter's bound, every symbol read blind by the C with all its blocks); every version in the default ring and
// ai:       a spread in the others (b % 9 == 2 of the 8-sub-channel counts b, and LIZARD-1024), `all` every version in
// ai:       every ring; LIZ_CODES=2 two codes a frame. Exit 1 on a fault
// ai:   tx_check gpucheck <subch> [ring] [frames] [codes] [encode]   the GPU's frames of the test stream's blocks against
// ai:       the C's paint at the same counts (the border and the anchors to the byte, every symbol read blind by the C
// ai:       with all its blocks; the square's pixels off the C reported, the largest and the RMS: GpuPainter::check, the
// ai:       sender's auto rule), then its GPU ms a frame over encodes of its most, the test stream made on the device and the blocks
// ai:       given whole alternately (their difference the device's BLAKE3), `encode` frames an encode (4 unset; 2026-11-18);
// ai:       LIZ_ENCODES=<k> the encodes of each kind timed (16 unset), LIZ_SEND_STAGES=1 each stage's GPU ms a frame too
// ai:       (GpuPainter's per-stage timestamp pairs; 2026-11-24)
// ai:       ring an index into src/lizard_phy.h
// ai:       LIZARD_RING, the default ring (LIZARD_RING_DEFAULT, the 128) unset
// ai:   LIZ_PAINTER=gpu|auto tx_check paint ...   the Sender painting on the GPU (or auto), every frame read blind as before
// ai:   LIZ_CODES=2 [LIZ_GAP=<modules>] tx_check paint ...   two codes a frame (2026-10-03), the gap between them set
// ai:       (2026-10-04, TxFormat.gap; GAP_MODULES unset), each read blind at its own offset
// ai:   LIZ_RUN_OUT=<file> tx_check paint ...   every symbol read written as a recording too (a .lizard, rx/lizfile.h:
// ai:       W x W luma, a code a frame, and its meta.json), which lizard_gpu_check replay reads (2026-10-07)
// ai:   LIZ_SPAN=<2 B> tx_check paint ...   the symbols in ring B (2026-10-10; 128, the 64 ring's, unset): 512 the 256 ring
// ai:   LIZ_SPILL=<dir> tx_check ids|paint <file> ...   the transfer's spill there (2026-11-14, xfer_tx.h: no encoder kept,
// ai:   its symbols in segments on disk), the apps' path: the ids and bytes the same
// ai:   LIZ_REFRESH=<6/7>,<3/4>,<2/3>,<1/2> tx_check paint ...   Refresh (2026-11-17, Sender::setRefresh; Rate refresh's
// ai:       1-to-4 holds of 2026-11-12 before) set before the first frame, each rate's delay, a whole number of pictures:
// ai:       every read block's id held against the frame before's at its place, the same exactly where its delay holds
// ai:       (core/tx/refresh.h refresh::due false: the count no multiple of it), a fault otherwise; the repeats a rate counted.
// ai:       LIZ_REFRESH_BLOCKS=<delay>,... in its place (Sender::setRefreshBlocks after the configure): place b takes entry
// ai:       b mod their count
// ai:   tx_check gpucheck ...   the GPU's frames against the C's: the anchors' squares (SPEC 6.13) to the byte besides
// ai:   the border, their pixels and the dark ones counted
// ai:   LIZ_REPAINT=<frame> tx_check paint ...   at that frame the Sender configured again on the other painter (the GPU's
// ai:       becomes the CPU's, the CPU's the GPU's; 2026-11-16): a file the GPU made read back for the CPU's painters
// ai:       (tx/gpu_stream.h host), the rest of the run read blind as before
// ai:   tx_check flights <subch> [encodes] [frames an encode]   the GPU painter's encodes in flight (2026-11-16,
// ai:       GpuPainter::submit): the test stream's frames painted GpuPainter::FLIGHTS at a time, nothing waited for
// ai:       between, then painted again one encode at a time and waited for: every ring frame byte-equal (6 encodes of 8
// ai:       unless given; LIZ_SPAN, LIZ_CODES as paint's). Exit 1 on a fault
// ai:   tx_check rqcheck <file | <MiB>m> [frames] [blocks a frame]   the sender's blocks on the GPU (2026-11-16,
// ai:       tx/gpu_stream.h) held to the CPU's: the stream's self-test, a file (or that many MiB of seeded random bytes)
// ai:       solved by the CPU's workers and its source blocks handed up (timed), then frames of the carousel's ids (60 of 248
// ai:       unless given) and chosen ones (each source block's first 300 source symbols, its last, K, K + 1, 2^24 - 1) made on
// ai:       the device and read back, every payload byte-equal to XferTx::block's (a reference transfer's encoders); frames
// ai:       0 the making alone, no reference. LIZ_VK_DEVICE the device. Exit 1 on a fault
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "codec.h"
#include "gpu_stream.h"
#include "gpu_painter.h"
#include "json.hpp"
#include "lizfile.h"
#include "send_tables.h"
#include "refresh.h"
#include "sender.h"
#include "receiver.h"
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
// ai: count frames of the test stream's blocks from id 0, blocks x 472 B each (id then BLAKE3 of it)
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
    XferTx x(m.p, m.n, std::filesystem::path(argv[2]).filename().string(), -1, spill());
    x.reseed(argc >= 7 ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 1u);   // ai: the carousel's seed, gen/xfer_tx_ref.mjs's
    x.waitMade();   // ai: every source block made before the first frame, so the frames are the serial sender's, as gen/xfer_tx_ref.mjs's (no warm frames)
    const int frames = atoi(argv[3]), T = atoi(argv[4]);
    FILE* out = fopen(argv[5], "wb");
    std::vector<xfer_id_t> ids(T);
    uint8_t b[XFER_PAYLOAD], idb[XFER_ID_BYTES];
    for (int f = 0; f < frames; f++) {
      x.frameIds(ids.data(), T);
      for (xfer_id_t id : ids) { xfer_id_put(idb, id); xfer_toi_put(idb, x.toiFor(id)); fwrite(idb, 1, XFER_ID_BYTES, out); }
      for (xfer_id_t id : ids) { x.block(id, b); fwrite(b, 1, XFER_PAYLOAD, out); }
    }
    fclose(out);
    printf("ids: %d frames of %d, source blocks %u, cycle %u, root %s, seed %u\n", frames, T, x.sourceBlocks(), x.cycle(), x.rootHex().c_str(), x.seed());
    return 0;
  }
  // ai: tx_check broadcast <file> <store dir> [key=value ...]: the broadcast test (the mode list above)
  if (mode == "broadcast" && argc >= 4) {
    Mapped m(argv[2]);
    const std::string dir = argv[3];
    std::map<std::string, std::string> kv;
    static const std::set<std::string> known{"subch", "refresh", "block", "frame", "burst", "tier", "tierframe", "join", "chunk", "seed", "frames", "decoder", "batch", "pause"};
    for (int a = 4; a < argc; a++) {
      const char* e = strchr(argv[a], '=');
      const std::string key = e ? std::string(argv[a], size_t(e - argv[a])) : "";
      if (!e || !known.count(key)) { fprintf(stderr, "broadcast: %s is not one of the keys\n", argv[a]); return 2; }
      kv[key] = e + 1;
    }
    auto num = [&](const char* k, double d) { auto it = kv.find(k); return it == kv.end() ? d : atof(it->second.c_str()); };
    auto four = [&](const char* k, double d, double out[4]) {
      for (int r = 0; r < 4; r++) out[r] = d;
      auto it = kv.find(k);
      if (it == kv.end()) return true;
      return sscanf(it->second.c_str(), "%lf,%lf,%lf,%lf", &out[0], &out[1], &out[2], &out[3]) == 4;
    };
    double refresh[4], tierLoss[4], tierFrame[4];
    const int subch = int(num("subch", 480)), chunk = int(num("chunk", -1));
    const double pBlock = num("block", 0), pFrame = num("frame", 0), burst = num("burst", 1), join = num("join", 0);
    const uint32_t seed = uint32_t(num("seed", 1));
    const uint64_t maxFrames = kv.count("frames") ? std::strtoull(kv["frames"].c_str(), nullptr, 10) : UINT64_MAX;
    const std::string decoder = kv.count("decoder") ? kv["decoder"] : "none";
    const int group = decoder == "none" ? 1 : std::max(1, int(num("batch", 32)));
    if (decoder != "none" && decoder != "gpu" && decoder != "cpu") { fprintf(stderr, "broadcast: decoder none, gpu or cpu\n"); return 2; }
    lizard_tier_t tier[LIZARD_TIERS];
    const int tiers = lizard_tiers_for(subch, tier);
    bool okArgs = four("refresh", 1, refresh) && four("tier", 0, tierLoss) && four("tierframe", 0, tierFrame) && tiers > 0 &&
                  pBlock >= 0 && pBlock < 1 && pFrame >= 0 && pFrame < 1 && burst >= 1 && join >= 0;
    for (int r = 0; r < 4; r++) okArgs = okArgs && refresh[r] >= 1 && refresh[r] <= refresh::MAX && tierLoss[r] >= 0 && tierLoss[r] <= 1 && tierFrame[r] >= 0 && tierFrame[r] <= 1;
    if (!okArgs) { fprintf(stderr, "broadcast: a format, delays of 1 to %d, losses in [0, 1) (a tier's in [0, 1]), burst 1 or more\n", refresh::MAX); return 2; }
    // ai: each block place's rate index (0 6/7 .. 3 1/2), the format's tiers inner first (liz_block_place_of's order)
    std::vector<int> rateOf;
    for (int t = 0; t < tiers; t++)
      for (int k = 0; k < tier[t].blocks; k++) rateOf.push_back(tier[t].rate == 6 ? 0 : tier[t].rate == 4 ? 1 : tier[t].rate == 3 ? 2 : 3);
    const int T = int(rateOf.size());
    double perPicture = 0;   // ai: fresh places a picture, on average
    for (int r : rateOf) perPicture += 1.0 / int(refresh[r]);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    double anonPeak = 0;
    auto anonMb = [&anonPeak] {
      double mb = 0;
      if (FILE* f = fopen("/proc/self/status", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f)) if (!strncmp(line, "RssAnon:", 8)) { mb = atof(line + 8) / 1024.0; break; }
        fclose(f);
      }
      anonPeak = std::max(anonPeak, mb);
      return mb;
    };
    const double t0 = nowMs();
    static const uint8_t emptyFile = 0;   // ai: an empty file is a file (liz_sender_new's rule): a pointer, no bytes
    XferTx x(m.n ? m.p : &emptyFile, m.n, std::filesystem::path(argv[2]).filename().string(), chunk, spill());
    x.reseed(seed);
    x.waitMade();
    if (!x.fault().empty()) { fprintf(stderr, "broadcast: the sender failed: %s\n", x.fault().c_str()); return 1; }
    const double tMade = nowMs();
    const uint64_t K = x.sourceSymbols();
    const uint64_t joinAt = uint64_t(std::ceil(join * double(x.cycleWhenMade()) / perPicture));
    printf("broadcast: %zu B, %u source blocks of 2^%d, K %llu, cycle %u; LIZARD-%d, %d places (%d %d %d %d), refresh %g,%g,%g,%g, "
           "%.2f fresh a picture; block %.3f, frame %.3f burst %g, tier %g,%g,%g,%g, tierframe %g,%g,%g,%g, join %.3f cycles (picture %llu), seed %u; "
           "made in %.1f s, anon %.0f MB\n",
           m.n, x.sourceBlocks(), int(x.sbSize() ? __builtin_ctzll(x.sbSize()) : 0), (unsigned long long)K, x.cycleWhenMade(), subch, T,
           tier[0].blocks, tiers > 1 ? tier[1].blocks : 0, tiers > 2 ? tier[2].blocks : 0, tiers > 3 ? tier[3].blocks : 0, refresh[0], refresh[1], refresh[2], refresh[3],
           perPicture, pBlock, pFrame, burst, tierLoss[0], tierLoss[1], tierLoss[2], tierLoss[3], tierFrame[0], tierFrame[1], tierFrame[2], tierFrame[3], join,
           (unsigned long long)joinAt, seed, (tMade - t0) / 1e3, anonMb());
    fflush(stdout);
    uint64_t logLines = 0;
    std::function<void(const std::string&)> logf = [&](const std::string& s) {
      if (logLines++ < 8 || s.find("rejected") != std::string::npos || s.find("stopped") != std::string::npos || s.find("rror") != std::string::npos) { printf("  [log] %s\n", s.c_str()); fflush(stdout); }
    };
    // ai: the transfer: the harness's own (decoder none: blocks straight in), or the receive path's (decoder gpu or cpu:
    // ai: Receiver, as the apps', its transfer in <store dir>), the pictures painted by the C (lizard_encode and
    // ai: lizard_paint_grey, the sender's count painted as Sender does, mod 4) on a painter a core, a lost block's place
    // ai: painted with a reserved control id (decoded as any block, ignored by the transfer: SPEC 7.5), a lost picture
    // ai: never pushed; the frames pushed `batch` at a time with full batches and room waited for (fillBatches,
    // ai: waitForRoom: every frame decoded, the GPU's launches whole), the receiver drained after each batch
    std::shared_ptr<XferRx> xr;
    std::unique_ptr<Receiver> recv;
    std::vector<lizard_t> painters;
    std::vector<std::vector<float>> drive;
    std::vector<std::vector<uint8_t>> grey;
    int W = 0;
    if (decoder == "none") xr = std::make_shared<XferRx>(dir, logf);
    else {
      painters.resize(std::max(1u, std::thread::hardware_concurrency() / 2));
      for (auto& f : painters) if (lizard_init(&f, lizard_n_for(subch), subch, 0)) { fprintf(stderr, "broadcast: the codec refused LIZARD-%d\n", subch); return 2; }
      if (painters[0].blocks != T || painters[0].block_bytes != XFER_BLOCK) { fprintf(stderr, "broadcast: the codec's layout is not the senders'\n"); return 1; }
      W = painters[0].px + 2 * LIZARD_QUIET * painters[0].pxm;
      drive.assign(painters.size(), std::vector<float>(size_t(painters[0].px) * painters[0].px));
      grey.assign(size_t(group), std::vector<uint8_t>(size_t(W) * W));
      ReceiverConfig cfg;
      cfg.assets = assets(); cfg.cacheDir = dir + ".cache"; cfg.storeDir = dir; cfg.decoder = decoder; cfg.layout = "1:1"; cfg.yieldGpu = false; cfg.log = logf;
      std::filesystem::create_directories(cfg.cacheDir);
      recv = Receiver::create(cfg, [](uint64_t) {});
      if (!recv || !recv->transfer()) { fprintf(stderr, "broadcast: no receiver (%s)\n", decoder.c_str()); return 1; }
      recv->waitForRoom(true);
      recv->fillBatches(true);
      xr = recv->transfer();
      const auto st = nlohmann::json::parse(recv->stats());
      printf("  the receive path: decoder %s%s, pictures %d x %d, %d a batch, %zu painters\n", st.value("decoder", std::string("?")).c_str(),
             st.contains("gpuWhy") && !st["gpuWhy"].is_null() && !st["gpuWhy"].get<std::string>().empty() ? (" (" + st["gpuWhy"].get<std::string>() + ")").c_str() : "",
             W, W, group, painters.size());
      fflush(stdout);
    }
    XferRx& rx = *xr;
    bool ok_decode = true;
    const double pauseS = num("pause", 0);
    bool pausedOnce = false;
    uint64_t r = 0x9e3779b97f4a7c15ull ^ (uint64_t(seed) << 32 | seed);   // ai: xorshift64*, the channel's, the carousel's seed apart
    auto unit = [&r] { r ^= r >> 12; r ^= r << 25; r ^= r >> 27; return double((r * 0x2545f4914f6cdd1dull) >> 11) / double(1ull << 53); };
    // ai: the picture losses' two states: a run ends with 1 / burst, one begins so the share lost is pFrame
    const double leave = 1 / burst, enter = pFrame > 0 ? pFrame / (1 - pFrame) * leave : 0;
    bool bad = false;
    std::vector<xfer_id_t> ids(T);
    std::vector<uint8_t> shown(size_t(T) * XFER_BLOCK);
    std::vector<uint64_t> cumFwd, cumCtl;   // ai: from the join, a picture each: blocks the judge passed on (new data and control), control ones
    // ai: freshShown, newIn: the fresh data ids shown from the join and the distinct ones delivered (a held place's second
    // ai: showing can deliver an id its first lost: the loss a receiver sees, Refresh's diversity in it)
    uint64_t pictures = 0, freshShown = 0, newIn = 0, fwd = 0, ctl = 0;
    double lastLine = nowMs();
    XferProgress p;
    // ai: decoder mode: the group's pictures (its painted blocks, empty for one lost whole; its control blocks: headers
    // ai: and the lost places' reserved ids, which the judge passes on), and the decode's health over the run
    struct Pic { std::vector<uint8_t> blocks; uint64_t seq = 0; int ctl = 0; };
    std::vector<Pic> pend;
    uint64_t pushed = 0, notFound = 0, blocksShort = 0, misses = 0;
    double since = 0;
    const uint8_t* const lostBlock = [] {
      static uint8_t b[XFER_BLOCK] = {};
      xfer_id_put(b, xfer_id(XFER_CONTROL, 1));   // ai: a reserved control id (SPEC 7.5): a receiver ignores it
      return b;
    }();
    auto flush = [&] {
      const size_t n = pend.size();
      std::vector<std::thread> th;
      for (size_t j = 0; j < painters.size(); j++)
        th.emplace_back([&, j] {
          for (size_t i = j; i < n; i += painters.size()) {
            if (pend[i].blocks.empty()) continue;
            lizard_parity(&painters[j], int(pend[i].seq & 3));
            lizard_encode(&painters[j], pend[i].blocks.data(), drive[j].data());
            lizard_paint_grey(&painters[j], drive[j].data(), grey[i].data());
          }
        });
      for (auto& h : th) h.join();
      for (size_t i = 0; i < n; i++) {
        if (pend[i].blocks.empty()) continue;
        CameraFrame f;
        f.luma = grey[i].data(); f.width = f.height = f.stride = uint32_t(W); f.whole = true; f.tag = i;
        f.timestampNs = int64_t(pictures - n + i + 1) * 1000000;   // ai: ms on the camera's clock: the picture's place from the join, from 1
        recv->push(f);
        pushed++;
      }
      recv->drain(600000);
      rx.drain();
      // ai: each pushed picture's row (ms its place): its new data blocks, found, verified against the T painted
      std::map<uint64_t, double> newAt;
      const std::vector<double> v = recv->series(since);
      for (size_t at = 1; at + 8 <= v.size(); at += 8) {
        newAt[uint64_t(v[at])] = v[at + 2];
        notFound += v[at + 3] == 0;
        blocksShort += v[at + 1] < T ? uint64_t(T - v[at + 1]) : 0;
        since = std::max(since, v[at]);
        // ai: a painted picture the decode did not read whole: its place from the join, the sender's count, and the
        // ai: picture itself kept beside the store (<store dir>.miss<place>.grey, W x W) for the decoders alone
        if (v[at + 3] == 0 || v[at + 1] < T) {
          const uint64_t place = uint64_t(v[at]);
          const size_t i = size_t(place - (pictures - n + 1));
          printf("  miss: picture %llu (count %llu, batch slot %zu): found %.0f, %.0f of %d verified\n", (unsigned long long)place,
                 (unsigned long long)(i < n ? pend[i].seq : 0), i, v[at + 3], v[at + 1], T);
          if (i < n && misses++ < 4) {
            if (FILE* g = fopen((dir + ".miss" + std::to_string(place) + ".grey").c_str(), "wb")) { fwrite(grey[i].data(), 1, grey[i].size(), g); fclose(g); }
            if (FILE* g = fopen((dir + ".miss" + std::to_string(place) + ".blocks").c_str(), "wb")) { fwrite(pend[i].blocks.data(), 1, pend[i].blocks.size(), g); fclose(g); }
          }
          fflush(stdout);
        }
      }
      for (size_t i = 0; i < n; i++) {
        if (!pend[i].blocks.empty()) {
          const auto it = newAt.find(pictures - n + i + 1);
          const uint64_t nw = it == newAt.end() ? 0 : uint64_t(it->second);
          newIn += nw;
          fwd += nw + uint64_t(pend[i].ctl);
          ctl += uint64_t(pend[i].ctl);
        }
        cumFwd.push_back(fwd); cumCtl.push_back(ctl);
      }
      pend.clear();
    };
    for (uint64_t seq = 0;; seq++) {
      int fresh = 0;
      std::vector<char> held(T);
      for (int t = 0; t < T; t++) { held[t] = seq > 0 && !refresh::due(int(refresh[rateOf[t]]), seq); fresh += !held[t]; }
      if (fresh) x.frameIds(ids.data(), fresh);
      for (int t = 0, k = 0; t < T; t++) {
        if (held[t]) continue;
        uint8_t* o = shown.data() + size_t(t) * XFER_BLOCK;
        xfer_id_put(o, ids[k]);
        xfer_toi_put(o, x.toiFor(ids[k]));
        x.block(ids[k++], o + XFER_ID_BYTES);
      }
      if (seq < joinAt) continue;
      if (pictures >= maxFrames) break;
      pictures++;
      bad = bad ? unit() >= leave : unit() < enter;
      Pic pic;
      pic.seq = seq;
      if (!bad) {
        bool down[4];
        for (int q = 0; q < 4; q++) down[q] = tierFrame[q] > 0 && unit() < tierFrame[q];
        if (recv) pic.blocks.resize(size_t(T) * XFER_BLOCK);
        for (int t = 0; t < T; t++) {
          const int q = rateOf[t];
          const uint8_t* o = shown.data() + size_t(t) * XFER_BLOCK;
          const bool isCtl = xfer_id_sbn(xfer_id_get(o)) == XFER_CONTROL;
          if (!held[t] && !isCtl) freshShown++;
          const bool lost = down[q] || (tierLoss[q] > 0 && unit() < tierLoss[q]) || (pBlock > 0 && unit() < pBlock);
          if (recv) {
            memcpy(pic.blocks.data() + size_t(t) * XFER_BLOCK, lost ? lostBlock : o, XFER_BLOCK);
            pic.ctl += lost || isCtl;
            continue;
          }
          if (lost) continue;
          const bool isNew = rx.take(o);
          if (isNew || isCtl) { fwd++; ctl += isCtl; }
          newIn += isNew;
        }
      } else for (int t = 0; t < T; t++) freshShown += !held[t] && xfer_id_sbn(xfer_id_get(shown.data() + size_t(t) * XFER_BLOCK)) != XFER_CONTROL;
      if (recv) {
        pend.push_back(std::move(pic));
        if (pend.size() < size_t(group)) continue;
        flush();
      } else {
        cumFwd.push_back(fwd); cumCtl.push_back(ctl);
        // ai: each picture taken in whole, its solves back, before the next: a source block a solve at K left short (RaptorQ
        // ai: wanting K + 1) is completed, and the file's doneAt stamped, within its picture, not when a busy solver returned
        // ai: while the producer ran on (the receiver stamps a fed-on source block at its recovery, xfer_rx_core.cpp recovered)
        rx.drain();
        // ai: pause=<s> (2026-11-24): the receiver paused that long at half the lossless receiver's pictures, as a camera
        // ai: stopped mid-transfer: the transfer's clock (XferProgress.secs) leaves it out
        if (pauseS > 0 && !pausedOnce && pictures >= uint64_t(double(K) / perPicture / 2)) {
          pausedOnce = true;
          rx.pause(true);
          std::this_thread::sleep_for(std::chrono::milliseconds(int64_t(pauseS * 1000)));
          rx.pause(false);
        }
      }
      if (pictures % (recv ? group : 64)) continue;
      p = rx.progress();
      if (p.done || p.failed) break;
      if (nowMs() - lastLine > 5000) {
        lastLine = nowMs();
        const double sec = (lastLine - tMade) / 1e3;
        printf("  %.0f s: %llu pictures from the join, %llu fresh data shown, %llu new in, %u of %u verified, %u solving, %.4f in, queued %u, lost %llu, %.0f k pictures/s, anon %.0f MB\n",
               sec, (unsigned long long)pictures, (unsigned long long)freshShown, (unsigned long long)newIn, p.verified, p.sourceBlocks, p.solving, p.fraction,
               p.queued, (unsigned long long)p.lost, pictures / sec / 1e3, anonMb());
        fflush(stdout);
      }
    }
    if (recv && !pend.empty()) flush();
    const double tFed = nowMs();
    rx.drain();
    p = rx.progress();
    if (recv)
      printf("  the decode: %llu pictures pushed, %llu not found, %llu painted blocks not verified (each picture %d, a lost place's reserved block among them)\n",
             (unsigned long long)pushed, (unsigned long long)notFound, (unsigned long long)blocksShort, T);
    ok_decode = !recv || (!notFound && !blocksShort);
    const double tDone = nowMs();
    bool ok = p.done && !p.failed && !p.lost && !p.rejected && ok_decode;
    bool same = false;
    if (p.done && !p.path.empty()) {
      Mapped out(p.path.c_str());
      same = out.n == m.n;
      const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
      std::vector<std::thread> th;
      std::atomic<bool> diff{false};
      for (unsigned t = 0; same && m.n && t < nt; t++)
        th.emplace_back([&, t] {
          const size_t a = m.n / nt * t, b = t + 1 == nt ? m.n : m.n / nt * (t + 1);
          if (b > a && memcmp(m.p + a, out.p + a, b - a)) diff = true;
        });
      for (auto& h : th) h.join();
      same = same && !diff;
    }
    ok = ok && same;
    // ai: the picture the last source block was complete in: the first whose blocks passed on reach doneAt
    const size_t at = size_t(std::lower_bound(cumFwd.begin(), cumFwd.end(), p.doneAt) - cumFwd.begin());
    const uint64_t toDone = p.done ? std::min<uint64_t>(at + 1, cumFwd.size()) : 0, ctlAt = p.done && at < cumCtl.size() ? cumCtl[at] : ctl;
    const double lossless = double(K) / perPicture, pEff = freshShown ? 1 - double(newIn) / double(freshShown) : 0;
    const uint64_t taken = p.doneAt > ctlAt ? p.doneAt - ctlAt : 0;   // ai: an empty file is done at its header: none
    const double over = K ? double(taken) / double(K) - 1 : 0;
    printf("broadcast: %s. %llu pictures from the join to the last source block complete: %.4f of a lossless receiver's %.0f, %.4f of 1 / (1 - p) of it "
           "(p %.4f measured: the fresh data ids never delivered); %llu fresh data blocks taken by then, %.3f%% past the %llu source symbols; %u source blocks verified, %u solved "
           "past K, %u rejected, %llu lost; made %.1f s, fed %.1f s, the solves' tail %.1f s, the compare %.1f s; anon peak "
           "%.0f MB; the file %s\n",
           ok ? "PASS" : "FAIL", (unsigned long long)toDone, toDone / lossless, lossless, toDone / (lossless / (1 - pEff)), pEff,
           (unsigned long long)taken, 100 * over, (unsigned long long)K, p.verified, p.over, p.rejected,
           (unsigned long long)p.lost, (tMade - t0) / 1e3, (tFed - tMade) / 1e3, (tDone - tFed) / 1e3, (nowMs() - tDone) / 1e3,
           (anonMb(), anonPeak), same ? "byte for byte the source's" : "NOT the source's");
    if (pauseS > 0) printf("  the transfer's clock %.2f s, the feed %.2f s with %.2f s paused at half\n", p.secs, (tFed - tMade) / 1e3, pausedOnce ? pauseS : 0.0);
    recv.reset();
    for (auto& f : painters) lizard_free(&f);
    return ok ? 0 : 1;
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
    XferTx x(data, len, "f.bin", -1, "", {}, true);
    const uint32_t C = x.sourceBlocks();
    printf("rqcheck: %zu B, %u source blocks of %zu B, %u controls\n", len, C, x.sbSize(), x.controls());
    const char* want = getenv("LIZ_VK_DEVICE");
    auto dev = wg::Device::create(want ? want : "", false, [](const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); });
    printf("device: %s\n", dev->name.c_str());
    std::unique_ptr<GpuStream> st;
    try { st = GpuStream::create(*dev, assets(), {len, x.sbSize(), C, x.controls()}); }
    catch (const std::exception& e) { printf("FAIL: no GPU stream: %s\n", e.what()); return 1; }
    if (const std::string why = st->selfTest(); !why.empty()) { printf("FAIL: the self-test: %s\n", why.c_str()); return 1; }
    printf("self-test ok; %.0f MB of symbols on the device\n", st->symbolBytes() / 1e6);
    // ai: the CPU's workers solving, each source block copied up by its worker on the device's second queue
    double t0 = nowMs();
    GpuStream* sp = st.get();
    x.makeOnCpu([sp](uint32_t c, const rq_enc_t* e) { return sp->put(c, e); });
    x.waitMade();
    if (!x.fault().empty() || !st->done()) { printf("FAIL: %s\n", x.fault().empty() ? "source blocks missing on the device" : x.fault().c_str()); return 1; }
    printf("every source block solved and on the device in %.0f ms\n", nowMs() - t0);
    if (!frames) { printf("timed alone (frames 0: no reference, nothing compared)\n"); return 0; }
    t0 = nowMs();
    XferTx ref(data, len, "f.bin", -1, "");
    ref.reseed(x.seed());   // ai: the same TOIs (the seed's)
    ref.waitMade();
    printf("the CPU's encoders made in %.0f ms (the reference)\n", nowMs() - t0);
    // ai: the carousel's frames, then the chosen ids
    std::vector<xfer_id_t> chosen;
    for (uint32_t c = 0; c < C; c++) {
      const uint32_t K = rq_symbols(x.sbBytes(c), XFER_PAYLOAD);
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
      for (xfer_id_t id : ids) { if (xfer_id_sbn(id) == XFER_CONTROL) ctl++; else data_.push_back(id); }
      got.resize(data_.size() * XFER_PAYLOAD);
      const double th = nowMs();
      st->host(data_.data(), static_cast<uint32_t>(data_.size()), got.data());
      hostMs += nowMs() - th;
      for (size_t i = 0; i < data_.size(); i++) {
        ref.block(data_[i], refb.data());
        if (std::memcmp(refb.data(), got.data() + i * XFER_PAYLOAD, XFER_PAYLOAD)) {
          if (wrong < 5) printf("block: source block %u symbol %u differs\n", xfer_id_sbn(data_[i]), xfer_id_esi(data_[i]));
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
  // ai: repick <subch A> <subch B> [painter 0|1|2] [codes] (2026-10-04, the desktop sender's stall at a re-pick): the
  // ai: test stream at A in the 128 ring, a few frames taken, then configured to B as a re-pick does: the configure's ms
  // ai: and the ms until B's first frame is ready, A to B and back, three times
  if (mode == "repick" && argc >= 4) {
    Sender s(nullptr, 0, "");
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
    Sender s(test ? nullptr : m->p, test ? 0 : m->n, test ? "" : std::filesystem::path(argv[2]).filename().string(), 0, test ? "" : spill());
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
    const int repaint = getenv("LIZ_REPAINT") ? atoi(getenv("LIZ_REPAINT")) : -1;
    const std::string err = s.configure(f);
    if (!err.empty()) { printf("FAIL configure: %s\n", err.c_str()); return 1; }
    printf("painter: %s%s%s\n", s.painter().c_str(), s.gpuWhy().empty() ? "" : ", not the GPU: ", s.gpuWhy().c_str());
    // ai: each block place's carousel as the sender holds it: its rate's, or LIZ_REFRESH_BLOCKS's
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
    int words = 0, faults = 0, clockWrong = 0;
    // ai: the clock every word must state (src/fmt.h): the format's fps and the largest Refresh delay asked, 1 with none
    int wantDelay = 1;
    for (const int e : byEntry.empty() ? byRate : byEntry) wantDelay = std::max(wantDelay, std::min(e, 30));
    long verified = 0, total = 0;
    // ai: LIZ_REFRESH: each block's code rate (0 6/7 ... 3 1/2) in the format's order, and the ids read a symbol before
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
      const int got = cpu_dec_frame(d, luma.data(), W, W, blocks.data(), ok.data(), &fr);
      words += fr.word;
      if (fr.word && (fr.fps != f.fps || fr.delay != wantDelay) && repaint < 0) { if (clockWrong++ < 3) printf("frame %d: the word's clock %d / %d, asked %d / %d\n", k, fr.fps, fr.delay, f.fps, wantDelay); }
      verified += got; total += fr.total;
      if (fr.version != wantSubch / LIZARD_STEP || got != wantBlocks) { faults++; if (faults < 5) printf("frame %d: word %d version %d, %d of %d blocks\n", k, fr.word, fr.version, got, fr.total); }
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
          const bool hold = !refresh::due(e, seq), rep = now[static_cast<size_t>(b)] == was[static_cast<size_t>(b)];
          // ai: a fresh block may name the control block the frame before named there (the header, every other control
          // ai: slot; the warm frames, xfer_tx.h, carry nothing else): no repeat
          if (rep && !hold && xfer_id_sbn(static_cast<xfer_id_t>(now[static_cast<size_t>(b)])) == XFER_CONTROL) continue;
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
    if (refresh) printf("refresh %s%s: repeated 6/7 %ld of %ld, 3/4 %ld of %ld, 2/3 %ld of %ld, 1/2 %ld of %ld\n", byEntry.empty() ? "by rate " : "by place ",
                        byEntry.empty() ? getenv("LIZ_REFRESH") : getenv("LIZ_REFRESH_BLOCKS"),
                        same[0], seen[0], same[1], seen[1], same[2], seen[2], same[3], seen[3]);
    if (!test) {
      rx->drain();
      const XferProgress p = rx->progress();
      printf("file: %s, done %d, root %s (the sender's %s)\n", p.name.c_str(), p.done ? 1 : 0, p.root.c_str(), s.xfer()->rootHex().c_str());
      if (!p.done) { printf("FAIL: the file did not finish in %d frames\n", frames); faults++; }
      else if (p.root != s.xfer()->rootHex()) { printf("FAIL: the root differs\n"); faults++; }
    }
    cpu_dec_free(d);
    if (clockWrong) { printf("FAIL: %d words stated another clock than %d / %d\n", clockWrong, f.fps, wantDelay); faults++; }
    printf(faults ? "FAILED %d\n" : "ok\n", faults);
    return faults ? 1 : 0;
  }
  // ai: decode <luma file> <width> <height>: one frame of 8-bit luma (a screenshot of the phone's Send screen, made
  // ai: grey by the caller), read blind by the C: the word, the blocks verified, and those that are the test stream's
  if (mode == "gputables") {
    const bool all = argc > 2 && std::string(argv[2]) == "all";
    const int codes = getenv("LIZ_CODES") ? atoi(getenv("LIZ_CODES")) : 1, frames = 4;
    std::unique_ptr<GpuPainter> g;
    try { g = GpuPainter::create(assets(), [](const std::string& m) { printf("%s\n", m.c_str()); }); }
    catch (const std::exception& e) { printf("FAIL no GPU painter: %s\n", e.what()); return 1; }
    int done = 0, bad = 0, worst = 0;
    long blocks = 0, square = 0;
    double rms = 0;
    for (int ring = 0; ring < LIZARD_RINGS; ring++)
      for (int v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX; v++) {
        const int subch = LIZARD_STEP * v, b = subch / 8;
        if (!all && ring != LIZARD_RING_DEFAULT && b % 9 != 2 && b != 128) continue;
        const std::string name = std::to_string(subch) + "_" + std::to_string(ring);
        const std::string ce = g->configure(subch, 2 * LIZARD_RING[ring], 60, frames, codes, GAP_MODULES, 0, 0, 0, false);
        if (!ce.empty()) { printf("FAIL %s: configure: %s\n", name.c_str(), ce.c_str()); bad++; done++; continue; }
        const auto c = g->check(testBlocks(std::max(frames, g->frames()), subch / 8 * codes).data(), frames);
        if (!c.ok) {
          printf("FAIL %s: border %ld pixels apart, the anchors %ld, the square %ld of %ld off the C (at most %d levels, RMS %.3f), read blind %ld of %ld blocks\n", name.c_str(),
                 c.borderDiff, c.anchorDiff, c.squareDiff, c.squarePx, c.squareMax, c.squareRms, c.readBlocks - c.readBad, c.readBlocks);
          bad++;
        }
        blocks += c.readBlocks - c.readBad; square += c.squareDiff; worst = std::max(worst, c.squareMax); rms = std::max(rms, c.squareRms);
        done++;
      }
    printf("%d formats x %d codes on %s: %d differ; %ld blocks read blind, the square %ld pixels off the C (at most %d levels, RMS at most %.3f)\n%s\n", done, codes,
           g->device().c_str(), bad, blocks, square, worst, rms, bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
  }
  if (mode == "gpucheck" && argc >= 3) {
    const int subch = atoi(argv[2]), ring = argc > 3 ? atoi(argv[3]) : LIZARD_RING_DEFAULT, frames = argc > 4 ? atoi(argv[4]) : 8, codes = argc > 5 ? atoi(argv[5]) : 1;
    const int encode = argc > 6 ? std::max(1, atoi(argv[6])) : 4;   // ai: frames an encode (2026-11-18: batching's GPU cost a frame)
    if (ring < 0 || ring >= LIZARD_RINGS) { printf("FAIL ring %d: an index of LIZARD_RING, 0 to %d\n", ring, LIZARD_RINGS - 1); return 1; }
    std::unique_ptr<GpuPainter> g;
    try { g = GpuPainter::create(assets(), [](const std::string& m) { printf("%s\n", m.c_str()); }); }
    catch (const std::exception& e) { printf("FAIL no GPU painter: %s\n", e.what()); return 1; }
    const std::string ce = g->configure(subch, 2 * LIZARD_RING[ring], 60, encode, codes, GAP_MODULES, 0, 0, 0, false);
    if (!ce.empty()) { printf("FAIL configure: %s\n", ce.c_str()); return 1; }
    const int V = subch / 8 * codes, R = g->frames();
    const auto blocks = testBlocks(std::max(frames, R), V);
    const auto c = g->check(blocks.data(), frames);
    char an[160] = "";
    snprintf(an, sizeof an, ", the anchors %ld pixels (%ld dark) %ld apart", c.anchorPx, c.anchorDark, c.anchorDiff);
    printf("check, %d frames of LIZARD-%d x %d in the %d ring on %s: border %ld pixels apart, the square %ld of %ld pixels off the C, at most %d levels, RMS %.3f%s; read blind: %ld of %ld blocks: %s\n",
           frames, subch, codes, LIZARD_RING[ring], g->device().c_str(), c.borderDiff, c.squareDiff, c.squarePx, c.squareMax, c.squareRms, an, c.readBlocks - c.readBad, c.readBlocks, c.ok ? "ok" : "FAILED");
    std::vector<std::vector<uint8_t>> out(R);
    // ai: the encodes alternate the test stream made on the device from the ids (the sender's test path) and the same
    // ai: blocks given whole (a file's path): the difference a frame is the device's BLAKE3 (2026-11-15)
    double gpu[2] = {0, 0}, wall[2] = {0, 0};
    const char* ne = getenv("LIZ_ENCODES");   // ai: encodes of each kind (16 unset; 2026-11-24, the timing A/B's)
    const int encodes = ne && atoi(ne) > 0 ? atoi(ne) : 16;
    std::array<double, GpuPainter::STAGES> st[2] = {};
    g->takeStageMs();   // ai: the check's encodes left out
    for (int i = 0; i < 2 * encodes; i++) {
      const int k = i & 1;
      const double t0 = nowMs();
      gpu[k] += g->encode(blocks.data(), R, (i >> 1) & 3, out.data(), 0, k == 0);
      wall[k] += nowMs() - t0;
      const auto s = g->takeStageMs();
      for (int j = 0; j < GpuPainter::STAGES; j++) st[k][j] += s[j];
    }
    if (g->staged()) for (int k = 0; k < 2; k++) {
      printf("stages (%s), ms a frame:", k ? "blocks given" : "test stream");
      for (int j = 0; j < GpuPainter::STAGES; j++) printf(" %s %.4f", GpuPainter::STAGE_NAMES[j], st[k][j] / (encodes * R));
      printf("\n");
    }
    const double perFrame = encodes * R, nb = static_cast<double>(g->blocks()) * g->codes();
    printf("time: %.3f ms of GPU a frame making the test stream, %.3f with the blocks given: BLAKE3 %+.3f ms a frame, %+.2f us a block "
           "(%g blocks); %.2f and %.2f ms an encode of %d frames end to end (readback included), %d px a side\n",
           gpu[0] / perFrame, gpu[1] / perFrame, (gpu[0] - gpu[1]) / perFrame, 1000 * (gpu[0] - gpu[1]) / (perFrame * nb), nb,
           wall[0] / encodes, wall[1] / encodes, R, g->side());
    return c.ok ? 0 : 1;
  }
  // ai: tx_check noise <subch> [ring] [codes] [frames] [sigma,...] (lab, 2026-11-24): the GPU painter's margin against the
  // ai: C's: the same test blocks painted by both, the same Gaussian noise (sigma grey levels, rounded, clamped to 0..255,
  // ai: a seed a symbol and sigma) added to each symbol, each read blind by the C; by sigma, each side's blocks verified
  // ai: and the LDPC's mean iterations a verified block (the GPU's square is its own arithmetic: is its margin the C's?)
  if (mode == "noise" && argc >= 3) {
    const int subch = atoi(argv[2]), ring = argc > 3 ? atoi(argv[3]) : LIZARD_RING_DEFAULT, codes = argc > 4 ? atoi(argv[4]) : 1, frames = argc > 5 ? atoi(argv[5]) : 32;
    std::vector<double> sigmas;
    for (std::string list = argc > 6 ? argv[6] : "20,30,40,50,60"; !list.empty();) {
      const size_t c = list.find(',');
      sigmas.push_back(atof(list.substr(0, c).c_str()));
      list = c == std::string::npos ? "" : list.substr(c + 1);
    }
    std::unique_ptr<GpuPainter> g;
    try { g = GpuPainter::create(assets(), [](const std::string&) {}); }
    catch (const std::exception& e) { printf("FAIL no GPU painter: %s\n", e.what()); return 1; }
    const std::string ce = g->configure(subch, 2 * LIZARD_RING[ring], 60, 4, codes, GAP_MODULES, 0, 0, 0, false);
    if (!ce.empty()) { printf("FAIL configure: %s\n", ce.c_str()); return 1; }
    lizard_t p{};
    if (lizard_init(&p, lizard_n_for(subch), subch, 2 * LIZARD_RING[ring])) { printf("FAIL the codec refused the format\n"); return 1; }
    const int R = g->frames(), W = g->side(), FW = g->width(), nb = g->blocks(), gap = codes > 1 ? (FW - codes * W) / (codes - 1) : 0;
    const auto blocks = testBlocks(frames + R, nb * codes);
    const size_t per = static_cast<size_t>(nb) * XFER_BLOCK;
    cpu_dec_t* d = cpu_dec_new(1536);
    const int top = cpu_dec_top(d), BB = cpu_dec_block_bytes(d);
    std::vector<uint8_t> rb(static_cast<size_t>(top) * BB), ok(top), rgba(static_cast<size_t>(W) * W * 4), luma[2], noisy(static_cast<size_t>(W) * W);
    luma[0].resize(noisy.size()); luma[1].resize(noisy.size());
    std::vector<float> drive(static_cast<size_t>(p.px) * p.px);
    std::vector<std::vector<uint8_t>> out(R);
    const size_t S = sigmas.size();
    std::vector<long> verified(2 * S, 0), its(2 * S, 0), total(2 * S, 0);
    for (int f0 = 0; f0 < frames; f0 += R) {
      const int k = std::min(R, frames - f0);
      g->encode(blocks.data() + static_cast<size_t>(f0) * codes * per, k, f0 & 3, out.data(), 0, true);
      for (int i = 0; i < k; i++) {
        lizard_parity(&p, (f0 + i) & 3);
        for (int cd = 0; cd < codes; cd++) {
          lizard_encode(&p, blocks.data() + (static_cast<size_t>(f0 + i) * codes + cd) * per, drive.data());
          lizard_paint_rgba(&p, drive.data(), rgba.data());
          const int x0 = cd * (W + gap);
          for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) {
            luma[0][static_cast<size_t>(y) * W + x] = rgba[4 * (static_cast<size_t>(y) * W + x)];
            luma[1][static_cast<size_t>(y) * W + x] = out[i][static_cast<size_t>(y) * FW + x0 + x];
          }
          for (size_t si = 0; si < S; si++) for (int side = 0; side < 2; side++) {
            std::mt19937_64 rng(1000003ull * (f0 + i) + 7919ull * cd + si);
            std::normal_distribution<double> nd(0.0, sigmas[si]);
            for (size_t j = 0; j < noisy.size(); j++) noisy[j] = static_cast<uint8_t>(std::clamp(std::lround(luma[side][j] + nd(rng)), 0L, 255L));
            cpu_frame_t fr{};
            const int got = cpu_dec_frame(d, noisy.data(), W, W, rb.data(), ok.data(), &fr);
            const int8_t* bi = nullptr; const float* be = nullptr;
            cpu_dec_block_stats(d, &bi, &be);
            verified[2 * si + side] += got; total[2 * si + side] += nb;
            if (bi) for (int b = 0; b < fr.total; b++) if (ok[static_cast<size_t>(b)]) its[2 * si + side] += bi[b];
          }
        }
      }
    }
    lizard_parity(&p, 0);
    printf("noise, %d frames of LIZARD-%d x %d in the %d ring on %s, read blind by the C (blocks verified, the LDPC's mean iterations a verified block):\n", frames, subch, codes, LIZARD_RING[ring], g->device().c_str());
    for (size_t si = 0; si < S; si++)
      printf("  sigma %5.1f: the C's paint %ld of %ld, %.2f; the GPU's %ld of %ld, %.2f\n", sigmas[si], verified[2 * si], total[2 * si], verified[2 * si] ? double(its[2 * si]) / verified[2 * si] : 0.0,
             verified[2 * si + 1], total[2 * si + 1], verified[2 * si + 1] ? double(its[2 * si + 1]) / verified[2 * si + 1] : 0.0);
    cpu_dec_free(d);
    lizard_free(&p);
    return 0;
  }
  if (mode == "decode" && argc >= 5) {
    const int iw = atoi(argv[3]), ih = atoi(argv[4]);
    std::vector<uint8_t> img(static_cast<size_t>(iw) * ih);
    FILE* in = fopen(argv[2], "rb");
    if (!in || fread(img.data(), 1, img.size(), in) != img.size()) { printf("FAIL: %s is not %d x %d bytes\n", argv[2], iw, ih); return 2; }
    fclose(in);
    cpu_dec_t* d = cpu_dec_new(1536);
    const int top = cpu_dec_top(d), BB = cpu_dec_block_bytes(d);
    std::vector<uint8_t> blocks(static_cast<size_t>(top) * BB), ok(top), want(XFER_PAYLOAD);
    cpu_frame_t fr{};
    const int got = cpu_dec_frame(d, img.data(), iw, ih, blocks.data(), ok.data(), &fr);
    int stream = 0;
    for (int b = 0; b < fr.total; b++) {
      if (!ok[b]) continue;
      const uint8_t* blk = blocks.data() + static_cast<size_t>(b) * BB;
      stream_fill(xfer_id_get(blk), want.data(), XFER_PAYLOAD);
      if (!memcmp(want.data(), blk + XFER_ID_BYTES, XFER_PAYLOAD)) stream++;
    }
    printf("decode: found %d, ring %d, n %d, word %d (version %d = LIZARD-%d), %d blocks verified, %d of them the test stream's, pilots %d sub-channels r %.2f %.2f\n",
           fr.found, fr.ring, fr.n, fr.word, fr.version, LIZARD_STEP * fr.version, got, stream, fr.pilot_subch, fr.pilot_r[0], fr.pilot_r[1]);
    cpu_dec_free(d);
    return got > 0 ? 0 : 1;
  }
  if (mode == "colours" && argc >= 5) {
    const int subch = atoi(argv[2]);
    const uint32_t ink = static_cast<uint32_t>(strtoul(argv[3], nullptr, 16)) & 0xffffffu, paper = static_cast<uint32_t>(strtoul(argv[4], nullptr, 16)) & 0xffffffu;
    Sender s(nullptr, 0, "");
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
#if LIZ_VK
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
#else
    printf("memory: shared with the host\ncooperative matrix: none (Metal)\n");
#endif
    return 0;
  }
  if (mode == "bench" && argc >= 3) {
    const int subch = atoi(argv[2]), frames = argc > 3 ? atoi(argv[3]) : 32;
    Sender s(nullptr, 0, "");
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
  // ai: presenter takes them, a frame each 1/fps by takeSlot on a steady carousel; each second the sender's stats line
  // ai: (paintMs: GPU ms a frame; paintBatch: frames an encode), then "pace:" with the frames late (not painted when
  // ai: due). The sender's batching A/B: LIZ_TX_BATCH=0 a frame a submission.
  if (mode == "pace" && argc >= 3) {
    const int subch = atoi(argv[2]), secs = argc > 3 ? atoi(argv[3]) : 10, fps = argc > 4 ? atoi(argv[4]) : 60, codes = argc > 5 ? atoi(argv[5]) : 1;
    // ai: LIZ_PACE_FILE=<file>: a file sent in place of the test stream (2026-11-16), the process's CPU said: until every
    // ai: source block is made (from the configure), then a frame (LIZ_TX_RQ=cpu the CPU's encoders, else the GPU's); the test
// ai: stream's a frame from 2 s on (the painting alone)
    const char* pf = getenv("LIZ_PACE_FILE");
    std::unique_ptr<Mapped> pm;
    if (pf) pm = std::make_unique<Mapped>(pf);
    const auto cpuMs = [] { rusage r{}; getrusage(RUSAGE_SELF, &r); return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1e3 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e3; };
    Sender s(pf ? pm->p : nullptr, pf ? pm->n : 0, pf ? std::filesystem::path(pf).filename().string() : "");
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
      printf("cpu: every source block made %.0f ms after the configure for %.0f core-ms; then %.3f core-ms a frame over %ld frames (%.1f%% of a core)\n",
             wMade - w0, cMade - c0, (cpuMs() - cMade) / after, after, (cpuMs() - cMade) / (nowMs() - wMade) * 100);
    } else if (pf) printf("cpu: not every source block made in the run\n");
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
