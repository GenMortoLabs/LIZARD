// ai: lizard_stream (2026-11-18): stream files, a sender's frames painted ahead into a .lizard ("LIZARD stream 1",
// ai: core/rx/lizfile.h) that a player shows at its fps without painting, on the C API (liz_sender_export) and the
// ai: .lizard reader.
// ai:   lizard_stream export FILE OUT.lizard [--blocks N] [--ring 0|1|2] [--fps F] [--codes 1|2] [--frames N | --cycles L]
// ai:                 [--painter auto|gpu|cpu] [--seed S] [--assets DIR] [--level L]
// ai:     FILE's frames (the default: LIZARD-480 in the 128 ring at 15 a second, one code, 2.5 cycles of the file's
// ai:     blocks); the count made odd, so a camera taking every other frame sees every frame across two cycles of the loop;
// ai:     packed by zstd at L (1 by default) on every core (sender.h exportStart: the GPU's frames drained, or each CPU
// ai:     painter packing its own).
// ai:     A line each tenth: frames, ms a frame; the file's bytes and a frame's raw size at the end.
// ai:   lizard_stream check OUT.lizard [--threads N]
// ai:     every frame read by the C (liz_decode on each entry's centre square, a 2:1 picture's two squares their own
// ai:     entries, nothing held) into a transfer, three times: all
// ai:     the frames, the even ones alone, the odd ones alone (a camera at half the rate in step with the loop): whether
// ai:     each makes the file whole (its BLAKE3 root verified) and after how many frames; the read rate a player needs at
// ai:     the stream's fps. Exit 0 where all three make it whole.
// ai:   lizard_stream play OUT.lizard [--frames N] [--threads N] [--assets DIR]
// ai:     the stream played (liz_sender_play: unpacked into the frame ring, taken as a screen takes them), N frames (the
// ai:     default: a loop and a half), each held to the file's own picture at its place in the loop, byte for byte (at
// ai:     2:1 its two squares and the gap between); the frames a second the player takes. Exit 0 where every one is.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "lizard.h"
#include "lizfile.h"

namespace {

int fail(const std::string& why) { std::fprintf(stderr, "lizard_stream: %s\n", why.c_str()); return 1; }

std::string arg(int argc, char** argv, const char* name, const char* dflt) {
  for (int i = 1; i + 1 < argc; i++) if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  return dflt;
}

// ai: a meta.json number by its key (the stream's own writer wrote it: no general JSON needed)
double metaNum(const std::string& m, const std::string& key, double dflt) {
  const size_t p = m.find("\"" + key + "\":");
  return p == std::string::npos ? dflt : std::atof(m.c_str() + p + key.size() + 3);
}

int doExport(int argc, char** argv) {
  const std::string in = argv[2], out = argv[3];
  std::ifstream f(in, std::ios::binary);
  if (!f) return fail("cannot read " + in);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const std::string name = in.substr(in.find_last_of("/\\") + 1);
  liz_format fmt{std::atoi(arg(argc, argv, "--blocks", "30").c_str()), std::atoi(arg(argc, argv, "--ring", "1").c_str()),
                 std::atoi(arg(argc, argv, "--fps", "15").c_str()), std::atoi(arg(argc, argv, "--codes", "1").c_str())};
  const std::string painter = arg(argc, argv, "--painter", "auto"), assets = arg(argc, argv, "--assets", LIZ_ASSETS);
  liz_sender* s = liz_sender_new(data.data(), data.size(), name.c_str(), LIZ_TX_COPY);
  if (!s) return fail(std::string("sender: ") + liz_last_error());
  const std::string seed = arg(argc, argv, "--seed", "");
  if (!seed.empty() && liz_sender_seed(s, uint32_t(std::stoul(seed))) != LIZ_OK) return fail(std::string("seed: ") + liz_last_error());
  // ai: configured (the source blocks are made from then), every source block made, then configured again: the frames painted ahead
  // ai: meanwhile (the session header alone) dropped, so the stream's first frame carries data; the cycle known
  const int p = painter == "gpu" ? LIZ_PAINTER_GPU : painter == "cpu" ? LIZ_PAINTER_CPU : LIZ_PAINTER_AUTO;
  const char* as = assets.c_str();   // ai: the frame ring is the GPU painter's, whatever paints
  if (liz_sender_configure(s, &fmt, p, 0, as, nullptr) != LIZ_OK) return fail(std::string("configure: ") + liz_last_error());
  int64_t cycle = 0;
  const auto tw = std::chrono::steady_clock::now();
  while ((cycle = liz_sender_cycle(s)) <= 0) {
    if (std::chrono::steady_clock::now() - tw > std::chrono::minutes(30)) return fail("the file's source blocks were not made in 30 minutes");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::printf("source blocks made in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count());
  if (liz_sender_configure(s, &fmt, p, 0, as, nullptr) != LIZ_OK) return fail(std::string("configure: ") + liz_last_error());
  liz_geometry g{};
  liz_sender_geometry(s, &g);
  int frames = std::atoi(arg(argc, argv, "--frames", "0").c_str());
  if (frames <= 0) {
    const double cycles = std::atof(arg(argc, argv, "--cycles", "2.5").c_str());   // ai: each half (every other frame) 1.25 cycles: whole for a camera in step at half the rate
    frames = int(std::ceil(cycles * double(cycle) / g.frame_blocks));
  }
  frames |= 1;
  std::printf("export: %s (%zu B) as LIZARD-%d, ring %d, %d fps, %d code%s, %d x %d frames, %d of them\n", name.c_str(), data.size(), 16 * fmt.blocks, fmt.ring,
              fmt.fps, fmt.codes, fmt.codes > 1 ? "s" : "", g.width, g.height, frames);
  struct P { std::chrono::steady_clock::time_point t0; int last; } st{std::chrono::steady_clock::now(), 0};
  const int level = std::atoi(arg(argc, argv, "--level", "1").c_str());
  const int rc = liz_sender_export(s, out.c_str(), frames, level, [](void* u, int done, int all) {
    auto* q = static_cast<P*>(u);
    if (done * 10 / all > q->last || done == all) {
      q->last = done * 10 / all;
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - q->t0).count();
      std::printf("  %d of %d frames, %.2f ms a frame\n", done, all, ms / done);
      std::fflush(stdout);
    }
    return 1;
  }, &st);
  if (rc < 0) return fail(std::string("export: ") + liz_last_error());
  std::ifstream o(out, std::ios::binary | std::ios::ate);
  const double bytes = double(o.tellg()), raw = double(g.width) * g.height;
  std::printf("wrote %s: %.1f MB, %.0f%% of the raw frames (%.2f MB a frame); a player reads %.1f MB/s raw at %d fps\n", out.c_str(), bytes / 1e6,
              100.0 * bytes / (raw * frames), raw / 1e6, raw * fmt.fps / 1e6, fmt.fps);
  liz_sender_free(s);
  return 0;
}

int doPlay(int argc, char** argv) {
  liz_sender* s = liz_sender_play(argv[2]);
  if (!s) return fail(std::string("play: ") + liz_last_error());
  const int threads = std::max(1, std::atoi(arg(argc, argv, "--threads", "3").c_str()));
  const std::string assets = arg(argc, argv, "--assets", LIZ_ASSETS);   // ai: the frame ring is the GPU painter's, whatever paints
  if (liz_sender_configure(s, nullptr, LIZ_PAINTER_CPU, threads, assets.c_str(), nullptr) != LIZ_OK) return fail(std::string("configure: ") + liz_last_error());
  liz_geometry g{};
  liz_sender_geometry(s, &g);
  std::string why;
  auto r = lizard::lizfile::Reader::open(argv[2], why);
  if (!r) return fail(why);
  const std::string meta = liz_sender_stream(s);
  const int codes = std::max(1, int(metaNum(meta, "codes", 1))), gap = int(metaNum(meta, "gap", 0));
  const int pics = int(metaNum(meta, "pictures", double(r->frames().size() / size_t(codes))));
  int frames = std::atoi(arg(argc, argv, "--frames", "0").c_str());
  if (frames <= 0) frames = pics + pics / 2 + 1;
  std::printf("play: %d pictures of %d x %d, %d code%s, %d frames taken\n", pics, g.width, g.height, codes, codes > 1 ? "s" : "", frames);
  std::vector<uint8_t> got(size_t(g.width) * g.height), part;
  long bad = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < frames; i++) {
    const auto tw = std::chrono::steady_clock::now();
    while (liz_sender_take(s, got.data(), g.width, LIZ_GREY8) != 1) {
      if (std::chrono::steady_clock::now() - tw > std::chrono::seconds(10)) return fail("no frame for 10 s at frame " + std::to_string(i) + ": " + liz_sender_stats(s));
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    const int pic = i % pics;
    long off = 0;
    for (int k = 0; k < codes; k++) {
      if (!r->frame(size_t(pic) * codes + k, part, why)) return fail(why);
      const int side = g.height, x0 = k * (side + gap);
      for (int y = 0; y < side; y++) {
        off += std::memcmp(got.data() + size_t(y) * g.width + x0, part.data() + size_t(y) * side, side) != 0;
        if (k + 1 < codes) for (int x = 0; x < gap; x++) off += got[size_t(y) * g.width + x0 + side + x] != part[size_t(y) * side + side - 1];
      }
    }
    if (off) { bad++; if (bad <= 3) std::printf("  frame %d (picture %d): %ld rows apart\n", i, pic, off); }
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  %d frames, %ld not the file's; taken at %.0f a second (%d unpacking threads); stats %s\n", frames, bad, frames / secs, threads, liz_sender_stats(s));
  liz_sender_free(s);
  return bad ? 2 : 0;
}

int doCheck(int argc, char** argv) {
  std::string why;
  auto r = lizard::lizfile::Reader::open(argv[2], why);
  if (!r) return fail(why);
  if (r->kind() != lizard::lizfile::Kind::Stream) std::printf("(a replay, not a stream: read all the same)\n");
  const size_t n = r->frames().size();
  const int threads = std::max(1, std::atoi(arg(argc, argv, "--threads", "8").c_str()));
  // ai: a 2:1 stream keeps a picture's two squares as two entries: each decoded whole, the pair fed together
  const int pics = int(metaNum(r->meta(), "pictures", double(n)));
  const size_t per = pics > 0 && size_t(pics) * 2 == n ? 2 : 1;
  // ai: every frame decoded once, on the threads (each its own decoder; nothing held, as a receiver joining anywhere)
  std::vector<std::vector<uint8_t>> blocks(n);
  std::vector<int> counts(n, 0);
  std::atomic<size_t> next{0};
  std::atomic<bool> bad{false};
  std::string err;
  std::vector<std::thread> pool;
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < threads; t++) pool.emplace_back([&] {
    std::string w;
    auto rd = lizard::lizfile::Reader::open(argv[2], w);
    liz_decoder* d = liz_decoder_new(0);
    std::vector<uint8_t> luma, ver(size_t(liz_decoder_max_blocks(d)) * LIZ_BLOCK);
    for (size_t i; (i = next++) < n;) {
      if (!rd || !rd->frame(i, luma, w)) { bad = true; err = w; break; }
      const auto& fi = rd->frames()[i];
      int rect[2][4];
      const int k = liz_layout_rects(int(fi.w), int(fi.h), 1, rect);
      for (int j = 0; j < k; j++) {
        const int got = liz_decode(d, luma.data() + size_t(rect[j][1]) * fi.w + rect[j][0], rect[j][2], rect[j][3], int(fi.w), LIZ_GREY8, ver.data(), nullptr);
        if (got > 0) { blocks[i].insert(blocks[i].end(), ver.begin(), ver.begin() + size_t(got) * LIZ_BLOCK); counts[i] += got; }
      }
    }
    liz_decoder_free(d);
  });
  for (auto& t : pool) t.join();
  if (bad) return fail("reading the frames: " + err);
  const double decMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  long total = 0;
  for (int c : counts) total += c;
  const double fps = metaNum(r->meta(), "fps", 15), fb = metaNum(r->meta(), "frame_blocks", 0);
  std::printf("check: %zu pictures, %ld blocks verified (%.1f a picture%s), decoded in %.1f s on %d threads\n", n / per, total, double(total) * per / std::max<size_t>(1, n),
              fb > 0 ? (", of " + std::to_string(int(fb))).c_str() : "", decMs / 1000, threads);
  // ai: the transfer fed the frames of each pass in order, in memory
  int whole = 0;
  for (int pass = 0; pass < 3; pass++) {
    liz_store st{nullptr, 0};
    liz_rx* rx = liz_rx_new(&st);
    liz_progress pr{};
    size_t at = 0;
    long seen = 0;
    for (size_t pic = (pass == 2 ? 1 : 0); pic * per < n; pic += (pass == 0 ? 1 : 2)) {
      liz_verdict v{};
      seen++;
      for (size_t i = pic * per; i < pic * per + per; i++)
        if (counts[i]) liz_rx_frame(rx, blocks[i].data(), counts[i], &v);
      liz_rx_progress(rx, &pr);
      if (pr.done) { at = size_t(seen); break; }
    }
    const char* what[3] = {"all the frames", "the even frames alone", "the odd frames alone"};
    if (pr.done) { whole++; std::printf("  %-22s the file whole and verified after %zu of them (%.1f s at %.0f fps)\n", what[pass], at, at / fps, fps); }
    else std::printf("  %-22s NOT whole: %.1f%% of %llu B in\n", what[pass], 100.0 * pr.fraction, (unsigned long long)pr.length);
    liz_rx_free(rx);
  }
  return whole == 3 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "export" && argc >= 4) return doExport(argc, argv);
  if (cmd == "check" && argc >= 3) return doCheck(argc, argv);
  if (cmd == "play" && argc >= 3) return doPlay(argc, argv);
  std::fprintf(stderr, "lizard_stream export FILE OUT.lizard [--blocks N] [--ring 0|1|2] [--fps F] [--codes 1|2] [--frames N | --cycles L]\n"
                       "              [--painter auto|gpu|cpu] [--seed S] [--assets DIR] [--level L]\n"
                       "lizard_stream check OUT.lizard [--threads N]\n"
                       "lizard_stream play OUT.lizard [--frames N] [--threads N] [--assets DIR]\n");
  return 2;
}
