// ai: zstd_check (2026-11-18): stream files' frames packed on the GPU (tx/gpu_zstd.h), held to the CPU twin and to
// ai: libzstd, and timed.
// ai:   zstd_check FILE.lizard [--frames N] [--batch F] [--level L] [--assets DIR (liblizard/out, from the root)] [--device NAME]
// ai: The file's first N pictures (48 unless named; a stream's squares, 2:1 its two side by side as the sender's ring
// ai: holds them, the gap the margin's byte as the painter lays it) put in a ring on the device, packed a batch of F
// ai: frames at a time (0: as many as the device's share holds); each square's frame: the same bytes as packCpu's,
// ai: decoded by libzstd (lizfile's unpack: zstd, then the predictor undone) back to its pixels exactly. Then the bytes
// ai: against libzstd at level L (1) on one core, and the ms a frame each way. Exit 0 where every frame holds.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpu_zstd.h"
#include "lizfile.h"
#include "wg.h"
#include "json.hpp"

namespace {
std::string arg(int argc, char** argv, const char* name, const char* dflt) {
  for (int i = 1; i + 1 < argc; i++) if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  return dflt;
}
double ms(std::chrono::steady_clock::time_point a) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); }
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "zstd_check FILE.lizard [--frames N] [--batch F] [--level L] [--assets DIR] [--device NAME]\n"); return 2; }
  std::string why;
  auto r = lizard::lizfile::Reader::open(argv[1], why);
  if (!r) { std::fprintf(stderr, "zstd_check: %s\n", why.c_str()); return 1; }
  const auto m = nlohmann::json::parse(r->meta());
  const int codes = m.value("format", nlohmann::json::object()).value("codes", 1), gap = m.value("gap", 0);
  const int side = int(r->frames()[0].w);
  const int pics = std::min<int>(std::atoi(arg(argc, argv, "--frames", "48").c_str()), int(r->frames().size()) / codes);
  const int W = codes * side + (codes - 1) * gap, rw = (W + 3) / 4;
  const uint32_t frameWords = uint32_t(rw) * uint32_t(side);
  // ai: the pictures as the ring holds them
  std::vector<uint8_t> ring(size_t(frameWords) * 4 * pics, 0), part;
  for (int p = 0; p < pics; p++)
    for (int k = 0; k < codes; k++) {
      if (!r->frame(size_t(p) * codes + k, part, why)) { std::fprintf(stderr, "zstd_check: %s\n", why.c_str()); return 1; }
      for (int y = 0; y < side; y++) {
        uint8_t* o = ring.data() + size_t(p) * frameWords * 4 + size_t(y) * rw * 4 + size_t(k) * (side + gap);
        std::memcpy(o, part.data() + size_t(y) * side, side);
        if (k + 1 < codes) std::memset(o + side, part[size_t(y) * side + side - 1], gap);
      }
    }
  auto dev = wg::Device::create(arg(argc, argv, "--device", ""));
  auto z = lizard::GpuZstd::create(*dev, arg(argc, argv, "--assets", "liblizard/out"), side);
  const int batch = std::max(1, std::min(std::atoi(arg(argc, argv, "--batch", "0").c_str()) > 0 ? std::atoi(arg(argc, argv, "--batch", "0").c_str()) : z->squares() / codes,
                                         z->squares() / codes));
  auto rb = dev->createBuffer(ring.size(), wg::STORAGE | wg::COPY_DST, "check ring", false);
  dev->upload(*rb, 0, ring.data(), ring.size());
  std::mutex mu;
  std::printf("zstd_check: %d pictures of %d x %d (%d code%s), batches of %d frames (%d squares at most) on %s\n", pics, W, side, codes, codes > 1 ? "s" : "",
              batch, z->squares(), dev->name.c_str());
  // ai: a warm batch (the pipelines' first use), then the timed ones
  { std::vector<int> sl(size_t(std::min(batch, pics))); for (size_t i = 0; i < sl.size(); i++) sl[i] = int(i); z->pack(*rb, sl, frameWords, uint32_t(rw), codes, side + gap, mu); }
  std::vector<std::vector<uint8_t>> gpu;
  auto t0 = std::chrono::steady_clock::now();
  for (int p = 0; p < pics; p += batch) {
    std::vector<int> sl;
    for (int i = p; i < std::min(pics, p + batch); i++) sl.push_back(i);
    auto f = z->pack(*rb, sl, frameWords, uint32_t(rw), codes, side + gap, mu);
    for (auto& e : f) gpu.push_back(std::move(e));
  }
  const double gpuMs = ms(t0);
  const auto& tl = z->last();
  std::printf("  the last batch's ms: counts %.2f, codes %.2f, streams %.2f, frames %.2f\n", tl.counts, tl.codes, tl.streams, tl.frames);
  int bad = 0;
  size_t gpuBytes = 0, cpuBytes = 0;
  double cpuMs = 0, zMs = 0;
  const int level = std::atoi(arg(argc, argv, "--level", "1").c_str());
  std::vector<uint8_t> back(size_t(side) * side);
  for (int p = 0; p < pics; p++)
    for (int k = 0; k < codes; k++) {
      const size_t q = size_t(p) * codes + k;
      const uint8_t* px = ring.data() + size_t(p) * frameWords * 4 + size_t(k) * (side + gap);
      auto tc = std::chrono::steady_clock::now();
      const auto twin = lizard::GpuZstd::packCpu(px, side, size_t(rw) * 4);
      cpuMs += ms(tc);
      auto tz = std::chrono::steady_clock::now();
      const auto ref = lizard::lizfile::packFrame(px, uint32_t(side), uint32_t(side), uint32_t(rw) * 4, level);
      zMs += ms(tz);
      gpuBytes += gpu[q].size();
      cpuBytes += ref.size();
      const bool same = twin == gpu[q];
      std::string w;
      const bool dec = lizard::lizfile::unpackFrame(gpu[q].data(), gpu[q].size(), uint32_t(side), uint32_t(side), back.data(), w);
      bool exact = dec;
      for (int y = 0; exact && y < side; y++) exact = std::memcmp(back.data() + size_t(y) * side, px + size_t(y) * rw * 4, side) == 0;
      if (!same || !exact) {
        if (++bad <= 3) std::printf("  square %zu: %s, %s%s\n", q, same ? "the twin's bytes" : "NOT the twin's bytes", dec ? (exact ? "decodes exactly" : "decodes to OTHER pixels") : "does NOT decode: ", dec ? "" : w.c_str());
      }
    }
  const double raw = double(pics) * codes * side * side;
  std::printf("  %d squares: %d not the twin's or not decoding to their pixels\n", pics * codes, bad);
  std::printf("  GPU: %.1f%% of the raw bytes, %.3f ms a frame (batches, the CPU's tables and frames included)\n", 100.0 * gpuBytes / raw, gpuMs / pics);
  std::printf("  libzstd level %d on one core: %.1f%% of the raw bytes, %.3f ms a frame; the CPU twin %.3f ms a frame\n", level, 100.0 * cpuBytes / raw, zMs / pics, cpuMs / pics);
  return bad ? 2 : 0;
}
