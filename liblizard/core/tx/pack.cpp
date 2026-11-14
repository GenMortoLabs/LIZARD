// ai: The default client's compression (pack.h): a file packed as a zstd stream where that pays, and a packed file
// ai: unpacked. sim/pack.mjs is its twin, byte for byte.
#include "pack.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "xfer_tx.h"   // ai: physicalCores

extern "C" {
#include "shim.h"
#include "xfer.h"
}

namespace fs = std::filesystem;

namespace lizard {

namespace {
const std::string PREFIX = "application/zstd; original=\"";
constexpr size_t SAMPLE = size_t{1} << 20, HEAD = 18;   // ai: a sample's slice; a frame's header at most (ZSTD_FRAMEHEADERSIZE_MAX)
constexpr double SAMPLE_MAX = 0.9, WHOLE_MAX = 0.97;

// ai: n bytes of the file at `at`, or a throw
std::vector<uint8_t> readAt(std::ifstream& f, uint64_t at, size_t n) {
  std::vector<uint8_t> b(n);
  f.clear();
  f.seekg(static_cast<std::streamoff>(at));
  f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(n));
  if (static_cast<size_t>(f.gcount()) != n) throw std::runtime_error("the file could not be read");
  return b;
}
std::vector<uint8_t> frameOf(const uint8_t* p, size_t n) {
  std::vector<uint8_t> o(lizard_zstd_bound(static_cast<uint32_t>(n)));
  const int32_t z = lizard_zstd_compress(p, static_cast<uint32_t>(n), o.data(), static_cast<uint32_t>(o.size()));
  if (z < 0) throw std::runtime_error("zstd refused a frame");
  o.resize(static_cast<size_t>(z));
  return o;
}
bool printable(const std::string& s) {
  return std::all_of(s.begin(), s.end(), [](char c) { return c >= 0x20 && c <= 0x7e; });
}
// ai: UTF-8 cut to n bytes at a character's start
std::string cutName(const std::string& name, size_t n) {
  n = std::min(n, name.size());
  while (n < name.size() && n > 0 && (static_cast<uint8_t>(name[n]) & 0xc0) == 0x80) n--;
  return name.substr(0, n);
}
}  // namespace

std::string packType(const std::string& type) {
  std::string t = PREFIX;
  for (char c : type) {
    if (c == '"' || c == '\\') t += '\\';
    t += c;
  }
  t += '"';
  return t.size() <= XFER_TYPE_MAX ? t : "";
}

bool packedType(const std::string& type, std::string& original) {
  if (type.size() <= PREFIX.size() || type.compare(0, PREFIX.size(), PREFIX) || type.back() != '"') return false;
  std::string o;
  for (size_t i = PREFIX.size(); i + 1 < type.size(); i++) {
    char c = type[i];
    if (c == '"') return false;
    if (c == '\\') {
      if (i + 2 >= type.size()) return false;
      c = type[++i];
    }
    o += c;
  }
  original = o;
  return true;
}

std::string unpackedName(const std::string& name) {
  return name.size() > 4 && name.compare(name.size() - 4, 4, ".zst") == 0 ? name.substr(0, name.size() - 4) : name;
}

Packed pack(const std::string& path, const std::string& name, const std::string& type, const std::string& dir) {
  Packed r;
  r.path = path; r.name = name; r.type = type;
  std::ifstream f(fs::u8path(path), std::ios::binary);
  if (!f) throw std::runtime_error("the file could not be read");
  f.seekg(0, std::ios::end);
  const uint64_t L = static_cast<uint64_t>(f.tellg());
  r.size = r.original = L;
  // ai: a type the header would drop (not printable ASCII) packs as none
  const std::string pt = packType(printable(type) ? type : "");
  if (!L || pt.empty()) return r;
  {
    // ai: the sample: three slices of 1 MiB (start, middle, end) as one frame, the whole file under 3 MiB
    std::vector<uint8_t> s;
    if (L <= 3 * SAMPLE) s = readAt(f, 0, static_cast<size_t>(L));
    else
      for (uint64_t at : {uint64_t{0}, L / 2 - SAMPLE / 2, L - SAMPLE}) {
        const auto b = readAt(f, at, SAMPLE);
        s.insert(s.end(), b.begin(), b.end());
      }
    if (static_cast<double>(frameOf(s.data(), s.size()).size()) > SAMPLE_MAX * static_cast<double>(s.size())) return r;
  }
  static std::atomic<uint32_t> made{0};
  const fs::path out = fs::u8path(dir) / ("pack-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                                          std::to_string(made++) + ".zst");
  std::error_code e;
  fs::create_directories(fs::u8path(dir), e);
  std::ofstream o(out, std::ios::binary | std::ios::trunc);
  if (!o) throw std::runtime_error("the packed file could not be made in " + dir);
  const uint64_t F = (L + PACK_FRAME - 1) / PACK_FRAME;
  const unsigned P = static_cast<unsigned>(std::min<uint64_t>(F, physicalCores()));
  uint64_t size = 0;
  try {
    // ai: P frames at a time, a worker each (each its own reader), written in order
    for (uint64_t b = 0; b < F; b += P) {
      const unsigned n = static_cast<unsigned>(std::min<uint64_t>(P, F - b));
      std::vector<std::vector<uint8_t>> z(n);
      std::vector<std::string> why(n);
      auto one = [&](unsigned k) {
        try {
          std::ifstream in(fs::u8path(path), std::ios::binary);
          if (!in) throw std::runtime_error("the file could not be read");
          const uint64_t at = (b + k) * PACK_FRAME;
          const auto plain = readAt(in, at, static_cast<size_t>(std::min<uint64_t>(PACK_FRAME, L - at)));
          z[k] = frameOf(plain.data(), plain.size());
        } catch (const std::exception& x) {
          why[k] = x.what();
        }
      };
      if (n == 1) one(0);
      else {
        std::vector<std::thread> ts;
        for (unsigned k = 0; k < n; k++) ts.emplace_back(one, k);
        for (auto& t : ts) t.join();
      }
      for (unsigned k = 0; k < n; k++) {
        if (!why[k].empty()) throw std::runtime_error(why[k]);
        o.write(reinterpret_cast<const char*>(z[k].data()), static_cast<std::streamsize>(z[k].size()));
        size += z[k].size();
      }
      if (!o) throw std::runtime_error("the packed file could not be written");
      if (static_cast<double>(size) > WHOLE_MAX * static_cast<double>(L)) break;   // ai: past the bar already
    }
    o.close();
    if (!o) throw std::runtime_error("the packed file could not be written");
  } catch (...) {
    o.close();
    fs::remove(out, e);
    throw;
  }
  if (static_cast<double>(size) > WHOLE_MAX * static_cast<double>(L)) {
    fs::remove(out, e);
    return r;
  }
  r.packed = true;
  r.path = out.u8string();
  r.name = name.empty() ? name : cutName(name, XFER_NAME_MAX - 4) + ".zst";
  r.type = pt;
  r.size = size;
  return r;
}

std::string unpack(const std::string& from, const std::string& to) {
  std::ifstream in(fs::u8path(from), std::ios::binary);
  if (!in) return "the packed file could not be read";
  in.seekg(0, std::ios::end);
  const uint64_t L = static_cast<uint64_t>(in.tellg());
  std::ofstream out(fs::u8path(to), std::ios::binary | std::ios::trunc);
  if (!out) return "the file could not be made";
  std::string why;
  try {
    std::vector<uint8_t> plain;
    for (uint64_t at = 0; at < L && why.empty();) {
      const size_t head = static_cast<size_t>(std::min<uint64_t>(HEAD, L - at));
      const auto h = readAt(in, at, head);
      const int32_t c = lizard_zstd_frame_content(h.data(), static_cast<uint32_t>(head));
      if (c < 0 || static_cast<size_t>(c) > PACK_FRAME_MAX) { why = "a frame that does not say its bytes, or holds more than 64 MiB"; break; }
      // ai: a frame of c bytes takes at most zstd's bound for c
      const size_t want = static_cast<size_t>(std::min<uint64_t>(lizard_zstd_bound(static_cast<uint32_t>(c)), L - at));
      const auto b = readAt(in, at, want);
      const int32_t fb = lizard_zstd_frame_bytes(b.data(), static_cast<uint32_t>(want));
      plain.resize(static_cast<size_t>(c));
      if (fb < 0 || lizard_zstd_decompress(b.data(), static_cast<uint32_t>(fb), plain.data(), static_cast<uint32_t>(c))) { why = "a frame damaged or cut short"; break; }
      out.write(reinterpret_cast<const char*>(plain.data()), static_cast<std::streamsize>(plain.size()));
      if (!out) why = "the file could not be written";
      at += static_cast<uint64_t>(fb);
    }
  } catch (const std::exception& x) {
    why = x.what();
  }
  out.close();
  if (why.empty() && !out) why = "the file could not be written";
  if (!why.empty()) {
    std::error_code e;
    fs::remove(fs::u8path(to), e);
  }
  return why;
}

}  // namespace lizard
