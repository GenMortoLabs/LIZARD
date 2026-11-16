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

std::string Unpacker::open(const std::string& path) {
  in_.open(fs::u8path(path), std::ios::binary);
  if (!in_) return "the packed file could not be read";
  in_.seekg(0, std::ios::end);
  const uint64_t L = static_cast<uint64_t>(in_.tellg());
  try {
    for (uint64_t at = 0; at < L;) {
      const size_t head = static_cast<size_t>(std::min<uint64_t>(HEAD, L - at));
      const auto h = readAt(in_, at, head);
      int32_t sum = 0;
      const int32_t c = lizard_zstd_frame_content(h.data(), static_cast<uint32_t>(head));
      const int32_t hb = lizard_zstd_frame_head(h.data(), static_cast<uint32_t>(head), &sum);
      if (c < 0 || hb < 0 || static_cast<size_t>(c) > PACK_FRAME_MAX) return "a frame that does not say its bytes, or holds more than 64 MiB";
      // ai: the blocks (RFC 8878 3.1.1.2): a 3-byte header, little-endian: bit 0 the last, bits 1 and 2 the type (raw,
      // ai: RLE, compressed, reserved), the rest the size; an RLE block's content is its one byte
      uint64_t p = at + static_cast<uint64_t>(hb);
      for (bool last = false; !last;) {
        if (p + 3 > L) return "a frame cut short";
        const auto b = readAt(in_, p, 3);
        const uint32_t w = uint32_t{b[0]} | uint32_t{b[1]} << 8 | uint32_t{b[2]} << 16;
        const uint32_t type = w >> 1 & 3;
        if (type == 3) return "a frame damaged";
        last = (w & 1) != 0;
        p += 3 + (type == 1 ? 1 : w >> 3);
      }
      if (sum) p += 4;
      if (p > L) return "a frame cut short";
      if (p - at > lizard_zstd_bound(static_cast<uint32_t>(c))) return "a frame damaged";
      frames_.push_back({at, static_cast<uint32_t>(p - at), static_cast<uint32_t>(c)});
      total_ += static_cast<uint64_t>(c);
      at = p;
    }
  } catch (const std::exception& x) {
    return x.what();
  }
  return "";
}

const uint8_t* Unpacker::take(size_t& n) {
  while (pos_ == plain_.size()) {
    if (next_ == frames_.size()) {
      n = 0;
      return nullptr;
    }
    const Frame& f = frames_[next_++];
    const auto z = readAt(in_, f.at, f.bytes);
    plain_.resize(f.content);
    pos_ = 0;
    if (lizard_zstd_decompress(z.data(), f.bytes, plain_.data(), f.content)) {
      plain_.clear();
      next_--;   // ai: the same frame again at a next call, which throws again: never a file cut short in silence
      throw std::runtime_error("a frame damaged or cut short");
    }
  }
  n = std::min(n, plain_.size() - pos_);
  const uint8_t* p = plain_.data() + pos_;
  pos_ += n;
  return p;
}

std::string unpack(const std::string& from, const std::string& to) {
  Unpacker u;
  std::string why = u.open(from);
  if (!why.empty()) return why;
  std::ofstream out(fs::u8path(to), std::ios::binary | std::ios::trunc);
  if (!out) return "the file could not be made";
  try {
    for (;;) {
      size_t n = PACK_FRAME;
      const uint8_t* p = u.take(n);
      if (!n) break;
      out.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
      if (!out) {
        why = "the file could not be written";
        break;
      }
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
