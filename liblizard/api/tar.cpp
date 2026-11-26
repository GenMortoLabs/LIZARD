// ai: The senders' bundle and its reader (lizard.h, the transfer's layer): a choice of files and folders as one POSIX
// ai: tar, laid out for the caller to write (the file's bytes never pass through here), and a tar read entry by entry for
// ai: a receiver unpacking one. The one copy: the apps reach it through JNI (jni_lib.cpp), the web through the wasm.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "api.hpp"

namespace liz {
namespace {

constexpr int BLOCK = 512;
constexpr int64_t MAX_OCTAL_11 = int64_t(1) << 33;   // ai: 11 octal digits: sizes and times to 8 GiB - 1 and 2242
constexpr int NAME_MAX_BYTES = 255;                  // ai: the session header's name (src/xfer.h XFER_NAME_MAX)
constexpr int64_t EXT_HEADER_MAX = int64_t(1) << 20; // ai: an extended header's body read whole, at most 1 MiB

int padOf(int64_t n) { return int((BLOCK - n % BLOCK) % BLOCK); }

// ai: UTF-8 a code point at a time: the code point at s[i], i moved past it (a byte that is no UTF-8 one U+FFFD)
uint32_t nextCp(const std::string& s, size_t& i) {
  const uint8_t b = uint8_t(s[i]);
  int len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
  if (len == 0 || i + len > s.size()) { i++; return 0xFFFD; }
  uint32_t c = len == 1 ? b : len == 2 ? (b & 0x1F) : len == 3 ? (b & 0x0F) : (b & 0x07);
  for (int k = 1; k < len; k++) {
    const uint8_t x = uint8_t(s[i + k]);
    if ((x & 0xC0) != 0x80) { i++; return 0xFFFD; }
    c = (c << 6) | (x & 0x3F);
  }
  i += len;
  return c;
}
std::vector<uint32_t> cps(const std::string& s) {
  std::vector<uint32_t> o;
  for (size_t i = 0; i < s.size();) o.push_back(nextCp(s, i));
  return o;
}
void putUtf8(std::string& o, uint32_t c) {
  if (c < 0x80) o += char(c);
  else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
  else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3F)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
}
std::string utf8Of(const std::vector<uint32_t>& c, size_t from, size_t to) {
  std::string o;
  for (size_t k = from; k < to; k++) putUtf8(o, c[k]);
  return o;
}
// ai: a path's UTF-16 code units: the order the entries take (the web's string order, the JVM's)
std::u16string utf16(const std::string& s) {
  std::u16string o;
  for (size_t i = 0; i < s.size();) {
    const uint32_t c = nextCp(s, i);
    if (c < 0x10000) o += char16_t(c);
    else { o += char16_t(0xD800 + ((c - 0x10000) >> 10)); o += char16_t(0xDC00 + ((c - 0x10000) & 0x3FF)); }
  }
  return o;
}

bool isAscii(const std::string& s) {
  for (char ch : s) if (uint8_t(ch) < 0x20 || uint8_t(ch) > 0x7e) return false;
  return true;
}

// ai: The ustar name a pax header stands in front of: the path's last max characters, each not plain ASCII a '_', its
// ai: leading slashes dropped (for a reader with no pax)
std::string asciiTail(const std::string& path, size_t max = 99) {
  std::string t;
  for (uint32_t c : cps(path)) t += (c >= 0x20 && c <= 0x7e) ? char(c) : '_';
  if (t.size() > max) t = t.substr(t.size() - max);
  size_t k = 0;
  while (k < t.size() && t[k] == '/') k++;
  return t.substr(k);
}

// ai: ustar's split of a plain ASCII path: the name field holds up to 100 bytes, the prefix up to 155 joined by a '/'
bool splitUstar(const std::string& path, std::string& prefix, std::string& name) {
  if (path.size() <= 100) { prefix.clear(); name = path; return true; }
  size_t from = std::min<size_t>(155, path.size() - 1);
  for (long i = long(path.rfind('/', from)); i > 0; i = long(path.rfind('/', size_t(i - 1)))) {
    const std::string n = path.substr(size_t(i) + 1);
    if (!n.empty() && n.size() <= 100) { prefix = path.substr(0, size_t(i)); name = n; return true; }
  }
  return false;
}

std::string octal(int64_t v, int width) {
  std::string d;
  do { d.insert(d.begin(), char('0' + (v & 7))); v >>= 3; } while (v > 0);
  if (int(d.size()) < width) d.insert(0, size_t(width) - d.size(), '0');
  return d;
}

// ai: One pax record, "<length> <key>=<value>\n", its length counting its own digits
std::string paxRecord(const std::string& key, const std::string& value) {
  const size_t rest = 1 + key.size() + 1 + value.size() + 1;
  size_t n = rest + 1;
  while (std::to_string(n).size() + rest != n) n = std::to_string(n).size() + rest;
  return std::to_string(n) + " " + key + "=" + value + "\n";
}

void field(uint8_t* h, int off, int len, const std::string& s) {
  std::memcpy(h + off, s.data(), std::min<size_t>(size_t(len), s.size()));
}
void header(std::string& out, const std::string& name, int64_t size, int64_t mtime, char type, const char* mode, const std::string& prefix) {
  uint8_t h[BLOCK] = {};
  field(h, 0, 100, name);
  field(h, 100, 8, std::string(mode) + '\0');
  field(h, 108, 8, std::string("0000000") + '\0');
  field(h, 116, 8, std::string("0000000") + '\0');
  field(h, 124, 12, octal(size, 11) + '\0');
  field(h, 136, 12, octal(mtime, 11) + '\0');
  field(h, 148, 8, "        ");   // ai: the checksum's field is spaces while the sum is taken
  h[156] = uint8_t(type);
  field(h, 257, 8, std::string("ustar\0" "00", 8));
  field(h, 345, 155, prefix);
  unsigned sum = 0;
  for (uint8_t b : h) sum += b;
  field(h, 148, 8, octal(sum, 6) + '\0' + ' ');
  out.append(reinterpret_cast<const char*>(h), BLOCK);
}

// ai: an entry's head: the pax header and its body where the path fits neither ustar field or is not plain ASCII, or
// ai: the size is past 8 GiB; then the ustar header
std::string headOf(const std::string& path, int64_t size, double mtimeSec, char type, const char* mode) {
  const int64_t mtime = int64_t(std::max(0.0, std::min(double(MAX_OCTAL_11 - 1), std::floor(mtimeSec))));
  std::string prefix, name, pax, out;
  const bool split = isAscii(path) && splitUstar(path, prefix, name);
  if (!split) pax += paxRecord("path", path);
  const bool big = size >= MAX_OCTAL_11;
  if (big) pax += paxRecord("size", std::to_string(size));
  if (!pax.empty()) {
    header(out, "PaxHeaders/" + asciiTail(path, 80), int64_t(pax.size()), mtime, 'x', "0000644", "");
    out += pax;
    out.append(size_t(padOf(int64_t(pax.size()))), '\0');
  }
  if (!split) { prefix.clear(); name = asciiTail(path); }
  header(out, name, big ? 0 : size, mtime, type, mode, prefix);
  return out;
}

std::vector<std::string> partsOf(const std::string& path) {
  std::vector<std::string> o;
  size_t i = 0;
  while (i <= path.size()) {
    size_t j = path.find('/', i);
    if (j == std::string::npos) j = path.size();
    if (j > i) o.push_back(path.substr(i, j - i));
    i = j + 1;
  }
  return o;
}
std::string joined(const std::vector<std::string>& p, size_t n) {
  std::string o;
  for (size_t k = 0; k < n; k++) { if (k) o += '/'; o += p[k]; }
  return o;
}

// ai: "a/b.txt" as "a/b (k).txt": a second file of one path (two picks of the same name from different folders)
std::string numbered(const std::string& path, int k) {
  const size_t slash = path.rfind('/') == std::string::npos ? 0 : path.rfind('/') + 1;
  const size_t dot = path.rfind('.');
  const size_t cut = dot != std::string::npos && dot > slash ? dot : path.size();
  return path.substr(0, cut) + " (" + std::to_string(k) + ")" + path.substr(cut);
}

// ai: The session header's description holds a name of 255 bytes of UTF-8 at most: the stem cut on a character so the
// ai: suffix (".tar", " and N more.tar") always ends it
std::string fitName(const std::string& stem, const std::string& suffix) {
  std::vector<uint32_t> c = cps(stem);
  size_t n = c.size();
  while (n > 0 && utf8Of(c, 0, n).size() + suffix.size() > size_t(NAME_MAX_BYTES)) n--;
  return utf8Of(c, 0, n) + suffix;
}

struct Item {
  std::string path;
  bool dir;
  int64_t size;
  double mtime;   // ai: NaN none
};
struct Entry {
  std::string path;   // ai: a folder's without its trailing '/'
  std::u16string key;
  bool dir;
  int item;           // ai: a file's item, -1 a folder
  int64_t size;
  double mtime;
};

}  // namespace
}  // namespace liz

struct liz_bundle {
  std::vector<liz::Item> items;
  std::vector<liz::Entry> entries;
  std::string name, head;
  liz_bundle_info info{0, -1, 0, 0};
};

struct liz_untar {
  enum Phase { HEADER, BODY, PAD, DONE } phase = HEADER;
  int64_t want = liz::BLOCK;
  int zeros = 0;
  char bodyType = 0;
  int64_t bodySize = 0;
  std::optional<std::string> paxPath, longName;
  std::optional<int64_t> paxSize;
  std::string path;
  bool dir = false;
  int64_t size = 0;
};

using namespace liz;

namespace {

// ai: The choice as one payload (the senders' rule, lizard.h)
void make(liz_bundle& b, double now) {
  need(!b.items.empty(), "nothing chosen");
  std::vector<Item> all = b.items;
  std::vector<std::string> used;
  for (Item& e : all) {
    if (e.dir) continue;
    std::string p = e.path;
    for (int k = 2; std::find(used.begin(), used.end(), p) != used.end(); k++) p = numbered(e.path, k);
    used.push_back(p);
    e.path = p;
  }
  std::vector<int> files;
  for (size_t i = 0; i < all.size(); i++) if (!all[i].dir) files.push_back(int(i));
  b.info = liz_bundle_info{0, -1, int(files.size()), 0};
  b.entries.clear();
  if (b.items.size() == 1 && files.size() == 1 && b.items[0].path.find('/') == std::string::npos) {
    b.info.single = 0;
    b.info.size = b.items[0].size;
    b.name = b.items[0].path;
    return;
  }
  // ai: each file's folders, and each folder chosen, in the order first named; a chosen folder's own time
  std::vector<Entry> dirs;
  for (const Item& e : b.items) {
    const std::vector<std::string> parts = partsOf(e.path);
    const size_t last = e.dir ? parts.size() + 1 : parts.size();
    for (size_t i = 1; i < last; i++) {
      const std::string p = joined(parts, i);
      if (std::none_of(dirs.begin(), dirs.end(), [&](const Entry& d) { return d.path == p; }))
        dirs.push_back(Entry{p, {}, true, -1, 0, e.dir && i == parts.size() ? e.mtime : NAN});
    }
  }
  // ai: a folder no time was given for (a picked folder's, which a browser does not list) takes the newest of what it
  // ai: holds, an empty one the newest of the choice, so the same choice makes the same archive, and so the same transfer
  auto newest = [&](const std::string* under) {
    double t = NAN;
    for (int f : files) {
      const Item& x = all[size_t(f)];
      if (std::isnan(x.mtime)) continue;
      if (under && x.path.compare(0, under->size() + 1, *under + "/") != 0) continue;
      if (std::isnan(t) || x.mtime > t) t = x.mtime;
    }
    return t;
  };
  for (Entry& d : dirs) if (std::isnan(d.mtime)) { d.mtime = newest(&d.path); if (std::isnan(d.mtime)) d.mtime = newest(nullptr); }
  std::vector<Entry> ordered = dirs;
  for (int f : files) ordered.push_back(Entry{all[size_t(f)].path, {}, false, f, all[size_t(f)].size, all[size_t(f)].mtime});
  for (Entry& e : ordered) e.key = utf16(e.path);
  std::stable_sort(ordered.begin(), ordered.end(), [](const Entry& x, const Entry& y) { return x.key < y.key; });
  for (Entry& e : ordered) {
    need(!e.path.empty() && e.path[0] != '/', "tar path must be relative: " + e.path);
    if (std::isnan(e.mtime)) e.mtime = now;
  }
  // ai: named after its one top folder or file, else its first and how many more
  std::vector<std::string> tops;
  for (const Item& e : all) {
    const std::vector<std::string> p = partsOf(e.path);
    const std::string t = p.empty() ? "" : p[0];
    if (std::find(tops.begin(), tops.end(), t) == tops.end()) tops.push_back(t);
  }
  b.name = tops.size() == 1 ? fitName(tops[0], ".tar") : fitName(tops[0], " and " + std::to_string(tops.size() - 1) + " more.tar");
  int64_t size = LIZ_TAR_END;
  for (const Entry& e : ordered) {
    size += int64_t(headOf(e.dir ? e.path + "/" : e.path, e.size, e.mtime, e.dir ? '5' : '0', e.dir ? "0000755" : "0000644").size());
    if (!e.dir) size += e.size + padOf(e.size);
  }
  b.entries = std::move(ordered);
  b.info.entries = int(b.entries.size());
  b.info.size = size;
}

// ai: the reader's parts of a header: a field's text up to its first NUL; trim as the JVM's (every char at most ' ')
std::string cstr(const uint8_t* h, int off, int len) {
  int end = off;
  while (end < off + len && h[end] != 0) end++;
  return std::string(reinterpret_cast<const char*>(h + off), size_t(end - off));
}
std::string trim(const std::string& s) {
  size_t a = 0, z = s.size();
  while (a < z && uint8_t(s[a]) <= ' ') a++;
  while (z > a && uint8_t(s[z - 1]) <= ' ') z--;
  return s.substr(a, z - a);
}
// ai: a number in base 8 or 10 with an optional sign (the JVM's toLongOrNull), none where it is not one
std::optional<int64_t> number(const std::string& s, int base) {
  size_t i = 0;
  bool neg = false;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) { neg = s[i] == '-'; i++; }
  if (i == s.size()) return std::nullopt;
  int64_t v = 0;
  for (; i < s.size(); i++) {
    const int d = s[i] - '0';
    if (d < 0 || d >= base) return std::nullopt;
    v = v * base + d;
  }
  return neg ? -v : v;
}
bool checksumOk(const uint8_t* h) {
  int64_t sum = 0;
  for (int i = 0; i < BLOCK; i++) sum += (i >= 148 && i < 156) ? 32 : h[i];
  const std::optional<int64_t> v = number(trim(cstr(h, 148, 8)), 8);
  return v && *v == sum;
}
int64_t octalField(const uint8_t* h, int off, int len) {
  if (h[off] & 0x80) {   // ai: GNU's base-256 form for sizes past 8 GiB
    int64_t v = 0;
    for (int i = off + 1; i < off + len; i++) v = (v << 8) | h[i];
    return v;
  }
  std::string t = trim(cstr(h, off, len));
  const std::optional<int64_t> v = number(t.empty() ? "0" : t, 8);
  if (!v) throw Error(LIZ_E_FORMAT, "a damaged tar header");
  return *v;
}
std::map<std::string, std::string> parsePax(const std::string& body) {
  std::map<std::string, std::string> out;
  size_t i = 0;
  while (i < body.size()) {
    const size_t sp = body.find(' ', i);
    if (sp == std::string::npos) break;
    const std::optional<int64_t> len = number(body.substr(i, sp - i), 10);
    if (!len || *len <= 0 || i + size_t(*len) > body.size() || *len > INT32_MAX) break;
    const size_t end = i + size_t(*len) - 1;   // ai: the record less its newline
    if (end < sp + 1) break;
    const std::string rec = body.substr(sp + 1, end - sp - 1);
    const size_t eq = rec.find('=');
    if (eq != std::string::npos && eq > 0) out[rec.substr(0, eq)] = rec.substr(eq + 1);
    i += size_t(*len);
  }
  return out;
}
// ai: An entry's path as its parts: relative, no "..", no empty parts, no drive; refused otherwise rather than cleaned up
std::string checkedPath(const std::string& path) {
  const std::string p = path.compare(0, 2, "./") == 0 ? path.substr(2) : path;
  std::vector<std::string> parts;
  for (const std::string& x : partsOf(p)) if (x != ".") parts.push_back(x);
  bool drive = false;
  if (!parts.empty()) {
    const std::u16string first = utf16(parts[0]);
    drive = first.size() == 2 && first[1] == u':';
  }
  if (!path.empty() && path[0] == '/') drive = true;
  if (drive || parts.empty() || std::find(parts.begin(), parts.end(), "..") != parts.end())
    throw Error(LIZ_E_FORMAT, "tar path refused: " + path);
  return joined(parts, parts.size());
}

int take(liz_untar& u, const uint8_t* bytes, size_t n) {
  if (u.phase == liz_untar::DONE) return LIZ_UNTAR_END;
  if (u.phase == liz_untar::PAD) {
    if (int64_t(n) < u.want) throw Error(LIZ_E_FORMAT, "a tar cut short");
    u.phase = liz_untar::HEADER;
    u.want = BLOCK;
    return LIZ_UNTAR_MORE;
  }
  if (u.phase == liz_untar::BODY) {
    if (int64_t(n) < u.want) throw Error(LIZ_E_FORMAT, "a tar cut short");
    const std::string body(reinterpret_cast<const char*>(bytes), size_t(u.bodySize));
    if (u.bodyType == 'x') {
      const std::map<std::string, std::string> m = parsePax(body);
      if (m.count("path")) u.paxPath = m.at("path");
      if (m.count("size")) { const std::optional<int64_t> s = number(m.at("size"), 10); if (s) u.paxSize = *s; }
    } else if (u.bodyType == 'L') {
      std::string t = body;
      while (!t.empty() && t.back() == '\0') t.pop_back();
      u.longName = t;
    }
    u.phase = liz_untar::HEADER;
    u.want = BLOCK;
    return LIZ_UNTAR_MORE;
  }
  // ai: a header
  if (n == 0) { u.phase = liz_untar::DONE; u.want = 0; return LIZ_UNTAR_END; }
  if (n < size_t(BLOCK)) throw Error(LIZ_E_FORMAT, "a tar cut short");
  const uint8_t* h = bytes;
  if (std::all_of(h, h + BLOCK, [](uint8_t b) { return b == 0; })) {
    if (++u.zeros == 2) { u.phase = liz_untar::DONE; u.want = 0; return LIZ_UNTAR_END; }
    return LIZ_UNTAR_MORE;
  }
  u.zeros = 0;
  if (!checksumOk(h)) throw Error(LIZ_E_FORMAT, "a damaged tar header");
  const char type = char(h[156]);
  const int64_t size = u.paxSize ? *u.paxSize : octalField(h, 124, 12);
  const std::string name = cstr(h, 0, 100), prefix = cstr(h, 257, 6) == "ustar" ? cstr(h, 345, 155) : "";
  const std::string path = u.paxPath ? *u.paxPath : u.longName ? *u.longName : prefix.empty() ? name : prefix + "/" + name;
  if (type == 'x' || type == 'g' || type == 'L') {
    if (size > EXT_HEADER_MAX) throw Error(LIZ_E_FORMAT, "a tar extended header of " + std::to_string(size) + " bytes");
    u.bodyType = type;
    u.bodySize = std::max<int64_t>(0, size);
    u.want = u.bodySize + padOf(u.bodySize);
    u.phase = liz_untar::BODY;
    if (u.want == 0) return take(u, bytes, 0);
    return LIZ_UNTAR_MORE;
  }
  if (type == '0' || type == '\0' || type == '7' || type == '5') {
    u.path = checkedPath(path);
    u.dir = type == '5' || (!path.empty() && path.back() == '/');
    u.size = u.dir ? 0 : size;
    const int pad = u.dir ? (type == '5' ? padOf(size) : 0) : padOf(size);
    u.paxPath.reset();
    u.paxSize.reset();
    u.longName.reset();
    u.phase = pad ? liz_untar::PAD : liz_untar::HEADER;
    u.want = pad ? pad : BLOCK;
    return LIZ_UNTAR_ENTRY;
  }
  std::string t;
  putUtf8(t, uint8_t(type));
  throw Error(LIZ_E_FORMAT, "tar entry '" + t + "' at " + path + ": only files and folders unpack");
}

}  // namespace

extern "C" {

LIZ_API liz_bundle* liz_bundle_new(void) { return guardNew<liz_bundle>([] { return new liz_bundle(); }); }
LIZ_API void liz_bundle_free(liz_bundle* b) { delete b; }

LIZ_API int liz_bundle_add(liz_bundle* b, const char* path, int dir, int64_t size, double mtime) {
  return guard([&] {
    need(b && path, "a bundle and a path");
    need(size >= 0 && (!dir || size == 0), "a file's size is 0 or more, a folder's 0");
    b->items.push_back(Item{path, dir != 0, size, mtime});
    b->entries.clear();
    b->name.clear();
    return int(b->items.size()) - 1;
  });
}

LIZ_API int liz_bundle_make(liz_bundle* b, double now, liz_bundle_info* out) {
  return guard([&] {
    need(b, "a bundle");
    make(*b, now);
    if (out) *out = b->info;
    return LIZ_OK;
  });
}

LIZ_API const char* liz_bundle_name(liz_bundle* b) { return b ? b->name.c_str() : ""; }

LIZ_API const uint8_t* liz_bundle_entry_get(liz_bundle* b, int e, liz_bundle_entry* out) {
  const uint8_t* p = nullptr;
  guard([&] {
    need(b && out && e >= 0 && e < int(b->entries.size()), "an entry of a made bundle");
    const Entry& x = b->entries[size_t(e)];
    b->head = headOf(x.dir ? x.path + "/" : x.path, x.size, x.mtime, x.dir ? '5' : '0', x.dir ? "0000755" : "0000644");
    *out = liz_bundle_entry{x.dir ? 0 : x.size, x.item, x.dir ? 0 : padOf(x.size), int(b->head.size())};
    p = reinterpret_cast<const uint8_t*>(b->head.data());
    return LIZ_OK;
  });
  return p;
}

LIZ_API liz_untar* liz_untar_new(void) { return guardNew<liz_untar>([] { return new liz_untar(); }); }
LIZ_API void liz_untar_free(liz_untar* u) { delete u; }
LIZ_API int64_t liz_untar_want(const liz_untar* u) { return u ? u->want : 0; }

LIZ_API int liz_untar_take(liz_untar* u, const uint8_t* bytes, size_t n) {
  return guard([&] {
    need(u && (bytes || n == 0), "a reader and its bytes");
    return take(*u, bytes, n);
  });
}

LIZ_API const char* liz_untar_entry(const liz_untar* u, int* dir, int64_t* size) {
  if (!u) return "";
  if (dir) *dir = u->dir ? 1 : 0;
  if (size) *size = u->size;
  return u->path.c_str();
}

}  // extern "C"
