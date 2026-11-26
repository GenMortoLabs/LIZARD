#include "build.h"

#include "codes.h"
#include "plan.h"

extern "C" {
#include "fmt.h"
#include "fmt_tables.h"
#include "lizard_phy.h"
#include "rs.h"
// ai: the ring tables' C (src/internal.h, which C++ does not read: its thread-local storage is C11's)
int ob_test_mesh_tables(const ob_layout_t* L, int32_t* dims, float* nodeX, float* nodeY, uint8_t* kind, int32_t* mark);
int ob_thin_fmt_plan(const ob_layout_t* L, int cells, float* pts, int32_t* meta, int32_t* widx);
int ob_thin_score_plan(const ob_layout_t* L, float* pts, int32_t* meta, int32_t* signs);
void ob_mark_geom(const ob_layout_t* L, float* out);
}

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace lizard::build {

// ai: ECMAScript's Number::toString(10): the shortest digits that read back (std::to_chars' shortest), then its rules
// ai: for where the point goes: plain digits for an exponent from -7 to 20, else d.ddde+x
std::string js(double v) {
  if (std::isnan(v)) return "NaN";
  if (v == 0) return "0";
  if (std::isinf(v)) return v < 0 ? "-Infinity" : "Infinity";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, std::fabs(v), std::chars_format::scientific);
  std::string s(buf, r.ptr);   // ai: d[.ddd]e[+-]xx
  const size_t e = s.find('e');
  std::string digits = s.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  const int exp10 = std::stoi(s.substr(e + 1));
  const int k = static_cast<int>(digits.size()), n = exp10 + 1;   // ai: value = 0.digits x 10^n
  std::string out;
  if (k <= n && n <= 21) out = digits + std::string(static_cast<size_t>(n - k), '0');
  else if (0 < n && n <= 21) out = digits.substr(0, static_cast<size_t>(n)) + "." + digits.substr(static_cast<size_t>(n));
  else if (-6 < n && n <= 0) out = "0." + std::string(static_cast<size_t>(-n), '0') + digits;
  else {
    out = digits.substr(0, 1) + (k > 1 ? "." + digits.substr(1) : std::string());
    out += std::string("e") + (n - 1 >= 0 ? "+" : "-") + std::to_string(std::abs(n - 1));
  }
  return (v < 0 ? "-" : "") + out;
}
std::string defU(double v) { return v == std::floor(v) && std::fabs(v) < 9007199254740992.0 ? js(v) + "u" : js(v); }
std::string lit(double v) { return v == std::floor(v) && std::fabs(v) < 9007199254740992.0 ? js(v) + ".0" : js(v); }

// ai: a recipe's key: its file, then its constants sorted by name (gen's own order is the JS module's, which a key need
// ai: not carry: a file and its constants name one module, gen checks)
static std::string keyOf(const std::string& src, std::vector<std::pair<std::string, std::string>> defs) {
  std::sort(defs.begin(), defs.end());
  std::string k = src;
  for (const auto& [n, v] : defs) k += "\n" + n + "=" + v;
  return k;
}

Modules::Modules(const json& index) {
  for (const auto& j : index.at("modules")) {
    std::vector<std::pair<std::string, std::string>> defs;
    for (const auto& [n, v] : j.at("defs").items()) defs.emplace_back(n, v.get<std::string>());
    const std::string key = keyOf(j.at("src").get<std::string>(), defs), name = j.at("name").get<std::string>();
    if (byKey_.count(key) && byKey_[key] != name) throw std::runtime_error("two modules of one recipe: " + byKey_[key] + " and " + name);
    byKey_[key] = name;
    bySrc_.emplace(j.at("src").get<std::string>(), key);
    facts_[name] = {j.at("lanes").get<uint32_t>(), j.value("needs", json::array())};
  }
}

std::string recipeKey(const Recipe& r) { return keyOf(r.src, r.defs); }

std::shared_ptr<Modules> Modules::recording() {
  std::shared_ptr<Modules> m(new Modules());
  m->record_ = true;
  return m;
}

std::string Modules::name(const Recipe& r) const {
  const std::string key = keyOf(r.src, r.defs);
  if (record_) {
    auto it = recordedKeys_.find(key);
    if (it != recordedKeys_.end()) return it->second;
    recorded_.push_back(r);
    return recordedKeys_[key] = key;
  }
  auto it = byKey_.find(key);
  if (it != byKey_.end()) return it->second;
  // ai: the nearest of the file's recipes: the one sharing the most constants, and the first line that differs
  std::string best, why;
  size_t bestSame = 0;
  auto lines = [](const std::string& k) { std::vector<std::string> v; size_t a = 0, b; while ((b = k.find('\n', a)) != std::string::npos) { v.push_back(k.substr(a, b - a)); a = b + 1; } v.push_back(k.substr(a)); return v; };
  const auto want = lines(key);
  auto range = bySrc_.equal_range(r.src);
  for (auto i = range.first; i != range.second; ++i) {
    const auto have = lines(i->second);
    size_t same = 0;
    for (const auto& l : want) same += std::count(have.begin(), have.end(), l);
    if (same >= bestSame) {
      bestSame = same;
      best = byKey_.at(i->second);
      why.clear();
      for (const auto& l : want) if (!std::count(have.begin(), have.end(), l)) { why = "built " + l; break; }
      for (const auto& l : have) if (!std::count(want.begin(), want.end(), l)) { why += (why.empty() ? "" : ", ") + std::string("gen ") + l; break; }
    }
  }
  throw std::runtime_error("no module of " + r.src + " with these constants" + (best.empty() ? std::string(" (none of that file)") : " (nearest " + best + ": " + why + ")"));
}

std::string Tree::buffer(uint64_t size, uint32_t usage, std::vector<uint8_t> data) {
  const std::string id = "b" + std::to_string(nb_++);
  if (data.size() < size) data.resize(size, 0);
  buffers[id] = {{"size", size}, {"usage", usage}};
  bytes[id] = std::move(data);
  return id;
}

std::string Tree::layout(const std::vector<std::string>& entries) {
  const std::string id = "L" + std::to_string(nl_++);
  bgls[id] = entries;
  return id;
}

json Tree::pipe(const Recipe& r, const std::vector<std::string>& entries, const std::string& label) { return pipeOn(r, layout(entries), label); }

json Tree::pipeOn(const Recipe& r, const std::string& L, const std::string& label) {
  const std::string id = "p" + std::to_string(np_++), m = modules->name(r);
  json p = {{"module", m}, {"entry", "main"}, {"groups", json::array({L})}, {"label", label.empty() ? json(nullptr) : json(label)}, {"lanes", modules->lanes(m)}};
  if (!modules->needs(m).empty()) p["needs"] = modules->needs(m);
  pipelines[id] = p;
  return {{"p", id}, {"bgl", L}};
}

void Tree::part(const std::string& pointer, json value) {
  tree[json::json_pointer(pointer)] = std::move(value);
  parts.push_back(pointer);
}

namespace {

// ai: The variant's precision and features as gen.mjs's recording adapter gives them (its FEATURES, all three) and
// ai: The host takes them (front.cpp FrontHalf::create): the subgroups feature with -sg alone; the floats f16 but at f32.
struct Variant {
  std::string name, precision, floatPrecision;
  bool subgroups = false;
  json features;
};
Variant variantOf(const std::string& name) {
  Variant v;
  v.name = name;
  v.subgroups = name.size() > 3 && name.compare(name.size() - 3, 3, "-sg") == 0;
  v.precision = v.subgroups ? name.substr(0, name.size() - 3) : name;
  if (v.precision != "int8" && v.precision != "f16" && v.precision != "f32") throw std::runtime_error("no variant " + name);
  v.floatPrecision = v.precision == "f32" ? "f32" : "f16";
  v.features = json::array({"timestamp-query", "shader-f16"});
  if (v.subgroups) v.features.push_back("subgroups");
  return v;
}

// ai: the pictures: every size a version takes (lizard_n_for), with its versions and its top's
// ai: sub-channels
json pictures() {
  json out = json::array();
  for (int i = 0; i < LIZARD_NPICTURES; i++) {
    const int n = LIZARD_PICTURES[i];
    int lo = 0, hi = -1;
    for (int v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX; v++)
      if (lizard_n_for(16 * v) == n) { if (!lo) lo = v; hi = v; }
    if (hi < 0) continue;
    out.push_back({{"n", n}, {"subch", 16 * hi}, {"lo", lo}, {"hi", hi}});
  }
  return out;
}

// ai: each ring's modules a side (the codec's: its span and its margin either way), padded to four
json ringModules() {
  json out = json::array();
  for (int r = 0; r < LIZARD_RINGS; r++) {
    lizard_t f;
    if (lizard_init(&f, LIZARD_PICTURES[0], 16 * OB_FMT_VERSION_MIN, 2 * LIZARD_RING[r]) != 0) throw std::runtime_error("the codec made no format in ring " + std::to_string(r));
    out.push_back(2 * LIZARD_RING[r] + 2 * f.margin);
    lizard_free(&f);
  }
  while (out.size() < 4) out.push_back(0);
  return out;
}

// ai: A buffer's bytes: a vector's (f32, u32 or i32) as the JS typed arrays write them
template <class T> std::vector<uint8_t> raw(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) memcpy(b.data(), v.data(), b.size());
  return b;
}
// ai: WebGPU's usage bits (wg.h)
constexpr uint32_t STORAGE_DST = 128 | 8, UNIFORM_DST = 64 | 8;

// ai: A ring's tables (formatTables below prints them for the lab): read from the codec's format at the smallest picture and its
// ai: top version (the border does not depend on the picture), the same C the wasm wraps for the JS (src/wasm.c
// ai: ob_test_mesh_tables_out, ob_test_marks, ob_test_thin_plan, ob_test_fmt_cells, ob_test_fmt_plan,
// ai: lizard_anchor_geom_get, lizard_grid_out)
struct Ring {
  int ring = 0, cells = 0, span = 0, margin = 0, modules = 0;
  std::vector<float> nodeX, nodeY;
  std::vector<uint8_t> kind;
  float gappedMid = 0, mergedMid = 0;
  struct { int refSteps = 0, cells = 0; std::vector<float> pts; std::vector<int32_t> signs; } track;
  struct { int cells = 0, bytes = 0, refSteps = 0, reps = 0, mids = 0, words = 0; std::vector<float> pts; std::vector<int32_t> widx; } word;
  struct { int side = 0, k = 0; std::vector<int> m0; } anch;
};
std::vector<Ring> rings() {
  int lo = 0, hi = -1;
  for (int v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX; v++) if (lizard_n_for(16 * v) == LIZARD_PICTURES[0]) { if (!lo) lo = v; hi = v; }
  const int N0 = LIZARD_PICTURES[0], subch0 = 16 * hi;
  std::vector<Ring> out;
  for (int r = 0; r < LIZARD_RINGS; r++) {
    Ring t;
    t.ring = r;
    t.cells = LIZARD_RING[r];
    lizard_t f;
    if (lizard_init(&f, N0, subch0, 2 * t.cells) != 0) throw std::runtime_error("the codec made no format in ring " + std::to_string(r));
    int32_t dims[64] = {};
    ob_test_mesh_tables(&f.frame, dims, nullptr, nullptr, nullptr, nullptr);
    const int nx = dims[0], ny = dims[1], w = dims[2], h = dims[3];
    t.nodeX.resize(static_cast<size_t>(nx));
    t.nodeY.resize(static_cast<size_t>(ny));
    t.kind.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    std::vector<int32_t> mark(static_cast<size_t>(nx) * static_cast<size_t>(ny));
    ob_test_mesh_tables(&f.frame, dims, t.nodeX.data(), t.nodeY.data(), t.kind.data(), mark.data());
    float mg[64] = {};
    ob_mark_geom(&f.frame, mg);
    t.gappedMid = mg[4];
    t.mergedMid = mg[5];
    const size_t planCap = 16 * (4 * static_cast<size_t>(w) + 4 * static_cast<size_t>(h));
    std::vector<float> pts(2 * planCap);
    std::vector<int32_t> meta(32), sig(planCap), widx(3 * planCap);
    const int np = ob_thin_score_plan(&f.frame, pts.data(), meta.data(), sig.data());
    t.track.refSteps = meta[0];
    t.track.cells = meta[1];
    t.track.pts.assign(pts.begin(), pts.begin() + 2 * np);
    t.track.signs.assign(sig.begin(), sig.begin() + meta[1]);
    t.word.cells = ob_fmt_side_cells(f.frame.w < f.frame.h ? f.frame.w : f.frame.h, f.frame.reserve);
    t.word.bytes = ob_fmt_bytes(f.frame.w < f.frame.h ? f.frame.w : f.frame.h, f.frame.reserve);
    const int nw = ob_thin_fmt_plan(&f.frame, t.word.cells, pts.data(), meta.data(), widx.data());
    t.word.refSteps = meta[0];
    t.word.reps = meta[2];
    t.word.mids = meta[3];
    t.word.words = meta[4];
    t.word.pts.assign(pts.begin(), pts.begin() + 2 * nw);
    t.word.widx.assign(widx.begin(), widx.begin() + 3 * meta[4]);
    int ag[8] = {};
    t.anch.k = lizard_anchor_geom(&f, ag);
    t.anch.side = ag[0];
    t.anch.m0.assign(ag + 1, ag + 1 + t.anch.k);
    // ai: the span and margin from the codec's grid at n = N0 (both whole modules; its float32 grid holds them to a few 1e-7)
    const float g0 = static_cast<float>(f.margin) + 0.5f / f.scale, step = 1.0f / f.scale;
    t.span = static_cast<int>(std::round(N0 * static_cast<double>(step)));
    t.margin = static_cast<int>(std::round(static_cast<double>(g0) - 0.5 * static_cast<double>(step)));
    t.modules = w;
    lizard_free(&f);
    if (t.span != 2 * t.cells || w != t.span + 2 * t.margin) throw std::runtime_error("ring " + std::to_string(t.cells) + ": the codec's span, margin and side disagree");
    out.push_back(std::move(t));
  }
  return out;
}

// ai: F7's plan: a node's template a side, its samples a module each way, the nodes' room, the light
// ai: samples' room; its round (steps, stride, step) and the anchors' border search's
constexpr int WINDOW = 12, SUB = 2, LIGHT_MAX = 4 * plan::NODES_MAX;
constexpr double ROUND[3] = {6, SUB * SUB, 0.5}, ANCH_ROUND[3] = {2, 1, 0.25};
// ai: the ring's nodes in ring order: along the top, down the right, back along the bottom, up the left
std::vector<std::pair<int, int>> ringOrder(int nx, int ny) {
  std::vector<std::pair<int, int>> o;
  for (int i = 0; i < nx; i++) o.emplace_back(i, 0);
  for (int j = 1; j < ny; j++) o.emplace_back(nx - 1, j);
  for (int i = nx - 2; i >= 0; i--) o.emplace_back(i, ny - 1);
  for (int j = ny - 2; j >= 1; j--) o.emplace_back(0, j);
  return o;
}

// ai: The front's fixed buffers: the track plan and its uniform, P4, PICK's, F7's nodes, template,
// ai: rounds and fit uniform; nodesMax the most nodes a ring takes
struct FrontTables {
  std::vector<float> plan, nodes, tmpl;
  std::vector<uint32_t> planAt, count = std::vector<uint32_t>(4, 0), nbase = std::vector<uint32_t>(4, 0);
  std::vector<float> modules = std::vector<float>(4, 0);
  uint32_t nodesMax = 0;
  explicit FrontTables(const std::vector<Ring>& rs) {
    planAt.assign(4 * rs.size(), 0);
    for (size_t i = 0; i < rs.size(); i++) {
      const Ring& t = rs[i];
      planAt[4 * i] = static_cast<uint32_t>(plan.size() / 4);
      planAt[4 * i + 1] = static_cast<uint32_t>(t.track.refSteps);
      planAt[4 * i + 2] = static_cast<uint32_t>(t.track.cells);
      planAt[4 * i + 3] = static_cast<uint32_t>(t.modules);
      const size_t np = t.track.pts.size() / 2, nref = 4 * static_cast<size_t>(t.track.refSteps);
      for (size_t k = 0; k < np; k++) {
        plan.push_back(t.track.pts[2 * k]);
        plan.push_back(t.track.pts[2 * k + 1]);
        plan.push_back(k < nref ? 0.0f : static_cast<float>(t.track.signs[k - nref]));
        plan.push_back(0.0f);
      }
      modules[i] = static_cast<float>(t.modules);
    }
    for (size_t si = 0; si < rs.size(); si++) {
      const Ring& t = rs[si];
      const int nx = static_cast<int>(t.nodeX.size()), M = t.modules;
      count[si] = static_cast<uint32_t>(2 * nx + 2 * nx - 4);
      nbase[si] = static_cast<uint32_t>(nodes.size() / 4);
      for (const auto& [i, j] : ringOrder(nx, nx)) {
        const double sx = t.nodeX[static_cast<size_t>(i)], sy = t.nodeX[static_cast<size_t>(j)];
        const size_t first = tmpl.size() / 4;
        const int y0 = static_cast<int>(std::floor(sy - WINDOW / 2.0)), x0 = static_cast<int>(std::floor(sx - WINDOW / 2.0));
        for (int my = y0; my < y0 + WINDOW; my++)
          for (int mx = x0; mx < x0 + WINDOW; mx++) {
            if (mx < 0 || my < 0 || mx >= M || my >= M) continue;
            const int k = t.kind[static_cast<size_t>(my) * static_cast<size_t>(M) + static_cast<size_t>(mx)];
            if ((k & 8) || (k & 3) == 0) continue;   // ai: a format-word cell (its content is the sender's) or picture
            for (int b = 0; b < SUB; b++)
              for (int a = 0; a < SUB; a++) {
                tmpl.push_back(static_cast<float>(mx + (a + 0.5) / SUB));
                tmpl.push_back(static_cast<float>(my + (b + 0.5) / SUB));
                tmpl.push_back((k & 3) == 2 ? 0.0f : 1.0f);
                tmpl.push_back(0.0f);
              }
          }
        nodes.push_back(static_cast<float>(sx));
        nodes.push_back(static_cast<float>(sy));
        nodes.push_back(static_cast<float>(first));
        nodes.push_back(static_cast<float>(tmpl.size() / 4 - first));
      }
    }
    nodesMax = *std::max_element(count.begin(), count.end());
    if (nodesMax > static_cast<uint32_t>(plan::NODES_MAX)) throw std::runtime_error(std::to_string(nodesMax) + " border nodes, room for " + std::to_string(plan::NODES_MAX));
    if (nodes.size() / 4 > static_cast<size_t>(LIGHT_MAX)) throw std::runtime_error("the rings' nodes past LIGHT_MAX");
  }
  // ai: a round's uniform: steps, stride, step; the rings' counts and bases; each node's light
  // ai: samples at the round's stride
  std::vector<uint8_t> round(const double (&r)[3]) const {
    std::vector<uint32_t> u((64 + 4 * LIGHT_MAX) / 4, 0);
    u[0] = static_cast<uint32_t>(r[0]);
    u[1] = static_cast<uint32_t>(r[1]);
    const float step = static_cast<float>(r[2]);
    memcpy(&u[4], &step, 4);
    for (int k = 0; k < 4; k++) { u[8 + k] = count[static_cast<size_t>(k)]; u[12 + k] = nbase[static_cast<size_t>(k)]; }
    const size_t stride = static_cast<size_t>(r[1]);
    for (size_t q = 0; q < nodes.size() / 4; q++) {
      const size_t first = static_cast<size_t>(nodes[4 * q + 2]), P = (static_cast<size_t>(nodes[4 * q + 3]) + stride - 1) / stride;
      double sum = 0;
      for (size_t j = 0; j < P; j++) sum += tmpl[4 * (first + j * stride) + 2];
      u[16 + q] += static_cast<uint32_t>(sum);
    }
    return raw(u);
  }
};

using Defs = std::vector<std::pair<std::string, std::string>>;
// ai: ZEROS_<n> (n zero floats) and ZEROS_<n>_<m> (n of m) as initialiser lists
std::string zerosOf(int n) {
  std::string s = "{ ";
  for (int i = 0; i < n; i++) s += std::string(i ? ", " : "") + "0.0";
  return s + " }";
}
std::string zerosOf(int n, int m) {
  std::string s = "{ ";
  for (int i = 0; i < n; i++) s += std::string(i ? ", " : "") + zerosOf(m);
  return s + " }";
}
void addZeros(Defs& d, std::initializer_list<int> shapes) { for (int n : shapes) d.emplace_back("ZEROS_" + std::to_string(n), zerosOf(n)); }
// ai: lib/cfg.slang's defines: the four codes' slots, the versions' range, a version's slots
void addCfg(Defs& d) {
  for (int k = 0; k < OB_TIERS; k++) d.emplace_back("CODE" + std::to_string(k) + "_SLOTS", defU(lizard_block_slots(OB_TIER_RATE[k])));
  d.emplace_back("CFG_VERSION_MIN", defU(OB_FMT_VERSION_MIN));
  d.emplace_back("CFG_VERSION_MAX", defU(OB_FMT_VERSION_MAX));
  d.emplace_back("CFG_VERSION_SLOTS", defU(OB_FMT_SLOTS));
}

// ai: F7 (nodes.slang, refit.slang): NODES a round (its samples and offsets on one lattice, latticeSide) and REFIT
constexpr int ITERS = 2, LENS_P = 11, LATTICE_MAX = 36;
constexpr double WMIN = 0.3, DELTA = 0.4;
int latticeSide(const double (&r)[3]) {
  const double spacing = r[1] == SUB * SUB ? 1 : r[1] == 1 ? 1.0 / SUB : 0, ratio = spacing / r[2];
  return spacing > 0 && std::fabs(ratio - std::round(ratio)) < 1e-6 ? static_cast<int>((WINDOW / spacing - 1) * std::round(ratio) + 2 * r[0] + 1) : 0;
}
Recipe nodesRecipe(bool centred, const double (&round)[3], int chunk, int items) {
  const int grid = latticeSide(round);
  if (grid <= 0) throw std::runtime_error("a round whose samples and offsets share no lattice");
  const bool four = round[1] == SUB * SUB;
  return {"nodes.slang", {{"NODES_CENTRED", js(centred ? 1 : 0)}, {"NODES_CHUNK", defU(chunk)}, {"NODES_ITEMS", defU(items)}, {"NODES_GRID", defU(grid)},
                          {"NODES_MAX", defU(plan::NODES_MAX)}, {"LIGHT_VECS", js(LIGHT_MAX / 4)},
                          {"NODES_SPACING", four ? std::string("1.0") : "1.0 / " + std::to_string(SUB) + ".0"},
                          {"NODES_RATIO", defU(std::round((four ? 1.0 : 1.0 / SUB) / round[2]))},
                          {"HALF_WINDOW", std::to_string(WINDOW / 2) + ".0"}, {"SUB_F", std::to_string(SUB) + ".0"}}};
}
Recipe refitRecipe() {
  Defs d = {{"REFIT_ITERS", js(ITERS)}, {"NODES_MAX", defU(plan::NODES_MAX)}, {"LENS_SLOTS", defU(plan::LENS_SLOTS)}, {"RESID", defU(plan::RESID)},
            {"WMIN", js(WMIN)}, {"DELTA", js(DELTA)}};
  addZeros(d, {8, 9, 16, 36});
  d.emplace_back("ZEROS_8_9", zerosOf(8, 9));
  return {"refit.slang", d};
}

// ai: F8 (word.slang; the lab reads the code and gate through lizard_gpu_tree wordcode and word): the word's code over the rings, its search and its uniform's
// ai: layout; the word's numbers the codec's (fmt.h, fmt_tables.h)
struct WordPlan {
  static constexpr int WMAX = 128, MIDS_MAX = 512, SWAP = 6;
  static constexpr double CON_MIN = 0.08, TAU = 0.1, SIDE_KEEP = 0.75;
  const int ROWS = 1 + 8 * (OB_FMT_DATA - 1), VECS = (4 * WMAX) / 128, UNKNOWN = OB_FMT_DATA - 1, PER = 1 + UNKNOWN * SWAP + UNKNOWN;
  const int SETS = PER, TOP = UNKNOWN + SWAP, BYTES_MAX = (4 * WMAX) / 8, GATE = 4 * ((4 * WMAX + 2 + 7) / 8);
  const int TOPS = (LIZARD_NPICTURES + 3) / 4, BEFORE_N = OB_FMT_VERSION_MAX - OB_FMT_VERSION_MIN + 2, BEFORE_VECS = (BEFORE_N + 3) / 4;
  // ai: the Word uniform's layout in vec4u (wordTables prints it for the lab)
  const int range = LIZARD_RINGS * ROWS * VECS, planAt = range + LIZARD_RINGS, tops = planAt + LIZARD_RINGS, gate = tops + TOPS;
  const int gf = gate + (LIZARD_RINGS * GATE) / 4, cols = gf + 512 / 16, before = cols + (LIZARD_RINGS * BYTES_MAX * 2) / 4, size = before + BEFORE_VECS;
  Recipe recipe() const {
    const double words = static_cast<double>(OB_FMT_CONFIGS) * OB_FMT_CLOCKS;
    const double hi = std::floor(words / 4294967296.0), lo = words - hi * 4294967296.0;
    Defs d = {{"W_CODE", js(range)}, {"RING_COUNT", js(LIZARD_RINGS)}, {"W_TOPS", js(TOPS)}, {"W_GATE", js((LIZARD_RINGS * GATE) / 4)}, {"W_GF", js(cols - gf)},
              {"W_COLS", js(before - cols)}, {"W_BEFORE", js(BEFORE_VECS)}, {"LATTICE_VECS", js((LIZARD_RINGS * LATTICE_MAX) / 4)},
              {"MIDS_MAX", js(MIDS_MAX)}, {"WMAX", defU(WMAX)}, {"BYTES_MAX", defU(BYTES_MAX)}, {"TOP", defU(TOP)}, {"SETS", defU(SETS)},
              {"ROWS", defU(ROWS)}, {"VECS", defU(VECS)}, {"GATE", defU(GATE)}, {"UNKNOWN", defU(UNKNOWN)}, {"SWAP", defU(SWAP)},
              {"MAGIC", defU(OB_FMT_MAGIC)}, {"WORDS_HI", defU(hi)}, {"WORDS_LO", defU(lo)}, {"CLOCKS", defU(OB_FMT_CLOCKS)},
              {"DELAY_MAX", defU(OB_FMT_DELAY_MAX)}, {"VERSION_MIN", defU(OB_FMT_VERSION_MIN)}, {"VERSION_MAX", defU(OB_FMT_VERSION_MAX)},
              {"VERSION_SLOTS", defU(OB_FMT_SLOTS)}, {"UNIT", defU(OB_FMT_UNIT)}};
    for (int k = 0; k < OB_TIERS; k++) d.emplace_back("UNIT" + std::to_string(k), defU(lizard_block_slots(OB_TIER_RATE[k]) / OB_FMT_UNIT));
    d.emplace_back("TAU", js(TAU));
    d.emplace_back("SIDE_KEEP", js(SIDE_KEEP));
    d.emplace_back("CON_MIN", js(CON_MIN));
    d.emplace_back("RESID", defU(plan::RESID));
    d.emplace_back("SLOT_COUNT", defU(LIZARD_NPICTURES));
    addCfg(d);
    return {"word.slang", d};
  }
};

// ai: The word's code and gate (over the C's src/rs.c and src/fmt.c): a ring's codeword length for W cells
// ai: a side, its bits in the reader's order, the magic's base and the 40 data bits' rows, and the gate's table (the
// ai: most bit errors a match may hold at m bits compared within the false-word budget, -1 none)
int bytesFor(int W) { return W < 32 ? 0 : std::min(4 * (W / 8), static_cast<int>(OB_FMT_BYTES_MAX)); }
std::vector<uint8_t> wordBytes(const uint8_t (&data)[OB_FMT_DATA], int bytes) {
  std::vector<uint8_t> cw(static_cast<size_t>(bytes), 0);
  memcpy(cw.data(), data, OB_FMT_DATA);
  rs_encode(cw.data(), OB_FMT_DATA, bytes - OB_FMT_DATA);
  return cw;
}
std::vector<uint32_t> wordBits(const std::vector<uint8_t>& cw, int W) {
  std::vector<uint32_t> out(static_cast<size_t>((4 * W + 31) / 32), 0);
  for (int side = 0; side < 4; side++)
    for (int c = 0; c < W; c++) {
      const int i = side * W + c, k = 4 * (c >> 3) + side;
      if (k < static_cast<int>(cw.size()) && ((cw[static_cast<size_t>(k)] >> (7 - (c & 7))) & 1)) out[static_cast<size_t>(i >> 5)] |= 1u << (i & 31);
    }
  return out;
}
constexpr double WORDS = static_cast<double>(OB_FMT_CONFIGS) * OB_FMT_CLOCKS;
constexpr int GATE_CHOICES = 3;
double falseWord(int W) {
  const int bytes = bytesFor(W), nroots = bytes - OB_FMT_DATA, first = 2 - ((nroots - 1) & 1);
  auto binom = [](int n, int k) { double r = 1; for (int i = 0; i < k; i++) r = (r * (n - i)) / (i + 1); return r; };
  auto pass = [&](int f) {
    const int t = (nroots - f) / 2;
    double v = 0;
    for (int i = 0; i <= t; i++) v += binom(bytes - f, i) * std::pow(255.0, i);
    return v / std::pow(256.0, nroots - f);
  };
  double steps = pass(0);
  for (int f = first; f <= nroots - 1; f += 2) steps += pass(f);
  return (steps * (WORDS / std::pow(2.0, 40))) / 256;
}
std::vector<int> gateTable(int W) {
  if (W != 32 && W != 64 && W != 128) throw std::runtime_error("a ring of " + std::to_string(W) + " word cells a side: the gate holds 32, 64, 128");
  const int nb = 4 * W;
  const double budget = falseWord(W) / GATE_CHOICES;
  std::vector<int> E(static_cast<size_t>(nb + 1), -1);
  for (int m = 0; m <= nb; m++) {
    double sum = 0, c = 1;
    for (int d = 0; d <= m; d++) {
      sum += c;
      if ((WORDS * sum) / std::pow(2.0, m) > budget) break;
      E[static_cast<size_t>(m)] = d;
      c = (c * (m - d)) / (d + 1);
    }
  }
  return E;
}

// ai: F8's tables (the tree's wordPtsBuf, wordUni and wordLatUni): the plan's points as vec4f, every ring after
// ai: the one before (references, track pairs, then the word cells in bit order with their pair range lo + 4096 hi);
// ai: the Word uniform in LAYOUT's order; the lattice uniform
struct WordTables {
  std::vector<float> pts;
  std::vector<std::vector<uint32_t>> plans;
  std::vector<uint32_t> uni;
  std::vector<uint8_t> lattice;
  WordTables(const std::vector<Ring>& rs, const json& pics, const WordPlan& P) {
    for (const Ring& t : rs) {
      const auto& w = t.word;
      if (w.reps != 1 || w.words != 4 * w.cells || w.cells > P.WMAX || w.mids > P.MIDS_MAX) throw std::runtime_error("ring " + std::to_string(t.cells) + ": its word plan leaves F8's shape");
      plans.push_back({static_cast<uint32_t>(pts.size() / 4), static_cast<uint32_t>(w.refSteps), static_cast<uint32_t>(w.mids), static_cast<uint32_t>(w.words)});
      const int np = 2 * w.refSteps + 2 * w.mids;
      for (int k = 0; k < np; k++) { pts.push_back(w.pts[2 * static_cast<size_t>(k)]); pts.push_back(w.pts[2 * static_cast<size_t>(k) + 1]); pts.push_back(0); pts.push_back(0); }
      for (int k = 0; k < w.words; k++) {
        const int lo = w.widx[3 * static_cast<size_t>(k)], hi = w.widx[3 * static_cast<size_t>(k) + 1], bit = w.widx[3 * static_cast<size_t>(k) + 2];
        if (bit != k || lo < 0 || hi < lo || hi >= w.mids) throw std::runtime_error("ring " + std::to_string(t.cells) + ": word point " + std::to_string(k) + " out of order");
        pts.push_back(w.pts[2 * static_cast<size_t>(np + k)]);
        pts.push_back(w.pts[2 * static_cast<size_t>(np + k) + 1]);
        pts.push_back(static_cast<float>(lo + 4096.0 * hi));
        pts.push_back(0);
      }
    }
    uni.assign(4 * static_cast<size_t>(P.size), 0);
    for (size_t r = 0; r < rs.size(); r++) {
      const int W = rs[r].word.cells, bytes = bytesFor(W);
      uint8_t d[OB_FMT_DATA] = {OB_FMT_MAGIC, 0, 0, 0, 0, 0};
      std::vector<std::vector<uint32_t>> code = {wordBits(wordBytes(d, bytes), W)};
      for (int j = 1; j < OB_FMT_DATA; j++)
        for (int b = 0; b < 8; b++) {
          uint8_t e[OB_FMT_DATA] = {0, 0, 0, 0, 0, 0};
          e[j] = static_cast<uint8_t>(1u << b);
          code.push_back(wordBits(wordBytes(e, bytes), W));
        }
      for (size_t k = 0; k < code.size(); k++)
        for (size_t i = 0; i < code[k].size(); i++) uni[4 * static_cast<size_t>(P.VECS) * (r * static_cast<size_t>(P.ROWS) + k) + i] = code[k][i];
      const auto gate = gateTable(W);
      for (size_t m = 0; m < gate.size(); m++)
        uni[4 * static_cast<size_t>(P.gate) + static_cast<size_t>(P.GATE) * r + (m >> 1)] |= static_cast<uint32_t>(gate[m] < 0 ? 0xffff : gate[m]) << (16 * (m & 1));
      uni[4 * (static_cast<size_t>(P.range) + r)] = static_cast<uint32_t>(W);
      for (int k = 0; k < 4; k++) uni[4 * (static_cast<size_t>(P.planAt) + r) + static_cast<size_t>(k)] = plans[r][static_cast<size_t>(k)];
      std::vector<std::vector<uint8_t>> unit;
      for (int j = 0; j < OB_FMT_DATA; j++) {
        uint8_t e[OB_FMT_DATA] = {0, 0, 0, 0, 0, 0};
        e[j] = 1;
        unit.push_back(wordBytes(e, bytes));
      }
      for (int p = 0; p < bytes; p++) {
        const size_t at = 4 * static_cast<size_t>(P.cols) + 2 * (static_cast<size_t>(P.BYTES_MAX) * r + static_cast<size_t>(p));
        uni[at] = unit[0][static_cast<size_t>(p)] | (unit[5][static_cast<size_t>(p)] << 8);
        uni[at + 1] = unit[1][static_cast<size_t>(p)] | (unit[2][static_cast<size_t>(p)] << 8) | (unit[3][static_cast<size_t>(p)] << 16) | (static_cast<uint32_t>(unit[4][static_cast<size_t>(p)]) << 24);
      }
    }
    uint8_t exp[256] = {}, log[256] = {};
    for (int i = 0, x = 1; i < 255; i++) { exp[i] = static_cast<uint8_t>(x); log[x] = static_cast<uint8_t>(i); x <<= 1; if (x & 256) x ^= 0x11d; }
    for (int i = 0; i < 256; i++) {
      uni[4 * static_cast<size_t>(P.gf) + static_cast<size_t>(i >> 2)] |= static_cast<uint32_t>(exp[i]) << (8 * (i & 3));
      uni[4 * static_cast<size_t>(P.gf) + 64 + static_cast<size_t>(i >> 2)] |= static_cast<uint32_t>(log[i]) << (8 * (i & 3));
    }
    for (size_t i = 0; i < pics.size(); i++) uni[4 * static_cast<size_t>(P.tops) + i] = pics[i]["hi"].get<uint32_t>();
    for (int i = 0; i < P.BEFORE_N; i++) uni[4 * static_cast<size_t>(P.before) + static_cast<size_t>(i)] = FMT_BEFORE[i];
    const size_t R = rs.size();
    std::vector<uint32_t> lat((16 * R + 4 * R * LATTICE_MAX) / 4, 0);
    for (size_t r = 0; r < R; r++) {
      if (rs[r].nodeX.size() > static_cast<size_t>(LATTICE_MAX) || rs[r].nodeX.size() < 2) throw std::runtime_error("a ring's lattice leaves LT's room");
      lat[4 * r] = static_cast<uint32_t>(LATTICE_MAX * r);
      lat[4 * r + 1] = static_cast<uint32_t>(rs[r].nodeX.size());
      memcpy(&lat[4 * R + LATTICE_MAX * r], rs[r].nodeX.data(), 4 * rs[r].nodeX.size());
    }
    lattice = raw(lat);
  }
};

// ai: The anchors' uniform: a ring's knots (the lattice's ends and its anchors' centres,
// ai: the last repeated), a, span and margin; then each ring's inverse of the squares' fill's normal matrix (Gauss-Jordan
// ai: with partial pivots in float64 over the points in no square's band); then (k, the anchors the geometry takes)
constexpr int ANCH_MIN = 12;
std::vector<float> anchUniform(const std::vector<Ring>& rs, int FILL, int BAND2) {
  const size_t R = rs.size();
  std::vector<float> u(12 * R + 36 * R + 4 * R, 0);
  for (size_t r = 0; r < R; r++) {
    const Ring& t = rs[r];
    const int a = t.anch.side, g = t.anch.k;
    if (g < 3 || g > plan::ANCH_SIDE) throw std::runtime_error("an anchors' grid the uniform does not hold");
    std::vector<double> c, k;
    for (int i = 0; i < g; i++) c.push_back(t.anch.m0[static_cast<size_t>(i)] + a / 2.0);
    k.push_back(t.nodeX.front());
    for (double x : c) k.push_back(x);
    k.push_back(t.nodeX.back());
    while (k.size() < static_cast<size_t>(plan::ANCH_SIDE + 2)) k.push_back(k.back());
    for (size_t i = 0; i < k.size(); i++) u[12 * r + i] = static_cast<float>(k[i]);
    u[12 * r + 9] = static_cast<float>(a);
    u[12 * r + 10] = static_cast<float>(t.span);
    u[12 * r + 11] = static_cast<float>(t.margin);
    auto band = [&](int p) { for (double ci : c) if (std::fabs(2.0 * FILL * (ci - t.margin) - t.span * (2.0 * p + 1)) <= static_cast<double>(BAND2) * FILL * a) return true; return false; };
    double gm[6][12] = {};
    for (int i = 0; i < 6; i++) gm[i][6 + i] = 1;
    for (int py = 0; py < FILL; py++)
      for (int px = 0; px < FILL; px++) {
        if (band(px) && band(py)) continue;
        const double ux = (2.0 * px + 1 - FILL) / FILL, uy = (2.0 * py + 1 - FILL) / FILL, b[6] = {1, ux, ux * ux, uy, uy * uy, ux * uy};
        for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) gm[i][j] += b[i] * b[j];
      }
    for (int col = 0; col < 6; col++) {
      int piv = col;
      for (int i = col + 1; i < 6; i++) if (std::fabs(gm[i][col]) > std::fabs(gm[piv][col])) piv = i;
      for (int j = 0; j < 12; j++) std::swap(gm[col][j], gm[piv][j]);
      const double d = gm[col][col];
      if (!(std::fabs(d) > 1e-9)) throw std::runtime_error("the fill's normal matrix is singular");
      for (int j = 0; j < 12; j++) gm[col][j] /= d;
      for (int i = 0; i < 6; i++) if (i != col) { const double m = gm[i][col]; for (int j = 0; j < 12; j++) gm[i][j] -= m * gm[col][j]; }
    }
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) u[12 * R + 36 * r + 6 * static_cast<size_t>(i) + static_cast<size_t>(j)] = static_cast<float>(gm[i][6 + j]);
    u[48 * R + 4 * r] = static_cast<float>(g);
    u[48 * R + 4 * r + 1] = static_cast<float>(std::min(ANCH_MIN, g * g - 1));
  }
  return u;
}

// ai: A safetensors file (scripts/gpu/cnn/netfile.py read): its metadata and each tensor's shape and values, every
// ai: value as a double (each dtype's exact value)
struct Tensors {
  nlohmann::json meta;
  struct T { std::vector<int> shape; std::vector<double> v; };
  std::map<std::string, T> t;
};
Tensors readTensors(const std::vector<uint8_t>& b, const std::string& path) {
  if (b.size() < 8) throw std::runtime_error(path + ": no safetensors header");
  uint64_t n = 0;
  memcpy(&n, b.data(), 8);
  if (8 + n > b.size()) throw std::runtime_error(path + ": a header past the file");
  const auto h = nlohmann::json::parse(b.begin() + 8, b.begin() + 8 + static_cast<long>(n));
  const size_t base = 8 + n;
  Tensors out;
  for (auto it = h.begin(); it != h.end(); ++it) {
    if (it.key() == "__metadata__") { out.meta = it.value(); continue; }
    const auto& t = it.value();
    const std::string dt = t.at("dtype");
    const size_t a = t.at("data_offsets")[0], z = t.at("data_offsets")[1];
    Tensors::T& x = out.t[it.key()];
    size_t count = 1;
    for (const auto& d : t.at("shape")) { x.shape.push_back(d.get<int>()); count *= d.get<size_t>(); }
    const size_t size = dt == "F64" || dt == "I64" ? 8 : dt == "F32" || dt == "I32" ? 4 : dt == "I16" ? 2 : dt == "I8" ? 1 : 0;
    if (!size || z - a != count * size || base + z > b.size()) throw std::runtime_error(path + ": tensor " + it.key() + " does not read");
    x.v.resize(count);
    const uint8_t* p = b.data() + base + a;
    for (size_t i = 0; i < count; i++, p += size) {
      if (dt == "F32") { float y; memcpy(&y, p, 4); x.v[i] = y; }
      else if (dt == "F64") { double y; memcpy(&y, p, 8); x.v[i] = y; }
      else if (dt == "I64") { int64_t y; memcpy(&y, p, 8); x.v[i] = static_cast<double>(y); }
      else if (dt == "I32") { int32_t y; memcpy(&y, p, 4); x.v[i] = y; }
      else if (dt == "I16") { int16_t y; memcpy(&y, p, 2); x.v[i] = y; }
      else { int8_t y; memcpy(&y, p, 1); x.v[i] = y; }
    }
  }
  return out;
}

// ai: A nets' file (scripts/gpu/cnn/netfile.py load): the safetensors file whose metadata names the net (lizard.spec,
// ai: lizard.arch, lizard.quant for an int8 one), its tensors a layer's fields (w, b; an int8 layer's wq, sw, bq, m or s)
struct Net {
  std::string spec;
  nlohmann::json arch, quant;
  struct Layer { std::string name; std::vector<int> shape; std::map<std::string, std::vector<double>> f; };
  std::vector<Layer> layers;
};
Net readNet(const std::vector<uint8_t>& bytes, const std::string& path) {
  Tensors T = readTensors(bytes, path);
  const auto& m = T.meta;
  Net net;
  net.spec = m.at("lizard.spec").get<std::string>();
  net.arch = nlohmann::json::parse(m.at("lizard.arch").get<std::string>());
  if (m.contains("lizard.quant")) net.quant = nlohmann::json::parse(m.at("lizard.quant").get<std::string>());
  static const std::pair<const char*, const char*> FIELDS[] = {{"wq", "weight_q"}, {"sw", "weight_scale"}, {"bq", "bias_q"}, {"m", "requant_scale"},
                                                              {"s", "dequant_scale"}, {"w", "weight"}, {"b", "bias"}};
  // ai: a classifier's layers conv1.., fc1.. by the arch (netfile.py slots); a proposer's convs.<i> then head
  const bool proposer = net.spec.rfind("proposer-", 0) == 0;
  std::vector<std::pair<std::string, std::string>> slots;
  int nconv = 0, nfc = 0;
  for (const auto& l : net.arch.at("layers")) {
    if (proposer) { slots.emplace_back("conv" + std::to_string(nconv + 1), "convs." + std::to_string(nconv)); nconv++; continue; }
    const std::string name = l.at("type").get<std::string>() == "conv" ? "conv" + std::to_string(++nconv) : "fc" + std::to_string(++nfc);
    slots.emplace_back(name, name);
  }
  if (proposer) slots.emplace_back("head", "head");
  for (const auto& [name, prefix] : slots) {
    Net::Layer L;
    L.name = name;
    bool any = false;
    for (const auto& [f, t] : FIELDS) {
      const auto x = T.t.find(prefix + "." + t);
      if (x == T.t.end()) continue;
      L.f[f] = std::move(x->second.v);
      if (!any) L.shape = x->second.shape;
      any = true;
    }
    if (!any) throw std::runtime_error(path + ": no tensors for " + name);
    net.layers.push_back(std::move(L));
  }
  return net;
}
const Net::Layer& layerOf(const Net& net, const std::string& name) {
  for (const auto& l : net.layers) if (l.name == name) return l;
  throw std::runtime_error("weights: no layer " + name);
}

// ai: IEEE half, round to nearest even
uint16_t toHalf(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  const uint32_t s = (u >> 16) & 0x8000;
  int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  uint32_t m = u & 0x7fffff;
  if (((u >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(s | 0x7c00 | (m ? 0x200 : 0));
  if (e >= 31) return static_cast<uint16_t>(s | 0x7c00);
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(s);
    m |= 0x800000;
    const int shift = 14 - e;
    const uint32_t half = m >> shift, rem = m & ((1u << shift) - 1), mid = 1u << (shift - 1);
    return static_cast<uint16_t>(s | (half + (rem > mid || (rem == mid && (half & 1)) ? 1 : 0)));
  }
  uint32_t hh = s | (static_cast<uint32_t>(e) << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (hh & 1))) hh++;
  return static_cast<uint16_t>(hh);
}

// ai: The classifier (classify.slang; scripts/gpu/cnn/train.py plan the trainer's): an arch's layers (3 x 3 convs, padding 1;
// ai: the first fc the last conv's output flattened), its plan (patches a workgroup, the buffers, each conv's tile),
// ai: its weights' layout and packing (float: f32 or f16; int8: the codes four a word, their biases and scales)
constexpr int WG = 256, PATCH = 32, RED_MIN = 1024, TCF = 4, BUDGET = 32768, REDI_MAX = 3 * WG * TCF;
constexpr double KAPPA = 1.8;
struct ArchLayer { std::string name, type; int stride = 1, inC = 0, inS = 0, outC = 0, outS = 0, inN = 0, outN = 0; std::vector<int> shape; };
std::vector<ArchLayer> planArch(const nlohmann::json& arch) {
  std::vector<ArchLayer> out;
  int C = 1, S = PATCH, nconv = 0, nfc = 0;
  for (const auto& l : arch.at("layers")) {
    ArchLayer L;
    L.type = l.at("type");
    if (L.type == "conv") {
      L.stride = l.value("stride", 1);
      const int O = (S - 1) / L.stride + 1;
      L.name = "conv" + std::to_string(++nconv);
      L.inC = C; L.inS = S; L.outC = l.at("out"); L.outS = O;
      L.shape = {L.outC, C, 3, 3};
      C = L.outC; S = O;
    } else {
      L.name = "fc" + std::to_string(++nfc);
      L.inN = nfc > 1 ? C : C * S * S;
      L.outN = l.at("out");
      L.shape = {L.outN, L.inN};
      C = L.outN;
    }
    out.push_back(L);
  }
  return out;
}
struct Tile { int TC, TX, KS; double partials; };
bool convTile(const ArchLayer& L, int patches, double partialsMax, bool subgroups, bool int8, Tile* t) {
  const double T = static_cast<double>(patches) * L.outC * L.outS * L.outS / WG;
  if (T != std::floor(T) || T < 1) return false;
  for (int TC : int8 ? std::vector<int>{8, 4} : std::vector<int>{8, 4, 2, 1}) {
    if (L.outC % TC) continue;
    for (int KS : {1, 2, 4}) {
      if ((int8 ? (L.inC == 1 ? 1 : L.inC / 4) : L.inC) % KS) continue;
      const double TX = T * KS / (4 * TC);
      if (TX != std::floor(TX) || TX < 1 || TX > 2 || (L.outS / 4) % static_cast<int>(TX)) continue;
      const double partials = subgroups ? 0 : (static_cast<double>(KS - 1) / KS) * WG * 4 * TC * TX;
      if (partials > partialsMax) continue;
      *t = {TC, static_cast<int>(TX), KS, partials};
      return true;
    }
  }
  return false;
}
// ai: a plan: patches a workgroup, the buffers' vec4s a patch, the scratch, each conv's tile, fc1's (NB blocks of TCF
// ai: outputs, KSf slices of slice inputs; int8: vec4s of 16 codes) and fc1's and fc2's outputs' places in red
struct ClassifyPlan { int P = 0, XS = 0, YS = 0, zAt = 0, fc1At = 0, NB = 0, KSf = 0, slice = 0; double red = 0, redi = 0; std::vector<Tile> tiles; };
int pitchOf(int S, int pad) { return S / 4 + (S >= (pad >= 2 ? 8 : pad == 1 ? PATCH : 1 << 30) ? 1 : 0); }
ClassifyPlan classifyPlan(const std::vector<ArchLayer>& layers, bool f16, bool int8, bool subgroups) {
  std::vector<const ArchLayer*> convs, fcs;
  for (const auto& l : layers) (l.type == "conv" ? convs : fcs).push_back(&l);
  // ai: the kernel's shape (src/classify.slang): one to four convs, then fc1 and fc2
  if (convs.empty() || convs.size() > 4 || fcs.size() != 2) throw std::runtime_error("the classifier's kernel runs 1 to 4 convs, then fc1 and fc2");
  ClassifyPlan c;
  if (int8) {
    auto vecs = [&](const ArchLayer* l) { return (l->outC / 4) * l->outS * (l->outS / 4); };
    c.XS = PATCH * (PATCH / 16);
    for (size_t k = 1; k < convs.size(); k += 2) c.XS = std::max(c.XS, vecs(convs[k]));
    for (size_t k = 0; k < convs.size(); k += 2) c.YS = std::max(c.YS, vecs(convs[k]));
    for (int P : {4, 2, 1}) {
      if (WG % P || (WG / P) % 8) continue;
      double redi = WG * TCF;
      bool ok = true;
      c.tiles.clear();
      for (auto* l : convs) { Tile t; if (!convTile(*l, P, REDI_MAX, subgroups, true, &t)) { ok = false; break; } redi = std::max(redi, t.partials); c.tiles.push_back(t); }
      if (!ok) continue;
      const int NB = fcs[0]->outN / TCF;
      const double KSf = static_cast<double>(WG) / (P * NB), slice = fcs[0]->inN / 16.0 / KSf;
      if (KSf != std::floor(KSf) || KSf < 1 || slice != std::floor(slice) || slice < 1) continue;
      const double red = std::max(2.0 * WG + 64, static_cast<double>(P) * (fcs[0]->outN + fcs[1]->outN));
      if (P * (c.XS + c.YS) * 16 + 4 * (red + redi) > BUDGET) continue;
      c.P = P; c.red = red; c.redi = redi; c.fc1At = 0; c.zAt = P * fcs[0]->outN;
      c.NB = NB; c.KSf = static_cast<int>(KSf); c.slice = static_cast<int>(slice);
      return c;
    }
    throw std::runtime_error("the int8 classifier tiles no patch count within the budget");
  }
  const int pad = 1, elBytes = f16 ? 8 : 16, E = f16 ? 8 : 4;
  auto vecs = [&](const ArchLayer* l) { return l->outC * l->outS * pitchOf(l->outS, pad); };
  c.XS = PATCH * pitchOf(PATCH, pad);
  for (size_t k = 1; k < convs.size(); k += 2) c.XS = std::max(c.XS, vecs(convs[k]));
  for (size_t k = 0; k < convs.size(); k += 2) c.YS = std::max(c.YS, vecs(convs[k]));
  for (int P : {8, 4, 2, 1}) {
    if (WG % P || (WG / P) % 8) continue;
    double red = std::max(576, RED_MIN);
    bool ok = true;
    c.tiles.clear();
    for (auto* l : convs) { Tile t; if (!convTile(*l, P, RED_MIN, subgroups, false, &t)) { ok = false; break; } red = std::max(red, t.partials); c.tiles.push_back(t); }
    if (!ok) continue;
    const int NB = fcs[0]->outN / TCF;
    const double KSf = static_cast<double>(WG) / (P * NB), slice = fcs[0]->inN / KSf;
    if (KSf != std::floor(KSf) || KSf < 1 || slice != std::floor(slice) || static_cast<int>(slice) % E) continue;
    const int fc1At = WG * TCF, zAt = fc1At + P * fcs[0]->outN, at = zAt + P * fcs[1]->outN;
    red = std::max(red, static_cast<double>(at));
    if (P * (c.XS + c.YS) * elBytes + 4 * red > BUDGET) continue;
    c.P = P; c.red = red; c.fc1At = fc1At; c.zAt = zAt;
    c.NB = NB; c.KSf = static_cast<int>(KSf); c.slice = static_cast<int>(slice);
    return c;
  }
  throw std::runtime_error("the classifier tiles no patch count within the budget");
}
// ai: exactF32: an f32's literal in nine significant digits as JS toPrecision(9) writes them (glibc's rounding)
std::string exactF32(double v) {
  char buf[64];
  snprintf(buf, sizeof buf, "%.8e", v);
  std::string s(buf);
  const size_t e = s.find('e');
  std::string digits = s.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  const int x = std::stoi(s.substr(e + 1));
  std::string out;
  if (x < -6 || x >= 9) out = digits.substr(0, 1) + "." + digits.substr(1) + "e" + (x >= 0 ? "+" : "-") + std::to_string(std::abs(x));
  else if (x >= 0) out = digits.substr(0, static_cast<size_t>(x + 1)) + (x + 1 < 9 ? "." + digits.substr(static_cast<size_t>(x + 1)) : std::string());
  else out = "0." + std::string(static_cast<size_t>(-x - 1), '0') + digits;
  if (out.find('.') == std::string::npos && out.find('e') == std::string::npos) out += ".0";
  if (static_cast<double>(static_cast<float>(v)) != v || static_cast<double>(static_cast<float>(std::stod(out))) != v) throw std::runtime_error(out + " is not an f32 its literal gives back");
  return out;
}
// ai: The classifier's weights' layout in vec4s (float and int8), read by the packers
// ai: and the recipe alike: a layer's taps (a conv: per tap, a run of its output channels) or rows (an fc) from w, kx
// ai: vec4s a tap or a row, its biases at b and, int8, its scales at m. Float: E elements a vec4 (8 halves or 4 floats),
// ai: each run padded to whole vec4s; int8: conv1 three taps a word, the other convs four input channels a word a tap,
// ai: fc1 16 codes a vec4, then each layer's biases (i32) and scales (f32); fc2 floats.
struct WeightAt { size_t w = 0, b = 0, m = 0, kx = 0; };
std::map<std::string, WeightAt> weightLayout(const std::vector<ArchLayer>& layers, bool f16, bool int8, size_t* vecs = nullptr) {
  std::map<std::string, WeightAt> o;
  const size_t E = f16 ? 8 : 4;
  size_t at = 0;
  for (const auto& L : layers) {
    const size_t outC = static_cast<size_t>(L.outC), inC = static_cast<size_t>(L.inC), outN = static_cast<size_t>(L.outN), inN = static_cast<size_t>(L.inN);
    if (int8 && L.type == "conv") {
      const size_t kx = outC / 4, taps = inC == 1 ? 3 : (inC / 4) * 9;
      o[L.name] = {at, at + taps * kx, at + (taps + 1) * kx, kx};
      at += (taps + 2) * kx;
    } else if (int8 && L.name == "fc1") {
      const size_t rv = inN / 16;
      o[L.name] = {at, at + outN * rv, at + outN * rv + outN / 4, rv};
      at += outN * rv + outN / 2;
    } else if (int8) {
      const size_t rv = (inN + 3) / 4;
      o[L.name] = {at, at + outN * rv, 0, rv};
      at += outN * rv + (outN + 3) / 4;
    } else if (L.type == "conv") {
      const size_t kx = (outC + E - 1) / E;
      o[L.name] = {at, at + inC * 9 * kx, 0, kx};
      at += inC * 9 * kx + kx;
    } else {
      const size_t rv = (inN + E - 1) / E;
      o[L.name] = {at, at + outN * rv, 0, rv};
      at += outN * rv + (outN + E - 1) / E;
    }
  }
  if (vecs) *vecs = at;
  return o;
}
// ai: The kernel's constants (src/classify.slang's header: each layer's list of ints in its order)
Recipe classifyRecipe(const ClassifyPlan& c, const std::vector<ArchLayer>& layers, bool f16, bool int8, bool subgroups, double inv) {
  std::vector<const ArchLayer*> convs, fcs;
  for (const auto& l : layers) (l.type == "conv" ? convs : fcs).push_back(&l);
  const auto lw = weightLayout(layers, f16, int8);
  auto list = [](std::initializer_list<long long> v) { std::string s; for (long long x : v) s += (s.empty() ? "" : ", ") + std::to_string(x); return s; };
  auto ll = [](size_t x) { return static_cast<long long>(x); };
  auto pitch = [&](int S, bool first) { return int8 ? (first ? PATCH / 16 : S / 4) : pitchOf(S, 1); };
  Defs d = {{"INT8", js(int8 ? 1 : 0)}, {"F16", js(f16 ? 1 : 0)}, {"SUBGROUPS", js(subgroups ? 1 : 0)}};
  if (int8) {
    for (auto x : Defs{{"Q8_MAX", "127.0"}, {"Q8_INV", exactF32(inv)}, {"BUF_X", defU(c.P * c.XS)}, {"BUF_Y", defU(c.P * c.YS)}, {"RED", defU(c.red)},
                       {"REDI", defU(c.redi)}}) d.push_back(x);
  } else {
    for (auto x : Defs{{"VEC4_T", f16 ? "half4" : "float4"}, {"BUF_X", defU(c.P * c.XS)}, {"BUF_Y", defU(c.P * c.YS)}, {"RED", defU(c.red)}}) d.push_back(x);
  }
  for (auto x : Defs{{"PATCHES", defU(c.P)}, {"GROUP", defU(WG / c.P)}, {"PATCH", defU(PATCH)}, {"PATCH_PITCH", defU(pitch(PATCH, true))}, {"Z_AT", defU(c.zAt)},
                     {"PATCH_KAPPA", js(KAPPA)}, {"CONVS", js(static_cast<double>(convs.size()))}}) d.push_back(x);
  for (size_t k = 0; k < convs.size(); k++) {
    const ArchLayer& L = *convs[k];
    const Tile& t = c.tiles[k];
    const WeightAt& o = lw.at(L.name);
    d.emplace_back("CONV" + std::to_string(k + 1), list({L.inC, L.inS, L.outC, L.outS, L.stride, pitch(L.inS, k == 0), int8 ? L.outS / 4 : pitchOf(L.outS, 1), t.TC, t.TX, t.KS,
                                                       ll(o.w), ll(o.kx), ll(o.b), ll(o.m)}));
  }
  const ArchLayer &last = *convs.back(), &f1 = *fcs[0], &f2 = *fcs[1];
  const WeightAt &o1 = lw.at(f1.name), &o2 = lw.at(f2.name);
  d.emplace_back("FC1", list({f1.inN, f1.outN, last.outS, int8 ? last.outS / 4 : pitchOf(last.outS, 1), c.NB, c.KSf, c.slice, ll(o1.w), ll(o1.kx), ll(o1.b), ll(o1.m), c.fc1At}));
  d.emplace_back("FC2", list({f2.inN, f2.outN, ll(o2.w), ll(o2.kx), ll(o2.b), c.fc1At, c.zAt}));
  return {"classify.slang", d};
}
// ai: the float classifier's weights on weightLayout: a conv's weight for output channel co of
// ai: input ci at tap (ky, kx), an fc's row n, element k
std::vector<uint8_t> packFloat(const Net& net, const std::vector<ArchLayer>& layers, bool f16) {
  const int E = f16 ? 8 : 4;
  size_t at = 0;
  std::vector<float> data;
  const auto o = weightLayout(layers, f16, false, &at);
  data.assign(at * static_cast<size_t>(E), 0);
  for (const auto& L : layers) {
    const auto& l = layerOf(net, L.name);
    const auto& w = l.f.at("w");
    const auto& b = l.f.at("b");
    const WeightAt& q = o.at(L.name);
    if (L.type == "conv") {
      for (int co = 0; co < L.outC; co++)
        for (int ci = 0; ci < L.inC; ci++)
          for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++) data[(q.w + static_cast<size_t>((ci * 3 + ky) * 3 + kx) * q.kx) * E + static_cast<size_t>(co)] = static_cast<float>(w[static_cast<size_t>(((co * L.inC + ci) * 3 + ky) * 3 + kx)]);
      for (int co = 0; co < L.outC; co++) data[q.b * E + static_cast<size_t>(co)] = static_cast<float>(b[static_cast<size_t>(co)]);
    } else {
      for (int n = 0; n < L.outN; n++) {
        for (int k = 0; k < L.inN; k++) data[(q.w + static_cast<size_t>(n) * q.kx) * E + static_cast<size_t>(k)] = static_cast<float>(w[static_cast<size_t>(n) * L.inN + static_cast<size_t>(k)]);
        data[q.b * E + static_cast<size_t>(n)] = static_cast<float>(b[static_cast<size_t>(n)]);
      }
    }
  }
  if (!f16) return raw(data);
  std::vector<uint16_t> h(data.size());
  for (size_t i = 0; i < data.size(); i++) h[i] = toHalf(data[i]);
  return raw(h);
}
uint32_t packI8x4(double a, double b, double c, double d) {
  auto q = [](double x) { return static_cast<uint32_t>(static_cast<int32_t>(x)) & 255u; };
  return q(a) | (q(b) << 8) | (q(c) << 16) | (q(d) << 24);
}
// ai: the int8 classifier's weights: conv1 three taps a word, the other convs four
// ai: input channels a word a tap, fc1 four by its input's layout, each layer's biases (i32) and scales (f32); fc2 floats
std::vector<uint8_t> packInt8(const Net& net, const std::vector<ArchLayer>& layers) {
  size_t at = 0;
  const auto o = weightLayout(layers, false, true, &at);
  std::vector<uint32_t> u(at * 4, 0);
  auto putF = [&](size_t i, double v) { const float x = static_cast<float>(v); memcpy(&u[i], &x, 4); };
  int S = PATCH;
  for (size_t li = 0; li + 1 < layers.size(); li++) {
    const ArchLayer& L = layers[li];
    const auto& l = layerOf(net, L.name);
    const WeightAt& q = o.at(L.name);
    const int K = L.type == "conv" ? L.inC * 9 : L.inN, cout = L.type == "conv" ? L.outC : L.outN;
    const auto& wq = l.f.at("wq");
    const auto& bq = l.f.at("bq");
    const auto& mul = l.f.at(L.type == "conv" ? "m" : "s");
    auto w = [&](int n, int j) { return wq[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(j)]; };
    for (int n = 0; n < cout; n++) {
      if (L.type == "conv" && L.inC == 1) {
        for (int ky = 0; ky < 3; ky++) u[(q.w + static_cast<size_t>(ky) * q.kx) * 4 + static_cast<size_t>(n)] = packI8x4(w(n, 3 * ky), w(n, 3 * ky + 1), w(n, 3 * ky + 2), 0);
      } else if (L.type == "conv") {
        for (int g = 0; g < L.inC / 4; g++)
          for (int t = 0; t < 9; t++) u[(q.w + static_cast<size_t>(g * 9 + t) * q.kx) * 4 + static_cast<size_t>(n)] = packI8x4(w(n, (4 * g) * 9 + t), w(n, (4 * g + 1) * 9 + t), w(n, (4 * g + 2) * 9 + t), w(n, (4 * g + 3) * 9 + t));
      } else {
        const int G4 = S / 4;
        for (size_t kv = 0; kv < q.kx; kv++) {
          const int xg = static_cast<int>(kv) % G4, gy = (static_cast<int>(kv) - xg) / G4, y = gy % S, g = (gy - y) / S;
          for (int e = 0; e < 4; e++) {
            auto at4 = [&](int j) { return w(n, ((4 * g + j) * S + y) * S + 4 * xg + e); };
            u[(q.w + static_cast<size_t>(n) * q.kx + kv) * 4 + static_cast<size_t>(e)] = packI8x4(at4(0), at4(1), at4(2), at4(3));
          }
        }
      }
      const int32_t bi = static_cast<int32_t>(bq[static_cast<size_t>(n)]);
      memcpy(&u[q.b * 4 + static_cast<size_t>(n)], &bi, 4);
      putF(q.m * 4 + static_cast<size_t>(n), mul[static_cast<size_t>(n)]);
    }
    if (L.type == "conv") S = L.outS;
  }
  const ArchLayer& F2 = layers.back();
  const auto& l2 = layerOf(net, F2.name);
  const WeightAt& q2 = o.at(F2.name);
  for (int n = 0; n < 5; n++) {
    for (int k = 0; k < F2.inN; k++) putF((q2.w + static_cast<size_t>(n) * q2.kx) * 4 + static_cast<size_t>(k), l2.f.at("w")[static_cast<size_t>(n) * static_cast<size_t>(F2.inN) + static_cast<size_t>(k)]);
    putF(q2.b * 4 + static_cast<size_t>(n), l2.f.at("b")[static_cast<size_t>(n)]);
  }
  return raw(u);
}

}  // namespace

// ai: JavaScript's toPrecision(9) (build.h): the double's exact decimal expansion (glibc prints it exactly) cut to nine
// ai: digits, a tie rounded away from zero as toPrecision rounds it (printf's own rounding would take the even one)
std::string precision9(double v) {
  if (v == 0) return "0.00000000";
  const bool neg = v < 0;
  char buf[160];
  snprintf(buf, sizeof buf, "%.120e", std::fabs(v));
  std::string s(buf);
  const size_t e = s.find('e');
  std::string all = s.substr(0, e);
  all.erase(std::remove(all.begin(), all.end(), '.'), all.end());
  int x = std::stoi(s.substr(e + 1));
  std::string digits = all.substr(0, 9);
  if (all[9] >= '5') {
    int i = 8;
    while (i >= 0 && digits[static_cast<size_t>(i)] == '9') digits[static_cast<size_t>(i--)] = '0';
    if (i < 0) { digits = "1" + digits.substr(0, 8); x++; }
    else digits[static_cast<size_t>(i)]++;
  }
  std::string out;
  if (x < -6 || x >= 9) out = digits.substr(0, 1) + "." + digits.substr(1) + "e" + (x >= 0 ? "+" : "-") + std::to_string(std::abs(x));
  else if (x >= 0) out = digits.substr(0, static_cast<size_t>(x + 1)) + (x + 1 < 9 ? "." + digits.substr(static_cast<size_t>(x + 1)) : std::string());
  else out = "0." + std::string(static_cast<size_t>(-x - 1), '0') + digits;
  return (neg ? "-" : "") + out;
}

namespace {
// ai: a float literal of precision9's digits (the proposer's constants, Fcn2 below): ".0" where they leave no point,
// ai: 0 as 0.0
std::string lit9(double v) {
  if (v == 0) return "0.0";
  const std::string s = precision9(v);
  return s.find('.') == std::string::npos && s.find('e') == std::string::npos ? s + ".0" : s;
}
// ai: an f32 the kernel must have exactly, as lit9 writes it: refused where it is no f32 or
// ai: its literal does not give it back
std::string exact9(double v) {
  const std::string s = lit9(v);
  if (static_cast<double>(static_cast<float>(v)) != v || static_cast<double>(static_cast<float>(std::stod(s))) != v) throw std::runtime_error(s + " is not an f32 its literal gives back");
  return s;
}
// ai: a half's value (IEEE binary16 bits)
double fromHalf(uint16_t h) {
  const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
  const double s = h & 0x8000 ? -1 : 1;
  if (e == 0) return s * m * std::ldexp(1.0, -24);
  if (e == 31) return s * HUGE_VAL;
  return s * (1 + m / 1024.0) * std::ldexp(1.0, e - 15);
}

// ai: F2, the proposer (fcn2_x.slang, fcn2_net.slang): its plan from the file's arch (the box
// ai: K, its halo), a stage's planes' sizes (fcn2Layout), the forms a device can be given best first (fcn2Forms), the first
// ai: whose workgroup memory fits taken
struct Fcn2 {
  // ai: X_RR: the X pass's column run, rows a lane (the stats layout's column run at the x stage)
  static constexpr int TILE = 16, LOCAL = 64, WALK = 4, X_RR = 2;
  int K = 0, R = 0, halo = 0;
  std::vector<int> dil, outC;
  double eps = 0, nominal = 0, floor = plan::TAU;
  explicit Fcn2(const nlohmann::json& arch) {
    K = arch.at("window");
    R = (K - 1) / 2;
    halo = 1;
    for (const auto& l : arch.at("layers")) { dil.push_back(l.at("dil")); outC.push_back(l.at("out")); halo += l.at("dil").get<int>(); }
    eps = arch.at("eps");
    nominal = arch.at("nominal");
    if (arch.contains("floor") && !arch.at("floor").is_null()) floor = arch.at("floor");
  }
  // ai: a stage's layout: its pools, workgroup memory, X elements a row; the network's
  // ai: region, row pitch, tiles and halo, the window's rows XH, conv1's groups G1, the map's row MW, and each conv's
  // ai: geometry (dilation, in and out channels, in width, out width, out height)
  struct L { int nA = 0, nB = 0, bytes = 0, XE = 0, RW = 0, RH = 0, RS = 0, NT = 0, HALO = 0, XH = 0, G1 = 0, MW = 0; std::vector<std::array<int, 6>> convs; };
  L x(int XB0, int XB1) const {
    const int XE = (XB0 + 3) / 4, NE = (K + 3 + 3) / 4, VH = XB1 + 2 * R, VE = XE - 1 + NE;
    L z;
    z.XE = XE;
    z.nA = std::max(WG / 4, 2 * VH * XE);
    z.nB = VH * VE;
    z.bytes = 16 * (z.nA + z.nB) + 4 * 16;
    return z;
  }
  L net(int TX, int TY, bool halves, bool int8) const {
    L z;
    z.RW = TILE * TX; z.RH = TILE * TY; z.HALO = halo;
    const int XW = z.RW + 2 * halo, XH = z.RH + 2 * halo;
    z.XE = (XW + 3) / 4;
    int w = XW, h = XH;
    std::vector<int> outW, outH;
    for (size_t k = 0; k < dil.size(); k++) {
      z.convs.push_back({dil[k], k ? outC[k - 1] : 1, outC[k], w, w - 2 * dil[k], h - 2 * dil[k]});
      w -= 2 * dil[k]; h -= 2 * dil[k]; outW.push_back(w); outH.push_back(h);
    }
    if (w != z.RW + 2 || h != z.RH + 2) throw std::runtime_error("fcn2: the convs leave another region");
    const int G1 = (outW[0] + 3) / 4;
    z.RS = G1 + 1;
    z.NT = TX * TY;
    const int MW = z.RW + 2, MH = z.RH + 2;
    z.XH = XH; z.G1 = G1; z.MW = MW;
    int nA = 0;
    if (int8) {
      for (size_t k = 0; k + 1 < dil.size(); k++) nA = std::max(nA, outW[k] * outH[k]);
      z.nA = std::max(nA, 2 * z.NT * LOCAL);
      z.nB = std::max((XH * z.RS + 1) / 2, MW * MH);
      z.bytes = 8 * (z.nA + z.nB) + 4 * z.NT;
      return z;
    }
    auto epp = [&](int C) { return halves ? (C + 7) / 8 : C / 4; };
    for (size_t k = 0; k + 1 < dil.size(); k++) nA = std::max(nA, outW[k] * outH[k] * epp(outC[k]));
    z.nA = std::max(nA, z.NT * LOCAL);
    z.nB = std::max(XH * z.RS, (MW * MH + 1) / 2);
    z.bytes = 16 * (z.nA + z.nB) + 4 * z.NT;
    return z;
  }
  struct Form { std::string store; int TX, TY, XB0, XB1, bytes; bool f16, packed, int8; std::string name; };
  Form pick(bool f16, bool int8, int limit) const {
    static const int REGIONS[4][2] = {{2, 2}, {3, 1}, {2, 1}, {1, 1}}, XBLOCKS[2][2] = {{64, 32}, {32, 32}};
    const std::vector<std::string> stores = int8 ? std::vector<std::string>{"int8"} : f16 ? std::vector<std::string>{"f16"} : std::vector<std::string>{"f32", "packed"};
    for (const auto& st : stores)
      for (const auto& rg : REGIONS)
        for (const auto& xb : XBLOCKS) {
          const bool halves = st == "f16" || st == "packed";
          const int bytes = std::max(net(rg[0], rg[1], halves, int8).bytes, x(xb[0], xb[1]).bytes);
          if (bytes > limit) continue;
          return {st, rg[0], rg[1], xb[0], xb[1], bytes, st == "f16", st == "packed", int8,
                  st + " " + std::to_string(TILE * rg[0]) + "x" + std::to_string(TILE * rg[1]) + " plane " + std::to_string(xb[0]) + "x" + std::to_string(xb[1])};
        }
    throw std::runtime_error("fcn2: no form within the device's workgroup memory");
  }
  Defs common(const Form& f, const L& z, int bankHalo, double inv, int qmax) const {
    Defs d = {{"INT8", js(f.int8 ? 1 : 0)}, {"PLANE_T", f.f16 || f.packed || f.int8 ? "uint" : "float"}, {"POOL_A", js(z.nA)}, {"POOL_B", js(z.nB)},
              {"BANK_HALO", js(bankHalo)}, {"BANK_KEEP", defU(plan::KEEP)}, {"BANK_LOCAL", defU(LOCAL)}, {"BANK_KAPPA", lit9(KAPPA)},
              {"BANK_NOMINAL", lit9(nominal)}, {"BANK_EPS", lit9(eps)}, {"BANK_FLOOR", lit9(floor)}};
    if (f.int8) { d.emplace_back("Q8_MAX", std::to_string(qmax) + ".0"); d.emplace_back("Q8_INV", exactF32(inv)); }
    return d;
  }
  Recipe xRecipe(const Form& f, bool tex, double inv, int qmax) const {
    const L z = x(f.XB0, f.XB1);
    Defs d = {{"TEX", js(tex ? 1 : 0)}};
    for (auto e : common(f, z, 0, inv, qmax)) d.push_back(e);
    d.emplace_back("PLANE_XE", defU(z.XE));
    d.emplace_back("XB0", js(f.XB0));
    d.emplace_back("XB1", js(f.XB1));
    d.emplace_back("HALVES", js((f.f16 || f.packed) && !f.int8 ? 1 : 0));
    d.emplace_back("BOX", js(K));
    d.emplace_back("X_RR", js(X_RR));
    return {"fcn2_x.slang", d};
  }
  // ai: the network's recipe: the layout's sizes, each conv's geometry and the file's values in its order as the kernel
  // ai: reads them (src/fcn2_net.slang): f32 as lit9 writes them, f16 each rounded to a half and written as one ("hf"),
  // ai: int8 the codes, the multipliers and the head exactly
  Recipe netRecipe(const Form& f, const Net& n, double inv, int qmax) const {
    const L z = net(f.TX, f.TY, f.f16 || f.packed, f.int8);
    Defs d = {{"F16", js(f.f16 ? 1 : 0)}};
    for (auto e : common(f, z, z.HALO, inv, qmax)) d.push_back(e);
    for (auto e : Defs{{"POOL_T", f.int8 ? "uint2" : "uint4"}, {"NET_RW", js(z.RW)}, {"NET_WALK", defU(WALK)}, {"NET_RH", js(z.RH)}, {"NET_XE", defU(z.XE)},
                       {"NET_RS", defU(z.RS)}, {"NET_NT", defU(z.NT)}, {"HALVES", js(f.f16 || f.packed ? 1 : 0)}, {"NET_TX", js(f.TX)}, {"NET_TY", js(f.TY)},
                       {"NET_XH", js(z.XH)}, {"NET_G1", js(z.G1)}, {"NET_MW", js(z.MW)}, {"NET_CONVS", js(static_cast<double>(z.convs.size()))}}) d.push_back(e);
    auto list = [](const std::vector<double>& v, const std::function<std::string(double)>& fmt) {
      std::string s;
      for (double x : v) s += (s.empty() ? "" : ", ") + fmt(x);
      return s;
    };
    const auto num = [&](double v) { return f.f16 ? lit9(fromHalf(toHalf(static_cast<float>(v)))) + "hf" : lit9(v); };
    const auto code = [](double v) { return std::to_string(static_cast<long long>(v)); };
    for (size_t k = 0; k < z.convs.size(); k++) {
      const auto& g = z.convs[k];
      const std::string i = std::to_string(k + 1);
      d.emplace_back("NET_C" + i, std::to_string(g[0]) + ", " + std::to_string(g[1]) + ", " + std::to_string(g[2]) + ", " + std::to_string(g[3]) + ", " +
                                      std::to_string(g[4]) + ", " + std::to_string(g[5]));
      const std::string name = "conv" + i;
      const Net::Layer& l = layerOf(n, name);
      if (f.int8) {
        d.emplace_back("NET_W" + i, list(l.f.at("wq"), code));
        d.emplace_back("NET_B" + i, list(l.f.at("bq"), code));
        // ai: a conv into codes carries its requantisation (m), the last one its dequantisation (s): scripts/gpu/cnn/q8.py read_layer
        d.emplace_back("NET_M" + i, list(l.f.count("m") ? l.f.at("m") : l.f.at("s"), exact9));
      } else {
        d.emplace_back("NET_W" + i, list(l.f.at("w"), num));
        d.emplace_back("NET_B" + i, list(l.f.at("b"), num));
      }
    }
    const std::string headName = "head";
    const Net::Layer& head = layerOf(n, headName);
    d.emplace_back("NET_HW", f.int8 ? list(head.f.at("w"), exact9) : list(head.f.at("w"), num));
    d.emplace_back("NET_HB", f.int8 ? list(head.f.at("b"), exact9) : list(head.f.at("b"), [](double v) { return lit9(v); }));
    return {"fcn2_net.slang", d};
  }
};

// ai: The anchors' stages (anch_*.slang): their template's pattern at the coarse and fine sizes
struct Anchors {
  static constexpr int TS = 16, KNOT_ROUNDS = 6, FILL = 64, BAND2 = 3, SW_T = 16, SW_R = 4, SW_LAG = 128, SW_MIN = 8, SW_BINS = 64;
  static constexpr double PEAK = 0.6;
  Defs head() const { return {{"RING_COUNT", js(LIZARD_RINGS)}, {"LATTICE_VECS", js((LIZARD_RINGS * LATTICE_MAX) / 4)}, {"RESID", defU(plan::RESID)}}; }
  Defs fill() const { return {{"ANCH_FILL", js(FILL)}, {"ANCH_BAND2", js(BAND2)}, {"ANCH_GS", defU(plan::ANCH_GS)}}; }
  static void pattern(Defs& d, const std::string& P, int ts) {
    auto dist = [&](int t) { return std::fabs((t + 0.5) / (ts / 2.0) - 1); };
    auto span = [&](double lim) { int a = -1, b = -1; for (int t = 0; t < ts; t++) if (dist(t) < lim) { if (a < 0) a = t; b = t; } return std::pair<int, int>(a, b); };
    const auto [i25, j25] = span(0.25);
    const auto [i6, j6] = span(0.6);
    const double dark = static_cast<double>((j6 - i6 + 1) * (j6 - i6 + 1) - (j25 - i25 + 1) * (j25 - i25 + 1)), light = ts * ts - dark;
    const double tm = (light - dark) / (ts * ts);
    const double tn = static_cast<float>(std::sqrt(light * ((1 - tm) * (1 - tm)) + dark * ((1 + tm) * (1 + tm))));
    d.emplace_back(P + "_TS", defU(ts));
    d.emplace_back(P + "_AREA", lit(ts * ts));
    d.emplace_back(P + "_I25", defU(i25));
    d.emplace_back(P + "_N25", defU(j25 - i25 + 1));
    d.emplace_back(P + "_I6", defU(i6));
    d.emplace_back(P + "_N6", defU(j6 - i6 + 1));
    d.emplace_back(P + "_TM1", lit(1 - tm));
    d.emplace_back(P + "_TN", lit(tn));
  }
  Recipe find() const {
    const int RR = TS, TC = TS / 2, RC = RR / 2, WC = TC + 2 * RC, FD = 3, WF = TS + 2 * FD;
    Defs d = head();
    for (auto x : Defs{{"ANCH_TS", defU(TS)}, {"ANCH_K", defU(plan::ANCH_K)}, {"ANCH_SIDE", defU(plan::ANCH_SIDE)}, {"ANCH_PEAK", lit(PEAK)},
                       {"WC", defU(WC)}, {"W1C", defU(WC + 1)}, {"OC", defU(2 * RC + 1)}, {"RC", js(RC)}, {"WF", defU(WF)}, {"W1F", defU(WF + 1)},
                       {"OF", defU(2 * FD + 1)}}) d.push_back(x);
    pattern(d, "PC", TC);
    pattern(d, "PF", TS);
    return {"anch_find.slang", d};
  }
  Recipe sums() const {
    Defs d = head();
    for (auto x : fill()) d.push_back(x);
    d.emplace_back("FILL_PER", defU((FILL * FILL) / (64 * plan::ANCH_GS)));
    return {"anch_sums.slang", d};
  }
  Recipe fit() const {
    Defs d = head();
    for (auto x : fill()) d.push_back(x);
    for (auto x : Defs{{"ANCH_K", defU(plan::ANCH_K)}, {"ANCH_SIDE", defU(plan::ANCH_SIDE)}, {"ANCH_REC", defU(plan::ANCH_REC)}, {"ANCH_SURF", defU(plan::ANCH_SURF)},
                       {"ANCH_GAIN", defU(plan::ANCH_GAIN)}, {"ANCH_KNOT_ROUNDS", defU(KNOT_ROUNDS)}, {"SW_T", defU(SW_T)}, {"SW_R", defU(SW_R)},
                       {"SW_LAG", defU(SW_LAG)}, {"SW_MIN", defU(SW_MIN)}, {"SW_BINS", defU(SW_BINS)}, {"SPREADS3", lit(3 * 1.4826)}}) d.push_back(x);
    return {"anch_fit.slang", d};
  }
  Recipe refit(bool predict) const {
    Defs d = {{"PREDICT", js(predict ? 1 : 0)}, {"ANCH_K", defU(plan::ANCH_K)}, {"ANCH_SIDE", defU(plan::ANCH_SIDE)}, {"NODES_MAX", defU(plan::NODES_MAX)},
              {"LENS_P", defU(LENS_P)}, {"RESID", defU(plan::RESID)}, {"WMIN", js(WMIN)}, {"TAU_SPREAD", js(static_cast<float>(3.0 * 1.4826))},
              {"RING_COUNT", js(LIZARD_RINGS)}, {"LATTICE_VECS", js((LIZARD_RINGS * LATTICE_MAX) / 4)}};
    addZeros(d, {77, LENS_P});
    d.emplace_back("ZEROS_11_11", zerosOf(LENS_P, LENS_P));
    return {"anch_refit.slang", d};
  }
};

// ai: The finder's stages (vote, gather, complete, score, pick, redo, peaks, trace .slang): its constants as defines, an
// ai: integer a uint
struct Finder {
  // ai: COMPLETE's side reads (complete.slang): pass 0 at REACH0 either way of its guessed line, the later passes each
  // ai: about the line the pass before fitted; the ruler's scan and refine; the fit's weights
  static constexpr int PROFILE = 13, RULER_CELLS0 = 64, RULER_NP = 1024, RULER_COARSE = 3, RULER_NDEP = 9, TR_MAX = 24, TR_STEP = 12, TR_L = 32;
  static constexpr double NCC_MIN = 0.6, REACH0 = 3.0, RULER_LO = 0.6, RULER_HI = 1.6, RULER_DMAX = 2.0, RULER_STEP1 = 0.0005, FIT_PIVOT = 1e-6;
  // ai: the later passes: over PASS_EXT0 and PASS_EXT1 modules of the corner's (at most the run),
  // ai: then the whole run, each at PASS_REACH
  static constexpr double PASS_REACH = 1.5, PASS_EXT0 = 36, PASS_EXT1 = 108, RULER_DEPTH0 = 2.5, RULER_DEPTH1 = 6.5;
  static constexpr int ALONG0_LO = 4, ALONG0_HI = 15, ALONG1_LO = 13, ALONG1_HI = 25;
  const int RING_COUNT = LIZARD_RINGS, HYP = plan::K * plan::ORIENT * LIZARD_RINGS * plan::BENDS, GROUP = 256 / plan::K;
  const double REACH_MAX = std::max(REACH0, PASS_REACH);
  const double RULER_STEP0 = 0.5 / (16 + 2 * RULER_CELLS0), RULER_STEP0C = RULER_COARSE * RULER_STEP0;
  const int RULER_N0 = static_cast<int>(std::ceil(std::log(RULER_HI / RULER_LO) / RULER_STEP0C));
  const int RULER_ND = static_cast<int>(std::round((2 * RULER_DMAX) / 0.5)) + 1;
  const int RULER_N1 = static_cast<int>(std::ceil((6 * RULER_STEP0) / RULER_STEP1)) + 1;
  const double FIT_LINE = 1 / (0.15 * 0.15), FIT_MARK = 1, FIT_RULER = 1 / (0.3 * 0.3);
  const int FIT_ROWS = 16 + 4 * plan::STATIONS;

  using Defs = std::vector<std::pair<std::string, std::string>>;
  Defs centres() const { return {{"CENTRE0", lit(plan::CENTRE[0])}, {"CENTRE1", lit(plan::CENTRE[1])}}; }
  static Defs zeros(std::initializer_list<int> shapes) {
    Defs d;
    for (int n : shapes) {
      std::string s = "{ ";
      for (int i = 0; i < n; i++) s += std::string(i ? ", " : "") + "0.0";
      d.emplace_back("ZEROS_" + std::to_string(n), s + " }");
    }
    return d;
  }
  Recipe vote() const {
    Defs d = {{"CELL", defU(plan::CELL)}, {"RAY_HI", defU(plan::RAY_HI)}, {"RAY_LO", defU(plan::RAY_LO)}, {"RAY_SPREAD", defU(plan::RAY_SPREAD)},
              {"RING_COUNT", defU(RING_COUNT)}, {"SMIN", defU(plan::SMIN)}, {"VOTE_READS", defU(plan::VOTE_READS)}};
    for (auto& c : centres()) d.push_back(c);
    return {"vote.slang", d};
  }
  Recipe peaks() const { return {"peaks.slang", {{"CELL", defU(plan::CELL)}, {"PEAK_LIST", defU(plan::PEAK_LIST)}, {"RING_COUNT", defU(RING_COUNT)}, {"SUPPRESS", defU(plan::SUPPRESS)}}}; }
  Recipe gather() const {
    Defs d = {{"CELL", defU(plan::CELL)}, {"GROUP", defU(GROUP)}, {"K", defU(plan::K)}, {"PEAK_LIST", defU(plan::PEAK_LIST)}, {"RAY_HI", defU(plan::RAY_HI)},
              {"RAY_LO", defU(plan::RAY_LO)}, {"RAY_SPREAD", defU(plan::RAY_SPREAD)}, {"RING_COUNT", defU(RING_COUNT)}, {"SMIN", defU(plan::SMIN)}};
    for (auto& c : centres()) d.push_back(c);
    return {"gather.slang", d};
  }
  Recipe complete() const {
    Defs d = {{"FIT_LINE", defU(FIT_LINE)}, {"FIT_PIVOT", defU(FIT_PIVOT)}, {"FIT_ROWS", defU(FIT_ROWS)}, {"K", defU(plan::K)}, {"NCC_MIN", defU(NCC_MIN)},
              {"PROFILE", defU(PROFILE)}, {"RULER_CELLS0", defU(RULER_CELLS0)}, {"RULER_HI", defU(RULER_HI)}, {"RULER_N0", defU(RULER_N0)},
              {"RULER_N1", defU(RULER_N1)}, {"RULER_ND", defU(RULER_ND)}, {"RULER_NDEP", defU(RULER_NDEP)}, {"RULER_NP", defU(RULER_NP)},
              {"RULER_STEP0", defU(RULER_STEP0)}, {"RULER_STEP0C", defU(RULER_STEP0C)}, {"RULER_STEP1", defU(RULER_STEP1)}, {"STATIONS", defU(plan::STATIONS)},
              {"ALONG0_LO", defU(ALONG0_LO)}, {"ALONG0_HI", defU(ALONG0_HI)}, {"ALONG1_LO", defU(ALONG1_LO)}, {"ALONG1_HI", defU(ALONG1_HI)},
              {"LANES", js(2 * plan::STATIONS)}, {"REACH0", lit(REACH0)}, {"RULER_DMAX", lit(RULER_DMAX)}, {"FIT_MARK", lit(FIT_MARK)},
              {"PROF_LEN", js(2 * std::round(4 * REACH_MAX) + 2 * PROFILE + 1)}, {"RULER_LOG_LO", lit(std::log(RULER_LO))},
              {"FIT_RULER_CELL", lit(FIT_RULER / 4)}, {"RULER_DEPTH0", lit(RULER_DEPTH0)},
              {"RULER_DEPTH_STEP", lit((RULER_DEPTH1 - RULER_DEPTH0) / (RULER_NDEP - 1))}, {"PASS0_EXT", lit(PASS_EXT0)}, {"PASS0_REACH", lit(PASS_REACH)},
              {"PASS1_EXT", lit(PASS_EXT1)}, {"PASS1_REACH", lit(PASS_REACH)}, {"PASS2_REACH", lit(PASS_REACH)}};
    return {"complete.slang", d};
  }
  Recipe trace() const { return {"trace.slang", {{"K", defU(plan::K)}, {"TR_L", defU(TR_L)}, {"TR_MAX", defU(TR_MAX)}, {"TR_STEP", defU(TR_STEP)}}}; }
  Recipe score() const {
    Defs d = {{"HYP", defU(HYP)}, {"K", defU(plan::K)}, {"NEAR", defU(plan::NEAR)}, {"ORIENT", defU(plan::ORIENT)}, {"RING_COUNT", defU(RING_COUNT)}};
    for (auto& z : zeros({9})) d.push_back(z);
    return {"score.slang", d};
  }
  Recipe pick() const {
    Defs d = {{"HYP", defU(HYP)}, {"K", defU(plan::K)}, {"ORIENT", defU(plan::ORIENT)}, {"RING_COUNT", defU(RING_COUNT)}};
    for (auto& z : zeros({9, 16, 32})) d.push_back(z);
    return {"pick.slang", d};
  }
  Recipe redo() const { return {"redo.slang", {{"K", defU(plan::K)}, {"R2_N", defU(plan::R2_N)}, {"RING_COUNT", defU(RING_COUNT)}}}; }
};

// ai: ---- The back half (backHalf), at each batch size gen writes ----

// ai: The back half's constants: a sub-channel's data slots, the 8-sub-channel chunks a version, a
// ai: sub-channel's data coefficients, the whitening's slots (every version's at the top), the LDPC's forms (the
// ai: first the one every device runs, each other a twin the host may time), the codes' names in OB_TIER_RATE's order
constexpr int SUB_SLOTS = OB_FMT_SLOTS / LIZARD_STEP, CHUNKS = LIZARD_STEP / 8, DCOEF = SUB_SLOTS / 2;
constexpr double PI = 3.141592653589793;   // ai: Math.PI
using codes::WHITE_SLOTS;
const char* const LDPC_FORMS[] = {"reg", "reg+i32", "reg+i16"};
const auto& CODE_NAMES = codes::NAMES;

// ai: the coefficient at pos p as (u, v), the C's half-plane block layout (src/lizard_phy.c init)
void uvOf(int p, int n, int bw, int& u, int& v) {
  const int nb = n * bw, uw = (p % nb) / bw;
  v = (p / nb) * bw + (p % nb) % bw;
  u = uw < n / 2 ? uw : uw - n;
}

// ai: A picture size's transform tables (the lab's through lizard_gpu_tree disc) from the codec's format in its
// ai: default ring (a picture's tables do not depend on its ring): the disc's rows (off, count, nonnegative entries,
// ai: umaxN a row), each coefficient's S entry (UV) and its (u, v) (UC: u signed in the high half, v in the low), the
// ai: rows the first c chunks take, the detrend's basis (x12), the twiddles, sx2 and sq2
struct Disc {
  int n = 0, subch = 0, blocks = 0, npos = 0, V = 0;
  std::vector<uint32_t> rows, UV, UC, rowsBy;
  std::vector<float> x12, tw;
  float sx2 = 0, sq2 = 0;
  Disc(int n_, int subch_) : n(n_), subch(subch_), blocks(subch_ / 8), npos(subch_ * LIZARD_SUB) {
    const std::string at = std::to_string(n) + "/" + std::to_string(subch);
    lizard_t f;
    if (lizard_init(&f, n, subch, 0) != 0) throw std::runtime_error("the codec made no format of " + at);
    std::vector<int> us(static_cast<size_t>(npos)), vs(static_cast<size_t>(npos));
    for (size_t i = 0; i < us.size(); i++) uvOf(f.pos[i], n, f.bw, us[i], vs[i]);
    std::vector<float> tab(6 * static_cast<size_t>(n) + 2);
    lizard_tables(&f, tab.data());
    lizard_free(&f);
    V = *std::max_element(vs.begin(), vs.end()) + 1;
    std::vector<std::vector<int>> byV(static_cast<size_t>(V));
    for (size_t i = 0; i < us.size(); i++) byV[static_cast<size_t>(vs[i])].push_back(us[i]);
    rows.assign(4 * static_cast<size_t>(V), 0);
    std::vector<int> off(static_cast<size_t>(V)), np(static_cast<size_t>(V)), un(static_cast<size_t>(V));
    int count = 0;
    for (int v = 0; v < V; v++) {
      const auto& row = byV[static_cast<size_t>(v)];
      const int ustart = v == 0 ? 1 : 0;
      int maxP = ustart - 1, maxN = 0;
      for (int u : row) { if (u >= 0) maxP = std::max(maxP, u); else maxN = std::max(maxN, -u); }
      std::vector<char> has(static_cast<size_t>(maxP + 1 + maxN + 1), 0);
      for (int u : row) has[static_cast<size_t>(u >= 0 ? u : maxP + 1 - u)] = 1;
      bool ok = static_cast<int>(row.size()) == maxP + 1 - ustart + maxN;
      for (int u = ustart; u <= maxP && ok; u++) ok = has[static_cast<size_t>(u)];
      for (int u = -maxN; u < 0 && ok; u++) ok = has[static_cast<size_t>(maxP + 1 - u)];
      if (!ok) throw std::runtime_error("disc row v = " + std::to_string(v) + " of " + at + " is not u = " + std::to_string(ustart) + ".." + std::to_string(maxP) + ", -" + std::to_string(maxN) + "..-1");
      off[static_cast<size_t>(v)] = count;
      np[static_cast<size_t>(v)] = maxP + 1 - ustart;
      un[static_cast<size_t>(v)] = maxN;
      const uint32_t r[4] = {static_cast<uint32_t>(count), static_cast<uint32_t>(row.size()), static_cast<uint32_t>(maxP + 1 - ustart), static_cast<uint32_t>(maxN)};
      std::copy(r, r + 4, rows.begin() + 4 * v);
      count += static_cast<int>(row.size());
    }
    if (count != npos) throw std::runtime_error("the disc of " + at + " holds " + std::to_string(count) + " entries for " + std::to_string(npos) + " coefficients");
    UV.resize(us.size());
    UC.resize(us.size());
    for (size_t i = 0; i < us.size(); i++) {
      const int u = us[i], v = vs[i], ustart = v == 0 ? 1 : 0;
      const size_t w = static_cast<size_t>(v);
      UV[i] = static_cast<uint32_t>(off[w] + (u >= 0 ? u - ustart : np[w] + un[w] + u));
      UC[i] = (static_cast<uint32_t>(u & 0xffff) << 16) | static_cast<uint32_t>(v & 0xffff);
    }
    const int per = npos / blocks;
    rowsBy.assign(static_cast<size_t>(blocks) + 1, 0);
    int top = -1;
    for (int c = 1; c <= blocks; c++) {
      for (int i = per * (c - 1); i < per * c; i++) top = std::max(top, vs[static_cast<size_t>(i)]);
      rowsBy[static_cast<size_t>(c)] = static_cast<uint32_t>(top + 1);
    }
    if (rowsBy[static_cast<size_t>(blocks)] != static_cast<uint32_t>(V)) throw std::runtime_error(at + ": the blocks take " + std::to_string(rowsBy[static_cast<size_t>(blocks)]) + " rows of the disc's " + std::to_string(V));
    const size_t N = static_cast<size_t>(n);
    x12.resize(4 * N);
    for (size_t u = 0; u < N; u++) { x12[4 * u] = tab[2 * N + u]; x12[4 * u + 1] = tab[3 * N + u]; x12[4 * u + 2] = tab[4 * N + u]; x12[4 * u + 3] = tab[5 * N + u]; }
    tw.resize(2 * N);
    for (int m = 0; m < n; m++) {
      const double th = -2 * PI * m / n;
      tw[2 * static_cast<size_t>(m)] = static_cast<float>(std::cos(th));
      tw[2 * static_cast<size_t>(m) + 1] = static_cast<float>(std::sin(th));
    }
    sx2 = tab[6 * N];
    sq2 = tab[6 * N + 1];
  }
  // ai: the disc rows a frame of version v takes: its first CHUNKS v chunks'
  uint32_t rowsOf(int v) const { return rowsBy[static_cast<size_t>(std::min(CHUNKS * v, blocks))]; }
};

// ai: The back half's strides: the sizes served, the largest sub-channels and blocks a frame, S's
// ai: entries, L's words and the records a batch
struct Dims {
  int B = 0, subchMax = 0, blocksMax = 0, sStride = 0, lStride = 0, recCap = 0;
  std::vector<int> served;
  Dims(const std::vector<const Disc*>& sizes, int B_) : B(B_) {
    for (size_t s = 0; s < sizes.size(); s++) if (sizes[s] && sizes[s]->blocks > 0) served.push_back(static_cast<int>(s));
    if (served.empty()) throw std::runtime_error("no size served");
    for (int s : served) {
      const Disc& z = *sizes[static_cast<size_t>(s)];
      subchMax = std::max(subchMax, z.subch);
      blocksMax = std::max(blocksMax, lizard_blocks_max_for(z.subch));
      sStride = std::max(sStride, z.subch * LIZARD_SUB);
    }
    lStride = subchMax * SUB_SLOTS / 4;
    recCap = B * blocksMax;
  }
  json out() const {
    return {{"B", B}, {"served", served}, {"subchMax", subchMax}, {"blocksMax", blocksMax}, {"sStride", sStride}, {"lStride", lStride},
            {"pilotRead", plan::PILOT_READ}, {"blkStride", blocksMax}, {"recCap", recCap}};
  }
};

// ai: the four codes, the bit map, the whitening and the CRC's powers: codes.h, the sender's too
using codes::Code;
using codes::crcPowers;
using codes::perm;
using codes::whiten;

}  // namespace

// ai: A code's learned stop (build.h Stop): the GPU's stops' (each code's file by its checks) and the C decoder's
// ai: (tools/ldpc_stop.cpp writes src/ldpc_stop.h from it), one fold: in float64 in the trainer's order
// ai: (scripts/gpu/back/stop/net.py fold), then each value rounded once to float32
std::vector<uint32_t> Stop::words() const {
  const size_t H = static_cast<size_t>(hidden);
  std::vector<float> f(H * H + 9 * H + 1, 0.0f);
  for (size_t k = 0; k < 6; k++) for (size_t j = 0; j < H; j++) f[k * H + j] = W1[6 * j + k];
  std::copy(b1.begin(), b1.end(), f.begin() + static_cast<long>(6 * H));
  for (size_t k = 0; k < H; k++) for (size_t j = 0; j < H; j++) f[7 * H + k * H + j] = W2[H * j + k];
  std::copy(b2.begin(), b2.end(), f.begin() + static_cast<long>(7 * H + H * H));
  std::copy(w3.begin(), w3.end(), f.begin() + static_cast<long>(8 * H + H * H));
  f[9 * H + H * H] = T;
  std::vector<uint32_t> w(f.size());
  memcpy(w.data(), f.data(), 4 * f.size());
  return w;
}
Stop readStop(const std::vector<uint8_t>& bytes, const std::string& path) {
  static const char* const FEATURES[6] = {"bad", "low", "first", "fall", "it", "grow"}, *const INPUTS[6] = {"bad", "low", "first", "prev", "t", "grow"};
  constexpr double GROW_ONE = 256;
  const Tensors T = readTensors(bytes, path);
  const std::string& where = path;
  if (T.meta.value("lizard.spec", "") != "ldpc-stop-mlp-v2") throw std::runtime_error(where + ": spec " + T.meta.value("lizard.spec", "") + ", not ldpc-stop-mlp-v2");
  const auto arch = nlohmann::json::parse(T.meta.at("lizard.arch").get<std::string>());
  if (arch.at("features") != nlohmann::json(std::vector<std::string>(FEATURES, FEATURES + 6)) || arch.at("inputs") != nlohmann::json(std::vector<std::string>(INPUTS, INPUTS + 6)))
    throw std::runtime_error(where + ": features " + arch.at("features").dump() + ", inputs " + arch.at("inputs").dump());
  Stop s;
  s.hidden = arch.at("hidden")[0].get<int>();
  s.m = arch.at("m").get<int>();
  const int H = s.hidden;
  const std::vector<std::pair<std::string, std::vector<int>>> want = {{"mu", {6}}, {"sd", {6}}, {"fc1.weight", {H, 6}}, {"fc1.bias", {H}}, {"fc2.weight", {H, H}},
                                                                       {"fc2.bias", {H}}, {"out.weight", {1, H}}, {"out.bias", {1}}, {"theta", {1}}};
  for (const auto& [k, shape] : want) {
    const auto x = T.t.find(k);
    if (x == T.t.end() || x->second.shape != shape) throw std::runtime_error(where + ": " + k + " is not " + nlohmann::json(shape).dump());
  }
  if (T.t.size() != want.size()) throw std::runtime_error(where + ": " + std::to_string(T.t.size()) + " tensors, not " + std::to_string(want.size()));
  const auto& W = T.t.at("fc1.weight").v;
  const auto& b = T.t.at("fc1.bias").v;
  const auto& mu = T.t.at("mu").v;
  const auto& sd = T.t.at("sd").v;
  const double m = s.m;
  s.W1.resize(6 * static_cast<size_t>(H));
  s.b1.resize(static_cast<size_t>(H));
  enum { BAD, LOW, FIRST, FALL, IT, GROW };
  for (size_t j = 0; j < static_cast<size_t>(H); j++) {
    auto c = [&](int k) { return W[6 * j + static_cast<size_t>(k)] / sd[static_cast<size_t>(k)]; };
    const double row[6] = {c(BAD) / m - c(FALL) / m, c(LOW) / m, c(FIRST) / m, c(FALL) / m, c(IT), c(GROW) / GROW_ONE};
    for (size_t k = 0; k < 6; k++) s.W1[6 * j + k] = static_cast<float>(row[k]);
    double shift = 0;
    for (int k = 0; k < 6; k++) shift = shift + c(k) * mu[static_cast<size_t>(k)];
    s.b1[j] = static_cast<float>(b[j] - shift);
  }
  auto f32 = [](const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); };
  s.W2 = f32(T.t.at("fc2.weight").v);
  s.b2 = f32(T.t.at("fc2.bias").v);
  s.w3 = f32(T.t.at("out.weight").v);
  s.T = static_cast<float>(T.t.at("theta").v[0] - T.t.at("out.bias").v[0]);
  s.code = T.meta.value("lizard.code", "3/4");
  s.budget = std::stod(T.meta.value("lizard.budget", "0"));
  return s;
}

namespace {
// ai: a code's stop file by its checks (plan.h STOPS)
std::string stopFile(int m) {
  for (const auto& x : plan::STOPS) if (x.m == m) return x.file;
  throw std::runtime_error("stop rule: no file for a code of " + std::to_string(m) + " checks");
}


// ai: A code's shape in an LDPC form and its workgroup memory (tierWgBytes)
int packWords(int n) { return std::max(120, (n + 31) / 32 + 1); }
struct Shape {
  int n = 0, k = 0, z = 0, mb = 0, ns = 0, slots = 0, threads = 0, rowv = 0, pairv = 0;
  bool one = false, wide = false, narrow = false;
  std::string form;
  Shape(const Code& c, const std::string& form_) : n(c.n), k(c.k), z(c.z), mb(c.mb), form(form_) {
    int nd = 0;
    for (int r = 0; r < c.mb; r++) nd = std::max(nd, c.lay[static_cast<size_t>(r) + 1] - c.lay[static_cast<size_t>(r)]);
    ns = nd + 2;
    slots = c.lay[static_cast<size_t>(c.mb)];
    threads = std::max(128, 32 * ((z + 31) / 32));
    one = ns <= 12;
    rowv = (mb + 2 + 2) / 4;
    pairv = (slots + 3) / 4;
    const bool i32 = form == "reg+i32", i16 = form == "reg+i16";
    if (!i32 && !i16 && form != "reg") throw std::runtime_error("LDPC form " + form + ": one of reg, reg+i32, reg+i16");
    wide = i32 && 4 * n + 4 * packWords(n) + 12 + 4 * (6 + 2 * 64) <= 32768;
    narrow = i16;
  }
  int wgBytes(int hidden) const { return (wide ? 4 * n : 4 * ((n + 1) / 2)) + 4 * packWords(n) + 20 + 4 * (6 + 2 * hidden); }
};

// ai: The rule stage's tables (rule() gives them to the lab): TAB (each code's codeword bit to its slot, two a word,
// ai: 0xffff a bit never sent; the whitening's bits and a word past them; the learned stops' words) and each code's
// ai: constants (its sent bits, k, a block's slots, the decline's bar at the sent rate)
struct Rule {
  std::vector<Code> codes;
  std::vector<Stop> stops;
  std::vector<int> inv, netAt;
  int WHITE = 0, NET = 0;
  std::vector<uint32_t> tab;
  std::vector<float> bar;
  explicit Rule(const std::function<std::vector<uint8_t>(const std::string&)>& net) : codes(lizard::codes::read()) {
    int o = 0, netWords = 0;
    for (const Code& c : codes) {
      Stop s = readStop(net(stopFile(c.m)), stopFile(c.m));
      if (s.m != c.m) throw std::runtime_error("stop rule: the file for " + std::to_string(c.m) + " checks was trained on a code of " + std::to_string(s.m));
      netWords += static_cast<int>(s.words().size());
      stops.push_back(std::move(s));
      inv.push_back(o);
      o += (c.n + 1) / 2;
    }
    WHITE = o;
    o += WHITE_SLOTS / 32 + 1;
    NET = o;
    o += netWords;
    tab.assign(static_cast<size_t>(o), 0);
    for (size_t c = 0, at = static_cast<size_t>(NET); c < codes.size(); c++) {
      const Code& x = codes[c];
      const auto p = perm(x.nt);
      std::vector<int> iv(p.size());
      for (size_t i = 0; i < p.size(); i++) iv[static_cast<size_t>(p[i])] = static_cast<int>(i);
      for (int j = 0; j < x.n; j++) {
        const bool unsent = j >= x.k - x.np && j < x.k;
        const uint32_t slot = unsent ? 0xffffu : static_cast<uint32_t>(iv[static_cast<size_t>(j < x.k - x.np ? j : j - x.np)]);
        tab[static_cast<size_t>(inv[c] + (j >> 1))] |= slot << (16 * (j & 1));
      }
      netAt.push_back(static_cast<int>(at));
      const auto w = stops[c].words();
      std::copy(w.begin(), w.end(), tab.begin() + static_cast<long>(at));
      at += w.size();
      if (x.blockSlots != x.nt) throw std::runtime_error(std::string("the ") + CODE_NAMES[c] + " code sends " + std::to_string(x.nt) + " bits, its block " + std::to_string(x.blockSlots) + " slots: a tail the rule stage does not read");
      bar.push_back(static_cast<float>(static_cast<double>(static_cast<float>(0.2 * x.k)) / x.nt));
    }
    const auto wb = whiten(WHITE_SLOTS);
    for (size_t i = 0; i < wb.size(); i++) if (wb[i]) tab[static_cast<size_t>(WHITE) + (i >> 5)] |= 1u << (i & 31);
  }
  // ai: TAB's readers' constants (lib/tab.slang), each code's sent bits and bar (its float32 bits)
  void tabDefs(Defs& d) const {
    d.emplace_back("TAB_WHITE_AT", defU(WHITE));
    d.emplace_back("TAB_PB_REF", defU(plan::PB_REF));
    d.emplace_back("TAB_SUB_SLOTS", defU(SUB_SLOTS));
    d.emplace_back("TAB_NCODES", defU(plan::NC));
    addCfg(d);
    for (size_t c = 0; c < codes.size(); c++) {
      uint32_t bits;
      memcpy(&bits, &bar[c], 4);
      d.emplace_back("CODE" + std::to_string(c) + "_N", defU(codes[c].nt));
      d.emplace_back("CODE" + std::to_string(c) + "_BAR", defU(bits));
    }
  }
  // ai: a code's uniform at batch size B (tldpc.slang's): B and the strides, the rows' pointers, the (col,
  // ai: shift) pairs, the CRC's powers
  std::vector<uint32_t> uniform(size_t c, const Shape& sh, const Dims& D) const {
    const Code& x = codes[c];
    std::vector<uint32_t> w(4 + 4 * static_cast<size_t>(sh.rowv) + 4 * static_cast<size_t>(sh.pairv) + plan::PW_WORDS, 0);
    w[0] = static_cast<uint32_t>(D.B); w[1] = static_cast<uint32_t>(D.lStride); w[2] = static_cast<uint32_t>(D.blocksMax); w[3] = static_cast<uint32_t>(D.recCap);
    size_t o = 4;
    for (int r = 0; r <= x.mb; r++) w[o + static_cast<size_t>(r)] = static_cast<uint32_t>(x.lay[static_cast<size_t>(r)]);
    o += 4 * static_cast<size_t>(sh.rowv);
    for (int e = 0; e < x.slots; e++) w[o + static_cast<size_t>(e)] = static_cast<uint32_t>(x.lay[static_cast<size_t>(x.mb + 1 + 2 * e)]) | (static_cast<uint32_t>(x.lay[static_cast<size_t>(x.mb + 2 + 2 * e)]) << 16);
    o += 4 * static_cast<size_t>(sh.pairv);
    const auto pw = crcPowers(plan::PAYLOAD);
    std::copy(pw.begin(), pw.end(), w.begin() + static_cast<long>(o));
    return w;
  }
  Recipe ldpcRecipe(size_t c, const Shape& sh) const {
    const int H = stops[c].hidden;
    if (sh.k < plan::KNOWN || sh.ns > 32 || (sh.one && sh.ns > 12)) throw std::runtime_error(std::string("the ") + CODE_NAMES[c] + " code's shape");
    const char* lw = sh.wide ? "i32" : sh.narrow ? "i16" : "pair";
    Defs d = {{"LW_PAIR", js(std::string(lw) == "pair")}, {"LW_I32", js(std::string(lw) == "i32")}, {"LW_I16", js(std::string(lw) == "i16")}, {"ONE", js(sh.one)},
              {"TH", defU(sh.threads)}, {"LDPC_CODE", defU(static_cast<double>(c))}, {"LDPC_INV_AT", defU(inv[c])}, {"LDPC_N", defU(sh.n)}, {"LDPC_K", defU(sh.k)},
              {"LDPC_Z", defU(sh.z)}, {"LDPC_MB", defU(sh.mb)}, {"LDPC_NS", defU(sh.ns)}, {"LDPC_KNOWN", defU(plan::KNOWN)}, {"LDPC_NET_H", defU(H)},
              {"LDPC_NET_AT", defU(netAt[c])}, {"CRC64_CHUNKS", defU(plan::PAYLOAD / 4)}, {"PACK_WORDS", js(packWords(sh.n))}, {"ROWV", js(sh.rowv)},
              {"PAIRV", js(sh.pairv)}, {"PW_VECS", js(plan::PW_WORDS / 4)}, {"SLOT_COUNT", defU(LIZARD_NPICTURES)}};
    tabDefs(d);
    return {"tldpc.slang", d};
  }
};

// ai: The back half at batch size B: the transform (the gate, pass 1 fused, reduce and pass 2 for every size built,
// ai: their tables, PIC and the gate's uniform), the soft values' tables, the LDPC's sizes and the rule stage (the four
// ai: codes' kernels and their twins in the other LDPC forms, its steps for front.cpp). A size whose pass 1 does not fit
// ai: the device's workgroup memory is not built.
json backHalf(Tree& t, int B, const std::string& precision, const std::vector<Disc>& discs, const std::vector<Ring>& R, const Rule& rule, int wgLimit) {
  const bool f16 = precision == "f16";
  const int SLOTS = LIZARD_NPICTURES;
  constexpr int ANCH_ROWS = plan::ANCH_SIDE + 4, ANCH_P1_BYTES = 8 * 4 * ANCH_ROWS + 4 * (plan::ANCH_SIDE + 2) + 2 * 4 * plan::ANCH_SIDE + 4 * 4;
  std::vector<const Disc*> sizes(static_cast<size_t>(SLOTS), nullptr);
  for (size_t s = 0; s < discs.size() && s < sizes.size(); s++) {
    const int n = discs[s].n, need = 2 * n * (f16 ? 4 : 8) + 16 * (n / 8) + 16 * (n / 8 + 1) + ANCH_P1_BYTES;
    if (need <= wgLimit) sizes[s] = &discs[s];
  }
  const Dims D(sizes, B);
  std::vector<int> built;
  for (size_t s = 0; s < sizes.size(); s++) if (sizes[s]) built.push_back(static_cast<int>(s));
  const int listsHead = SLOTS * (1 + B), listsRows = listsHead + 3 * B, listsRing = listsHead + 4 * B, argsLdpc0 = SLOTS * plan::ARGS_WORDS;
  json T = json::object();
  // ai: the transform's layouts and kernels (gate, pass1, reduce, pass2 .slang)
  const std::string gateBgl = t.layout({"ro", "ro", "rw", "rw", "uniform", "rw", "rw", "rw", "rw"});
  const std::string p1fBgl = t.layout({"tex", "ro", "ro", "rw", "rw", "rw", "uniform", "ro", "ro", "uniform", "ro", "sampler"});
  const std::string redBgl = t.layout({"ro", "ro", "rw", "uniform"});
  const std::string p2Bgl = t.layout({"ro", "ro", "ro", "ro", "ro", "ro", "uniform", "rw", "rw", "uniform"});
  int counts = 0;
  for (int s : built) counts = std::max(counts, 1 + sizes[static_cast<size_t>(s)]->blocks / CHUNKS);
  {
    Defs d = {{"BATCH", defU(B)}, {"SLOT_COUNT", defU(SLOTS)}, {"GATE_ROWS", js((counts * SLOTS + 3) / 4)}, {"LISTS_HEAD", defU(listsHead)},
              {"ARGS_LDPC", defU(argsLdpc0)}, {"ARGS_WORDS", defU(plan::ARGS_WORDS)}, {"NC", defU(plan::NC)}, {"CHUNKS", defU(CHUNKS)}, {"BLK_STRIDE", defU(D.blocksMax)}};
    addCfg(d);
    T["gate"] = t.pipeOn({"gate.slang", d}, gateBgl, "back gate")["p"];
  }
  json pass1f = json::array(), reduce = json::array(), pass2 = json::array();
  std::vector<std::string> p1(static_cast<size_t>(SLOTS)), rd(static_cast<size_t>(SLOTS)), p2(static_cast<size_t>(SLOTS));
  for (int s : built) {
    const int n = sizes[static_cast<size_t>(s)]->n;
    const std::string ns = std::to_string(n);
    rd[static_cast<size_t>(s)] = t.pipeOn({"reduce.slang", {{"PICTURE_N", defU(n)}, {"BATCH", defU(B)}}}, redBgl, "back reduce " + ns)["p"];
    p2[static_cast<size_t>(s)] = t.pipeOn({"pass2.slang", {{"F16", js(f16)}, {"LANES", js(n / 8)}, {"PICTURE_N", defU(n)}, {"BATCH", defU(B)}, {"LISTS_ROWS", defU(listsRows)}}}, p2Bgl, "back pass2 " + ns)["p"];
    p1[static_cast<size_t>(s)] = t.pipeOn({"pass1.slang", {{"F16", js(f16)}, {"LANES", js(n / 8)}, {"PICTURE_N", defU(n)}, {"BATCH", defU(B)}, {"LISTS_HEAD", defU(listsHead)},
                                                          {"LISTS_ROWS", defU(listsRows)}, {"LISTS_RING", defU(listsRing)}, {"ANCH_SIDE", defU(plan::ANCH_SIDE)},
                                                          {"ANCH_ROWS", defU(ANCH_ROWS)}, {"ANCH_REC", defU(plan::ANCH_REC)}, {"ANCH_SURF", defU(plan::ANCH_SURF)},
                                                          {"ANCH_GAIN", defU(plan::ANCH_GAIN)}, {"ANCH_ERASE", js(plan::ANCH_ERASE)}, {"RING_COUNT", js(LIZARD_RINGS)},
                                                          {"LATTICE_VECS", js(LIZARD_RINGS * LATTICE_MAX / 4)}, {"RESID", defU(plan::RESID)}, {"LUMH_BINDING", js(11)}}},
                                        p1fBgl, "back pass1 fused " + ns)["p"];
  }
  // ai: the shared tables (the twiddles, x12, the rows uniform of 2048 words, PIC, the gate's uniform)
  auto perSlot = [&](const std::vector<std::string>& ids) { json a = json::array(); for (const auto& x : ids) a.push_back(x.empty() ? json(nullptr) : json(x)); return a; };
  std::vector<std::string> tw(static_cast<size_t>(SLOTS)), x12(static_cast<size_t>(SLOTS)), rows(static_cast<size_t>(SLOTS)), pic(static_cast<size_t>(SLOTS));
  auto up = [&](std::vector<uint8_t> b, uint32_t usage) { const size_t size = std::max<size_t>(16, b.size()); return t.buffer(size, usage, std::move(b)); };
  for (int s : built) {
    const Disc& z = *sizes[static_cast<size_t>(s)];
    tw[static_cast<size_t>(s)] = up(raw(z.tw), STORAGE_DST);
    x12[static_cast<size_t>(s)] = up(raw(z.x12), STORAGE_DST);
    if (z.V > 512) throw std::runtime_error("V = " + std::to_string(z.V) + " rows at n = " + std::to_string(z.n) + ": the rows uniform holds 512");
    std::vector<uint32_t> r(2048, 0);
    std::copy(z.rows.begin(), z.rows.end(), r.begin());
    rows[static_cast<size_t>(s)] = up(raw(r), UNIFORM_DST);
  }
  // ai: PIC a size (pass1.slang's PICTURE): per ring (g0, step, nodes a side, first coordinate), then every
  // ai: ring's lattice, LATTICE_MAX a ring (a picture's grid in a ring: g0 = margin + 0.5 span / n, step span / n)
  for (int s : built) {
    std::vector<float> fl(4 * R.size() + R.size() * LATTICE_MAX, 0.0f);
    for (size_t r = 0; r < R.size(); r++) {
      const auto& lat = R[r].nodeX;
      if (lat.size() > static_cast<size_t>(LATTICE_MAX) || lat.size() < 2) throw std::runtime_error(std::to_string(lat.size()) + " lattice nodes a side in ring " + std::to_string(r));
      const double n = sizes[static_cast<size_t>(s)]->n, g0 = R[r].margin + (0.5 * R[r].span) / n, step = R[r].span / n;
      fl[4 * r] = static_cast<float>(g0);
      fl[4 * r + 1] = static_cast<float>(step);
      fl[4 * r + 2] = static_cast<float>(lat.size());
      fl[4 * r + 3] = static_cast<float>(r * LATTICE_MAX);
      std::copy(lat.begin(), lat.end(), fl.begin() + static_cast<long>(4 * R.size() + r * LATTICE_MAX));
    }
    pic[static_cast<size_t>(s)] = up(raw(fl), UNIFORM_DST);
  }
  {
    // ai: the gate's uniform: (n / CPW, V, the size's top version, 0) a slot, then
    // ai: the rows each version takes at each slot
    std::vector<uint32_t> g(4 * static_cast<size_t>(SLOTS) + 4 * static_cast<size_t>((counts * SLOTS + 3) / 4), 0);
    for (int s : built) {
      const Disc& z = *sizes[static_cast<size_t>(s)];
      const uint32_t at[4] = {static_cast<uint32_t>(z.n / plan::CPW), static_cast<uint32_t>(z.V), static_cast<uint32_t>(z.blocks / CHUNKS), 0};
      std::copy(at, at + 4, g.begin() + 4 * s);
      for (int v = 0; v < counts; v++) g[static_cast<size_t>(4 * SLOTS + v * SLOTS + s)] = z.rowsOf(v);
    }
    T["gateUni"] = up(raw(g), UNIFORM_DST);
  }
  int yStride = 0, partStride = 0;
  json tsizes = json::array();
  for (int s = 0; s < SLOTS; s++) {
    const Disc* z = sizes[static_cast<size_t>(s)];
    if (!z) { tsizes.push_back(nullptr); continue; }
    yStride = std::max(yStride, z->V * z->n);
    partStride = std::max(partStride, z->n);
    tsizes.push_back({{"n", z->n}, {"subch", z->subch}, {"blocks", z->blocks}, {"m2", static_cast<double>(z->sx2) / z->n}, {"sx2", z->sx2}, {"sq2", z->sq2}});
  }
  T["built"] = built;
  T["cbytes"] = f16 ? 4 : 8;
  T["yStride"] = yStride;
  T["sStride"] = D.sStride;
  T["partStride"] = partStride;
  T["gateBgl"] = gateBgl;
  T["p1fBgl"] = p1fBgl;
  T["redBgl"] = redBgl;
  T["p2Bgl"] = p2Bgl;
  T["pass1f"] = perSlot(p1);
  T["reduce"] = perSlot(rd);
  T["pass2"] = perSlot(p2);
  T["twBufs"] = perSlot(tw);
  T["x12Bufs"] = perSlot(x12);
  T["rowsBufs"] = perSlot(rows);
  T["picUnis"] = perSlot(pic);
  T["sizes"] = tsizes;
  // ai: the soft values' tables: UV a size, each at its own offset, then UC at the same offsets again; a
  // ai: size's params (the slot, UV's offset, the strides, B, UC's offset, n)
  int uvTotal = 0;
  std::vector<int> uvOff(static_cast<size_t>(SLOTS), 0);
  for (int s : D.served) { uvOff[static_cast<size_t>(s)] = uvTotal; uvTotal += static_cast<int>(sizes[static_cast<size_t>(s)]->UV.size()); }
  std::vector<uint32_t> uv(2 * static_cast<size_t>(uvTotal), 0);
  for (int s : D.served) {
    const Disc& z = *sizes[static_cast<size_t>(s)];
    std::copy(z.UV.begin(), z.UV.end(), uv.begin() + uvOff[static_cast<size_t>(s)]);
    std::copy(z.UC.begin(), z.UC.end(), uv.begin() + uvTotal + uvOff[static_cast<size_t>(s)]);
  }
  const std::string uvBuf = t.buffer(8 * static_cast<uint64_t>(uvTotal), STORAGE_DST, raw(uv));
  std::vector<std::string> params(static_cast<size_t>(SLOTS));
  for (int s : D.served) {
    const std::vector<uint32_t> p = {static_cast<uint32_t>(s), static_cast<uint32_t>(uvOff[static_cast<size_t>(s)]), static_cast<uint32_t>(D.sStride), static_cast<uint32_t>(D.lStride),
                                     static_cast<uint32_t>(D.subchMax), static_cast<uint32_t>(D.blocksMax), static_cast<uint32_t>(B),
                                     static_cast<uint32_t>(uvTotal + uvOff[static_cast<size_t>(s)]), static_cast<uint32_t>(sizes[static_cast<size_t>(s)]->n), 0, 0, 0};
    params[static_cast<size_t>(s)] = t.buffer(48, UNIFORM_DST, raw(p));
  }
  json blocks = json::array();
  for (int s = 0; s < SLOTS; s++) blocks.push_back(sizes[static_cast<size_t>(s)] ? sizes[static_cast<size_t>(s)]->blocks : 0);
  const json pilot = 4 * plan::PILOT_READ * B + 8 * B;
  const json soft = {{"uvBuf", uvBuf}, {"params", perSlot(params)}, {"served", D.served}, {"blocks", blocks},
                     {"bytes", {{"L", 4 * B * D.lStride}, {"BLK", 8 * B * D.blocksMax}, {"PILOT", pilot}}}};
  const json ldBytes = {{"V", 4 * B * D.blocksMax}, {"ITS", 4 * B * D.blocksMax}, {"REC", 4 + plan::REC_BYTES * D.recCap}};
  const json ldpc = {{"bytes", ldBytes}, {"blocksMax", D.blocksMax}, {"recCap", D.recCap}};
  // ai: the rule stage: five layouts, TAB, each code's uniform, the kernels
  const std::vector<Shape> shapes = [&] { std::vector<Shape> v; for (const Code& c : rule.codes) v.emplace_back(c, LDPC_FORMS[0]); return v; }();
  std::vector<std::vector<Shape>> alts;
  for (size_t f = 1; f < std::size(LDPC_FORMS); f++) { alts.emplace_back(); for (const Code& c : rule.codes) alts.back().emplace_back(c, LDPC_FORMS[f]); }
  auto fits = [&](const std::vector<Shape>& set) {
    for (size_t c = 0; c < set.size(); c++)
      if (set[c].wgBytes(rule.stops[c].hidden) > wgLimit) throw std::runtime_error(std::string("the ") + CODE_NAMES[c] + " code (" + set[c].form + ") needs " + std::to_string(set[c].wgBytes(rule.stops[c].hidden)) + " B of workgroup memory, the device offers " + std::to_string(wgLimit));
  };
  fits(shapes);
  for (const auto& set : alts) fits(set);
  const std::string noiseBgl = t.layout({"ro", "ro", "ro", "rw", "uniform"});
  const std::string alignBgl = t.layout({"ro", "ro", "ro", "rw", "ro", "uniform", "ro"});
  const std::string softBgl = t.layout({"ro", "ro", "ro", "rw", "rw", "rw", "ro", "ro", "uniform"});
  const std::string blocksBgl = t.layout({"ro", "ro", "rw", "rw", "ro", "uniform"});
  const std::string ldpcBgl = t.layout({"ro", "ro", "ro", "ro", "rw", "rw", "rw", "rw", "uniform"});
  const std::string tabBuf = t.buffer(4 * rule.tab.size(), STORAGE_DST, raw(rule.tab));
  std::vector<std::string> tBufs;
  for (size_t c = 0; c < rule.codes.size(); c++) { const auto w = rule.uniform(c, shapes[c], D); tBufs.push_back(t.buffer(4 * w.size(), UNIFORM_DST, raw(w))); }
  const Defs lists = {{"SLOT_COUNT", defU(SLOTS)}};
  auto with = [](Defs a, const Defs& b) { a.insert(a.end(), b.begin(), b.end()); return a; };
  Defs tab;
  rule.tabDefs(tab);
  Defs noiseD = with({{"F16", js(f16)}, {"CHUNKS", defU(CHUNKS)}}, lists);
  addCfg(noiseD);
  const std::string TWO_PI = "6.283185307";   // ai: 2 pi to nine places
  const std::string noise = t.pipeOn({"tnoise.slang", noiseD}, noiseBgl, "tnoise")["p"];
  const std::string align = t.pipeOn({"talign.slang", with(with({{"F16", js(f16)}, {"TIER_BLOCKS", defU(plan::TIER_BLOCKS)}, {"STEP", defU(LIZARD_STEP)}, {"PILOT_READ", defU(plan::PILOT_READ)}, {"TWO_PI", TWO_PI}}, lists), tab)}, alignBgl, "talign")["p"];
  const std::string softP = t.pipeOn({"tsoft.slang", with(with({{"F16", js(f16)}, {"CHUNKS", defU(CHUNKS)}, {"PILOT_READ", defU(plan::PILOT_READ)}, {"DCOEF", defU(DCOEF)}, {"TWO_PI", TWO_PI}}, lists), tab)}, softBgl, "tsoft")["p"];
  const std::string blocksP = t.pipeOn({"tblocks.slang", with(with({{"TIER_BLOCKS", defU(plan::TIER_BLOCKS)}, {"STEP", defU(LIZARD_STEP)}, {"PILOT_READ", defU(plan::PILOT_READ)}}, lists), tab)}, blocksBgl, "tblocks")["p"];
  std::vector<std::string> ldpcP;
  for (size_t c = 0; c < rule.codes.size(); c++) ldpcP.push_back(t.pipeOn(rule.ldpcRecipe(c, shapes[c]), ldpcBgl, std::string("tldpc ") + CODE_NAMES[c])["p"]);
  std::vector<std::vector<std::string>> altP;
  for (const auto& set : alts) {
    altP.emplace_back();
    for (size_t c = 0; c < set.size(); c++) altP.back().push_back(t.pipeOn(rule.ldpcRecipe(c, set[c]), ldpcBgl, std::string("tldpc ") + CODE_NAMES[c] + " " + set[c].form)["p"]);
  }
  // ai: its steps: every dispatch indirect from the lane's ARGS at argsAt (a slot's soft or reduce entry, a code's LDPC's)
  auto argsAt = [&](int s, int slot) { return 4 * (plan::ARGS_WORDS * s + slot); };
  auto step = [](const std::string& p, const std::string& bgl, json binds, int at) { return json{{"pipeline", p}, {"bgl", bgl}, {"binds", std::move(binds)}, {"argsAt", at}}; };
  json softSteps = json::array();
  for (int s : D.served) softSteps.push_back(step(noise, noiseBgl, {"lane:S", uvBuf, "lane:LISTS", "lane:EP", params[static_cast<size_t>(s)]}, argsAt(s, plan::ARGS_SOFT)));
  for (int s : D.served) softSteps.push_back(step(align, alignBgl, {"lane:S", uvBuf, "lane:LISTS", "lane:PILOT", tabBuf, params[static_cast<size_t>(s)], "lane:EP"}, argsAt(s, plan::ARGS_REDUCE)));
  for (int s : D.served) softSteps.push_back(step(softP, softBgl, {"lane:S", uvBuf, "lane:LISTS", "lane:L", "lane:EP", "lane:BCOUNTS", "lane:PILOT", tabBuf, params[static_cast<size_t>(s)]}, argsAt(s, plan::ARGS_SOFT)));
  for (int s : D.served) softSteps.push_back(step(blocksP, blocksBgl, {"lane:LISTS", "lane:EP", "lane:BLK", "lane:PILOT", tabBuf, params[static_cast<size_t>(s)]}, argsAt(s, plan::ARGS_REDUCE)));
  auto ldpcOf = [&](const std::vector<std::string>& pipes) {
    json a = json::array();
    for (size_t c = 0; c < pipes.size(); c++)
      a.push_back(step(pipes[c], ldpcBgl, {"lane:L", "lane:BLK", "lane:LISTS", tabBuf, "lane:V", "lane:ITS", "lane:REC", "lane:BCOUNTS", tBufs[c]}, 4 * (argsLdpc0 + 3 * static_cast<int>(c))));
    return a;
  };
  json ruleOut = {{"bytes", {{"EP", 16 * B * D.subchMax}}}, {"soft", softSteps}, {"ldpc", ldpcOf(ldpcP)}, {"ldpcForms", json::array({LDPC_FORMS[0], LDPC_FORMS[1], LDPC_FORMS[2]})}};
  if (!altP.empty()) { json a = json::array(); for (const auto& set : altP) a.push_back(ldpcOf(set)); ruleOut["ldpcAlts"] = a; }
  const int rec = 4 + plan::REC_BYTES * D.recCap, v = 4 * B * D.blocksMax, bc = 4 * plan::COUNTS * B;
  const json bytes = {{"REC", rec}, {"V", v}, {"ITS", v}, {"BCOUNTS", bc}, {"PILOT", pilot}};
  return {{"B", B}, {"precision", precision}, {"dims", D.out()}, {"bytes", bytes}, {"readBytes", (rec + v + v + bc + pilot.get<int>() + 7) & ~7},
          {"transform", T}, {"soft", soft}, {"ldpc", ldpc}, {"rule", ruleOut}};
}

}  // namespace

// ai: What every back half is made from: each picture size's tables at
// ai: its top, the codes, their stops and TAB, the rings' grids, the floats' precision
struct BackPlan {
  std::vector<Disc> discs;
  Rule rule;
  std::vector<Ring> R;
  std::string precision;
};

json Tree::back(int B) {
  if (!backPlan) throw std::runtime_error("a tree made without its back half's plan");
  return backHalf(*this, B, backPlan->precision, backPlan->discs, backPlan->R, backPlan->rule, plan::WG_BYTES);
}

namespace {
Tree makeWith(std::shared_ptr<const Modules> modules, const std::function<std::vector<uint8_t>(const std::string&)>& net, const std::string& variantName,
              const std::vector<int>& backs, int cap);
}  // namespace

Tree make(const ReadFile& read, const std::string& variantName, const std::vector<int>& backs, int cap) {
  const Variant v = variantOf(variantName);
  const auto ib = read("setup/" + v.name + ".json");
  const json index = json::parse(ib.begin(), ib.end());
  const int version = index.value("treeVersion", 0);
  if (version != plan::TREE_VERSION)
    throw std::runtime_error("the setup is tree version " + std::to_string(version) + ", this host reads " + std::to_string(plan::TREE_VERSION) + " (liblizard/gen/gen.mjs on another tree): generate it again");
  // ai: a net's file by its library path, through the index's blob for it
  const auto net = [&](const std::string& path) {
    const json& nets = index.at("nets");
    if (!nets.contains(path)) throw std::runtime_error("the setup carries no " + path + " (liblizard/gen/gen.mjs on another tree): generate it again");
    return read("blobs/" + nets[path].get<std::string>() + ".bin");
  };
  return makeWith(std::make_shared<const Modules>(index), net, variantName, backs, cap);
}

const char* const VARIANTS[6] = {"int8-sg", "int8", "f16-sg", "f16", "f32-sg", "f32"};
const int BACK_SIZES[6] = {plan::B_CEIL, 16, 8, 4, 2, 1};

Listing recipes(const ReadFile& readLib, const std::string& variant, int cap) {
  const auto rec = Modules::recording();
  Listing out;
  const auto net = [&](const std::string& path) {
    if (std::find(out.nets.begin(), out.nets.end(), path) == out.nets.end()) out.nets.push_back(path);
    return readLib(path);
  };
  out.tree = makeWith(rec, net, variant, std::vector<int>(std::begin(BACK_SIZES), std::end(BACK_SIZES)), cap);
  out.recipes = rec->recorded();
  return out;
}

json formatTables() {
  json rs = json::array();
  for (const Ring& t : rings()) {
    json order = json::array();
    for (const auto& [i, j] : ringOrder(static_cast<int>(t.nodeX.size()), static_cast<int>(t.nodeX.size()))) order.push_back({i, j});
    rs.push_back({{"ring", t.ring}, {"cells", t.cells}, {"span", t.span}, {"margin", t.margin}, {"modules", t.modules}, {"nodeX", t.nodeX}, {"nodeY", t.nodeY},
                  {"kind", t.kind}, {"mark", {{"gappedMid", t.gappedMid}, {"mergedMid", t.mergedMid}}},
                  {"track", {{"refSteps", t.track.refSteps}, {"cells", t.track.cells}, {"pts", t.track.pts}, {"signs", t.track.signs}}},
                  {"word", {{"cells", t.word.cells}, {"refSteps", t.word.refSteps}, {"reps", t.word.reps}, {"mids", t.word.mids}, {"words", t.word.words},
                            {"pts", t.word.pts}, {"widx", t.word.widx}}},
                  {"anch", {{"side", t.anch.side}, {"k", t.anch.k}, {"m0", t.anch.m0}}}, {"order", order}});
  }
  return {{"rings", rs}, {"pictures", pictures()}};
}

json wordTables() {
  const WordPlan P;
  const WordTables WT(rings(), pictures(), P);
  return {{"consts", {{"WMAX", P.WMAX}, {"ROWS", P.ROWS}, {"VECS", P.VECS}, {"UNKNOWN", P.UNKNOWN}, {"SWAP", P.SWAP}, {"PER", P.PER}, {"SETS", P.SETS},
                      {"TOP", P.TOP}, {"BYTES_MAX", P.BYTES_MAX}, {"GATE", P.GATE}, {"CON_MIN", P.CON_MIN}}},
          {"layout", {{"range", P.range}, {"planAt", P.planAt}, {"tops", P.tops}, {"gate", P.gate}, {"gf", P.gf}, {"cols", P.cols}, {"before", P.before}, {"size", P.size}}},
          {"uni", WT.uni}};
}

json codesTable() {
  const auto cs = codes::read();
  json out = json::array();
  for (size_t q = 0; q < cs.size(); q++) {
    const auto& c = cs[q];
    out.push_back({{"name", codes::NAMES[q]}, {"rate", OB_TIER_RATE[q]}, {"n", c.n}, {"k", c.k}, {"m", c.m}, {"z", c.z}, {"zp", c.zp}, {"mb", c.mb}, {"kb", c.kb},
                   {"norm", c.norm}, {"slotsMax", c.slotsMax}, {"slotsTotal", c.slotsTotal}, {"slots", c.slots}, {"np", c.np}, {"nt", c.nt},
                   {"blockSlots", c.blockSlots}, {"lay", c.lay}, {"perm", codes::perm(c.nt)}});
  }
  const auto w = codes::whiten(codes::WHITE_SLOTS);
  std::vector<uint32_t> white(static_cast<size_t>(codes::WHITE_SLOTS / 32 + 1), 0);
  for (size_t i = 0; i < w.size(); i++) if (w[i]) white[i >> 5] |= 1u << (i & 31);
  return {{"codes", out}, {"white", white},
          {"consts", {{"VERSION_SLOTS", OB_FMT_SLOTS}, {"SUB_SLOTS", SUB_SLOTS}, {"DCOEF", DCOEF}, {"PB_REF", plan::PB_REF}, {"KNOWN", plan::KNOWN},
                      {"WHITE_SLOTS", codes::WHITE_SLOTS}, {"PAYLOAD", plan::PAYLOAD}}}};
}

json stopsTable(const ReadFile& readLib) {
  json out = json::object();
  for (const auto& x : plan::STOPS) {
    const Stop s = readStop(readLib(x.file), x.file);
    out[std::to_string(x.m)] = {{"file", x.file}, {"m", s.m}, {"code", s.code}, {"hidden", s.hidden}, {"budget", s.budget}, {"T", s.T}, {"words", s.words()}};
  }
  return out;
}

json discTable(int n, int subch) {
  const Disc d(n, subch);
  json versions = json::array();
  for (int v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX && LIZARD_STEP * v <= subch; v++)
    if (lizard_n_for(LIZARD_STEP * v) == n) versions.push_back({{"v", v}, {"V", Disc(n, LIZARD_STEP * v).V}, {"rows", d.rowsOf(v)}});
  return {{"n", d.n}, {"subch", d.subch}, {"blocks", d.blocks}, {"npos", d.npos}, {"V", d.V}, {"rows", d.rows}, {"rowsBy", d.rowsBy}, {"UV", d.UV}, {"UC", d.UC},
          {"versions", versions}};
}

json wordCodes(const json& req) {
  const std::vector<Ring> R = rings();
  // ai: a codeword of `bytes` from six data bytes (src/rs.c rs_encode) and its 4 W bits as the C lays them (ob_fmt_bit)
  const auto cwOf = [](const uint8_t* data, int bytes) { std::vector<uint8_t> cw(static_cast<size_t>(bytes), 0); memcpy(cw.data(), data, OB_FMT_DATA); rs_encode(cw.data(), OB_FMT_DATA, bytes - OB_FMT_DATA); return cw; };
  const auto bitsOf = [](const std::vector<uint8_t>& cw, int W) {
    std::vector<uint32_t> b(static_cast<size_t>((4 * W + 31) / 32), 0);
    for (int s = 0; s < 4; s++) for (int c = 0; c < W; c++) if (ob_fmt_bit(cw.data(), static_cast<int>(cw.size()), s, c)) b[static_cast<size_t>((s * W + c) >> 5)] |= 1u << ((s * W + c) & 31);
    return b;
  };
  json rings = json::object();
  int Wmax = 0, Bmax = 0;
  for (const Ring& t : R) {
    const int W = t.word.cells, n = t.word.bytes;
    if (W > Wmax) { Wmax = W; Bmax = n; }
    uint8_t d[OB_FMT_DATA] = {OB_FMT_MAGIC, 0, 0, 0, 0, 0};
    json rows = json::array();
    for (int j = 1; j < OB_FMT_DATA; j++)
      for (int b = 0; b < 8; b++) { uint8_t e[OB_FMT_DATA] = {0, 0, 0, 0, 0, 0}; e[j] = static_cast<uint8_t>(1u << b); rows.push_back(bitsOf(cwOf(e, n), W)); }
    rings[std::to_string(W)] = {{"W", W}, {"bytes", n}, {"base", bitsOf(cwOf(d, n), W)}, {"rows", rows}, {"gate", gateTable(W)}};
  }
  json words = json::array();
  for (const auto& w : req.value("words", json::array())) {
    ob_fmt_t f{};
    f.version = w.at(0);
    for (int k = 0; k < OB_TIERS; k++) f.count[k] = w.at(1 + static_cast<size_t>(k));
    f.fps = w.at(5);
    f.delay = w.at(6);
    uint8_t data[OB_FMT_DATA];
    if (ob_fmt_pack(&f, data)) { words.push_back(nullptr); continue; }
    json by = json::object();
    for (const Ring& t : R) {
      std::vector<uint8_t> cw(static_cast<size_t>(t.word.bytes));
      if (ob_fmt_encode(&f, cw.data(), t.word.bytes)) throw std::runtime_error("ob_fmt_encode refused a word ob_fmt_pack took");
      by[std::to_string(t.word.cells)] = {{"cw", cw}, {"bits", bitsOf(cw, t.word.cells)}};
    }
    // ai: the word as a reader states it (ob_fmt_decode of its clean codeword in the largest ring: a still's delay 1)
    std::vector<float> q(4 * static_cast<size_t>(Wmax));
    {
      const auto bits = bitsOf(cwOf(data, Bmax), Wmax);
      for (size_t i = 0; i < q.size(); i++) q[i] = (bits[i >> 5] >> (i & 31)) & 1 ? 0.5f : -0.5f;
    }
    ob_fmt_t g{};
    if (ob_fmt_decode(q.data(), Wmax, Bmax, &g)) throw std::runtime_error("ob_fmt_decode refused a clean word");
    words.push_back({{"data", std::vector<uint8_t>(data, data + OB_FMT_DATA)}, {"rings", by},
                     {"read", {g.version, g.count[0], g.count[1], g.count[2], g.count[3], g.fps, g.delay}}});
  }
  json reads = json::array();
  for (const auto& d : req.value("reads", json::array())) {
    uint8_t data[OB_FMT_DATA];
    for (int j = 0; j < OB_FMT_DATA; j++) data[j] = d.at(static_cast<size_t>(j));
    const auto bits = bitsOf(cwOf(data, Bmax), Wmax);
    std::vector<float> q(4 * static_cast<size_t>(Wmax));
    for (size_t i = 0; i < q.size(); i++) q[i] = (bits[i >> 5] >> (i & 31)) & 1 ? 0.5f : -0.5f;
    ob_fmt_t f{};
    if (ob_fmt_decode(q.data(), Wmax, Bmax, &f)) reads.push_back(nullptr);
    else reads.push_back({{"version", f.version}, {"count", std::vector<int>(f.count, f.count + OB_TIERS)}, {"fps", f.fps}, {"delay", f.delay}});
  }
  json codes = json::array();
  for (const auto& c : req.value("codes", json::array())) {
    uint8_t data[OB_FMT_DATA];
    for (int j = 0; j < OB_FMT_DATA; j++) data[j] = c.at(0).at(static_cast<size_t>(j));
    const int W = c.at(1);
    const auto it = std::find_if(R.begin(), R.end(), [&](const Ring& t) { return t.word.cells == W; });
    if (it == R.end()) throw std::runtime_error("no ring of " + std::to_string(W) + " word cells a side");
    const auto cw = cwOf(data, it->word.bytes);
    codes.push_back({{"cw", cw}, {"bits", bitsOf(cw, W)}});
  }
  return {{"consts", {{"MAGIC", OB_FMT_MAGIC}, {"DATA", OB_FMT_DATA}, {"BYTES_MAX", OB_FMT_BYTES_MAX}, {"VERSION_MIN", OB_FMT_VERSION_MIN}, {"VERSION_MAX", OB_FMT_VERSION_MAX},
                      {"FPS_MAX", OB_FMT_FPS_MAX}, {"DELAY_MAX", OB_FMT_DELAY_MAX}, {"CLOCKS", OB_FMT_CLOCKS}, {"CONFIGS", OB_FMT_CONFIGS},
                      {"WORDS", static_cast<double>(OB_FMT_CONFIGS) * OB_FMT_CLOCKS}, {"TAU", WordPlan::TAU}, {"SIDE_KEEP", WordPlan::SIDE_KEEP}, {"CON_MIN", WordPlan::CON_MIN}}},
          {"rings", rings}, {"words", words}, {"reads", reads}, {"codes", codes}};
}

json planValues(const ReadFile& readLib) {
  const Finder F;
  json stops = json::object();
  for (const auto& s : plan::STOPS) stops[std::to_string(s.m)] = s.file;
  json proposer = json::object();
  for (const char* f : {plan::PROPOSER, plan::PROPOSER_INT8}) {
    const Net n = readNet(readLib(f), f);
    const Fcn2 P(n.arch);
    proposer[f] = {{"eps", P.eps}, {"nominal", P.nominal}, {"floor", P.floor}};
    // ai: an int8 file's input quantisation as the X pass takes it (Q8_INV, Q8_MAX)
    if (!n.quant.is_null()) {
      proposer[f]["inv"] = n.quant.at("input").at("inv").get<double>();
      proposer[f]["qmax"] = (1 << (n.quant.at("bits").get<int>() - 1)) - 1;
    }
  }
  return {{"files", {{"classifier", plan::CLASSIFIER}, {"classifierInt8", plan::CLASSIFIER_INT8}, {"proposer", plan::PROPOSER}, {"proposerInt8", plan::PROPOSER_INT8}, {"stops", stops}}},
          {"consts", {{"K", plan::K}, {"CELL", plan::CELL}, {"ORIENT", plan::ORIENT}, {"RING_COUNT", F.RING_COUNT}, {"HYP", F.HYP}, {"NODES_MAX", plan::NODES_MAX},
                      {"WMIN", WMIN}, {"KEEP", plan::KEEP}, {"TILE", Fcn2::TILE}, {"TAU", plan::TAU}, {"PAYLOAD", plan::PAYLOAD}, {"KAPPA", KAPPA}, {"PATCH", PATCH}}},
          {"proposer", proposer}};
}

json weightsTable(const ReadFile& readLib) {
  // ai: the forms as makeWith packs them: the installed file's arch for both files, a form's layout and words
  const Net cls = readNet(readLib(plan::CLASSIFIER), plan::CLASSIFIER);
  const auto archL = planArch(cls.arch);
  json layers = json::array();
  for (const auto& L : archL)
    layers.push_back({{"name", L.name}, {"type", L.type}, {"stride", L.stride}, {"inC", L.inC}, {"inS", L.inS}, {"outC", L.outC}, {"outS", L.outS}, {"inN", L.inN},
                      {"outN", L.outN}, {"shape", L.shape}});
  auto form = [&](const Net& n, bool half, bool int8) {
    size_t vecs = 0;
    json layout = json::object();
    for (const auto& [name, w] : weightLayout(archL, half, int8, &vecs)) layout[name] = {{"w", w.w}, {"b", w.b}, {"m", w.m}, {"kx", w.kx}};
    const auto bytes = int8 ? packInt8(n, archL) : packFloat(n, archL, half);
    std::vector<uint32_t> words(bytes.size() / 4);
    memcpy(words.data(), bytes.data(), 4 * words.size());
    return json{{"layout", layout}, {"vecs", vecs}, {"words", words}};
  };
  Net twin = readNet(readLib(plan::CLASSIFIER_INT8), plan::CLASSIFIER_INT8);
  if (twin.quant.is_null()) throw std::runtime_error(std::string(plan::CLASSIFIER_INT8) + " carries no lizard.quant");
  return {{"layers", layers},
          {"files", {{plan::CLASSIFIER, {{"spec", cls.spec}, {"forms", {{"f32", form(cls, false, false)}, {"f16", form(cls, true, false)}}}}},
                     {plan::CLASSIFIER_INT8, {{"spec", twin.spec}, {"quant", twin.quant}, {"forms", {{"int8", form(twin, false, true)}}}}}}}};
}

json rule(const ReadFile& readLib) {
  const Rule r(readLib);
  return {{"inv", r.inv}, {"white", r.WHITE}, {"net", r.NET}, {"netAt", r.netAt}, {"tab", r.tab}};
}

json dump(const Tree& t) {
  json d = t.tree;
  d["objects"] = {{"buffers", t.buffers}, {"bgls", t.bgls}, {"pipelines", t.pipelines}};
  return d;
}

namespace {
Tree makeWith(std::shared_ptr<const Modules> modules, const std::function<std::vector<uint8_t>(const std::string&)>& net, const std::string& variantName,
              const std::vector<int>& backs, int cap) {
  const Variant v = variantOf(variantName);
  Tree t;
  t.modules = std::move(modules);
  const int CAP = cap > 0 ? cap : plan::CAP;
  // ai: gen.mjs exportTree's scalars and constants
  t.part("/treeVersion", plan::TREE_VERSION);
  t.part("/variant", v.name);
  t.part("/precision", v.precision);
  t.part("/floatPrecision", v.floatPrecision);
  t.part("/subgroups", v.subgroups);
  t.part("/features", v.features);
  t.part("/cap", CAP);
  t.part("/modules", ringModules());
  t.part("/pictures", pictures());
  const Finder F;
  json consts = {{"LEVELS", plan::LEVELS}, {"COUNTERS", plan::COUNTERS}, {"RESULT", plan::RESULT}, {"SEL_WORDS", plan::SEL_WORDS}, {"TS_MAX", plan::TS_MAX},
                 {"TS_PLAIN", plan::TS_PLAIN}, {"RING_SLACK", plan::RING_SLACK}, {"K", plan::K}, {"CELL", plan::CELL}, {"HYP", F.HYP}, {"RING_COUNT", F.RING_COUNT},
                 {"VOTE_READS", plan::VOTE_READS}, {"PEAK_LIST", plan::PEAK_LIST}, {"R2_N", plan::R2_N}, {"NODES_MAX", plan::NODES_MAX}, {"RESID", plan::RESID},
                 {"PYRAMID_BLOCK", plan::PYRAMID_BLOCK}, {"TAU", plan::TAU}, {"TILE", plan::TILE}, {"B_CEIL", plan::B_CEIL}, {"CAP_MS", plan::CAP_MS},
                 {"TWIN_FRAMES", plan::TWIN_FRAMES}, {"TWIN_ROUNDS", plan::TWIN_ROUNDS}, {"VERSION_MIN", OB_FMT_VERSION_MIN}, {"VERSION_MAX", OB_FMT_VERSION_MAX},
                 {"SLOTS", LIZARD_NPICTURES}, {"ARGS_WORDS", plan::ARGS_WORDS},
                 {"ARGS_SLOT", {{"pass1", plan::ARGS_PASS1}, {"reduce", plan::ARGS_REDUCE}, {"pass2", plan::ARGS_PASS2}, {"soft", plan::ARGS_SOFT}}},
                 {"argsWords", LIZARD_NPICTURES * plan::ARGS_WORDS + 3 * plan::NC}, {"COUNTS", plan::COUNTS}, {"REC_BYTES", plan::REC_BYTES}, {"PAYLOAD", plan::PAYLOAD},
                 {"ANCH_K", plan::ANCH_K}, {"ANCH_REC", plan::ANCH_REC}, {"ANCH_GS", plan::ANCH_GS}};
  t.part("/consts", consts);
  // ai: the front's plain stages (in creation order: the pyramid, then the finder's and
  // ai: SELECT, PEAKS, TRACE); LV the bindings every stage that reads the pyramid takes
  const std::vector<std::string> LV = {"tex", "ro", "ro", "ro", "ro", "ro", "uniform"};
  auto with = [&](std::vector<std::string> a, std::initializer_list<std::string> b) { a.insert(a.end(), b); return a; };
  t.part("/front/pyramid", t.pipe({"pyramid.slang", {}}, {"tex", "ro", "rw", "rw", "rw", "rw", "uniform"}));
  t.part("/front/vote", t.pipe(F.vote(), {"ro", "ro", "rw", "uniform"}));
  t.part("/front/gather", t.pipe(F.gather(), {"ro", "ro", "ro", "rw", "rw", "uniform", "ro", "rw"}));
  t.part("/front/complete", t.pipe(F.complete(), with(LV, {"ro", "rw", "uniform"})));
  t.part("/front/score", t.pipe(F.score(), with(LV, {"ro", "ro", "rw", "uniform"})));
  t.part("/front/pick", t.pipe(F.pick(), {"ro", "ro", "ro", "rw", "rw", "rw", "uniform"}));
  t.part("/front/redo", t.pipe(F.redo(), {"ro", "ro", "rw", "rw", "rw", "rw", "rw", "uniform", "rw", "uniform"}));
  t.part("/front/select", t.pipe({"select.slang", {}}, {"ro", "ro", "rw", "rw", "uniform"}));
  t.part("/front/peaks", t.pipe(F.peaks(), {"ro", "ro", "rw", "rw", "uniform"}));
  t.part("/front/trace", t.pipe(F.trace(), with(LV, {"rw"})));
  // ai: the front's fixed buffers (after the stages)
  const std::vector<Ring> R = rings();
  const FrontTables T(R);
  t.part("/front/buffers/planBuf", t.buffer(4 * T.plan.size(), STORAGE_DST, raw(T.plan)));
  t.part("/front/buffers/planUni", t.buffer(4 * T.planAt.size(), UNIFORM_DST, raw(T.planAt)));
  t.part("/front/buffers/p4Uni", t.buffer(16, UNIFORM_DST, raw(std::vector<uint32_t>{static_cast<uint32_t>(CAP), 0, 0, 0})));
  {
    std::vector<float> pk(8, 0);
    pk[0] = static_cast<float>(plan::TAU_TRACK);
    for (int k = 0; k < 4; k++) pk[4 + static_cast<size_t>(k)] = T.modules[static_cast<size_t>(k)];
    t.part("/front/buffers/pickUni", t.buffer(32, UNIFORM_DST, raw(pk)));
  }
  t.part("/front/buffers/nodesBuf", t.buffer(4 * T.nodes.size(), STORAGE_DST, raw(T.nodes)));
  t.part("/front/buffers/tmplBuf", t.buffer(4 * T.tmpl.size(), STORAGE_DST, raw(T.tmpl)));
  t.part("/nodesMax", T.nodesMax);
  t.part("/front/round", t.buffer(64 + 4 * LIGHT_MAX, UNIFORM_DST, T.round(ROUND)));
  {
    std::vector<uint32_t> fit(12, 0);
    for (int k = 0; k < 4; k++) { fit[static_cast<size_t>(k)] = T.count[static_cast<size_t>(k)]; fit[4 + static_cast<size_t>(k)] = T.nbase[static_cast<size_t>(k)]; memcpy(&fit[8 + static_cast<size_t>(k)], &T.modules[static_cast<size_t>(k)], 4); }
    t.part("/front/buffers/fitUni", t.buffer(48, UNIFORM_DST, raw(fit)));
  }
  t.part("/anch/buffers/round", t.buffer(64 + 4 * LIGHT_MAX, UNIFORM_DST, T.round(ANCH_ROUND)));
  // ai: F7 and F8's stages, and the anchors' (ANCH_PREDICT on ANCH_REFIT's layout,
  // ai: the border's second search on NODES', ANCH_SUMS on ANCH_FIND's)
  const json nodesP = t.pipe(nodesRecipe(false, ROUND, WINDOW * WINDOW * SUB * SUB / 2, 512), with(LV, {"ro", "ro", "ro", "ro", "rw", "uniform"}));
  t.part("/front/nodesP", nodesP);
  t.part("/front/refit", t.pipe(refitRecipe(), {"ro", "ro", "ro", "ro", "rw", "rw", "uniform"}));
  t.part("/front/wordP", t.pipe(WordPlan().recipe(), with(LV, {"ro", "rw", "ro", "ro", "rw", "uniform", "uniform"})));
  const Anchors A;
  const json find = t.pipe(A.find(), {"tex", "ro", "ro", "ro", "ro", "uniform", "uniform", "rw", "rw"});
  const json refit = t.pipe(A.refit(false), {"ro", "ro", "ro", "rw", "rw", "rw", "uniform", "rw", "uniform", "uniform"});
  t.part("/anch/find", find);
  t.part("/anch/fit", t.pipe(A.fit(), {"tex", "ro", "ro", "ro", "ro", "uniform", "uniform", "ro", "rw", "rw", "ro"}));
  t.part("/anch/refit", refit);
  t.part("/anch/predict", t.pipeOn(A.refit(true), refit["bgl"].get<std::string>()));
  t.part("/anch/nodes", t.pipeOn(nodesRecipe(true, ANCH_ROUND, 144, 256), nodesP["bgl"].get<std::string>()));
  t.part("/anch/sums", t.pipeOn(A.sums(), find["bgl"].get<std::string>()));
  const WordTables WT(R, t.tree["pictures"], WordPlan());
  t.part("/front/buffers/wordPtsBuf", t.buffer(4 * WT.pts.size(), STORAGE_DST, raw(WT.pts)));
  t.part("/front/buffers/wordUni", t.buffer(4 * WT.uni.size(), UNIFORM_DST, raw(WT.uni)));
  t.part("/front/buffers/wordLatUni", t.buffer(WT.lattice.size(), UNIFORM_DST, WT.lattice));
  const auto au = anchUniform(R, Anchors::FILL, Anchors::BAND2);
  t.part("/anch/buffers/anchUni", t.buffer(4 * au.size(), UNIFORM_DST, raw(au)));
  // ai: F2, the proposer: the int8 file at int8, else the float file in the floats' precision;
  // ai: the net, the X pass and the X pass from the frame's texture, at the first form the workgroup memory holds
  {
    const bool int8 = v.precision == "int8";
    const char* file = int8 ? plan::PROPOSER_INT8 : plan::PROPOSER;
    const Net pn = readNet(net(file), file);
    const Fcn2 P(pn.arch);
    const auto form = P.pick(!int8 && v.floatPrecision == "f16", int8, plan::WG_BYTES);
    const double inv = int8 ? pn.quant.at("input").at("inv").get<double>() : 0;
    const int qmax = int8 ? (1 << (pn.quant.at("bits").get<int>() - 1)) - 1 : 0;
    t.part("/front/fcn2", t.pipe(P.netRecipe(form, pn, inv, qmax), {"ro", "ro", "rw", "rw", "uniform"}));
    t.part("/front/fcn2X", t.pipe(P.xRecipe(form, false, inv, qmax), {"ro", "ro", "rw", "uniform"}));
    t.part("/front/fcn2X0", t.pipe(P.xRecipe(form, true, inv, qmax), {"tex", "ro", "rw", "uniform"}));
    t.part("/front/fcn2Form", {{"TX", form.TX}, {"TY", form.TY}, {"XB", {form.XB0, form.XB1}}, {"walk", Fcn2::WALK}, {"xplane", true}, {"int8", form.int8},
                               {"store", form.store}, {"name", form.name}});
    t.part("/nets", {{"bank", form.store == "packed" ? "f32" : form.store}, {"large", v.floatPrecision}});
    t.part("/bankKeep", plan::KEEP);
    t.part("/front/coarseFrom", 2);
  }
  // ai: F4, the classifier: the installed file packed at the floats' precision, and at int8 its
  // ai: int8 twin built beside for the host to measure (gen's twins "measure" under B auto)
  const Net cls = readNet(net(plan::CLASSIFIER), plan::CLASSIFIER);
  const auto archL = planArch(cls.arch);
  const bool f16 = v.floatPrecision == "f16";
  auto netForm = [&](const Net& n, bool int8, bool half) {
    const ClassifyPlan c = classifyPlan(archL, half, int8, v.subgroups);
    const double inv = int8 ? n.arch.at("quant").at("input").at("inv").get<double>() : 0;
    json pipe = t.pipe(classifyRecipe(c, archL, half, int8, v.subgroups, inv), with(LV, {"ro", "ro", "rw", "uniform", "ro"}));
    pipe["P"] = c.P;
    const auto bytes = int8 ? packInt8(n, archL) : packFloat(n, archL, half);
    return std::pair<json, std::string>(pipe, t.buffer(bytes.size(), STORAGE_DST, bytes));
  };
  const auto [clsPipe, clsBuf] = netForm(cls, false, f16);
  t.part("/front/classify", clsPipe);
  t.part("/front/buffers/weightsBuf", clsBuf);
  t.part("/weightsPrecision", v.floatPrecision);
  if (v.precision == "int8") {
    Net twin = readNet(net(plan::CLASSIFIER_INT8), plan::CLASSIFIER_INT8);
    if (twin.quant.is_null()) throw std::runtime_error(std::string(plan::CLASSIFIER_INT8) + " carries no lizard.quant");
    twin.arch["quant"] = twin.quant;
    const auto [tp, tb] = netForm(twin, true, false);
    t.part("/front/twins", {{"large", {{"pipe", tp}, {"buf", tb}, {"precision", "int8"}}}});
  } else t.part("/front/twins", nullptr);
  // ai: the back half's plan: in the floats' precision, every picture at its top, the rings' grids
  {
    auto bp = std::make_shared<BackPlan>(BackPlan{{}, Rule(net), R, v.floatPrecision});
    for (const auto& p : t.tree["pictures"]) bp->discs.emplace_back(p["n"].get<int>(), p["subch"].get<int>());
    t.backPlan = bp;
  }
  t.tree["backs"] = json::object();
  for (int B : backs) {
    // ai: a part each of the stages' fields, so one apart leaves the rest compared
    const std::string at = "/backs/" + std::to_string(B);
    const json bh = t.back(B);
    t.tree["backs"][std::to_string(B)] = json::object();
    for (auto it = bh.begin(); it != bh.end(); ++it) {
      if (!it.value().is_object()) { t.part(at + "/" + it.key(), it.value()); continue; }
      t.tree[json::json_pointer(at)][it.key()] = json::object();
      for (auto f = it.value().begin(); f != it.value().end(); ++f) t.part(at + "/" + it.key() + "/" + f.key(), f.value());
    }
  }
  return t;
}
}  // namespace

}  // namespace lizard::build
