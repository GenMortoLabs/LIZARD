// ai: The GPU decoder's setup made by the C++ host (since 2026-11-25), from the plan (plan.h), the C codec and the nets'
// ai: files: every buffer with its bytes, every layout and pipeline, and the values the lanes and batches read (the tree
// ai: core/dec/front.cpp reads). What the host takes from gen is the variant's index alone (setup/<variant>.json: each
// ai: module's recipe, name, lanes and needs, and the nets' files as blobs/<hash>.bin): a module is named through it (a
// ai: kernel's file and its constants name one module). gen's recipes are the same code's (recipes below,
// ai: tools/gpu_tree.cpp).
#pragma once
#include "json.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lizard {
// ai: A file's bytes by its path under the generated tree (a directory, or the app's assets).
using ReadFile = std::function<std::vector<uint8_t>(const std::string&)>;
}  // namespace lizard

namespace lizard::build {

using json = nlohmann::json;

// ai: a module's recipe as gen.mjs compiles it: the kernel's file under gpu/slang/src and
// ai: its constants, each value Slang's text (a uint `16u`, a float as JavaScript prints it: js below)
struct Recipe {
  std::string src;
  std::vector<std::pair<std::string, std::string>> defs;
};

// ai: a double as JavaScript's Number toString writes it (the shortest that reads back, ECMAScript's exponent rules), so
// ai: a constant's text is the JS plans' to the character
std::string js(double v);
// ai: a define's text: a whole number `<n>u`, any other its js text, and lit (a float literal: a whole number `<n>.0`)
std::string defU(double v);
std::string lit(double v);
// ai: a double as JavaScript's toPrecision(9) writes it: nine significant digits of its exact decimal expansion, a tie
// ai: rounded away from zero, d.dddddddde+x for an exponent under -6 or past 8, 0 as 0.00000000
std::string precision9(double v);

// ai: A code's learned LDPC stop (ldpc-stop-mlp-v2), folded into the form the kernels and the C's decoder run: the
// ai: inputs' standardisation, 1 / m, the fall (prev - bad) and the growth's 1 / 256 folded into the first layer in
// ai: float64, each value then rounded once to float32; T the file's theta less the output bias. W1 is H x 6 and W2
// ai: H x H, row-major; words() the words the GPU's kernel reads (W1 input-major, b1, W2 input-major, b2, w3, T).
struct Stop {
  int hidden = 0, m = 0;
  std::vector<float> W1, b1, W2, b2, w3;
  float T = 0;
  std::string code;   // ai: the file's lizard.code (its rate's name)
  double budget = 0;  // ai: the file's lizard.budget (the trainer's)
  std::vector<uint32_t> words() const;
};
// ai: a stop file's bytes (a safetensors file), checked against the spec, its features and its tensors' shapes; path
// ai: names it in what a refusal says
Stop readStop(const std::vector<uint8_t>& bytes, const std::string& path);

// ai: a recipe's key: its file, then its constants sorted by name (a file and its constants name one module)
std::string recipeKey(const Recipe& r);

// ai: A variant's modules as gen compiled them (its index, setup/<variant>.json modules): each recipe to its module's
// ai: name, and a pipeline's facts from its SPIR-V (its workgroup's lanes; its needs, "int16" where it declares Int16:
// ai: gen.mjs slangFacts). Or, made by recording(), no index: every recipe asked for is kept (in the order first
// ai: asked) under a name of its key, lanes 0 and no needs; what gen compiles (recipes below).
class Modules {
 public:
  explicit Modules(const json& index);
  static std::shared_ptr<Modules> recording();
  std::string name(const Recipe& r) const;   // ai: throws, naming the nearest recipe of that file, where gen made none
  uint32_t lanes(const std::string& name) const { return record_ ? 0 : facts_.at(name).first; }
  const json& needs(const std::string& name) const { static const json none = json::array(); return record_ ? none : facts_.at(name).second; }
  const std::vector<Recipe>& recorded() const { return recorded_; }
 private:
  Modules() = default;
  bool record_ = false;
  std::map<std::string, std::string> byKey_;
  std::multimap<std::string, std::string> bySrc_;   // ai: file -> key, for the nearest recipe's report
  std::map<std::string, std::pair<uint32_t, json>> facts_;
  mutable std::vector<Recipe> recorded_;
  mutable std::map<std::string, std::string> recordedKeys_;
};

struct BackPlan;   // ai: what every back half is made from (the sizes' tables, the codes and stops, the rings)

// ai: The objects a tree names, made in creation order (b<n>, L<n>, p<n>)
struct Tree {
  json tree = json::object();
  json buffers = json::object(), bgls = json::object(), pipelines = json::object();
  std::map<std::string, std::vector<uint8_t>> bytes;   // ai: a buffer's bytes by its id
  std::vector<std::string> parts;   // ai: the tree's parts made here (JSON pointers)
  std::shared_ptr<const Modules> modules;
  std::shared_ptr<const BackPlan> backPlan;

  std::string buffer(uint64_t size, uint32_t usage, std::vector<uint8_t> data);
  std::string layout(const std::vector<std::string>& entries);
  // ai: { p, bgl }: a pipeline of the recipe on a layout of these entries, its label
  // ai: the back half's (null where none)
  json pipe(const Recipe& r, const std::vector<std::string>& entries, const std::string& label = "");
  // ai: { p, bgl }: one on an existing layout
  json pipeOn(const Recipe& r, const std::string& bgl, const std::string& label = "");
  void part(const std::string& pointer, json value);
  // ai: the back half at batch size B (build.cpp backHalf), its objects made: the tree's backs[B]
  json back(int B);

 private:
  int nb_ = 0, nl_ = 0, np_ = 0;
};

// ai: A variant's setup (int8, f16, f32, each with -sg or not): the front half, the anchors and the back half at each
// ai: of backs (tree.backs; Tree::back makes another), read through read (the variant's index, the nets' blobs).
// ai: cap: SELECT's cap, plan.h CAP where 0.
Tree make(const ReadFile& read, const std::string& variant, const std::vector<int>& backs = {}, int cap = 0);

// ai: The variants a tree holds (gen writes each one's index) and the back halves' batch sizes gen compiles for.
extern const char* const VARIANTS[6];
extern const int BACK_SIZES[6];

// ai: What gen compiles for a variant: the setup made with no index (Modules::recording), its nets' files read through
// ai: readLib (a file by its path under liblizard/); the recipes its pipelines name (the front, the anchors and the
// ai: back half at every BACK_SIZES), the nets' paths it read, and the tree made (its modules named by their keys).
struct Listing {
  std::vector<Recipe> recipes;
  std::vector<std::string> nets;
  Tree tree;
};
Listing recipes(const ReadFile& readLib, const std::string& variant, int cap = 0);

// ai: The rule stage's TAB as the back half binds it, for the lab (scripts/gpu/back/check_tiers.mjs): { inv: each
// ai: code's codeword bits' slots' first word, white: the whitening's, net: the stops' first, netAt: each code's stop,
// ai: tab: the words }, its stops' files read through readLib (a file by its path under liblizard/)
json rule(const ReadFile& readLib);

// ai: The format's tables as the host reads them from the C codec (the lab's: scripts/gpu/native/host.mjs formatTables):
// ai: { rings: each ring's cells, span, margin, modules, node lattice, kind map, the marks' mids, track and word plans,
// ai: anchors and its nodes' ring order, pictures: each picture size's n, top sub-channels and versions lo to hi }
json formatTables();
// ai: The codes as the hosts take them from the C (dec/codes.h, for the lab: check_tiers, the stop's dumps): { codes: each
// ai: code's name, rate, shape, block-row tables (lay) and LINEAR's map over its sent bits (perm: slot i's sent bit),
// ai: white: the whitening a bit a frame slot over the ladder's largest, packed 32 a word, consts: the slots and pilots }
json codesTable();
// ai: The GPU's learned stops (plan.h STOPS, each folded by readStop), for the lab's check_rule: { <m>: { file, code,
// ai: hidden, budget, T, words } }, read through readLib
json stopsTable(const ReadFile& readLib);
// ai: A picture size's transform tables as the back half takes them (Disc: its rows, UV and UC, rowsBy), and for each
// ai: version whose picture it is up to subch its own disc's V and the rows the gate gives it (rowsOf), for the lab
json discTable(int n, int subch);
// ai: The word reader's tables as the host makes them (the tree's wordUni, for the lab's scripts/gpu/word_check.mjs):
// ai: { consts: the reader's sizes (WMAX, ROWS, VECS, UNKNOWN, SWAP, PER, SETS, TOP, BYTES_MAX, GATE, CON_MIN), layout:
// ai: the uniform's sections in vec4u (range, planAt, tops, gate, gf, cols, before, size), uni: its words }
json wordTables();
// ai: The format word as the C has it, for the lab's word instruments (scripts/gpu/word_check.mjs, scripts/exp/word_arms.mjs):
// ai: { consts: the word's (MAGIC, DATA, the versions, the clock's range, CLOCKS, CONFIGS, WORDS) and the reader's (TAU,
// ai: SIDE_KEEP, CON_MIN), rings: by each ring's W (word cells a side) its codeword bytes (src/fmt.c ob_fmt_bytes), its
// ai: linear code (base: the magic alone, rows: each of the 40 data bits; each word's bits as the C lays them, ob_fmt_bit,
// ai: side * W + cell, a u32 a 32 bits) and the host's gate (E[m]), words: each req.words entry ([version, four counts, fps,
// ai: delay]) as src/fmt.c packs it (its six data bytes, null where it is no word) with its codeword and bits in each ring
// ai: and the word as a reader states it (read: [version, four counts, fps, delay]; a still's delay 1),
// ai: reads: each req.reads entry (six data bytes, the magic first) as src/fmt.c reads its clean codeword (ob_fmt_decode
// ai: in the largest ring: version, counts, fps, delay; null where it reads none), codes: each req.codes entry ([six data
// ai: bytes], W) as rs_encode makes its codeword and bits }
json wordCodes(const json& req);
// ai: The plan's values the lab reads (host.mjs planOf): the nets' files, the finder's and the registration's constants,
// ai: the bank's KEEP and TILE, TAU, the classifier's KAPPA and PATCH, and each proposer file's eps, nominal and floor as
// ai: the bank takes them (Fcn2; an int8 file's inv and qmax too), its files read through readLib (a file by its path
// ai: under liblizard/)
json planValues(const ReadFile& readLib);
// ai: The classifier's weights as the tree's buffers hold them (the lab's, scripts/gpu/cnn/pack_check.py): { layers:
// ai: planArch's (each layer's name, type, stride, sides, channels, shape), files: { <the installed file>: { spec,
// ai: forms: { f32, f16 } }, <its int8 twin>: { spec, quant, forms: { int8 } } } }, a form { layout: by layer its w, b,
// ai: m and kx in vec4s (weightLayout), vecs, words: the buffer as u32 }, read through readLib (a file by its path
// ai: under liblizard/)
json weightsTable(const ReadFile& readLib);

// ai: A made tree as JSON for the lab's scripts (scripts/gpu/native): its parts and its objects (each pipeline's module,
// ai: entry, label, lanes, needs and layouts; each layout's entries; each buffer's size and use)
json dump(const Tree& t);

}  // namespace lizard::build
