// ai: The GPU decoder's tree as the host makes it (dec/build.cpp), with no device: what gen compiles, and a made tree for
// ai: the lab's scripts.
//
//   lizard_gpu_tree recipes [lib] [variant...]   lib: liblizard/ (by default); every variant by default. JSON on stdout:
//       { treeVersion, variants: { <variant>: { recipes: [{ src, defs, key }], nets: [path], nets_used, pipelines,
//       buffers } } }: each recipe the variant's setup names (the front, the anchors, the back half at every batch size
//       gen compiles for), its constants in the host's order and its key (its file, its constants sorted), the nets'
//       files the setup read (paths under lib). LIZ_GEN_CAP=<n>: SELECT's cap (the lab's, a tree written elsewhere).
//   lizard_gpu_tree dump [out] <variant> [B...]  out: the tree (liblizard/out by default, LIZ_OUT). JSON on stdout: the
//       tree the host makes through gen's index (its parts and objects: build::dump), the back half at each B (none by
//       default).
//   lizard_gpu_tree format   the format's tables as the host reads them from the C (build::formatTables): JSON on
//       stdout, { rings, pictures }, the lab's (scripts/gpu/native/host.mjs formatTables).
//   lizard_gpu_tree codes   the codes, LINEAR's maps and the whitening as the hosts take them (build::codesTable).
//   lizard_gpu_tree stops [lib]   the GPU's learned stops, folded (build::stopsTable).
//   lizard_gpu_tree disc <n> <subch>   a picture size's transform tables and its versions' rows (build::discTable).
//   lizard_gpu_tree wordcode   the format word as the C has it (build::wordCodes), for the cases a JSON request on stdin
//       names ({ words: [[version, c0, c1, c2, c3, fps, delay]], reads: [[six data bytes]], codes: [[[six bytes], W]] }).
//   lizard_gpu_tree word   the word reader's packed uniform and its layout (build::wordTables): JSON on stdout,
//       { consts, layout, uni }, the lab's word_check.
//   lizard_gpu_tree plan [lib]   the plan's values the lab reads (build::planValues): { files, consts, proposer }.
//   lizard_gpu_tree weights [lib]   the classifier's weights as the tree's buffers hold them (build::weightsTable):
//       { layers, files: { <file>: { spec, quant, forms: { <f32|f16|int8>: { layout, vecs, words } } } } }.
//   lizard_gpu_tree rule [lib]   the rule stage's TAB (build::rule), its stops read under lib (liblizard/ by default):
//       JSON on stdout, { inv, white, net, netAt, tab }.
//   lizard_gpu_tree send   the GPU sender's (tx/send_plan.h, tx/send_tables.h), with no device: JSON on stdout,
//       { treeVersion, modules: [{ name, src, defs, web, native }], setup }: each module's recipe (its constants in the
//       plan's order), the stage the web's encoder runs it at (null: the native hosts' alone) and whether the native
//       hosts do; setup what no format changes (the paint's uniform's layout, the kernels' lanes and bands, the codes,
//       the CRC-64's powers, TAB's words).
#include "build.h"
#include "send_plan.h"
#include "send_tables.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using lizard::build::json;

namespace {

std::vector<uint8_t> slurp(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

int usage() {
  fprintf(stderr, "usage: lizard_gpu_tree recipes [lib] [variant...] | dump [out] <variant> [B...] | format | word | wordcode | codes | stops [lib] | disc <n> <subch> | plan [lib] | weights [lib] | rule [lib] | send\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string cmd = argv[1];
  try {
    if (cmd == "recipes") {
      int i = 2;
      std::filesystem::path lib = "liblizard";
      if (i < argc && std::filesystem::is_directory(std::filesystem::u8path(argv[i]))) lib = std::filesystem::u8path(argv[i++]);
      std::vector<std::string> variants;
      for (; i < argc; i++) variants.push_back(argv[i]);
      if (variants.empty()) variants.assign(std::begin(lizard::build::VARIANTS), std::end(lizard::build::VARIANTS));
      const int cap = getenv("LIZ_GEN_CAP") ? atoi(getenv("LIZ_GEN_CAP")) : 0;
      const auto read = [&](const std::string& path) { return slurp(lib / std::filesystem::u8path(path)); };
      json out = {{"variants", json::object()}};
      for (const auto& v : variants) {
        const auto L = lizard::build::recipes(read, v, cap);
        json rs = json::array();
        for (const auto& r : L.recipes) {
          json defs = json::object();
          for (const auto& [n, x] : r.defs) defs[n] = x;
          rs.push_back({{"src", r.src}, {"defs", defs}, {"key", lizard::build::recipeKey(r)}});
        }
        out["variants"][v] = {{"recipes", rs}, {"nets", L.nets}, {"nets_used", L.tree.tree.value("nets", json())}, {"pipelines", L.tree.pipelines.size()},
                              {"buffers", L.tree.buffers.size()}};
        out["treeVersion"] = L.tree.tree.at("treeVersion");
      }
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (cmd == "format") {
      std::cout << lizard::build::formatTables().dump() << "\n";
      return 0;
    }
    if (cmd == "wordcode") {
      std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
      std::cout << lizard::build::wordCodes(in.empty() ? json::object() : json::parse(in)).dump() << "\n";
      return 0;
    }
    if (cmd == "codes") {
      std::cout << lizard::build::codesTable().dump() << "\n";
      return 0;
    }
    if (cmd == "stops") {
      const std::filesystem::path lib = argc > 2 ? std::filesystem::u8path(argv[2]) : std::filesystem::path("liblizard");
      std::cout << lizard::build::stopsTable([&](const std::string& path) { return slurp(lib / std::filesystem::u8path(path)); }).dump() << "\n";
      return 0;
    }
    if (cmd == "disc" && argc > 3) {
      std::cout << lizard::build::discTable(atoi(argv[2]), atoi(argv[3])).dump() << "\n";
      return 0;
    }
    if (cmd == "word") {
      std::cout << lizard::build::wordTables().dump() << "\n";
      return 0;
    }
    if (cmd == "plan") {
      const std::filesystem::path lib = argc > 2 ? std::filesystem::u8path(argv[2]) : std::filesystem::path("liblizard");
      std::cout << lizard::build::planValues([&](const std::string& path) { return slurp(lib / std::filesystem::u8path(path)); }).dump() << "\n";
      return 0;
    }
    if (cmd == "weights") {
      const std::filesystem::path lib = argc > 2 ? std::filesystem::u8path(argv[2]) : std::filesystem::path("liblizard");
      std::cout << lizard::build::weightsTable([&](const std::string& path) { return slurp(lib / std::filesystem::u8path(path)); }).dump() << "\n";
      return 0;
    }
    if (cmd == "rule") {
      const std::filesystem::path lib = argc > 2 ? std::filesystem::u8path(argv[2]) : std::filesystem::path("liblizard");
      std::cout << lizard::build::rule([&](const std::string& path) { return slurp(lib / std::filesystem::u8path(path)); }).dump() << "\n";
      return 0;
    }
    if (cmd == "send") {
      using oj = nlohmann::ordered_json;
      oj mods = oj::array();
      for (const auto& m : lizard::send::modules()) {
        oj defs = oj::object();
        for (const auto& [n, x] : m.defs) defs[n] = x;
        mods.push_back({{"name", m.name}, {"src", m.src}, {"defs", defs}, {"web", m.webStage.empty() ? oj() : oj(m.webStage)}, {"native", m.native}});
      }
      const lizard::SendConsts k;
      oj codes = oj::array();
      for (size_t c = 0; c < k.codeRate.size(); c++) codes.push_back({{"rate", k.codeRate[c]}, {"slots", k.codeSlots[c]}});
      std::vector<int> sizes(LIZARD_PICTURES, LIZARD_PICTURES + LIZARD_NPICTURES);
      namespace S = lizard::send;
      const oj setup = {{"sizes", sizes}, {"slots", S::SLOTS}, {"paramsAt", {{"sizes", S::P_SIZES}, {"dims", S::P_DIMS}, {"pw", S::P_PW}, {"words", S::P_WORDS}}},
                        {"rshThreads", S::RSH_THREADS}, {"rshBand", S::RSH_BAND}, {"blockBytes", S::BLOCK_BYTES}, {"payload", lizard::plan::PAYLOAD},
                        {"gapModules", lizard::GAP_MODULES}, {"codes", codes}, {"pw", k.pw}, {"gWords", S::G_WORDS}, {"tab", k.tab}};
      std::cout << oj{{"treeVersion", lizard::plan::TREE_VERSION}, {"modules", mods}, {"setup", setup}}.dump() << "\n";
      return 0;
    }
    if (cmd == "dump") {
      int i = 2;
      std::filesystem::path out = getenv("LIZ_OUT") ? std::filesystem::u8path(getenv("LIZ_OUT")) : std::filesystem::path("liblizard/out");
      if (i < argc && std::filesystem::is_directory(std::filesystem::u8path(argv[i]))) out = std::filesystem::u8path(argv[i++]);
      if (i >= argc) return usage();
      const std::string variant = argv[i++];
      std::vector<int> backs;
      for (; i < argc; i++) backs.push_back(atoi(argv[i]));
      const auto read = [&](const std::string& path) { return slurp(out / std::filesystem::u8path(path)); };
      const auto t = lizard::build::make(read, variant, backs);
      std::cout << lizard::build::dump(t).dump() << "\n";
      return 0;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "lizard_gpu_tree %s: %s\n", cmd.c_str(), e.what());
    return 1;
  }
  return usage();
}
