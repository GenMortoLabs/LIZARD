// ai: The decoder's setup, made by the host (build.h: its tree, every buffer with its bytes, every layout and
// ai: pipeline) through the variant's index gen writes (setup/<variant>.json: its modules' names, the nets' blobs) and
// ai: built on a wg device from its pipelines' SPIR-V (spv/<module>.spv). Only the front half and the one back half in
// ai: use (a batch size) are made and built.
#pragma once
#include "build.h"
#include "wg.h"
#include "json.hpp"

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace lizard {

using json = nlohmann::json;

ReadFile readDir(const std::string& root);
// ai: A variant's index's path under the tree: setup/<variant>.json.
std::string setupName(const std::string& variant);

// ai: A pipeline and its one bind group layout, as the web host's { p, bgl } pairs.
struct Pipe {
  std::shared_ptr<wg::Pipeline> p;
  std::shared_ptr<wg::BindGroupLayout> bgl;
  // ai: a classifier's patches a workgroup (its dispatch is ceil(slots / P)); 1 on every other pipeline
  uint32_t P = 1;
  explicit operator bool() const { return !!p; }
};

// ai: The variants in the web's precision chain order, and which one a device can run: int8 needs f16, int8 and the
// ai: integer dot product (the int8 proposer); f16 needs f16 with 16-bit storage; -sg needs subgroup shuffles in
// ai: compute. Empty when the device misses the floors every variant needs (32 KB of shared memory, 10 storage
// ai: buffers a stage): the C decodes there.
std::vector<std::string> variantsFor(const wg::Device& d);
// ai: Of variants, in their order, those a generated tree holds (setup/<variant>.json; an install carries a variant
// ai: whole, its modules and blobs with it, cmake/gpu_files.cmake): an install may hold only some of the six
// ai: (LIZ_GPU_VARIANTS, the AAR's lizardVariants), and the receiver chooses among those.
std::vector<std::string> variantsIn(const std::string& root, const std::vector<std::string>& variants);

struct Setup {
  json tree;            // ai: the variant's tree (made.tree's values, the back half built at backB among its backs)
  build::Tree made;     // ai: the objects by id (buffers with their bytes until uploaded, bgls, pipelines)
  wg::Device* dev = nullptr;
  ReadFile read;
  std::map<std::string, std::shared_ptr<wg::Buffer>> buffers;
  std::map<std::string, std::shared_ptr<wg::BindGroupLayout>> bgls;
  std::map<std::string, std::shared_ptr<wg::Pipeline>> pipelines;
  std::set<std::string> unbuilt;   // ai: pipelines left out, their workgroups past the device's invocations
  std::string variant;
  double buildMs = 0;

  static std::unique_ptr<Setup> load(wg::Device& d, ReadFile read, const std::string& variant);
  // ai: The front half's objects and the back half at batch size B, built (a back half built before is dropped).
  void build(int B);
  void buildBack(int B);
  // ai: the back half at B made (its tree part and objects) if it is not, none of it built on the device
  const json& backTree(int B);
  const json& back() const { return tree["backs"][std::to_string(backB)]; }
  int backB = 0;

  wg::Buffer& buf(const std::string& id);
  std::shared_ptr<wg::Buffer> bufPtr(const std::string& id);
  std::shared_ptr<wg::BindGroupLayout> bgl(const std::string& id);
  std::shared_ptr<wg::Pipeline> pipe(const std::string& id);
  bool built(const std::string& id) const { return pipelines.count(id) > 0; }
  Pipe pair(const json& pb);   // ai: a tree's { p, bgl } (null gives an empty Pipe)
  void makeOne(const std::string& id) { make({id}); }

 private:
  void make(const std::vector<std::string>& ids);
  wg::Module module(const std::string& name);
};

}  // namespace lizard
