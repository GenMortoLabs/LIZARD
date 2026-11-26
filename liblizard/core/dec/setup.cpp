#include "setup.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace lizard {

// ai: The tree's files by their UTF-8 paths (fs::u8path): Windows reads a narrow path in the ANSI code page, so a tree
// ai: under a folder named outside it (the app's, under %LOCALAPPDATA%) held no variant there (2026-10-25).
ReadFile readDir(const std::string& root) {
  return [root](const std::string& path) {
    std::ifstream f(std::filesystem::u8path(root + "/" + path), std::ios::binary);
    if (!f) throw wg::Error("no " + root + "/" + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  };
}

std::vector<std::string> variantsFor(const wg::Device& d) {
  std::vector<std::string> out;
  const auto& F = d.features;
  const auto& L = d.limits;
  // ai: no module initialises workgroup memory (gen refuses one that does, gen.mjs slangSpv), so no device needs
  // ai: shaderZeroInitializeWorkgroupMemory (every variant did until the GLSL went, 2026-11-24)
  if (L.maxComputeSharedMemorySize < 32768 || L.maxPerStageStorageBuffers < 10) return out;
  const bool sg = F.subgroupShuffle;
  auto add = [&](const std::string& p) { if (sg) out.push_back(p + "-sg"); out.push_back(p); };
  const bool halfOk = F.f16;
  if (halfOk && F.int8 && F.dot) add("int8");
  if (halfOk) add("f16");
  add("f32");
  return out;
}

// ai: A variant's index: setup/<variant>.json.
std::string setupName(const std::string& variant) { return "setup/" + variant + ".json"; }

std::vector<std::string> variantsIn(const std::string& root, const std::vector<std::string>& variants) {
  std::vector<std::string> out;
  for (const auto& v : variants) if (std::ifstream(std::filesystem::u8path(root + "/" + setupName(v)))) out.push_back(v);
  return out;
}

// ai: The tree made (build.cpp) with no back half yet: buildBack makes the one in use. LIZ_CAP=<n> (lab): SELECT's
// ai: cap in place of plan.h CAP.
std::unique_ptr<Setup> Setup::load(wg::Device& d, ReadFile read, const std::string& variant) {
  auto s = std::make_unique<Setup>();
  s->dev = &d;
  s->read = read;
  s->variant = variant;
  const char* cap = getenv("LIZ_CAP");
  try {
    s->made = build::make(read, variant, {}, cap ? atoi(cap) : 0);
  } catch (const wg::Error&) {
    throw;
  } catch (const std::exception& e) {
    throw wg::Error(std::string("the GPU setup ") + variant + ": " + e.what());
  }
  s->tree = std::move(s->made.tree);
  return s;
}

// ai: the module as the device takes it (wg::readModule); LIZ_SPV (lab) names another SPIR-V folder of the tree
wg::Module Setup::module(const std::string& name) {
#if LIZ_VK
  static const std::string dir = getenv("LIZ_SPV") ? getenv("LIZ_SPV") : "spv";
  return wg::readModule(read, name, dir);
#else
  return wg::readModule(read, name);
#endif
}

static wg::Bind bindOf(const std::string& t) {
  if (t == "tex") return wg::Bind::Tex;
  if (t == "ro") return wg::Bind::RO;
  if (t == "rw") return wg::Bind::RW;
  if (t == "uniform") return wg::Bind::Uniform;
  if (t == "sampler") return wg::Bind::Sampler;
  throw wg::Error("bind group layout entry " + t);
}

// ai: The objects named, those a pipeline names (its layout) first; each made once.
void Setup::make(const std::vector<std::string>& ids) {
  for (const auto& id : ids) {
    if (id.empty()) continue;
    const char k = id[0];
    if (k == 'L' && !bgls.count(id)) {
      std::vector<wg::Bind> e;
      for (auto& t : made.bgls.at(id)) e.push_back(bindOf(t.get<std::string>()));
      bgls[id] = dev->createBindGroupLayout(e, id);
    } else if (k == 'p' && !pipelines.count(id) && !unbuilt.count(id)) {
      const auto& o = made.pipelines.at(id);
      if (o["groups"].size() != 1) throw wg::Error(id + ": " + std::to_string(o["groups"].size()) + " bind groups; the decoder uses one");
      const std::string g = o["groups"][0];
      make({g});
      const std::string label = o.value("label", json()).is_string() ? o["label"].get<std::string>() : "";
      // ai: a pipeline whose workgroup passes the device's invocations (its lanes, gen/gen.mjs slangFacts) is left out: only a form
      // ai: the host may do without names one (the LDPC's, front.cpp ldpcBuilt)
      const uint32_t lanes = o.value("lanes", 0u);
      if (lanes > dev->limits.maxComputeWorkGroupInvocations) {
        dev->log("pipeline " + id + " (" + label + ") left out: " + std::to_string(lanes) + " lanes a workgroup, the device " + std::to_string(dev->limits.maxComputeWorkGroupInvocations));
        unbuilt.insert(id);
        continue;
      }
      // ai: so is one whose module needs a device feature the device lacks (its needs, gen/gen.mjs slangFacts: "int16", shaderInt16)
      std::string lacks;
      for (const auto& n : o.value("needs", json::array())) {
        const std::string k = n.get<std::string>();
        if (!(k == "int16" && dev->features.int16)) lacks += (lacks.empty() ? "" : ", ") + k;
      }
      if (!lacks.empty()) {
        dev->log("pipeline " + id + " (" + label + ") left out: it needs " + lacks + ", which the device does not offer");
        unbuilt.insert(id);
        continue;
      }
      auto t0 = std::chrono::steady_clock::now();
      pipelines[id] = dev->createPipeline(module(o["module"]), bgls.at(g), id, label);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (ms > 200) dev->log("pipeline " + id + " (" + (label.empty() ? o["module"].get<std::string>() : label) + ", module " + o["module"].get<std::string>() + "): " + std::to_string((int)ms) + " ms to compile");
    } else if (k == 'b' && !buffers.count(id)) {
      // ai: its bytes uploaded and let go (a buffer of zeros made cleared); a back half made again makes them again
      const auto& o = made.buffers.at(id);
      auto it = made.bytes.find(id);
      if (it == made.bytes.end()) throw wg::Error(id + ": its bytes were let go (the back half it is in was dropped)");
      const bool zero = std::all_of(it->second.begin(), it->second.end(), [](uint8_t x) { return x == 0; });
      auto b = dev->createBuffer(o["size"].get<uint64_t>(), o["usage"].get<uint32_t>(), id, zero);
      if (it->second.size() != b->size) throw wg::Error(id + ": " + std::to_string(it->second.size()) + " B for a buffer of " + std::to_string(b->size));
      if (!zero) dev->upload(*b, 0, it->second.data(), it->second.size());
      made.bytes.erase(it);
      buffers[id] = b;
    }
  }
}

// ai: Every id a part of the tree names (strings like b12, L3, p40).
static void idsIn(const json& j, std::vector<std::string>& out) {
  if (j.is_string()) {
    const auto& s = j.get_ref<const std::string&>();
    if (s.size() > 1 && (s[0] == 'b' || s[0] == 'L' || s[0] == 'p') && std::all_of(s.begin() + 1, s.end(), ::isdigit)) out.push_back(s);
  } else if (j.is_array() || j.is_object()) {
    for (const auto& v : j) idsIn(v, out);
  }
}

void Setup::build(int B) {
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::string> ids;
  idsIn(tree["front"], ids);
  // ai: the anchors' stages (the tree's `anch`), with the front half
  idsIn(tree.at("anch"), ids);
  // ai: layouts before pipelines before buffers, so a pipeline's layout is there and its compile starts early
  std::stable_sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) { auto r = [](char c) { return c == 'L' ? 0 : c == 'p' ? 1 : 2; }; return r(a[0]) < r(b[0]); });
  make(ids);
  buildBack(B);
  buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Setup::buildBack(int B) {
  // ai: The previous back half's objects go (the front half's stay): the web's destroyBack; made again when wanted
  if (backB && backB != B) {
    std::vector<std::string> old;
    idsIn(back(), old);
    for (const auto& id : old) { buffers.erase(id); pipelines.erase(id); unbuilt.erase(id); }
    tree["backs"].erase(std::to_string(backB));
  }
  backTree(B);
  backB = B;
  std::vector<std::string> ids;
  idsIn(back(), ids);
  make(ids);
  dev->flush();
}

const json& Setup::backTree(int B) {
  const std::string k = std::to_string(B);
  if (!tree["backs"].contains(k)) {
    try {
      tree["backs"][k] = made.back(B);
    } catch (const std::exception& e) {
      throw wg::Error("the GPU setup's back half at B = " + k + ": " + e.what());
    }
  }
  return tree["backs"][k];
}

wg::Buffer& Setup::buf(const std::string& id) { return *bufPtr(id); }
std::shared_ptr<wg::Buffer> Setup::bufPtr(const std::string& id) {
  auto it = buffers.find(id);
  if (it == buffers.end()) throw wg::Error("setup buffer " + id + " not built");
  return it->second;
}
std::shared_ptr<wg::BindGroupLayout> Setup::bgl(const std::string& id) {
  auto it = bgls.find(id);
  if (it == bgls.end()) throw wg::Error("setup layout " + id + " not built");
  return it->second;
}
std::shared_ptr<wg::Pipeline> Setup::pipe(const std::string& id) {
  auto it = pipelines.find(id);
  if (it == pipelines.end()) throw wg::Error("setup pipeline " + id + " not built");
  return it->second;
}
Pipe Setup::pair(const json& pb) {
  if (pb.is_null()) return {};
  return {pipe(pb["p"]), bgl(pb["bgl"]), pb.value("P", 1u)};
}

}  // namespace lizard
