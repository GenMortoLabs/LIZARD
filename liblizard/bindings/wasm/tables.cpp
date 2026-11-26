// ai: The web sender's GPU encoder's tables (2026-11-25), the native painter's own: core/tx/send_tables.cpp (SendConsts:
// ai: the paint's TAB and the CRC's powers, the same for every format; sendTables: a format's disc rows and entries, the
// ai: twiddles, the inverse's sizes and clip, the resampler's taps, the geometry words and the anchors', the paint's
// ai: uniform) and core/tx/send_plan.h's constants, made for the module's one codec (src/wasm.c lizard_current: the latest
// ai: Lizard's). In every build of the module; gpu/encoder.mjs reads them through sim/ob.mjs sendTables. The web's in-tree
// ai: interface, as src/wasm.c is the codec's: no stable interface.
#include <emscripten.h>

#include <exception>
#include <memory>
#include <string>

#include "send_plan.h"
#include "send_tables.h"

extern "C" const lizard_t* lizard_current(void);

namespace {

std::unique_ptr<lizard::SendConsts> consts;
lizard::SendTables tables;
std::string json;

std::string quote(const std::string& s) {
  std::string o = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (static_cast<unsigned char>(c) >= 0x20) o += c;
  }
  return o + "\"";
}
template <class T>
std::string arr(const char* k, const std::vector<T>& v) {
  return ",\"" + std::string(k) + "\":[" + std::to_string(reinterpret_cast<uintptr_t>(v.data())) + "," + std::to_string(v.size()) + "]";
}
std::string num(const char* k, long long v) { return ",\"" + std::string(k) + "\":" + std::to_string(v); }

}  // namespace

extern "C" {

// ai: The tables of the module's codec for encodes of `frames` frames of `codes` symbols `gap` modules apart, as JSON:
// ai: {"why": "..."} where it cannot be painted (no codec, a tier the paint has no code for, a disc the shader does not
// ai: read), else the sizes (n, subch, blocks, blockBytes, npos, Vr, sdWords, q, sq, W, FW, RW, FS, codes, gap in
// ai: pixels, copy), each table as [its address in the heap, its count] (rows, uv, su, g, taps, pu, anch, tab: u32;
// ai: tw: f32), good until the next call, and the plan's constants (P_DIMS, P_WORDS, G_WORDS, RSH_THREADS, RSH_BAND,
// ai: BLOCK_BYTES).
EMSCRIPTEN_KEEPALIVE const char* lizw_send_tables(int frames, int codes, int gap) {
  try {
    if (!consts) consts = std::make_unique<lizard::SendConsts>();
    const lizard_t* f = lizard_current();
    const std::string why = f ? lizard::sendTables(*f, *consts, frames, tables, codes, gap) : "no codec";
    if (!why.empty()) return (json = "{\"why\":" + quote(why) + "}").c_str();
    const auto& t = tables;
    json = "{\"why\":\"\"" + num("n", t.n) + num("subch", t.subch) + num("blocks", t.blocks) + num("blockBytes", t.blockBytes) + num("npos", t.npos) +
           num("Vr", t.Vr) + num("sdWords", t.sdWords) + num("q", t.q) + num("sq", t.sq) + num("W", t.W) + num("FW", t.FW) + num("RW", t.RW) +
           num("FS", t.FS) + num("codes", t.codes) + num("gap", t.gap) + ",\"copy\":" + (t.copy ? "true" : "false") + arr("rows", t.rows) +
           arr("uv", t.uv) + arr("tw", t.tw) + arr("su", t.su) + arr("g", t.g) + arr("taps", t.taps) + arr("pu", t.pu) + arr("anch", t.anch) +
           arr("tab", consts->tab) + num("P_DIMS", lizard::send::P_DIMS) + num("P_WORDS", lizard::send::P_WORDS) + num("G_WORDS", lizard::send::G_WORDS) +
           num("RSH_THREADS", lizard::send::RSH_THREADS) + num("RSH_BAND", lizard::send::RSH_BAND) + num("BLOCK_BYTES", lizard::send::BLOCK_BYTES) + "}";
  } catch (const std::exception& e) {
    json = "{\"why\":" + quote(e.what()) + "}";
  }
  return json.c_str();
}

}  // extern "C"
