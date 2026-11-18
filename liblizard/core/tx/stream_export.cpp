// ai: stream_export.h. At 2:1 a picture is two entries, its squares, the gap between them meta.json's; the frames are
// ai: packed and written by the sender's export (sender.h exportStart).
#include "stream_export.h"

#include "lizfile.h"
#include "sender.h"
#include "xfer_tx.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace lizard {

namespace {
// ai: a JSON string's escapes (a file's name as meta.json carries it)
std::string jstr(const std::string& v) {
  std::string o = "\"";
  for (unsigned char c : v) {
    if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += char(c);
  }
  return o + "\"";
}
}  // namespace

int64_t exportStream(Sender& s, const StreamInfo& info, const std::string& path, int frames, const std::function<bool(int, int)>& progress,
                     std::string& why, const lizfile::Packing& packing) {
  if (s.width() <= 0) { why = "the sender is not configured"; return -1; }
  Sender::ExportJob j;
  j.path = path;
  j.frames = frames;
  j.packing = packing;
  j.display = false;
  j.info = info;
  why = s.exportStart(j);
  if (!why.empty()) return -1;
  // ai: the sender's own threads write it; here its progress, and a stall said (no frame written for 60 s once under way)
  int last = -1;
  auto at = std::chrono::steady_clock::now();
  for (;;) {
    const Sender::ExportState st = s.exportState();
    if (!st.error.empty()) { why = st.error; s.exportStop(); return -1; }
    if (st.written != last) { last = st.written; at = std::chrono::steady_clock::now(); }
    if (progress && st.written > 0 && !progress(st.written, st.frames)) { s.exportStop(); why = "the export was stopped"; return -1; }
    if (st.done) return st.written;
    if (std::chrono::steady_clock::now() - at > std::chrono::seconds(60)) { s.exportStop(); why = "no frame written for 60 s"; return -1; }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

std::string streamMeta(Sender& s, const StreamInfo& info, int frames) {
  const int W = s.width(), H = s.side();
  const bool two = info.codes == 2;
  const int side = H, gap = two ? W - 2 * side : 0;
  const XferTx* x = s.xfer();
  char root[65] = "";
  if (x) for (int k = 0; k < 32; k++) std::snprintf(root + 2 * k, 3, "%02x", x->root()[k]);
  const std::string st = s.stats();
  const size_t sp = st.find("\"seed\":");
  const std::string seed = sp == std::string::npos ? "null" : st.substr(sp + 7, st.find_first_of(",}", sp + 7) - (sp + 7));
  const int entries = two ? 2 * frames : frames, ew = two ? side : W;
  std::string m = "{\"kind\":\"stream\",\"run\":" + jstr(info.name) + ",\"w\":" + std::to_string(ew) + ",\"h\":" + std::to_string(H) +
                  ",\"frames\":" + std::to_string(entries) + ",\"pictures\":" + std::to_string(frames) + ",\"gap\":" + std::to_string(gap) +
                  ",\"layout\":\"" + (two ? "2:1" : "1:1") + "\",\"fps\":" + std::to_string(info.fps) +
                  ",\"format\":{\"blocks\":" + std::to_string(info.blocks) + ",\"ring\":" + std::to_string(info.ring) + ",\"codes\":" + std::to_string(info.codes) +
                  ",\"frame_blocks\":" + std::to_string(info.frameBlocks) + "},\"file\":{\"name\":" + jstr(info.name) + ",\"type\":" + jstr(info.type) +
                  ",\"length\":" + std::to_string(x ? x->length() : 0) + ",\"chunks\":" + std::to_string(x ? x->chunks() : 0) +
                  ",\"lap\":" + std::to_string(x ? x->lap() : 0) + ",\"root\":\"" + root + "\"},\"test\":" + (x ? "false" : "true") +
                  ",\"colours\":{\"ink\":" + std::to_string(s.ink()) + ",\"paper\":" + std::to_string(s.paper()) + "},\"seed\":" + seed +
                  ",\"painter\":\"" + s.painter() + "\",\"sizes\":[";
  for (int i = 0; i < entries; i++) m += std::string(i ? "," : "") + "[" + std::to_string(ew) + "," + std::to_string(H) + "]";
  m += "],\"ms\":[";
  for (int i = 0; i < entries; i++) { char b[32]; std::snprintf(b, sizeof b, "%s%.3f", i ? "," : "", 1000.0 * (two ? i / 2 : i) / info.fps); m += b; }
  m += "]";
  if (two) {
    // ai: a half's axis the middle of its inner edge (core/rx/receiver.cpp axisOf): the left half's right edge, the
    // ai: right half's left
    m += ",\"axes\":[";
    for (int i = 0; i < frames; i++) m += std::string(i ? "," : "") + "[" + std::to_string(side) + "," + std::to_string(side / 2) + "],[0," + std::to_string(side / 2) + "]";
    m += "]";
  }
  return m + "}";
}

}  // namespace lizard
