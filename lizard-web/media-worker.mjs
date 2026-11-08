// ai: The web's file camera (2026-11-07): the apps' reader in WebAssembly (liblizard/bindings/wasm/media.cpp,
// ai: lizard-media.mjs: core/cam/media_source.cpp over FFmpeg, decoders only), one file at a time, its frames one at a
// ai: time as the page asks, so a frame is decoded only when a decoder is about to take it and one is held at a time.
// ai:   page to worker: { type: "open", id, file } (a File, mounted read-only: WORKERFS reads it as FFmpeg asks, nothing
// ai:     copied whole); { type: "next", id, x, y, w, h } (the next frame's crop); { type: "close" }
// ai:   worker to page, each with the request's id: { type: "opened", w, h, image, frames, fps } or { type: "error", why };
// ai:     { type: "frame", luma, w, h } (the crop's bytes, transferred) or { type: "end", why }
// ai:     (why empty: the file's end)
import init from "../liblizard/bindings/wasm/dist/lizard-media.mjs";

let M = null, mounted = false;
const ready = init().then((m) => { M = m; M.FS.mkdir("/in"); });

function close() {
  M?._lizm_close();
  if (mounted) { M.FS.unmount("/in"); mounted = false; }
}

async function handle(m) {
  await ready;
  if (m.type === "close") return close();
  if (m.type === "open") {
    close();
    M.FS.mount(M.FS.filesystems.WORKERFS, { files: [m.file] }, "/in");
    mounted = true;
    if (!M.ccall("lizm_open", "number", ["string"], [`/in/${m.file.name}`])) return postMessage({ type: "error", id: m.id, why: M.UTF8ToString(M._lizm_why()) });
    return postMessage({ type: "opened", id: m.id, w: M._lizm_w(), h: M._lizm_h(), image: !!M._lizm_image(), frames: M._lizm_frames(), fps: M._lizm_fps() });
  }
  if (m.type === "next") {
    const p = M._lizm_next(m.x, m.y, m.w, m.h);
    if (!p) return postMessage({ type: "end", id: m.id, why: M.UTF8ToString(M._lizm_why()) });
    const luma = M.HEAPU8.slice(p, p + m.w * m.h).buffer;
    postMessage({ type: "frame", id: m.id, luma, w: m.w, h: m.h }, [luma]);
  }
}

// ai: in order, one at a time (a read waits for the open before it); a failure answers the request it was for
let chain = Promise.resolve();
onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", id: m.id, why: String(e?.message ?? e) }));
};
