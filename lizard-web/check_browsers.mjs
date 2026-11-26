// ai: The receiver in two browsers' engines, on each one's fake camera: the page decodes on the CPU pool and its stats
// ai: row says how the frames reached it.
// ai:   chrome: headless with the GPU off; the pool's VideoFrames come off the track (MediaStreamTrackProcessor)
// ai:   firefox: headless; the pool's VideoFrames come from new VideoFrame(video) (no track processor); skipped when
// ai:     firefox is not installed
// ai: Then the sender in each (2026-11-25): a picked file (3 MB, three source blocks of 1 MiB, ?chunk=20) read where it lies by
// ai: the C through the page's worker (send-worker.mjs Tx.prepare: blob-reader.mjs's slices, the pack, the transfer on
// ai: the threaded module's threads, the spill on OPFS): the readout's file line must carry the file's BLAKE3 root (the
// ai: plain module's, in node) and the page paint. Chrome over the DevTools protocol (phone.mjs), Firefox over WebDriver
// ai: BiDi (bidi.mjs); the pages cross-origin isolated by the rig server.
// ai: node lizard-web/check_browsers.mjs (BROWSERS=chrome,firefox); about 10 s a browser; exit 1 unless each decoded and sent.
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { tabs, open as openTab, evaluate, navigate } from "./phone.mjs";
import { firefox } from "./bidi.mjs";
import { init as initOb, hex } from "../liblizard/sim/ob.mjs";

const PORT = 8500 + (process.pid % 300), root = mkdtempSync(join(tmpdir(), "lizard-browsers-"));
const server = spawn("node", [fileURLToPath(new URL("./server.mjs", import.meta.url))], { env: { ...process.env, RIG_HTTP: PORT, RIG_HTTPS: PORT + 363 }, stdio: "ignore" });
process.on("exit", () => { server.kill(); rmSync(root, { recursive: true, force: true }); });
const api = (p, body) => fetch(`http://localhost:${PORT}${p}`, body ? { method: "POST", body: JSON.stringify(body) } : undefined).then((r) => r.json());
for (let i = 0; i < 40; i++) { try { await api("/api/info"); break; } catch { await new Promise((r) => setTimeout(r, 250)); } }
const url = `http://localhost:${PORT}/lizard-web/recv.html?auto&seen`;

function open(browser, prof) {
  if (browser === "chrome") return spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", "--enable-unsafe-swiftshader", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--user-data-dir=${prof}`, url], { stdio: "ignore", detached: true });
  const prefs = { "media.navigator.streams.fake": true, "media.navigator.permission.disabled": true, "browser.shell.checkDefaultBrowser": false };
  writeFileSync(join(prof, "user.js"), Object.entries(prefs).map(([k, v]) => `user_pref("${k}", ${v});\n`).join(""));
  return spawn("firefox", ["--headless", "--no-remote", "--profile", prof, url], { stdio: "ignore", detached: true });
}

for (const browser of (process.env.BROWSERS ?? "chrome,firefox").split(",")) {
  if (browser === "firefox" && spawnSync("firefox", ["--version"]).status !== 0) { console.log("firefox: not installed, skipped"); continue; }
  const prof = mkdtempSync(join(root, `${browser}-`)), b = open(browser, prof);
  // ai: Polled until the pool is decoding frames. A row is this browser's by its ua (the last browser's may still be the
  // ai: server's last row).
  const mine = (r) => (r?.ua ?? "").startsWith(browser === "chrome" ? "Chrome" : "Firefox");
  const decoding = (r) => r?.decoder === "cpu" && r.workersReady > 0 && r.processedFps > 0;
  let row = null;
  for (let t = 0; t < 60 && !decoding(row); t++) {
    await new Promise((r) => setTimeout(r, 1000));
    const got = (await api("/api/stats")).stats ?? null;
    row = mine(got) ? got : null;
  }
  try { process.kill(-b.pid, "SIGKILL"); } catch {}
  const ok = decoding(row);
  // ai: How the pool got its frames: as VideoFrames (from where, the format, copied to luma in the worker) or by the grab.
  const frames = !row ? "" : row.vfSent ? `; ${row.vfSent} frames in the window as VideoFrames from the ${row.vfSource} (${row.vfFormat || "format not reported"}), ${row.lumaSent} by the canvas grab` : `; frames by the canvas grab (${row.vfNote || "no reason given"})`;
  console.log(`${browser}: ${row ? `${row.ua}, decoder ${row.decoder}, ${row.workersReady} of ${row.workers} workers ready, ${row.processedFps.toFixed(1)} frames/s decoded${frames}` : "no stats row"}  ${ok ? "ok" : "FAILED"}`);
  if (!ok) process.exitCode = 1;
}

// ai: the sender, a picked file: its root from the plain module (src/xfer.h xfer_b3_hash, b3sum's)
const M = await initOb(), FILE_BYTES = 3 * 2 ** 20 - 777, file = new Uint8Array(FILE_BYTES);
for (let i = 0, v = 7; i < file.length; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; file[i] = v >>> 16; }
const filePath = join(root, "check-browsers.bin");
writeFileSync(filePath, file);
const fp = M._malloc(file.length), rp = M._malloc(32);
M.HEAPU8.set(file, fp);
M._xfer_b3_hash(fp, file.length, rp);
const want = hex(M.HEAPU8.slice(rp, rp + 32)).slice(0, 16);
M._free(fp); M._free(rp);
const sendUrl = `http://localhost:${PORT}/lizard-web/send.html?subch=96&chunk=20&fps=24`;
// ai: the page's readout (#tx) once its file line names a root and it paints, 60 s at most
const sendLine = `(() => { const t = document.getElementById("tx")?.textContent ?? ""; return /BLAKE3 [0-9a-f]{16}/.test(t) && /painted [1-9]/.test(t) ? t : ""; })()`;
for (const browser of (process.env.BROWSERS ?? "chrome,firefox").split(",")) {
  if (browser === "firefox" && spawnSync("firefox", ["--version"]).status !== 0) continue;
  let line = "", why = "";
  if (browser === "chrome") {
    const dev = 9400 + (process.pid % 300), prof = mkdtempSync(join(root, "chrome-send-"));
    const b = spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", `--remote-debugging-port=${dev}`, `--user-data-dir=${prof}`, "about:blank"], { stdio: "ignore", detached: true });
    try {
      let list = [];
      for (let i = 0; i < 50 && !list.length; i++) { await new Promise((r) => setTimeout(r, 200)); try { list = await tabs(dev); } catch {} }
      const s = await openTab(list[0], dev);
      await navigate(s, sendUrl);
      const { root: doc } = await s.send("DOM.getDocument");
      const { nodeId } = await s.send("DOM.querySelector", { nodeId: doc.nodeId, selector: "#file" });
      await s.send("DOM.setFileInputFiles", { nodeId, files: [filePath] });
      await new Promise((r) => setTimeout(r, 500));
      await evaluate(s, `document.getElementById("go").click()`);
      for (let t = 0; t < 60 && !line; t++) { await new Promise((r) => setTimeout(r, 1000)); line = await evaluate(s, sendLine); }
    } catch (e) { why = e.message; }
    try { process.kill(-b.pid, "SIGKILL"); } catch {}
  } else {
    const ff = await firefox();
    try {
      await ff.navigate(sendUrl);
      await ff.setFiles("#file", [filePath]);
      await new Promise((r) => setTimeout(r, 500));
      await ff.evaluate(`document.getElementById("go").click()`);
      for (let t = 0; t < 60 && !line; t++) { await new Promise((r) => setTimeout(r, 1000)); line = (await ff.evaluate(sendLine)).value; }
    } catch (e) { why = e.message; }
    ff.close();
  }
  const fileLine = line.split("\n").map((x) => x.trim()).find((x) => x.startsWith("file ")) ?? "";
  const ok = fileLine.includes(`BLAKE3 ${want}`);
  console.log(`${browser} sender: ${fileLine || `no file line${why ? ` (${why})` : ""}`}; the file's root ${want}...  ${ok ? "ok" : "FAILED"}`);
  if (!ok) process.exitCode = 1;
}
process.exit(process.exitCode ?? 0);
