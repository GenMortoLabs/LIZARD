// ai: The pages' UI (send.html and recv.html, rebuilt 2026-09-26; the app's shell and Home, 2026-10-01) in
// ai: headless Chrome, driven through phone.mjs's DevTools calls, at four sizes: phone portrait 412 x 915 at 2.625,
// ai: phone landscape 915 x 412, laptop 1366 x 768, desktop 1920 x 1080. Screenshots of every step go to research/build/ui/.
// ai:   0. The sender: idle (Start disabled until a file, a tap on the middle opens the file chooser; Start and
// ai:      Fullscreen in its controls, the other tool in the bar), then the test stream on auto, the settings' folds
// ai:      (Code, Refresh, Layout, Performance; the checks' "Advanced") closed then open: <main>'s box and the version
// ai:      unmoved by them (a resize re-picks
// ai:      mid-stream), #c inside <main>; #tx's first line names the version (the lab line #lab did until 2026-10-02), the
// ai:      numbers line (#nums) the speed;
// ai:      the defaults LIZARD-480 at 60; the folds and their controls in order, no group headings; Rate refresh's four
// ai:      sliders after FPS in Refresh, at 1, one kept over a reload and drawn before the first paint; Fullscreen, then
// ai:      its end clearing body.full.
// ai:   1. The receiver on Chrome's fake camera, its folds (Camera, Decoder) in order, closed then open: #v visible and above 100 x 100 px with
// ai:      more than 10 frame callbacks a second; at the phone sizes #go and #view on screen with no scrolling.
// ai:   2. getUserMedia made to throw NotAllowedError, then navigator.mediaDevices taken away: each sentence in red.
// ai:   3. LIZARD-96 through a 1920 x 1080 clip as the fake camera: the test stream read (its state, speed and lab line),
// ai:      then a 300,000 B file: receiving (the meter, progress, size, speed and time left), received (Save holds the
// ai:      file's bytes) with the camera stopped by itself, Save kept, the file named again after a restart with the
// ai:      camera left on (Receive again, which received it anew, went 2026-10-05).
// ai:   4. Home in the same profile at the four sizes: the received file listed, Save holding its bytes, Open, no tips
// ai:      card, Settings, Delete emptying the received files.
// ai: All pages: no horizontal scroll, the settings' folds closed on a fresh profile and each one's state remembered
// ai: over a reload (some open, some closed), a menu's
// ai: setting kept over a reload with a URL preset winning its own load (the sender's ring, the receiver's frame rate),
// ai: no console error but the ones the start errors put there on purpose. The sender's blocks label before the first
// ai: paint (send.html's inline script) the rate profile's, as send.mjs draws it.
// ai: node lizard-web/check_ui.mjs, from anywhere (about 70 s, a few of them making the clips). APP=1 runs the same
// ai: against the built app (lizard-web/app/, node lizard-web/pwa/build.mjs first), its worker serving the pages. Exit 1 on a failure.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, openSync, writeSync, closeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { tabs, open, evaluate, navigate, screenshot } from "./phone.mjs";
import { makePhy } from "../liblizard/sim/phy.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initWh, Encoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { XferSender, DEFAULT_LOG2, PAYLOAD } from "../liblizard/sim/xfer.mjs";
import { N_FOR } from "../liblizard/sim/lizard_pick.mjs";

const at = (p) => fileURLToPath(new URL(p, import.meta.url));
const DIR = process.env.APP ? "lizard-web/app" : "lizard-web", OUT = at(process.env.APP ? "../research/build/ui/app" : "../research/build/ui"), PORT = 8700 + (process.pid % 200);
const BASE = `http://localhost:${PORT}/${DIR}/recv.html`, SEND = `http://localhost:${PORT}/${DIR}/send.html`, HOME = `http://localhost:${PORT}/${DIR}/index.html`;
// ai: the column's block as the pages draw it (ui.mjs details and detailsOf, 2026-11-07: the state line and figures
// ai: before): { name, end, spin, frac, ghost, rows: { label: value } }, null while hidden
const DT = `(() => { const el = document.getElementById("dt"); if (!el || el.hidden) return null; const rows = {}; for (const r of el.querySelectorAll(".dt-row")) rows[r.dataset.k] = r.lastElementChild.textContent; const m = el.querySelector(".meter>span"); return { name: el.querySelector(".nm")?.textContent ?? "", end: el.querySelector(".pc")?.textContent ?? "", spin: !!el.querySelector(".spin"), frac: m ? parseFloat(m.style.width) / 100 : null, ghost: el.classList.contains("ghost"), rows }; })()`;
// ai: the settings' folds by topic (2026-11-12; one Settings fold, #dev, before), each kept as <page>:<id>; FOLDS_AT
// ai: each fold of the column as "<id> <title>: <its controls' ids>", and any group heading left inside one
const SEND_FOLDS = ["foldCode", "foldRefresh", "foldPerf"], RECV_FOLDS = ["foldCamera", "foldPerf"];
// ai: (open: one state for every fold, or a state a fold)
const statesOf = (ids, open) => ids.map((_, i) => (Array.isArray(open) ? open[i] : open));
const foldsOpen = (ids) => `${JSON.stringify(ids)}.map((id) => document.getElementById(id).open)`;
const setFolds = (ids, open) => `${JSON.stringify(ids)}.forEach((id, i) => { document.getElementById(id).open = ${JSON.stringify(statesOf(ids, open))}[i]; })`;
const foldsKept = (page, ids, open) => statesOf(ids, open).map((o, i) => `localStorage.getItem("${page}:${ids[i]}") === "${o ? 1 : 0}"`).join(" && ");
const FOLDS_AT = `[...document.querySelectorAll(".settings>details")].map((d) => d.id + " " + d.querySelector("summary").textContent + ": " + [...d.querySelectorAll("select, input")].map((e) => e.id).join(" ")).join(" | ") + (document.querySelector(".settings fieldset, .settings legend") ? " + a group heading" : "")`;
const scratch = mkdtempSync(join(tmpdir(), "lizard-ui-"));
mkdirSync(OUT, { recursive: true });
const server = spawn("node", [at("./server.mjs")], { env: { ...process.env, RIG_HTTP: PORT, RIG_HTTPS: PORT + 363 }, stdio: "ignore" });
const browsers = [];
process.on("exit", () => { server.kill(); for (const b of browsers) try { process.kill(-b.pid, "SIGKILL"); } catch {} rmSync(scratch, { recursive: true, force: true }); });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

async function chrome(name, extra = []) {
  const port = 9400 + browsers.length + (process.pid % 100);
  const b = spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", "--enable-unsafe-swiftshader", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", ...extra, `--remote-debugging-port=${port}`, `--user-data-dir=${join(scratch, name)}`, "about:blank"], { stdio: "ignore", detached: true });
  browsers.push(b);
  for (let t = 0; t < 50; t++) {
    try { const list = await tabs(port); if (list.length) { const s = await open(list[0], port); return watch(s); } } catch {}
    await sleep(200);
  }
  throw new Error(`chrome ${name}: no DevTools on ${port}`);
}
// ai: Every console error, uncaught exception and error log entry, into s.errors.
async function watch(s) {
  s.errors = [];
  s.on("Runtime.consoleAPICalled", (p) => { if (p.type === "error") s.errors.push(`console: ${p.args.map((a) => a.value ?? a.description).join(" ")}`); });
  s.on("Runtime.exceptionThrown", (p) => s.errors.push(`exception: ${p.exceptionDetails.exception?.description ?? p.exceptionDetails.text}`));
  s.on("Log.entryAdded", ({ entry: e }) => { if (e.level === "error") s.errors.push(`log: ${e.text} ${e.url ?? ""}`); });
  await s.send("Runtime.enable"); await s.send("Log.enable"); await s.send("Page.enable");
  return s;
}
// ai: a key pressed and let go as a keyboard does it (trusted events: a user activation); modifiers 2 is Ctrl
async function press(s, key, code, vk, modifiers = 0) {
  for (const type of ["keyDown", "keyUp"]) await s.send("Input.dispatchKeyEvent", { type, key, code, windowsVirtualKeyCode: vk, modifiers, ...(type === "keyDown" && !modifiers ? { text: key } : {}) });
}
async function until(s, expr, ms, what) {
  for (const t0 = Date.now(); Date.now() - t0 < ms; await sleep(100)) if (await evaluate(s, expr)) return true;
  // ai: what the page showed when it gave up, and its console errors so far, so a stall names itself
  const seen = await evaluate(s, `(() => { const t = (id) => document.getElementById(id)?.textContent ?? ""; return [t("state"), t("dt"), t("lab"), t("why"), t("file")].filter(Boolean).join(" | "); })()`).catch(() => "");
  throw new Error(`waited ${ms / 1000} s for ${what}; the page: ${seen}${s.errors?.length ? `; errors: ${s.errors.slice(-3).join("; ")}` : ""}`);
}
async function size(s, { w, h, dpr = 1, mobile = false }) {
  await s.send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: dpr, mobile, screenOrientation: w > h ? { type: "landscapePrimary", angle: 90 } : { type: "portraitPrimary", angle: 0 } });
  await s.send("Emulation.setTouchEmulationEnabled", { enabled: mobile, maxTouchPoints: mobile ? 5 : 1 });
}
// ai: The layout at the top of the page, and #v's frame callbacks a second over 2 s.
const MEASURE = `(async () => {
  scrollTo(0, 0);
  const $ = (id) => document.getElementById(id), box = (id) => { const b = $(id).getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, w: b.width, h: b.height }; };
  const v = $("v"), cs = getComputedStyle(v);
  const rate = await new Promise((ok) => { let n = 0; const t0 = performance.now(); const f = () => { n++; const t = performance.now() - t0; if (t < 2000) v.requestVideoFrameCallback(f); else ok(n / (t / 1000)); }; v.requestVideoFrameCallback(f); });
  return { rate, iw: innerWidth, ih: innerHeight, sw: document.documentElement.scrollWidth, view: box("view"), v: box("v"), vDisplay: cs.display, vVisibility: cs.visibility,
    dt: ${DT}, lab: $("lab").textContent, why: $("why").textContent, go: $("go").textContent, goBox: box("go"), swapBox: box("swap"), swap: $("swap").textContent + " " + $("swap").getAttribute("href"), dev: ${foldsOpen(RECV_FOLDS)} };
})()`;
// ai: The bar's Settings button (#set) and its check went 2026-10-02.

// ai: A Settings menu's setting kept over a reload (localStorage, ui.mjs persist, 2026-09-29), and a URL preset of it
// ai: winning for its own load without overwriting what was kept; then set back so the rest runs on the defaults.
// ai: the entries in an OPFS folder (the received files' lizard-files), -1 where it is missing
const opfsFiles = (dir) => `(async () => { try { const d = await (await navigator.storage.getDirectory()).getDirectoryHandle("${dir}"); let n = 0; for await (const k of d.keys()) n++; return n; } catch { return -1; } })()`;
async function remembered(s, page, id, value, other, tag) {
  await evaluate(s, `(() => { const el = document.getElementById("${id}"); el.value = "${value}"; el.dispatchEvent(new Event("change")); })()`);
  await navigate(s, page);
  await sleep(300);
  const kept = await evaluate(s, `document.getElementById("${id}").value`);
  await navigate(s, `${page}?${id}=${other}`);
  await sleep(300);
  const url = await evaluate(s, `document.getElementById("${id}").value`);
  await navigate(s, page);
  await sleep(300);
  const after = await evaluate(s, `document.getElementById("${id}").value`);
  check(kept === value && url === other && after === value, `${tag}: #${id} "${value}" kept over a reload ("${kept}"), "${other}" from the URL for that load ("${url}"), "${value}" again after it ("${after}")`);
  await evaluate(s, `(() => { const el = document.getElementById("${id}"); el.value = el.options[[...el.options].findIndex((o) => o.defaultSelected)]?.value ?? el.options[0].value; el.dispatchEvent(new Event("change")); })()`);
}

let sendTiles = null;   // ai: the sender's tiles' rectangle at the desktop size (its pick list), the receiver's card held to it
const SIZES = [
  { name: "phone-portrait", w: 412, h: 915, dpr: 2.625, mobile: true, phone: true },
  { name: "phone-landscape", w: 915, h: 412, dpr: 2.625, mobile: true, phone: true },
  { name: "laptop", w: 1366, h: 768 },
  { name: "desktop", w: 1920, h: 1080 },
];

// ai: 0. The sender at the four sizes.
{
  const s = await chrome("send");
  const MEASURE_SEND = `(() => {
    const $ = (id) => document.getElementById(id), box = (el) => { const b = el.getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, w: b.width, h: b.height }; };
    return { iw: innerWidth, ih: innerHeight, sw: document.documentElement.scrollWidth, main: box(document.querySelector("main")), c: box($("c")),
      state: $("state").textContent, dt: ${DT}, lab: $("tx").textContent.split("\\n")[0], go: $("go").disabled, full: document.body.classList.contains("full"), dev: ${foldsOpen(SEND_FOLDS)}, hint: !$("pick").hidden && $("sel").checkVisibility() && $("selSum").textContent === "0 items" && !$("pickList").children.length && $("clearPick").disabled,
      buttons: [...document.querySelectorAll("#side button:not([hidden]), #side .btn")].map((b) => b.textContent), bar: [...document.querySelectorAll(".bar .brand, .bar h1, .bar > .btn, .bar > button")].filter((b) => b.checkVisibility()).map((b) => b.textContent).join(","), railHome: document.getElementById("railHome")?.getAttribute("href") ?? "" };
  })()`;
  const same = (a, b) => ["top", "left", "w", "h"].every((k) => Math.abs(a[k] - b[k]) < 0.5);
  for (const [k, z] of SIZES.entries()) {
    await size(s, z);
    await navigate(s, SEND);
    await sleep(300);
    const idle = await evaluate(s, MEASURE_SEND);
    if (k === 0) check(idle.dev.every((o) => !o), `send ${z.name}: every settings fold closed on a fresh profile (${idle.dev.join()})`);
    if (k === 0) {
      const at = await evaluate(s, FOLDS_AT);
      check(at === "foldCode Code: blocks subch ring codes size spacing margin | foldRefresh Refresh: fps refresh78 refresh34 refresh23 refresh12 | foldPerf Performance: enc | logs Developer Tools: stats payload", `send ${z.name}: the folds and their controls in order, no group heading (${at})`);
    }
    check(idle.state === "" && idle.dt === null && idle.hint && idle.go && idle.buttons.join() === "Start,Fullscreen", `send ${z.name}: idle, no state line or block, the File and Folder tiles and the empty card ("0 items", Clear dimmed) shown, buttons ${idle.buttons.join(" and ")}, Start ${idle.go ? "disabled" : "ENABLED"}`);
    // ai: Home in the bar in portrait, in the rail's last square in landscape, where the bar has the collapser (2026-10-05);
    // ai: since 2026-11-01 a phone's landscape bar the collapser alone, a mouse's the switch alone (its edge a sash)
    check(idle.bar === (z.w > z.h ? (z.mobile ? "Send,Sidebar" : "Send,Receiver") : "Home,Send,Receiver") && idle.railHome === "index.html", `send ${z.name}: the bar "${idle.bar}", the rail's Home "${idle.railHome}"`);
    // ai: The File tile opens the chooser (2026-10-26; a tap on the middle of the page until then), which the DevTools
    // ai: protocol intercepts here instead of showing.
    if (k === 0) {
      await s.send("Page.setInterceptFileChooserDialog", { enabled: true });
      const opened = new Promise((ok) => { s.on("Page.fileChooserOpened", () => ok(true)); setTimeout(() => ok(false), 3000); });
      await s.send("Runtime.evaluate", { expression: `document.getElementById("addFiles").click()`, userGesture: true });
      check(await opened, `send ${z.name}: the File tile opens the file chooser`);
      // ai: Ctrl+O, as a video player opens a file (2026-10-23)
      const byKey = new Promise((ok) => { s.on("Page.fileChooserOpened", () => ok(true)); setTimeout(() => ok(false), 3000); });
      await press(s, "o", "KeyO", 79, 2);
      check(await byKey, `send ${z.name}: Ctrl+O opens the file chooser`);
      await s.send("Page.setInterceptFileChooserDialog", { enabled: false });
    }
    await screenshot(s, `${OUT}/send-${z.name}-idle.png`);
    await navigate(s, `${SEND}?fps=24&payload=test&auto`);
    await until(s, `/painted [\\d.]+\\/s/.test(document.getElementById("tx").textContent) && /^LIZARD-\\d+/.test(document.getElementById("tx").textContent) && !!(${DT})?.rows?.Speed`, 20000, "the test stream painted");
    const seen = [];
    for (const openPanel of [false, true]) {
      await evaluate(s, setFolds(SEND_FOLDS, openPanel));
      await sleep(500);
      const m = await evaluate(s, MEASURE_SEND), tag = `send ${z.name}, Advanced ${openPanel ? "open" : "closed"}`, v = +m.lab.match(/LIZARD-(\d+)/)[1];
      seen.push({ ...m, v });
      await screenshot(s, `${OUT}/send-${z.name}-sending-${openPanel ? "open" : "closed"}.png`);
      console.log(`     ${tag}: ${m.iw} x ${m.ih}, main ${m.main.w.toFixed(0)} x ${m.main.h.toFixed(0)} at ${m.main.left.toFixed(0)},${m.main.top.toFixed(0)}; ${JSON.stringify(m.dt)} / "${m.lab}"`);
      check(m.sw <= m.iw, `${tag}: no horizontal scroll (${m.sw} of ${m.iw})`);
      check(m.c.left >= m.main.left - 0.5 && m.c.right <= m.main.right + 0.5 && m.c.top >= m.main.top - 0.5 && m.c.bottom <= m.main.bottom + 0.5, `${tag}: #c inside <main>`);
      // ai: the test stream's block: its name and Speed alone (2026-11-07; "Sending" over the rate before); the version
      // ai: is #tx's first line's (the lab line's, 2026-10-01 to 2026-10-02)
      check(m.dt?.name === "Test stream" && !m.dt.spin && Object.keys(m.dt.rows).join() === "Speed" && /^[\d.]+\s(KB|MB)\/s$/.test(m.dt.rows.Speed), `${tag}: the block ${JSON.stringify(m.dt)}`);
    }
    check(same(seen[0].main, seen[1].main) && seen[0].v === seen[1].v, `send ${z.name}: Advanced moves neither <main> nor the version (LIZARD-${seen[0].v})`);
    if (k === 0) {
      // ai: the page stores each fold's state on its toggle event, a task a busy page can run late: reload after it;
      // ai: Code and Layout left open, Refresh and Performance closed, so each fold is seen to keep its own
      const MIX = [true, false, false];
      await evaluate(s, setFolds(SEND_FOLDS, MIX));
      await until(s, foldsKept("send", SEND_FOLDS, MIX), 5000, "the folds' open states stored");
      await navigate(s, SEND);
      // ai: send.mjs awaits the codec's wasm before anything else, and restores the panel at its end: read it once the
      // ai: module has run (the rate slider's label filled; About is empty on the rig's pages since 2026-10-02), not at the
      // ai: load event
      await until(s, `document.getElementById("fpsOut").textContent !== ""`, 10000, "send.mjs run");
      const kept = await evaluate(s, foldsOpen(SEND_FOLDS));
      check(kept.join() === MIX.join(), `send ${z.name}: each fold's open state remembered over a reload (${kept.join()})`);
      await remembered(s, SEND, "ring", "256", "64", `send ${z.name}`);   // ai: the options are the rings' cells since 2026-10-10
      // ai: the page's defaults, LIZARD-480 at 60 (2026-10-01)
      const defaults = await evaluate(s, `[document.getElementById("subch").value, document.getElementById("fps").value].join()`);
      check(defaults === "480,60", `send ${z.name}: the defaults LIZARD-480 at 60 (${defaults})`);
      // ai: the blocks label at the first paint, before send.mjs has its codec (DOMContentLoaded, the module still
      // ai: awaiting the wasm), and after: LIZARD-480's 58 blocks under the rate profile both times
      await s.send("Page.addScriptToEvaluateOnNewDocument", { source: `document.addEventListener("DOMContentLoaded", () => { window.__firstBlocks = document.getElementById("blocksOut")?.textContent ?? null; });` });
      // ai: Show statistics off (the default): Density and the bytes alone, the ring Thin, Medium, Thick (the 256, 128, 64);
      // ai: on (the receiver's key, recv:stats, one switch for both pages): Blocks with the count, the ring by its cells
      const labels = `[window.__firstBlocks, document.getElementById("blocksOut").textContent, document.getElementById("blocksTitle").textContent,
        [...document.getElementById("ring").options].map((o) => o.value + ":" + o.text).join(" "), [...document.getElementById("codes").options].map((o) => o.text).join(" ")].join(" / ")`;
      await navigate(s, SEND);
      await until(s, `typeof window.__firstBlocks === "string" && document.getElementById("blocksOut").textContent === "27.4 KB"`, 10000, "the density label").catch(() => {});
      const off = await evaluate(s, labels);
      check(off === "27.4 KB / 27.4 KB / Density / 256:Thin 128:Medium 64:Thick / 1 2", `send ${z.name}: Show statistics off, the labels before the first paint and after the codec, LIZARD-480 (${off})`);
      await evaluate(s, `localStorage.setItem("recv:stats", "1")`);
      await navigate(s, SEND);
      await until(s, `typeof window.__firstBlocks === "string" && document.getElementById("blocksOut").textContent === "58, 27.4 KB"`, 10000, "the blocks label").catch(() => {});
      const on = await evaluate(s, labels + ` + " / " + document.getElementById("stats").value`);
      check(on === "58, 27.4 KB / 58, 27.4 KB / Blocks / 64:64 128:128 256:256 / 1 2 / 1", `send ${z.name}: Show statistics on (the receiver's key), the labels before the first paint and after the codec (${on})`);
      const ringWas = await evaluate(s, `document.getElementById("ring").value`);   // ai: the kept ring (remembered above leaves one)
      await evaluate(s, `(() => { const e = document.getElementById("stats"); e.value = "0"; e.dispatchEvent(new Event("change")); })()`);
      const back = await evaluate(s, `[localStorage.getItem("recv:stats"), document.getElementById("blocksTitle").textContent, document.getElementById("blocksOut").textContent, document.getElementById("ring").value].join(" / ")`);
      check(back === `0 / Density / 27.4 KB / ${ringWas}`, `send ${z.name}: Show statistics turned off on the sender, kept under the receiver's key, the labels redrawn, the ring kept at ${ringWas} (${back})`);
      // ai: Rate refresh (2026-11-12, refresh.mjs): four sliders 1 to 4 in Refresh after FPS, 1 each by default; one
      // ai: moved to 3 kept over a reload (send:refresh34), its label 3 before the first paint (the inline script's) and
      // ai: after the codec; set back to 1 for the steps after
      const REF = `["refresh78", "refresh34", "refresh23", "refresh12"].map((id) => { const e = document.getElementById(id), o = document.getElementById(id + "Out"); return e ? [e.closest("details")?.id, e.type, e.min, e.max, e.value, o?.textContent].join(":") : "none"; }).join(" ")`;
      const refAt = await evaluate(s, REF);
      check(refAt === ["78", "34", "23", "12"].map(() => "foldRefresh:range:1:4:1:1").join(" "), `send ${z.name}: Rate refresh in Refresh, four sliders 1 to 4 at 1 (${refAt})`);
      await evaluate(s, `(() => { const e = document.getElementById("refresh34"); e.value = "3"; e.dispatchEvent(new Event("change")); })()`);
      await s.send("Page.addScriptToEvaluateOnNewDocument", { source: `document.addEventListener("DOMContentLoaded", () => { window.__firstRefresh = document.getElementById("refresh34Out")?.textContent ?? null; });` });
      await navigate(s, SEND);
      await until(s, `typeof window.__firstRefresh === "string" && document.getElementById("fpsOut").textContent !== ""`, 10000, "send.mjs run").catch(() => {});
      await sleep(300);
      const refKept = await evaluate(s, `[localStorage.getItem("send:refresh34"), window.__firstRefresh, document.getElementById("refresh34").value, document.getElementById("refresh34Out").textContent].join(" / ")`);
      check(refKept === "3 / 3 / 3 / 3", `send ${z.name}: a refresh of 3 kept over a reload, its label before the first paint and after (${refKept})`);
      await evaluate(s, `(() => { const e = document.getElementById("refresh34"); e.value = "1"; e.dispatchEvent(new Event("change")); })()`);
      await evaluate(s, setFolds(SEND_FOLDS, false));
      await navigate(s, `${SEND}?fps=24&payload=test&auto`);
      await until(s, `/^LIZARD-\\d+/.test(document.getElementById("tx").textContent)`, 20000, "the test stream painted");
    }
    // ai: Fullscreen from a click (a user gesture, as the Fullscreen API asks), then its end from outside the page.
    await s.send("Runtime.evaluate", { expression: `document.getElementById("fs").click()`, userGesture: true });
    await sleep(500);
    const f = await evaluate(s, `({ full: document.body.classList.contains("full"), api: !!document.fullscreenElement, main: document.querySelector("main").getBoundingClientRect().width, iw: innerWidth })`);
    await screenshot(s, `${OUT}/send-${z.name}-full.png`);
    check(f.full && Math.abs(f.main - f.iw) < 1, `send ${z.name}: Fullscreen shows <main> alone (${f.main.toFixed(0)} of ${f.iw} px wide, the API ${f.api ? "in fullscreen" : "refused, body.full alone"})`);
    // ai: a click on the code keeps full screen (2026-10-24, send.mjs); only the fallback, with no API behind it, leaves on one
    if (f.api) {
      await evaluate(s, `document.querySelector("main").click()`);
      await sleep(300);
      check(await evaluate(s, `document.body.classList.contains("full") && !!document.fullscreenElement`), `send ${z.name}: a click on the code keeps full screen`);
    }
    await evaluate(s, f.api ? `document.exitFullscreen()` : `document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))`);
    await sleep(500);
    check(!(await evaluate(s, `document.body.classList.contains("full")`)), `send ${z.name}: body.full cleared when ${f.api ? "fullscreen ends (fullscreenchange)" : "Escape is pressed"}`);
    // ai: Keys as video players have them (2026-10-23, send.mjs): Space and K pause and resume, Space on a focused button
    // ai: is the hotkey and not that button's press, F full screen and back, Ctrl+O nothing while the test stream is the payload
    if (k === 0) {
      const now = () => evaluate(s, `({ dt: ${DT}, resume: !document.getElementById("pause").hidden && document.getElementById("pause").textContent === "Resume", full: document.body.classList.contains("full") })`);
      await press(s, " ", "Space", 32); await sleep(200);
      const a = await now();
      await press(s, "k", "KeyK", 75); await sleep(200);
      const b = await now();
      await evaluate(s, `document.getElementById("fs").focus()`);
      await press(s, " ", "Space", 32); await sleep(400);
      const c = await now();
      await press(s, " ", "Space", 32); await sleep(200);
      await press(s, "f", "KeyF", 70); await sleep(500);
      const d = await now();
      await press(s, "f", "KeyF", 70); await sleep(500);
      const e = await now();
      await s.send("Page.setInterceptFileChooserDialog", { enabled: true });
      const none = new Promise((ok) => { s.on("Page.fileChooserOpened", () => ok(true)); setTimeout(() => ok(false), 1500); });
      await press(s, "o", "KeyO", 79, 2);
      const opened = await none;
      await s.send("Page.setInterceptFileChooserDialog", { enabled: false });
      // ai: a pause is told by the button alone (Pause turned Resume beside Stop); the block keeps its name and rows, its
      // ai: Speed blank while paused, no word of a pause
      check(a.resume && a.dt?.name === b.dt?.name && a.dt?.rows.Speed === "" && !/paused/i.test(JSON.stringify(a.dt)), `send ${z.name}: Space pauses, the block unchanged but its speed (${JSON.stringify(a.dt)})`);
      check(!b.resume, `send ${z.name}: K resumes (${JSON.stringify(b.dt)})`);
      check(c.resume && !c.full, `send ${z.name}: Space with Fullscreen focused pauses, not Fullscreen (full ${c.full})`);
      check(d.full && !e.full, `send ${z.name}: F enters full screen and leaves it`);
      check(!opened, `send ${z.name}: Ctrl+O opens nothing while the test stream is the payload`);
    }
  }
  // ai: The selection (2026-10-23): files and a folder added through the inputs (the DevTools protocol sets their
  // ai: files), a file added twice listed once, one removed, one dropped, the list cleared; the state line names the
  // ai: bundle and counts its files each step.
  {
    const { writeFileSync: put, mkdirSync: mk } = await import("node:fs");
    const src = join(scratch, "pick");
    mk(join(src, "album", "sub"), { recursive: true });
    put(join(src, "a.txt"), "a".repeat(10)); put(join(src, "b.bin"), Buffer.alloc(2000, 7)); put(join(src, "c.txt"), "c".repeat(30));
    put(join(src, "album", "x.txt"), "x".repeat(40)); put(join(src, "album", "sub", "y.txt"), "y".repeat(50));
    await size(s, SIZES[3]);
    await navigate(s, SEND);
    await until(s, `document.getElementById("fpsOut").textContent !== ""`, 10000, "send.mjs run");
    const setFiles = async (id, files) => {
      const { result } = await s.send("Runtime.evaluate", { expression: `document.getElementById("${id}")` });
      await s.send("DOM.setFileInputFiles", { objectId: result.objectId, files });
    };
    const LIST = `(() => ({ rows: [...document.querySelectorAll("#pickList .t")].map((t) => t.textContent), subs: [...document.querySelectorAll("#pickList .s")].map((t) => t.textContent),
      state: document.getElementById("selSum").textContent, hint: !document.getElementById("pick").hidden && !document.getElementById("pickList").children.length, pick: document.getElementById("pickList").children.length > 0, go: document.getElementById("go").disabled }))()`;
    const step = async (what, rows, state) => {
      await until(s, `(${LIST}).rows.join("|") === ${JSON.stringify(rows.join("|"))} && /^${state.replace(/[.()]/g, "\\$&")}/.test(document.getElementById("selSum").textContent)`, 8000, what).catch(() => {});
      const m = await evaluate(s, LIST);
      check(m.rows.join("|") === rows.join("|") && m.state.startsWith(state) && (rows.length ? m.pick && !m.hint && !m.go : m.hint && !m.pick && m.go),
        `send: ${what}: rows ${m.rows.join(", ") || "none"} (${m.subs.join("; ")}), "${m.state}"`);
    };
    await setFiles("file", [join(src, "a.txt"), join(src, "b.bin")]);
    await step("two files added", ["a.txt", "b.bin"], "2 items · ");
    await setFiles("folder", [join(src, "album")]);
    // ai: a folder's files come to the page a moment after the protocol sets them
    const folder = await until(s, `(${LIST}).rows.length === 3`, 5000, "the folder listed").catch(() => false);
    if (folder) await step("a folder added", ["a.txt", "b.bin", "album"], "3 items · ");
    else {   // ai: a protocol that cannot set a folder input: the folder through a drop below instead
      console.log("     DOM.setFileInputFiles took no folder here: the folder is dropped below");
    }
    await setFiles("file", [join(src, "a.txt")]);
    await sleep(300);
    await step("a file added again listed once", folder ? ["a.txt", "b.bin", "album"] : ["a.txt", "b.bin"], folder ? "3 items · " : "2 items · ");
    await evaluate(s, `document.querySelector('#pickList button[title="Remove b.bin"]').click()`);
    await step("one removed", folder ? ["a.txt", "album"] : ["a.txt"], folder ? "2 items · " : "1 item · ");
    await screenshot(s, `${OUT}/send-pick-list.png`);
    // ai: the tiles' rectangle at the desktop size, for the receiver's card to match (2026-11-06)
    sendTiles = await evaluate(s, `(() => { const r = document.querySelector("#pick .tiles").getBoundingClientRect(), m = document.querySelector("main").getBoundingClientRect(); return { w: r.width, h: r.height, x: r.left - m.left, y: r.top - m.top, mw: m.width, mh: m.height }; })()`);
    // ai: a drop on <main>: the protocol's drag events carry the files as paths
    const drop = async (files) => {
      const at = await evaluate(s, `(() => { const b = document.querySelector("main").getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + 20 }; })()`);
      const data = { items: [], files, dragOperationsMask: 1 };
      for (const type of ["dragEnter", "dragOver", "drop"]) await s.send("Input.dispatchDragEvent", { type, x: at.x, y: at.y, data });
    };
    await drop(folder ? [join(src, "c.txt")] : [join(src, "c.txt"), join(src, "album")]);
    await step("dropped", folder ? ["a.txt", "album", "c.txt"] : ["a.txt", "c.txt", "album"], "3 items · ");
    await evaluate(s, `document.getElementById("clearPick").click()`);
    await step("cleared", [], "0 items");
    await setFiles("file", [join(src, "c.txt")]);
    await step("one file alone, sent as itself", ["c.txt"], "1 item · ");
    // ai: the size of what goes in the card's head after the count (2026-11-07; a line under the card before)
    await until(s, `/^1 item · 30\\sB$/.test(document.getElementById("selSum").textContent)`, 10000, "the file's size");   // ai: no-break spaces in the figures
    check(/^1 item · 30\sB$/.test(await evaluate(s, `document.getElementById("selSum").textContent`)), `send: the card's head, the count and the size "${await evaluate(s, `document.getElementById("selSum").textContent`)}"`);
    // ai: long names (2026-10-23): the sending block's name (2026-11-07; the choice line under the card before), on one
    // ai: line, its stem cut with an ellipsis and its ending whole (".txt" for a file, " and 1 more.tar" for a bundle, its
    // ai: space kept); the tile's stem cut and its ending kept too
    const long = join(src, "a very long file name for the ellipsis check, longer than any row is wide.txt");
    const long2 = join(src, "another file with a long name, so the bundle's name is long too.txt");
    put(long, "long"); put(long2, "long too");
    const HEAD = `(() => { const nm = document.querySelector("#dt:not([hidden]) .nm"), f = nm?.querySelector(".fname"); if (!f) return null; const st = f.querySelector(".stem"), e = f.querySelector(".end"), fs = parseFloat(getComputedStyle(nm).fontSize);
      return { end: e.textContent, cut: st.scrollWidth > st.clientWidth, over: getComputedStyle(st).textOverflow, oneLine: nm.getBoundingClientRect().height < 2 * fs, ws: getComputedStyle(e).whiteSpace, state: nm.textContent }; })()`;
    // ai: Start, the block's name read, Stop, the pick back
    const sendOnce = async (end) => {
      await evaluate(s, `document.getElementById("go").click()`);
      await until(s, `(${HEAD})?.end === ${JSON.stringify(end)}`, 15000, "the block's name").catch(() => {});
      const h = await evaluate(s, HEAD);
      await evaluate(s, `document.getElementById("stop").click()`);
      await until(s, `!document.getElementById("pick").hidden && !document.getElementById("sel").hidden`, 8000, "the pick back").catch(() => {});
      return h;
    };
    await evaluate(s, `document.getElementById("clearPick").click()`);
    await setFiles("file", [long]);
    await until(s, `document.querySelectorAll("#pickList .t").length === 1`, 8000, "the long name's tile").catch(() => {});
    const row = await evaluate(s, `(() => { const t = document.querySelector("#pickList .t"), st = t.querySelector(".stem"); return { cut: st.scrollWidth > st.clientWidth, over: getComputedStyle(st).textOverflow, end: st.nextElementSibling.textContent, title: t.title }; })()`);
    const one = await sendOnce(".txt");
    check(one?.end === ".txt" && one.cut && one.over === "ellipsis" && one.oneLine && row.cut && row.end === ".txt" && row.title.endsWith("is wide.txt"),
      `send: a long name's block on one line, its stem cut and ".txt" whole (${JSON.stringify(one)}), its tile's stem cut and ".txt" kept (${row.over})`);
    await setFiles("file", [long2]);
    await until(s, `document.querySelectorAll("#pickList .t").length === 2`, 8000, "the second tile").catch(() => {});
    const two = await sendOnce(" and 1 more.tar");
    check(two?.end === " and 1 more.tar" && two.cut && two.oneLine && two.ws === "pre" && two.state.endsWith(" and 1 more.tar"), `send: a long bundle name on one line, " and 1 more.tar" whole with its space (${JSON.stringify(two)})`);
    await screenshot(s, `${OUT}/send-long-name.png`);
    // ai: an error as a popup (2026-11-07; a cover over the code area that morning, a red line in the column before): a
    // ai: file picked, then gone from disk before Start, so the browser refuses the read; the popup says what happened
    // ai: and what to do, the column nothing, the upload under it; OK closes it
    const gone = join(src, "gone.txt");
    put(gone, "gone");
    await evaluate(s, `document.getElementById("clearPick").click()`);
    await setFiles("file", [gone]);
    await until(s, `document.querySelectorAll("#pickList .t").length === 1`, 8000, "the file to lose").catch(() => {});
    rmSync(gone);
    await evaluate(s, `document.getElementById("go").click()`);
    await until(s, `document.getElementById("errDlg").open`, 8000, "the error's popup").catch(() => {});
    const er = await evaluate(s, `(() => { const $ = (id) => document.getElementById(id);
      return { title: $("errTitle").textContent, body: $("errBody").textContent, open: $("errDlg").open, pick: !$("pick").hidden && $("addFiles").checkVisibility(), state: $("state").textContent, go: $("go").hidden }; })()`);
    await screenshot(s, `${OUT}/send-error.png`);
    check(er.open && er.title === "File changed" && er.body === "Choose it again" && er.pick && er.state === "" && !er.go, `send: an error as a popup over the upload, the column silent (${JSON.stringify(er)})`);
    await evaluate(s, `document.getElementById("errOk").click()`);
    check(await evaluate(s, `!document.getElementById("errDlg").open`), `send: the error's popup closed by OK`);
  }
  // ai: Files from outside (2026-10-23): a file the system's Open with hands over (launchQueue, stubbed here) and one
  // ai: another app shared (left in the "lizard-share" cache as the worker leaves it, then send.html?shared), each added
  {
    await s.send("Page.addScriptToEvaluateOnNewDocument", { source: `Object.defineProperty(window, "launchQueue", { configurable: true, value: { setConsumer(fn) { window.__launch = fn; } } });` });
    await navigate(s, SEND);
    await until(s, `typeof window.__launch === "function"`, 10000, "the page's launch consumer");
    await evaluate(s, `window.__launch({ files: [{ getFile: async () => new File(["opened"], "opened.txt", { lastModified: 1700000000000 }) }] })`);
    await until(s, `[...document.querySelectorAll("#pickList .t")].map((t) => t.textContent).join() === "opened.txt"`, 8000, "the opened file listed").catch(() => {});
    check(await evaluate(s, `[...document.querySelectorAll("#pickList .t")].map((t) => t.textContent).join() === "opened.txt"`), `send: a file opened with the app is listed`);
    await evaluate(s, `caches.open("lizard-share").then((c) => c.put(new Request(location.origin + "/share/0"), new Response("shared!", { headers: { "content-type": "text/plain", "x-name": encodeURIComponent("from app.txt"), "x-modified": "1700000000000" } })))`);
    await navigate(s, `${SEND}?shared`);
    await until(s, `[...document.querySelectorAll("#pickList .t")].map((t) => t.textContent).join() === "from app.txt"`, 8000, "the shared file listed").catch(() => {});
    const sh = await evaluate(s, `(async () => ({ rows: [...document.querySelectorAll("#pickList .t")].map((t) => t.textContent).join(), left: (await (await caches.open("lizard-share")).keys()).length, url: location.search }))()`);
    check(sh.rows === "from app.txt" && sh.left === 0 && sh.url === "", `send: a shared file is listed, the share cache emptied and the URL plain (${JSON.stringify(sh)})`);
  }
  // ai: Power saving (2026-10-23): said in the rail's "?" alone since 2026-11-03 (no first-send question, no live line)
  {
    const tip = await evaluate(s, `document.getElementById("tipDisplay").textContent`);
    const extra = await evaluate(s, `!!document.getElementById("first") || !!document.getElementById("power")`);
    check(/Turn off power saving/.test(tip) && !extra, `send: power saving in the "?" alone ("${tip}"), no first-send question, no live line`);
  }
  // ai: the deliberate error's own line (the file gone before Start, above) is no fault
  const sendErrs = s.errors.filter((e) => !/File changed: choose it again/.test(e));
  check(!sendErrs.length, `send: no console errors${sendErrs.length ? `: ${sendErrs.join("; ")}` : ""}`);
  s.close();
}

// ai: 1. The receiver at the four sizes, the panel closed and open.
{
  const s = await chrome("sizes");
  for (const [k, z] of SIZES.entries()) {
    await size(s, z);
    await navigate(s, `${BASE}?auto`);
    // ai: the browser's limit (#first, 2026-10-09) before the first start on a fresh profile, Home's ?auto included;
    // ai: its Start starts the camera and is kept, so the later sizes' loads go straight to the camera
    if (k === 0) {
      await until(s, `document.getElementById("first").open`, 10000, "the browser's limit before the first start");
      check(!(await evaluate(s, `document.getElementById("v").videoWidth > 0`)), `${z.name}: no camera before the browser's limit is answered`);
      await screenshot(s, `${OUT}/recv-${z.name}-limit.png`);
      await evaluate(s, `document.getElementById("firstStart").click()`);
    } else check(!(await evaluate(s, `document.getElementById("first").open`)), `${z.name}: the browser's limit shown once`);
    await until(s, `document.getElementById("v").videoWidth > 0 && !document.getElementById("go").disabled && document.getElementById("lab").textContent !== ""`, 20000, "the camera and a stats tick");
    if (k === 0) {
      const shut = await evaluate(s, foldsOpen(RECV_FOLDS)), at = await evaluate(s, FOLDS_AT);
      check(shut.every((o) => !o), `${z.name}: every settings fold closed on a fresh profile (${shut.join()})`);
      check(at === "foldCamera Camera: cam res lock fps zoom | foldPerf Performance: dec | logs Developer Tools: stats", `${z.name}: the folds and their controls in order, no group heading (${at})`);
    }
    for (const openPanel of [false, true]) {
      await evaluate(s, setFolds(RECV_FOLDS, openPanel));
      await sleep(400);
      const m = await evaluate(s, MEASURE), tag = `${z.name}, Advanced ${openPanel ? "open" : "closed"}`;
      await screenshot(s, `${OUT}/recv-${z.name}-${openPanel ? "open" : "closed"}.png`);
      console.log(`     ${tag}: ${m.iw} x ${m.ih}, view ${m.view.w.toFixed(0)} x ${m.view.h.toFixed(0)} at ${m.view.top.toFixed(0)}, go ${m.goBox.top.toFixed(0)} to ${m.goBox.bottom.toFixed(0)}, ${m.rate.toFixed(1)} callbacks/s; ${JSON.stringify(m.dt)} / "${m.lab}" / "${m.why.replace(/\n/g, " | ")}"`);
      check(m.rate > 10, `${tag}: #v ${m.rate.toFixed(1)} frame callbacks a second`);
      check(m.v.w > 100 && m.v.h > 100 && m.vDisplay !== "none" && m.vVisibility !== "hidden", `${tag}: #v ${m.v.w.toFixed(0)} x ${m.v.h.toFixed(0)}, display ${m.vDisplay}, visibility ${m.vVisibility}`);
      check(m.sw <= m.iw, `${tag}: no horizontal scroll (${m.sw} of ${m.iw})`);
      if (z.phone) check(m.goBox.top >= 0 && m.goBox.bottom <= m.ih && m.view.top >= 0 && m.view.bottom <= m.ih, `${tag}: #go and #view on screen unscrolled`);
      // ai: the switch to the sender page (2026-09-30), in the bar since 2026-10-01, on screen with Start; not in a
      // ai: phone's landscape bar since 2026-11-01 (the collapser alone there), Home's tiles the way between the pages
      if (z.mobile && z.w > z.h) check(m.swapBox.w === 0, `${tag}: no switch in the landscape bar`);
      else check(m.swap === "Sender send.html" && m.swapBox.w > 0 && (!z.phone || (m.swapBox.top >= 0 && m.swapBox.bottom <= m.ih)), `${tag}: "${m.swap}" in the bar`);
    }
    // ai: Camera closed and Decoder open: each remembered over a reload (then closed again for the next size)
    if (k === 0) {
      const MIX = [false, true];
      await evaluate(s, setFolds(RECV_FOLDS, MIX));
      await until(s, foldsKept("recv", RECV_FOLDS, MIX), 5000, "the folds' open states stored");
      await navigate(s, BASE);
      const kept = await evaluate(s, foldsOpen(RECV_FOLDS));
      check(kept.join() === MIX.join(), `${z.name}: each fold's open state remembered over a reload (${kept.join()})`);
      // ai: a menu in a fold, kept as the sender's are (the decoder's and the GPU crop's went 2026-10-25)
      await remembered(s, BASE, "fps", "30", "60", z.name);
    }
    await evaluate(s, setFolds(RECV_FOLDS, false));
  }
  check(!s.errors.length, `no console errors over the four sizes${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);
  s.close();
}

// ai: 2. Start errors as popups (2026-11-07, the apps' ReceiveScreen; over the camera's box with the rail's red triangle
// ai: from 2026-11-06): the title and the body, the box "Camera off", the column's block empty; OK closes it.
{
  const s = await chrome("errors");
  await size(s, SIZES[0]);
  const cases = [
    ["notallowed", `navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("Permission denied", "NotAllowedError"));`, /^Camera blocked \| Allow it in site settings$/],
    ["nomediadevices", `Object.defineProperty(Navigator.prototype, "mediaDevices", { get: () => undefined, configurable: true });`, /^Insecure page \| Camera needs https or localhost$/],
  ];
  for (const [name, source, want] of cases) {
    const { identifier } = await s.send("Page.addScriptToEvaluateOnNewDocument", { source });
    await navigate(s, `${BASE}?seen`);
    await evaluate(s, `document.getElementById("go").click()`);
    await sleep(600);
    const r = await evaluate(s, `(() => { const $ = (id) => document.getElementById(id); return { text: $("errTitle").textContent + " | " + $("errBody").textContent, open: $("errDlg").open, idle: !$("idle").hidden && $("idleText").textContent, dt: ${DT}, rail: !!$("railAlert"), go: $("go").disabled }; })()`);
    await screenshot(s, `${OUT}/recv-error-${name}.png`);
    check(want.test(r.text) && r.open && r.idle === "Camera off" && r.dt === null && !r.rail && !r.go, `${name}: "${r.text}" in a popup, the box "${r.idle}", the column's block ${JSON.stringify(r.dt)}, ${r.rail ? "a rail triangle LEFT" : "no rail triangle"}, #go ${r.go ? "disabled" : "enabled"}`);
    await evaluate(s, `document.getElementById("errOk").click()`);
    check(await evaluate(s, `!document.getElementById("errDlg").open`), `${name}: OK closes the popup`);
    await s.send("Page.removeScriptToEvaluateOnNewDocument", { identifier });
  }
  const other = s.errors.filter((e) => !/camera did not start/.test(e));
  check(!other.length && s.errors.length === 2, `the start errors' only console errors are their own two (${s.errors.length})${other.length ? `: ${other.join("; ")}` : ""}`);
  s.close();
}

// ai: 2b. Zoom where the camera exposes it (2026-11-06, the apps' Zoom): Chrome's fake camera exposes none, so a zoom is
// ai: given to every track (getCapabilities) and applyConstraints recorded: the row shown at "1.0x", a drag applied live
// ai: with the square's constraints kept in the same set, its value kept for the camera and set again at the next start;
// ai: without the capability the row stays hidden.
{
  const s = await chrome("zoom");
  await size(s, SIZES[2]);
  const { identifier } = await s.send("Page.addScriptToEvaluateOnNewDocument", { source: `(() => {
    const caps = MediaStreamTrack.prototype.getCapabilities, apply = MediaStreamTrack.prototype.applyConstraints;
    window.applied = [];
    MediaStreamTrack.prototype.getCapabilities = function () { return { ...caps.call(this), zoom: { min: 1, max: 5, step: 0.1 } }; };
    MediaStreamTrack.prototype.applyConstraints = function (c) { window.applied.push(JSON.parse(JSON.stringify(c ?? {}))); return apply.call(this, c); };
  })();` });
  await evaluate(s, `localStorage.clear()`).catch(() => {});
  await navigate(s, `${BASE}?seen&auto`);
  await until(s, `!document.getElementById("zoomRow").hidden`, 20000, "the zoom row");
  const a = await evaluate(s, `document.getElementById("zoomOut").textContent`);
  check(a === "1.0x", `zoom: the row shown where the track has a zoom, at "${a}"`);
  await evaluate(s, `(() => { const z = document.getElementById("zoom"); z.value = "2"; z.dispatchEvent(new Event("input")); z.dispatchEvent(new Event("change")); })()`);
  await until(s, `window.applied.some((c) => (c.advanced ?? []).some((x) => x.zoom === 2))`, 5000, "zoom 2 applied");
  // ai: the set in force kept: the zoom's set is the one applied before it (the square, or the first constraints where
  // ai: the square was put back, as on Chrome's fake camera) with the zoom added to its advanced
  const b = await evaluate(s, `(() => { const A = window.applied, i = A.findLastIndex((c) => (c.advanced ?? []).some((x) => x.zoom === 2)), c = A[i], before = A.slice(0, i).findLast((c) => !(c.advanced ?? []).some((x) => "zoom" in x));
    const strip = (c) => { const { advanced, ...rest } = c; return JSON.stringify(rest); };
    return { label: document.getElementById("zoomOut").textContent, same: !!before && strip(before) === strip(c) && JSON.stringify(c.advanced.filter((x) => !("zoom" in x))) === JSON.stringify(before.advanced ?? []),
      kept: Object.keys(localStorage).filter((k) => k.startsWith("recv:zoom@")).map((k) => localStorage.getItem(k)).join() }; })()`);
  check(b.label === "2.0x" && b.same && b.kept === "2", `zoom: a drag to 2 applied live as "${b.label}" with the constraints in force kept (${b.same}), kept as ${b.kept}`);
  await screenshot(s, `${OUT}/recv-laptop-zoom.png`);
  await navigate(s, `${BASE}?seen&auto`);
  await until(s, `window.applied.some((c) => (c.advanced ?? []).some((x) => x.zoom === 2))`, 20000, "the kept zoom set again at the start");
  check(true, "zoom: the kept 2 set again at the next start");
  await s.send("Page.removeScriptToEvaluateOnNewDocument", { identifier });
  await navigate(s, `${BASE}?seen&auto`);
  await until(s, `document.getElementById("go").textContent === "Stop" && !document.getElementById("go").disabled`, 20000, "the camera without a zoom");
  await sleep(500);
  check(await evaluate(s, `document.getElementById("zoomRow").hidden`), "zoom: no row for a camera that exposes none");
  check(!s.errors.length, `zoom: no console errors${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);
  s.close();
}

// ai: 3. The light through the fake camera: 1920 x 1080 clips at 30 fps of a sender at 24, LIZARD-96, one clip pixel a
// ai: sample, as check_rates.mjs builds them. First the test stream (2.5 s, looped), then a file (a lap and 8%).
{
  const CW = 1920, CH = 1080, FD = 24, FC = 30, SUBCH = 96, LEN = 300000, NAME = "holiday-photo.jpg";
  await initWh(); await initZstd();
  const M = await initOb(), bytes = new Uint8Array(LEN);
  for (let i = 0, v = 7; i < LEN; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; bytes[i] = v >>> 16; }
  const phy = await makePhy({ phy: "lizard", stream: "shake256", n: N_FOR(SUBCH), subch: SUBCH, mode: 1, fps: FD, variants: [{ name: "rx" }] }), B = phy.blocksPerFrame;
  async function clip(path, frameOf, n) {
    const fd = openSync(path, "w"), y = Buffer.alloc(CW * CH), chroma = Buffer.alloc((CW * CH) / 2, 128);
    writeSync(fd, `YUV4MPEG2 W${CW} H${CH} F${FC}:1 Ip A1:1 C420jpeg\n`);
    let shown = -1;
    for (let i = 0; i < n; i++) {
      const seq = Math.floor((i * FD) / FC);
      if (seq !== shown) {
        const fr = await frameOf(seq), ox = (CW - fr.w) >> 1, oy = (CH - fr.h) >> 1;
        y.fill(235);
        for (let r = 0; r < fr.h; r++) for (let c = 0; c < fr.w; c++) y[(oy + r) * CW + ox + c] = Math.round(20 + 215 * fr.drive[r * fr.w + c]);
        shown = seq;
      }
      writeSync(fd, "FRAME\n"); writeSync(fd, y); writeSync(fd, chroma);
    }
    closeSync(fd);
  }
  const testPath = join(scratch, "test.y4m");
  await clip(testPath, (seq) => phy.frame(seq), 75);
  const xs = new XferSender(bytes, { name: NAME, type: "image/jpeg", chunkLog2: DEFAULT_LOG2, M, Encoder, Z });
  phy.setSource((id, out) => xs.block(id, out));
  const frames = Math.ceil((xs.sched.lap * 1.08) / (B - 1)) + 24, clipFrames = Math.ceil((frames * FC) / FD), path = join(scratch, "file.y4m");
  await clip(path, (seq) => phy.frame(seq, xs.frameIds(B)), clipFrames);
  console.log(`     clips: the test stream, 75 frames; a ${LEN} B file, ${clipFrames} frames (${frames} display frames of LIZARD-${SUBCH})`);

  const t = await chrome("test", [`--use-file-for-fake-video-capture=${testPath}`]);
  await size(t, SIZES[0]);
  await navigate(t, `${BASE}?auto&seen`);
  await until(t, `(${DT})?.name === "Test stream" && document.getElementById("lab").textContent.includes("% registered, LIZARD")`, 30000, "the test stream read");
  await sleep(1200);
  const ts = await evaluate(t, `(() => { const $ = (id) => document.getElementById(id); return { dt: ${DT}, lab: $("lab").textContent, why: $("why").textContent }; })()`);
  await screenshot(t, `${OUT}/recv-phone-portrait-test.png`);
  // ai: the test stream's block: its name and Speed, no meter (2026-11-07; "Reading the test stream" before); the lab
  // ai: line the rate, the registered share and the format (Developer Tools)
  check(ts.dt?.name === "Test stream" && ts.dt.frac === null && Object.keys(ts.dt.rows).join() === "Speed" && /^\d+\sKB\/s, \d+% registered, LIZARD-96 at 24\sfps$/.test(ts.lab), `test stream: ${JSON.stringify(ts.dt)} / "${ts.lab}" / "${ts.why}"`);
  // ai: The camera's rate under twice the word's (2026-11-11, the apps' rule for a camera no phase lock holds; a fixed 24
  // ai: before), a hint (its row's class h, the info icon): the fake camera runs at 30, so the test stream at 24 is told
  // ai: "recommended 15", as a clip whose word says 30; a clip whose word says 15, half the camera's rate, nothing.
  const rows = `[...document.querySelectorAll("#why .wr")].map((r) => (r.classList.contains("h") ? "hint:" : "") + r.dataset.k + "|" + r.querySelector(".v").textContent).join(" / ")`;
  const tw = await evaluate(t, rows);
  check(/(^| \/ )hint:Sender at 24\sfps\|recommended 15( \/ |$)/.test(tw), `the test stream at 24 fps on a 30 fps camera: the hint row "${tw}"`);
  for (const wf of [FC, FC / 2]) {
    const clipPhy = await makePhy({ phy: "lizard", stream: "shake256", n: N_FOR(SUBCH), subch: SUBCH, mode: 1, fps: wf, variants: [{ name: "rx" }] }), clipPath = join(scratch, `word${wf}.y4m`);
    const fd = openSync(clipPath, "w"), y = Buffer.alloc(CW * CH), chroma = Buffer.alloc((CW * CH) / 2, 128);
    writeSync(fd, `YUV4MPEG2 W${CW} H${CH} F${FC}:1 Ip A1:1 C420jpeg\n`);
    for (let i = 0; i < 60; i++) {
      const fr = await clipPhy.frame(i), ox = (CW - fr.w) >> 1, oy = (CH - fr.h) >> 1;
      y.fill(235);
      for (let r = 0; r < fr.h; r++) for (let c = 0; c < fr.w; c++) y[(oy + r) * CW + ox + c] = Math.round(20 + 215 * fr.drive[r * fr.w + c]);
      writeSync(fd, "FRAME\n"); writeSync(fd, y); writeSync(fd, chroma);
    }
    closeSync(fd);
    const f = await chrome(`word${wf}`, [`--use-file-for-fake-video-capture=${clipPath}`]);
    await size(f, SIZES[0]);
    await navigate(f, `${BASE}?auto&seen`);
    await until(f, `/at ${wf}\\sfps/.test(document.getElementById("lab").textContent)`, 30000, `the ${wf} fps test stream read`);
    await sleep(1200);
    // ai: the hint as a row (2026-11-11; a warning from 2026-11-07, a sentence naming the LIZARD app before): the sender's
    // ai: rate and the most the camera reads
    const fw = await evaluate(f, rows);
    if (wf > FC / 2) check(new RegExp(`(^| / )hint:Sender at ${wf}\\sfps\\|recommended 15( / |$)`).test(fw), `a sender at ${wf} fps on a 30 fps camera: the hint row "${fw}"`);
    else check(!/Sender at/.test(fw), `a sender at ${wf} fps on a 30 fps camera: no hint, "${fw}"`);
  }
  // ai: The light gone (the video paused, as a camera turned to a wall): the state falls back to looking, which says
  // ai: nothing (2026-11-06, as the apps'; its words drawn clear while the run holds the line's room), the code's name
  // ai: leaves the lab line (until 2026-09-26 the band, once read, stayed on both for good).
  await evaluate(t, `document.getElementById("v").pause()`);
  await until(t, `!/LIZARD/.test(document.getElementById("lab").textContent) && (${DT})?.ghost === true`, 8000, "the block silent once the light is gone");
  const gone = await evaluate(t, `[JSON.stringify(${DT}), document.getElementById("lab").textContent].join(" / ")`);
  check(!/LIZARD/.test(gone) && /"ghost":true/.test(gone), `the light gone: the block drawn clear, its room held / ${gone}`);
  check(!t.errors.length, `no console errors over the test stream${t.errors.length ? `: ${t.errors.join("; ")}` : ""}`);
  t.close();

  const s = await chrome("file", [`--use-file-for-fake-video-capture=${path}`]);
  await size(s, SIZES[0]);
  await navigate(s, `${BASE}?auto&seen`);
  const STATE = `(() => { const $ = (id) => document.getElementById(id); return { dt: ${DT}, why: $("why").textContent, lab: $("lab").textContent, deliver: !$("done").hidden, doneName: $("doneName").textContent, doneTitle: $("doneTitle").innerText, clear: !!$("clear"), open: !$("open").hidden && $("open").classList.contains("primary"), go: $("go").textContent, goSolid: $("go").classList.contains("primary"), save: $("save").textContent, href: $("save").href, download: $("save").download,  railRate: $("railRate").textContent, railPct: $("railPct").hidden ? null : $("railPct").textContent, railDone: !$("railDone").hidden, railActs: [...document.querySelectorAll("#rail button[title], #rail a[title]")].map((b) => b.title).filter((t) => t !== "Home" && !/^(Start|Stop|Expand|Collapse|Help)/.test(t)), railOff: $("railCam").classList.contains("off") }; })()`;
  await until(s, `(${DT})?.name === ${JSON.stringify(NAME)} && (${DT}).frac > 0.4`, 60000, "Receiving, two fifths of the way");
  const a = await evaluate(s, STATE);
  await screenshot(s, `${OUT}/recv-phone-portrait-receiving.png`);
  console.log(`     receiving: ${JSON.stringify(a.dt)}`);
  // ai: the block (2026-11-07): the name with its percentage, the meter, Received, Size, Time left and Speed (the last
  // ai: two once a second has closed); progress, size, speed and time left since 2026-10-01
  const ar = a.dt?.rows ?? {};
  check(a.dt?.name === NAME && /^\d+%$/.test(a.dt.end) && a.dt.frac > 0 && Object.keys(ar).join() === "Received,Size,Time left,Speed" && /^300\sKB$/.test(ar.Size)
    && /^[\d.]+\s(B|KB)$/.test(ar.Received) && (ar["Time left"] === "" || /^\d+\s(s|min)$/.test(ar["Time left"])) && (ar.Speed === "" || /^[\d.]+\s(KB|MB)\/s$/.test(ar.Speed)) && !a.deliver,
    `receiving: the block ${JSON.stringify(a.dt)}`);
  // ai: the collapsed rail's squares follow the page whether or not it is shown (2026-10-02): the pause square, the
  // ai: last second's rate where the line has one, no tick yet
  check(!a.railOff && !a.railDone && (ar.Speed ? /^[\d.]+(KB|MB)\/s$/.test(a.railRate) : true), `receiving: the rail's camera on, its rate "${a.railRate}", no check, Open or Save`);
  // ai: the share of the file in under the rate (2026-10-09), the figures line's own percent
  check(a.railPct === a.dt?.end, `receiving: the rail's percent "${a.railPct}" as the block's "${a.dt?.end}"`);
  await until(s, `!document.getElementById("done").hidden`, 60000, "Received");
  const b = await evaluate(s, STATE);
  check(b.railOff && b.railDone && !b.railActs.length && b.railRate === "" && b.railPct === null, `received: the rail's play square and the green check alone, no Open or Save there (${b.railActs.join(", ") || "none"}), no rate or percent`);
  await screenshot(s, `${OUT}/recv-phone-portrait-received.png`);
  const got = await evaluate(s, `fetch(document.getElementById("save").href).then((r) => r.arrayBuffer()).then((x) => { const u = new Uint8Array(x); let v = 7, same = u.length === ${LEN}; for (let i = 0; same && i < u.length; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; same = u[i] === ((v >>> 16) & 255); } return { n: u.length, same }; })`);
  console.log(`     received: ${JSON.stringify(b.dt)}, "${b.save}" ${b.href.slice(0, 5)}... as ${b.download}`);
  // ai: the received file in the camera's place (#done, 2026-11-06): "Received 1 file" and the file in the card, no block
  // ai: in the column with Show statistics off (2026-11-07: how it went is the column's block with it on, below)
  check(b.dt === null && b.deliver && b.doneName === NAME && /^Received 1\sfile$/.test(b.doneTitle) && b.download === NAME && b.href.startsWith("blob:"),
    `received: the card in the camera's place ("${b.doneName}", "${b.doneTitle}"), no block in the column, Save offered as ${b.download}`);
  await evaluate(s, `(() => { const e = document.getElementById("stats"); e.value = "1"; e.dispatchEvent(new Event("change")); })()`);
  const bs = await evaluate(s, DT);
  await screenshot(s, `${OUT}/recv-phone-portrait-received-stats.png`);
  check(bs?.name === NAME && bs.end === "100%" && bs.frac === 1 && /^300\sKB$/.test(bs.rows.Size) && (!("Time" in bs.rows) || (/^[\d.]+\ss$/.test(bs.rows.Time) && /^[\d.]+\s(KB|MB)\/s$/.test(bs.rows.Speed))),
    `received, Show statistics on: the column's block ${JSON.stringify(bs)}`);
  await evaluate(s, `(() => { const e = document.getElementById("stats"); e.value = "0"; e.dispatchEvent(new Event("change")); })()`);
  check(b.open && !b.goSolid && !b.clear, `received: Open the one solid button, "${b.go}" outlined, no Receive again`);
  // ai: the camera off by itself once the file is in (2026-10-01, to save heat)
  check(b.go === "Start" && b.why === "", `received: the camera stopped by itself ("${b.go}")`);
  check(got.n === LEN && got.same, `Save's file is the ${LEN} bytes sent (${got.n}, ${got.same ? "the same" : "NOT the same"})`);
  await sleep(500);
  const c = await evaluate(s, STATE);
  await screenshot(s, `${OUT}/recv-phone-portrait-received-stopped.png`);
  check(c.dt === null && c.deliver && c.go === "Start" && c.why === "", `stopped: the card and Save kept, no block, "${c.go}"`);
  // ai: started again on the same clip: the camera's view back in the card's place, the finished file named again by the
  // ai: light, Save's file the same, and the camera left running (it stops itself once a file, not each time the light
  // ai: names one)
  await evaluate(s, `document.getElementById("go").click()`);
  await until(s, `document.getElementById("go").textContent === "Stop" && document.getElementById("done").hidden && /LIZARD-/.test(document.getElementById("lab").textContent)`, 30000, "the code read again after a restart");
  const d = await evaluate(s, STATE);
  await sleep(1500);
  const d2 = await evaluate(s, STATE);
  check(d.dt === null && !d.deliver && d.href === b.href && d2.go === "Stop", `started again: no block for the file already in (${JSON.stringify(d.dt)}), the camera's view in the card's place, the same Save, the camera still on 1.5 s later ("${d2.go}")`);
  await evaluate(s, `document.getElementById("go").click()`);
  await size(s, SIZES[2]);
  await sleep(300);
  await screenshot(s, `${OUT}/recv-laptop-received-stopped.png`);
  // ai: The received file's card at a laptop's size, a mouse's (2026-11-06): Open, Save (and Share where the browser
  // ai: shares files) as tiles on one row in equal cells, Open solid, in a column at most 36rem wide centred in the area
  // ai: beside the column, the card under them; the column's one button Start, its label's width.
  const DONE = `(() => { const $ = (id) => document.getElementById(id), r = (el) => el.getBoundingClientRect(), d = r($("done")), side = r($("side"));
    const tiles = [...document.querySelectorAll("#done .tile")].filter((t) => !t.hidden).map(r), near = (a, b) => Math.abs(a - b) < 1;
    return { shown: !$("done").hidden, n: tiles.length, row: tiles.every((t) => near(t.top, tiles[0].top) && near(t.width, tiles[0].width)), w: d.width,
      centred: near(d.left - side.right, innerWidth - d.right) || Math.abs((d.left + d.right) / 2 - (side.right + innerWidth) / 2) < 2, card: r(document.querySelector("#done .sel")).top >= tiles[0].bottom,
      solid: $("open").classList.contains("primary"), go: r($("go")).width, spilled: [...document.querySelectorAll("#done .tile")].filter((t) => t.scrollWidth > t.clientWidth + 1).length }; })()`;
  const dn = await evaluate(s, DONE);
  check(dn.shown && dn.n >= 2 && dn.row && dn.w <= 576.5 && dn.centred && dn.card && dn.solid && dn.go < 120 && !dn.spilled,
    `received at a laptop's size: ${dn.n} tiles on a row in equal cells, Open solid, the column ${dn.w} px wide centred in the area, the card under them, Start ${Math.round(dn.go)} px`);
  // ai: the same rectangle as the sender's tiles at the desktop size (2026-11-06; the buttons' rectangle the same on Send
  // ai: and Receive): width and height, and its place in the area beside the column
  await size(s, SIZES[3]);
  await sleep(300);
  await screenshot(s, `${OUT}/recv-desktop-received-stopped.png`);
  const rt = await evaluate(s, `(() => { const r = document.querySelector("#done .tiles").getBoundingClientRect(), side = document.getElementById("side").getBoundingClientRect(); return { w: r.width, h: r.height, x: r.left - side.right, mw: innerWidth - side.right }; })()`);
  const same = sendTiles && Math.abs(rt.w - sendTiles.w) < 1 && Math.abs(rt.h - sendTiles.h) < 1 && Math.abs(rt.x - sendTiles.x) < 2 && Math.abs(rt.mw - sendTiles.mw) < 2;
  check(same, `received at the desktop size: the tiles ${rt.w} x ${rt.h} px at ${Math.round(rt.x)} px into the area, the sender's ${sendTiles ? `${sendTiles.w} x ${sendTiles.h} at ${Math.round(sendTiles.x)}` : "not measured"}`);
  check(!s.errors.length, `no console errors over the transfer${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);
  // ai: The column's floor (recv.mjs actionsFloor): Start or Stop with no word broken; a column saved at 180 px is lifted
  // ai: to it, at 500 px kept; Start its label's width either way, under a mouse.
  const ACTS = `(() => { const $ = (id) => document.getElementById(id); return { spilled: $("go").scrollWidth > $("go").clientWidth + 1, w: $("go").getBoundingClientRect().width,
      side: parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--side")), go: $("go").textContent }; })()`;
  for (const w of [500, 180]) {
    await evaluate(s, `localStorage.setItem("recv:side", "${w}")`);
    await navigate(s, BASE);
    await until(s, `document.documentElement.style.getPropertyValue("--side") !== "" && document.getElementById("go").textContent === "Start"`, 10000, "the column's width restored");
    const m = await evaluate(s, ACTS);
    await screenshot(s, `${OUT}/recv-laptop-actions-${w}.png`);
    check(!m.spilled && m.w < 120 && (w === 180 ? m.side >= 180 : m.side === 500), `the column at ${w} px: Start its label's width (${Math.round(m.w)} px), the column ${m.side} px`);
  }
  await evaluate(s, `localStorage.removeItem("recv:side")`);
  // ai: The column's edge as a sash under a mouse (2026-11-01, ui.mjs sideResizer): pulled under half its floor it closes to
  // ai: the rail, pulled out from the rail it opens at the width it was dragged to; a phone has the collapser instead.
  {
    await navigate(s, BASE);
    await until(s, `document.getElementById("go").textContent === "Start"`, 10000, "the receiver up");
    const drag = async (from, to) => {
      const y = 300;
      await s.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: from, y });
      await s.send("Input.dispatchMouseEvent", { type: "mousePressed", x: from, y, button: "left", clickCount: 1 });
      for (let i = 1; i <= 8; i++) await s.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: from + ((to - from) * i) / 8, y, button: "left" });
      await s.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: to, y, button: "left", clickCount: 1 });
      await sleep(200);
    };
    const STATE = `(() => ({ shut: document.body.classList.contains("collapsed"), kept: localStorage.getItem("recv:collapsed"),
      side: parseFloat(getComputedStyle(document.body).getPropertyValue("--side")), edge: getComputedStyle(document.querySelector(".side-edge")).display,
      button: getComputedStyle(document.getElementById("collapse")).display, rail: getComputedStyle(document.getElementById("rail")).display,
      at: ((r) => r.left + r.width / 2)(document.querySelector(".side-edge").getBoundingClientRect()) }))()`;
    const open0 = await evaluate(s, STATE);
    check(!open0.shut && open0.edge === "block" && open0.button === "none", `the receiver under a mouse: the column's edge shown, no collapser (edge ${open0.edge}, collapser ${open0.button})`);
    await drag(open0.at, 40);
    const shut = await evaluate(s, STATE);
    await screenshot(s, `${OUT}/recv-laptop-sash-closed.png`);
    check(shut.shut && shut.kept === "1" && shut.rail === "flex" && shut.side === 56 && shut.button === "flex", `the edge pulled to 40 px closes the column to the rail, its expander shown (collapsed ${shut.shut}, kept ${shut.kept}, rail ${shut.rail}, --side ${shut.side}, expander ${shut.button})`);
    await drag(shut.at, 300);
    const opened = await evaluate(s, STATE);
    check(!opened.shut && opened.kept === "0" && Math.abs(opened.side - 300) < 1 && opened.button === "none", `the rail's edge pulled to 300 px opens the column there, no collapser (collapsed ${opened.shut}, kept ${opened.kept}, --side ${opened.side}, collapser ${opened.button})`);
    // ai: the rail's expander opens it too (2026-11-01: the collapse button alone went from a mouse's column)
    await drag(opened.at, 40);
    await evaluate(s, `document.getElementById("collapse").click()`);
    await sleep(200);
    const clicked = await evaluate(s, STATE);
    check(!clicked.shut && clicked.kept === "0", `the rail's expander clicked opens the column (collapsed ${clicked.shut}, kept ${clicked.kept})`);
    await evaluate(s, `localStorage.removeItem("recv:side"); localStorage.removeItem("recv:collapsed")`);
  }

  // ai: 4. Home (2026-10-01) in the same profile: the file just received is listed, its Save holds its bytes and Open opens
  // ai: it in a tab of its own; no tips card or Show tips again (2026-11-03); Delete (asked on the button, then done)
  // ai: empties the received files.
  await s.send("Target.setDiscoverTargets", { discover: true });
  const opened = [];
  for (const ev of ["Target.targetCreated", "Target.targetInfoChanged"]) s.on(ev, ({ targetInfo: i }) => { if (i.type === "page") opened.push(i.url); });
  const HOMEM = `(() => { const $ = (id) => document.getElementById(id), rows = [...document.querySelectorAll("#list li")];
    return { iw: innerWidth, sw: document.documentElement.scrollWidth, rows: rows.map((r) => r.querySelector(".name").textContent), acts: rows[0] ? [...rows[0].querySelectorAll(".acts > *")].map((a) => a.textContent).join(",") : "",
      href: rows[0]?.querySelector("a[download]")?.href ?? "", download: rows[0]?.querySelector("a[download]")?.download ?? "", tips: !!$("tips") || !!$("tipsAgain"), empty: !$("empty").hidden, clear: $("clearAllTop").checkVisibility(), heads: [...document.querySelectorAll("main h2")].filter((x) => x.checkVisibility()).map((x) => x.querySelector(".ft")?.textContent ?? x.textContent).join(","), meta: rows[0]?.querySelector(".meta")?.textContent ?? "", solid: [...document.querySelectorAll(".primary")].map((b) => b.id).join(), bar: [...document.querySelectorAll(".bar .brand, .bar > button, .bar > a.btn")].filter((b) => b.checkVisibility()).map((b) => b.textContent).join(",") }; })()`;
  for (const [k, z] of SIZES.entries()) {
    await size(s, z);
    await navigate(s, HOME);
    await until(s, `document.querySelectorAll("#list li").length > 0`, 10000, "the received file listed on Home");
    // ai: a phone's lists folds (2026-11-07): closed at first (the phone's portrait, the first size), opened by a tap on
    // ai: the heading's row, kept open over the next load (the phone's landscape); the count beside the title; Delete
    // ai: all only while open. A PC's never fold.
    if (z.mobile || z.w < 900) {
      const F = `(() => { const $ = (id) => document.getElementById(id); return { shut: $("files").classList.contains("shut"), list: $("list").checkVisibility(), clear: $("clearAllTop").checkVisibility(), n: $("filesN").textContent, exp: $("filesFold").getAttribute("aria-expanded") }; })()`;
      const before = await evaluate(s, F);
      if (before.shut) { await screenshot(s, `${OUT}/home-${z.name}-closed.png`); await evaluate(s, `document.getElementById("filesFold").click()`); }
      const after = await evaluate(s, F);
      check((k === 0 ? before.shut && !before.list && !before.clear : !before.shut) && before.n === "1" && !after.shut && after.list && after.clear && after.exp === "true",
        `home ${z.name}: the received files a fold, ${k === 0 ? "closed at first" : "kept open"} ("${before.n}" beside the title), open with the list and Delete all`);
    }
    const h = await evaluate(s, HOMEM);
    await screenshot(s, `${OUT}/home-${z.name}.png`);
    console.log(`     home ${z.name}: ${h.rows.join(" | ")}; ${h.acts}; headings ${h.heads}; solid ${h.solid}`);
    check(h.sw <= h.iw, `home ${z.name}: no horizontal scroll (${h.sw} of ${h.iw})`);
    check(h.rows.length === 1 && h.rows[0].startsWith(NAME) && /300\sKB/.test(h.rows[0]) && /^Open,Save,(Share,)?Delete$/.test(h.acts) && h.download === NAME && !h.empty, `home ${z.name}: "${h.rows[0]}" listed with ${h.acts}`);
    // ai: a PC's Home (2026-11-06, a window 900 px wide or more without a touch screen): About and Source code in the bar
    const pc = !z.mobile && z.w >= 900;
    check(h.bar === (pc ? "LIZARD,About,Source code" : "LIZARD") && h.solid === (z.mobile ? "recv" : "send"), `home ${z.name}: the bar "${h.bar}", the solid one ${h.solid}`);
    // ai: a phone's (2026-11-07, the apps' the same): Delete all on the files' heading, no Settings section's heading (its
    // ai: Received files row gone), the row's date short (the file came in today: its time, no year)
    if (!pc) check(h.clear && h.heads === "Received files" && /\d:\d\d/.test(h.meta) && !/20\d\d/.test(h.meta),
      `home ${z.name}: Delete all on the heading (${h.clear}), headings "${h.heads}", the row's line "${h.meta}"`);
    if (pc) {
      // ai: Send and Receive two cards across the content, 200 px tall; the file's row in columns with Open beside its
      // ai: menu; Delete all on the heading; no Settings section
      const p = await evaluate(s, `(() => { const $ = (id) => document.getElementById(id), r = (el) => el.getBoundingClientRect(), a = r($("send")), b = r($("recv")), m = r(document.querySelector("main")), row = document.querySelector("#list li"), vis = (el) => !!el && el.checkVisibility();
        const pad = parseFloat(getComputedStyle(document.querySelector("main")).paddingLeft);
        return { cards: Math.round(a.height) === 200 && Math.round(b.height) === 200 && Math.abs(a.width - b.width) < 1 && Math.abs(a.left - (m.left + pad)) < 1 && Math.abs(b.right - (m.right - pad)) < 1,
          size: row.querySelector(".c-size").textContent, when: vis(row.querySelector(".c-when")) && row.querySelector(".c-when").textContent !== "", open: vis(row.querySelector(".c-open")), meta: vis(row.querySelector(".meta")),
          clear: vis($("clearAllTop")), settings: vis(document.querySelector(".settings")), cols: vis(document.querySelector("#files .cols")) }; })()`);
      check(p.cards && /^300\sKB$/.test(p.size) && p.when && p.open && !p.meta && p.clear && !p.settings && p.cols, `home ${z.name}: the PC's layout (cards ${p.cards}, size "${p.size}", date ${p.when}, Open ${p.open}, Delete all on the heading ${p.clear}, Settings ${p.settings ? "shown" : "gone"})`);
      const ab = await evaluate(s, `(() => { document.getElementById("aboutTop").click(); const d = document.getElementById("aboutDlg"), open = d.open, text = d.textContent; document.getElementById("aboutClose").click(); return { open, mpl: /Mozilla Public License 2\.0/.test(text), credits: /BSD 3-Clause: Wirehair, Zstandard/.test(text), shut: !d.open }; })()`);
      check(ab.open && ab.mpl && ab.credits && ab.shut, `home ${z.name}: the bar's About opens the credits in a dialog and Close shuts it`);
    }
    if (k === 0) {
      const bytesOk = await evaluate(s, `fetch(${JSON.stringify(h.href)}).then((r) => r.arrayBuffer()).then((x) => { const u = new Uint8Array(x); let v = 7, same = u.length === ${LEN}; for (let i = 0; same && i < u.length; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; same = u[i] === ((v >>> 16) & 255); } return same; })`);
      check(bytesOk, `home: the listed file's Save holds the ${LEN} bytes sent`);
      await s.send("Runtime.evaluate", { expression: `document.querySelector("#list li .acts button").click()`, userGesture: true });
      await sleep(800);
      check(opened.some((u) => u.startsWith("blob:")), `home: Open opens the file in a tab of its own (${opened.join(", ") || "none"})`);
      // ai: back from the tab Open made: the return redraws the list (library.mjs watch) and keeps a row's link while its
      // ai: entry stands, so that tab goes on reading (2026-10-01; before, every drawing let the links go)
      const hidden = await evaluate(s, `document.querySelector("#list li").dataset.mark = "1"; document.visibilityState`);
      await s.send("Page.bringToFront");
      await until(s, `!document.querySelector("#list li")?.dataset.mark`, 5000, "Home redrawn on a return to the page");
      const redrawn = await evaluate(s, HOMEM), readable = await evaluate(s, `fetch(${JSON.stringify(h.href)}).then((r) => r.ok, () => false)`);
      check(hidden === "hidden" && redrawn.href === h.href && readable, `home: back from the opened tab (Home ${hidden} meanwhile), the list redrawn with the same link (${redrawn.href === h.href}), which still reads (${readable})`);
      check(h.tips === false, `home: no tips card and no Show tips again (2026-11-03)`);
    }
  }
  await evaluate(s, `(() => { const b = [...document.querySelectorAll("#list li .acts button")].find((x) => x.textContent === "Delete"); b.click(); return b.textContent; })()`);
  await sleep(200);
  const asked = await evaluate(s, `[...document.querySelectorAll("#list li .acts button")].map((x) => x.textContent).join(",")`);
  await evaluate(s, `[...document.querySelectorAll("#list li .acts button")].find((x) => x.textContent === "Delete it?").click()`);
  await until(s, `document.querySelectorAll("#list li").length === 0`, 5000, "the list empty after Delete");
  const left = await evaluate(s, opfsFiles("lizard-files"));
  check(/Delete it\?/.test(asked) && left === 0, `home: Delete asks on itself ("${asked}") and then empties the received files (${left} left)`);
  // ai: at the desktop size (the PC's layout), the section stays with its empty box and no Delete all (2026-11-06);
  // ai: on a phone's size the section goes, as before
  const em = await evaluate(s, `(() => { const $ = (id) => document.getElementById(id); return { files: !$("files").hidden, box: $("emptyPc").checkVisibility(), text: $("emptyPc").textContent, clear: $("clearAllTop").checkVisibility(), cols: document.querySelector("#files .cols").checkVisibility() }; })()`);
  await screenshot(s, `${OUT}/home-desktop-empty.png`);
  check(em.files && em.box && em.text === "No received files" && !em.clear && !em.cols, `home desktop, no files: the section kept, "${em.text}" in its box, no Delete all or columns`);
  await size(s, SIZES[0]);
  await sleep(300);
  check(await evaluate(s, `document.getElementById("files").hidden && !document.getElementById("emptyPc").checkVisibility()`), `home phone-portrait, no files: the section gone`);
  await size(s, SIZES[3]);
  // ai: An unfinished transfer (2026-10-22, library.mjs unfinished): a folder under lizard-xfer as the fountain worker
  // ai: leaves one (its header and meta.json), listed under Unfinished with its share and size, Continue and Delete;
  // ai: Delete asks on itself, then removes the folder.
  const MAKE_PART = `(async () => { const d = await (await (await navigator.storage.getDirectory()).getDirectoryHandle("lizard-xfer", { create: true })).getDirectoryHandle("00112233aabbccdd", { create: true });
    for (const [n, b] of [["header", new Uint8Array(${PAYLOAD})], ["meta.json", new TextEncoder().encode(JSON.stringify({ name: "half.bin", type: "", length: 3133383, chunks: 3, verified: 1, root: "00112233aabbccdd", fraction: 0.37, sent: 3133383, sentIn: 1159352 }))]]) {
      const w = await (await d.getFileHandle(n, { create: true })).createWritable(); await w.write(b); await w.close(); } return true; })()`;
  await evaluate(s, MAKE_PART);
  await navigate(s, HOME);
  await until(s, `!document.getElementById("partial").hidden && document.querySelectorAll("#partialList li").length === 1`, 5000, "the unfinished transfer listed on Home");
  // ai: on a phone (2026-11-07): Unfinished a fold, closed at first, "1" beside it; open, its Delete all asks on itself and
  // ai: then forgets every unfinished transfer; the folder made again for the PC's row checks below
  await size(s, SIZES[0]); await sleep(300);
  const pf0 = await evaluate(s, `(() => { const $ = (id) => document.getElementById(id); return { shut: $("partial").classList.contains("shut"), n: $("partialN").textContent, clear: $("clearPartial").checkVisibility() }; })()`);
  await evaluate(s, `document.getElementById("partialFold").click()`);
  const pf1 = await evaluate(s, `(() => { const b = document.getElementById("clearPartial"), vis = b.checkVisibility(); b.click(); return { vis, ask: b.textContent }; })()`);
  await evaluate(s, `document.getElementById("clearPartial").click()`);
  await until(s, `document.getElementById("partial").hidden`, 5000, "Unfinished gone after its Delete all");
  const pfLeft = await evaluate(s, opfsFiles("lizard-xfer"));
  check(pf0.shut && pf0.n === "1" && !pf0.clear && pf1.vis && pf1.ask === "Delete 1 unfinished transfer?" && pfLeft === 0,
    `home phone-portrait: Unfinished a fold, closed with "${pf0.n}", open its Delete all asking "${pf1.ask}" then forgetting it (${pfLeft} left)`);
  await evaluate(s, MAKE_PART);
  await size(s, SIZES[3]);
  await navigate(s, HOME);
  await until(s, `!document.getElementById("partial").hidden && document.querySelectorAll("#partialList li").length === 1`, 5000, "the unfinished transfer listed again on Home");
  const part = await evaluate(s, `(() => { const r = document.querySelector("#partialList li"); return { text: r.querySelector(".name").textContent, acts: [...r.querySelectorAll(".acts > *")].map((a) => a.textContent).join(","), href: r.querySelector(".acts a")?.getAttribute("href") }; })()`);
  await screenshot(s, `${OUT}/home-unfinished.png`);
  check(/^half\.bin37% of 3\.1\sMB/.test(part.text) && part.acts === "Continue,Delete" && part.href === "recv.html?auto", `home: an unfinished transfer listed, "${part.text}", ${part.acts} (Continue to ${part.href})`);
  await evaluate(s, `[...document.querySelectorAll("#partialList .acts button")].find((x) => x.textContent === "Delete").click()`);
  await sleep(200);
  await evaluate(s, `[...document.querySelectorAll("#partialList .acts button")].find((x) => x.textContent === "Delete it?").click()`);
  await until(s, `document.getElementById("partial").hidden`, 5000, "the unfinished list empty after Delete");
  const partLeft = await evaluate(s, opfsFiles("lizard-xfer"));
  check(partLeft === 0, `home: Delete removes the unfinished transfer's folder (${partLeft} left in lizard-xfer)`);
  check(!s.errors.length, `no console errors on Home${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);
  s.close();
}

console.log(failed ? `${failed} FAILED` : "all passed");
process.exit(failed ? 1 : 0);
