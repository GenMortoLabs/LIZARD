// ai: The helpers every LIZARD page shares, for the parts in ui.css. Page-only: no worker imports this.

// ai: remember(details, key): keeps a <details>' open state across reloads (a phone session is a series of reloads).
// ai: Storage can throw (a private window, blocked site data); the page then opens as its markup says.
// ai: collapser(button, key): the tool pages' sidebar collapser (2026-10-02): body.collapsed toggled and kept under key (each
// ai: page's inline script restores it before the first paint); landscape only, ui.css draws the rail.
export function collapser(button, key) {
  const show = () => {
    const on = document.body.classList.contains("collapsed");
    button.title = on ? "Show sidebar" : "Hide sidebar"; button.setAttribute("aria-expanded", String(!on));
  };
  try { document.body.classList.toggle("collapsed", localStorage.getItem(key) === "1"); } catch {}
  show();
  button.addEventListener("click", () => {
    const on = document.body.classList.toggle("collapsed");
    try { localStorage.setItem(key, on ? "1" : "0"); } catch {}
    show();
  });
}

// ai: sideResizer(key, floor, shutKey): the landscape column's width set by dragging its edge (2026-10-05, as the Android
// ai: app's; a mouse's alone since 2026-11-01, ui.css): a strip on the column's right edge (.side-edge), dragged sideways,
// ai: sets --side, within 180 px (or floor(), px, where a page gives one: the least its content needs with no word
// ai: broken, read at each drag's start and at the restore; recv.mjs actionsFloor) and three fifths of the window; kept
// ai: under `key` on release (each page's inline script restores it before the first paint). An editor's sash: dragged
// ai: under half the floor the column closes to the rail (body.collapsed, kept under shutKey as the collapser keeps
// ai: it), and dragged back over it opens again at the floor or wider; the width kept is the last open one. The
// ai: sender re-picks as it does for a resized window.
export const SIDE_MIN = 180, sideClamp = (px, min = SIDE_MIN) => Math.max(min, Math.min(px, innerWidth * 0.6));
export function sideResizer(key, floor, shutKey) {
  const root = document.documentElement, edge = document.createElement("div"), lo = () => Math.max(SIDE_MIN, floor?.() || 0);
  edge.className = "side-edge";
  edge.setAttribute("role", "separator"); edge.setAttribute("aria-orientation", "vertical"); edge.setAttribute("aria-label", "Sidebar width");
  document.body.append(edge);
  // ai: lit after 300 ms under the pointer (the apps' SideEdge hoverLit), unlit as it leaves; a drag lights it at once (ui.css)
  let lit = 0;
  const light = (on) => { edge.classList.toggle("lit", on); document.body.classList.toggle("edge-lit", on); };   // ai: body's for the column's scrollbar (ui.css)
  edge.addEventListener("pointerenter", () => { clearTimeout(lit); lit = setTimeout(() => light(true), 300); });
  edge.addEventListener("pointerleave", () => { clearTimeout(lit); light(false); });
  const restore = () => { try { const v = +localStorage.getItem(key); if (v > 0) root.style.setProperty("--side", `${sideClamp(v, lo())}px`); } catch {} };
  restore();
  document.fonts?.ready.then(restore);   // ai: a floor measured in the page's font: again once it is loaded (ui.css, Inter)
  edge.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    edge.setPointerCapture(e.pointerId);
    document.body.classList.add("resizing");
    const min = lo(), move = (m) => {
      const shut = m.clientX < min / 2;
      if (shut !== document.body.classList.contains("collapsed")) {
        document.body.classList.toggle("collapsed", shut);
        try { localStorage.setItem(shutKey, shut ? "1" : "0"); } catch {}
      }
      if (!shut) root.style.setProperty("--side", `${sideClamp(m.clientX, min)}px`);
    };
    const up = () => {
      for (const [t, f] of [["pointermove", move], ["pointerup", up], ["pointercancel", up]]) edge.removeEventListener(t, f);
      document.body.classList.remove("resizing");
      try { localStorage.setItem(key, String(parseFloat(root.style.getPropertyValue("--side")) || "")); } catch {}
    };
    for (const [t, f] of [["pointermove", move], ["pointerup", up], ["pointercancel", up]]) edge.addEventListener(t, f);
  });
}

export function remember(details, key) {
  try { const v = localStorage.getItem(key); if (v !== null) details.open = v === "1"; } catch {}
  details.addEventListener("toggle", () => { try { localStorage.setItem(key, details.open ? "1" : "0"); } catch {} });
}

// ai: persist(el, key): keeps a setting's value across reloads in localStorage (2026-09-29), a
// ai: <select> or an <input>: restored now where the saved value is one it can take (one of a select's options), saved
// ai: whenever the user changes it (a page's own assignments fire no change and save nothing). A page presets from its
// ai: URL after this, so a URL still wins for that load. restoreSaved(el, key): the restore alone, for a select whose
// ai: options come later (the receiver's camera list). Storage that throws (a private window, blocked site data) keeps nothing.
export function restoreSaved(el, key) {
  try {
    const v = localStorage.getItem(key);
    if (v === null) return;
    if (el.tagName === "SELECT" ? [...el.options].some((o) => o.value === v) : el.type !== "file") el.value = v;
  } catch {}
}
export function persist(el, key) {
  restoreSaved(el, key);
  el.addEventListener("change", () => { try { localStorage.setItem(key, el.value); } catch {} });
}

// ai: say(el, text, tone): sets one status line; tone "bad" or "good" colours it, anything else leaves it black.
export function say(el, text, tone = null) {
  el.textContent = text;
  el.classList.toggle("bad", tone === "bad");
  el.classList.toggle("good", tone === "good");
}

// ai: meter(el, frac): a .meter element at frac (0 to 1) of full; null hides it.
export function meter(el, frac) {
  el.hidden = frac == null;
  if (frac == null) return;
  const f = Math.max(0, Math.min(1, frac));
  (el.firstElementChild ?? el.appendChild(document.createElement("span"))).style.width = `${f * 100}%`;
  el.setAttribute("aria-valuenow", String(Math.round(f * 100)));
}

// ai: bytes(n): a size for people, 1 KB = 1000 B as the rates count it: "812 B", "7.4 MB", "31 MB".
export function bytes(n) {
  if (n < 1000) return `${n} B`;
  // ai: the unit chosen after rounding, so 999,500 B reads "1.0 MB", not "1000 KB"
  const at = (v) => (v < 9.95 ? v.toFixed(1) : String(Math.round(v)));
  const [v, u] = n < 999.5e3 ? [n / 1e3, "KB"] : n < 999.5e6 ? [n / 1e6, "MB"] : [n / 1e9, "GB"];
  return `${at(v)} ${u}`;
}

// ai: The transfer's figures for people (2026-10-01: progress, size, speed, time left), the Android app's
// ai: Readout.kt the same: rate(kbs) a speed from KB/s (1 KB = 1000 B), "640 KB/s" under 1,000 and "1.57 MB/s" from
// ai: there; left(secs) the time still to go, "4 s", "2 min" from 100 s (a Time left row's value). A figure keeps its unit on its line
// ai: (a no-break space).
export function rate(kbs) {
  return kbs < 999.5 ? `${Math.round(kbs)} KB/s` : `${(kbs / 1000).toFixed(2)} MB/s`;
}
export function left(secs) {
  const s = Math.max(1, Math.ceil(secs));
  return s < 100 ? `${s} s` : `${Math.round(s / 60)} min`;
}

// ai: The icons a script puts on a page (2026-10-01; Feather's, MIT): ui.css svg.i draws them on a 24 grid, 2 px strokes
// ai: in the text's colour. The pages inline the same set in their static markup (back, gear, swap, camera, monitor,
// ai: camera-off, info), and the Android app has them as vector drawables (res/drawable/ic_*.xml).
const ICONS = {
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
  more: '<circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>',
  folder: '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
  alert: '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
};
// ai: A line that names a file (2026-10-23; the PC's rule since 2026-11-06, lizard-ui Parts.kt NameLine): on one line
// ai: (.fname) its head, the name's stem, its ending and a comma after it, then the rest; the head, the ending and the
// ai: rest measured first, the stem cut with an ellipsis where the line is too narrow, the rest cut only once even the
// ai: stem is gone (ui.css). A row cuts its title to its width instead (ui.css .item .t, .files .name).
export function sayName(el, before, name, after = "", tone = null) {
  const [stem, end] = nameParts(name), comma = after.startsWith(","), rest = (comma ? after.slice(1) : after).trim();
  const part = (cls, text) => { const p = document.createElement("span"); p.className = cls; p.textContent = text; return p; };
  const line = part("fname", "");
  line.append(...(before ? [part("head", before)] : []), part("stem", stem), part("end", `${end}${comma ? "," : ""}`), ...(rest ? [part("rest", ` ${rest}`)] : []));
  el.replaceChildren(line);
  el.classList.toggle("bad", tone === "bad");
  el.classList.toggle("good", tone === "good");
}
// ai: What a send or a receive moves, as elements (2026-11-07, the apps' lizard-ui Details; a sentence of state and
// ai: figures before, "Sending x, 2.4 GB (4.4 GB raw)"): a head, the file's icon and name (its stem cut, its ending
// ai: kept) with at its end a percentage or, while it is prepared, a spinner; the meter where there is one; then a row a
// ai: fact (data-k its label), the label muted at the left, the value at the right. d: { name, test (the test stream:
// ai: its name not a file's), end, spin, frac, rows: [[label, value]] }, or null to hide the block (el, a .dt). clear:
// ai: drawn clear, its room kept (a run's block once shown). Rebuilt only when it changes (a screen reader reads each
// ai: rewrite of a live region).
export function details(el, d, clear = false) {
  el.hidden = !d;
  el.classList.toggle("ghost", !!d && clear);
  if (!d) { el.dataset.key = ""; return; }
  const key = JSON.stringify(d);
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  const span = (cls, text) => { const p = document.createElement("span"); p.className = cls; p.textContent = text; return p; };
  const head = document.createElement("div"), nm = span("nm", "");
  head.className = "dt-head";
  if (d.test) nm.textContent = d.name; else sayName(nm, "", d.name);
  head.append(icon("file"), nm);
  if (d.spin) head.append(span("spin", "")); else if (d.end) head.append(span("pc", d.end));
  const parts = [head];
  if (d.frac != null) {
    const m = document.createElement("div");
    m.className = "meter"; m.setAttribute("role", "progressbar"); m.setAttribute("aria-valuemin", "0"); m.setAttribute("aria-valuemax", "100");
    meter(m, d.frac); parts.push(m);
  }
  for (const [k, v] of d.rows ?? []) { const r = document.createElement("div"); r.className = "dt-row"; r.dataset.k = k; r.append(span("k", k), span("v", v)); parts.push(r); }
  el.replaceChildren(...parts);
}
// ai: Warnings as status labels under the block (2026-11-07, the apps' lizard-ui Warn and WarnRow; sentences before,
// ai: "30 pictures a second is too fast for a browser: use the LIZARD app, or send at 24."): a row each, the red
// ai: triangle, the label, and at the right a value where a figure helps. list: [[label, value]]; clear: drawn clear,
// ai: its room kept. Rebuilt only when it changes.
export function warnings(el, list, clear = false) {
  el.classList.toggle("ghost", clear);
  const key = JSON.stringify(list);
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  el.replaceChildren(...list.map(([l, v]) => {
    const r = document.createElement("div"), a = document.createElement("span"), b = document.createElement("span");
    r.className = "wr"; r.dataset.k = l; a.className = "l"; a.textContent = l; b.className = "v"; b.textContent = v ?? "";
    r.append(icon("alert"), a, b);
    return r;
  }));
}
// ai: a block's facts as the checks read them: { name, end, frac, rows: { label: value } }, null where hidden
export function detailsOf(el) {
  if (!el || el.hidden) return null;
  const rows = {};
  for (const r of el.querySelectorAll(".dt-row")) rows[r.dataset.k] = r.lastElementChild.textContent;
  const m = el.querySelector(".meter>span");
  return { name: el.querySelector(".nm")?.textContent ?? "", end: el.querySelector(".pc")?.textContent ?? "", spin: !!el.querySelector(".spin"),
    frac: m ? parseFloat(m.style.width) / 100 : null, ghost: el.classList.contains("ghost"), rows };
}
// ai: a name's stem and the ending kept whole when it is cut: a bundle's " and N more.tar" (tar.mjs bundle), else an
// ai: extension of at most 10 characters, else nothing
export function nameParts(name) {
  const m = /^(.+)( and \d+ more\.tar)$/s.exec(name);
  if (m) return [m[1], m[2]];
  const dot = name.lastIndexOf(".");
  return dot > 0 && name.length - dot <= 10 ? [name.slice(0, dot), name.slice(dot)] : [name, ""];
}
export function icon(name) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("class", "i");
  s.setAttribute("aria-hidden", "true");
  s.innerHTML = ICONS[name];
  return s;
}



// ai: confirmTap(button, ask, act): a destructive button asks on itself ("Delete 3 files?") for 4 s, and a second tap
// ai: within them acts; ask may be a function, for a count that changes.
export function confirmTap(btn, ask, act) {
  let armed = 0;
  const label = btn.textContent;
  const disarm = () => { if (armed) { clearTimeout(armed); armed = 0; btn.textContent = label; } };
  btn.addEventListener("click", () => {
    if (armed) { disarm(); act(); return; }
    btn.textContent = typeof ask === "function" ? ask() : ask;
    armed = setTimeout(disarm, 4000);
  });
  return disarm;   // ai: for a menu that closes under an armed button (home.mjs), so reopening it asks again
}

// ai: The built app (lizard-web/pwa/build.mjs puts <meta name="lizard-build"> in each page it ships): "<when> <cache name>", or
// ai: null on the rig's source pages, which are not the app and register no worker.
export function buildInfo() { return document.querySelector('meta[name="lizard-build"]')?.content ?? null; }
// ai: about(el): the About line, the build this page is.
// ai: About's text (2026-11-06, the apps' AboutFold, lizard-ui About.kt): the build line (the build's time alone, none on
// ai: the rig's pages, 2026-10-02), LIZARD's license, the design it is based on, then a line a license of what the web
// ai: app carries (its LICENSES.txt: BLAKE3, Wirehair and Zstandard in the codec's wasm, the two fonts), in the order the
// ai: apps' CREDITS first name them. el: the fold's .about-text.
const CREDITS = [["BLAKE3", "Apache 2.0 / CC0 1.0"], ["Wirehair", "BSD 3-Clause"], ["Zstandard", "BSD 3-Clause"], ["Inter", "SIL OFL 1.1"], ["JetBrains Mono", "SIL OFL 1.1"]];
export function about(el, failed = false) {
  const b = buildInfo(), groups = new Map();
  for (const [name, license] of CREDITS) groups.set(license, [...(groups.get(license) ?? []), name]);
  const p = (text, cls) => Object.assign(document.createElement("p"), { textContent: text, className: cls ?? "" });
  el.replaceChildren(...[
    b ? p(`Build: ${b.split(" ")[0].replace("T", " ").replace(/:\d\d(\.\d+)?Z$/, " UTC")}${failed ? ". Update failed" : ""}`, "build") : null,
    p("LIZARD is licensed under the Mozilla Public License 2.0 (MPL 2.0)."),
    p("The design is based on FOCUS: Robust Visual Codes for Everyone (Hermans et al., MobiSys 2016). The implementation is independent."),
    p("Third-party software and licenses:"),
    ...[...groups].map(([license, names]) => p(`${license}: ${names.join(", ")}`)),
  ].filter(Boolean));
}

// ai: slider(el): a range input's fill (ui.css input[type=range], the apps' Bar) kept at its value: --f, the value's
// ai: fraction of its range, set at each input and at each assignment of value by a page (an own-property setter over
// ai: the prototype's), so a value a page sets fills as one dragged does.
export function slider(el) {
  const d = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  const fill = () => { const lo = +el.min || 0, hi = el.max === "" ? 100 : +el.max, v = +d.get.call(el); el.style.setProperty("--f", String(hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0)); };
  Object.defineProperty(el, "value", { configurable: true, get() { return d.get.call(this); }, set(v) { d.set.call(this, v); fill(); } });
  el.addEventListener("input", fill);
  el.slide = fill;   // ai: for a page that moves min or max
  fill();
}

// ai: registerApp(onFail): the installable app's worker (lizard-web/pwa/service-worker.js, a ported scheme), only on a built
// ai: page. Like that scheme's page script it only induces and observes: an update check whenever the page comes back into view
// ai: (an installed app that never navigates hears of a new build no other way), and onFail when an update's install
// ai: failed (a worker that went from installing straight to redundant; one superseded after it activated did not fail).
// ai: Nothing here reloads a page: a new build is served from the next navigation.
export function registerApp(onFail) {
  if (!buildInfo() || !navigator.serviceWorker || location.protocol === "file:") return;
  navigator.serviceWorker.register("service-worker.js").then((reg) => {
    const watch = (w) => {
      if (!w) return;
      let passed = w.state !== "installing";
      w.addEventListener("statechange", () => {
        if (w.state !== "installing" && w.state !== "redundant") passed = true;
        if (w.state === "redundant" && !passed) { console.warn("lizard: an update of the app failed to install"); onFail?.(); }
      });
    };
    watch(reg.installing);
    reg.addEventListener("updatefound", () => watch(reg.installing));
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") reg.update().catch(() => {}); });
  }).catch((e) => console.warn("lizard: the app's worker did not register:", e));
}
