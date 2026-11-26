// ai: Headless Firefox driven over WebDriver BiDi (its remote agent, Node's own WebSocket; phone.mjs is Chrome's over the
// ai: DevTools protocol): navigate, evaluate an expression (promises awaited), set a file input's files; firefox() starts
// ai: one on a fresh profile of its own (prefs as user.js), its processes one group (pid), close() ends them.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
export async function firefox(prefs = {}) {
  const port = 9700 + (process.pid % 200), prof = mkdtempSync(join(tmpdir(), "lizard-firefox-"));
  writeFileSync(`${prof}/user.js`, Object.entries({ "browser.shell.checkDefaultBrowser": false, ...prefs }).map(([k, v]) => `user_pref("${k}", ${JSON.stringify(v)});\n`).join(""));
  const p = spawn("firefox", ["--headless", "--no-remote", "--profile", prof, `--remote-debugging-port=${port}`], { stdio: "ignore", detached: true });
  let ws = null;
  for (let i = 0; i < 100 && !ws; i++) {
    await new Promise((r) => setTimeout(r, 200));
    try { ws = await new Promise((ok, no) => { const w = new WebSocket(`ws://127.0.0.1:${port}/session`); w.onopen = () => ok(w); w.onerror = () => no(new Error("no")); }); } catch {}
  }
  if (!ws) throw new Error("no BiDi");
  let id = 0;
  const waiting = new Map();
  ws.onmessage = ({ data }) => { const m = JSON.parse(data); if (m.id && waiting.has(m.id)) { const w = waiting.get(m.id); waiting.delete(m.id); m.type === "error" ? w.no(new Error(`${m.error}: ${m.message}`)) : w.ok(m.result); } };
  const send = (method, params = {}) => new Promise((ok, no) => { const k = ++id; waiting.set(k, { ok, no }); ws.send(JSON.stringify({ id: k, method, params })); });
  await send("session.new", { capabilities: {} });
  const context = (await send("browsingContext.getTree", {})).contexts[0].context;
  return {
    navigate: (url) => send("browsingContext.navigate", { context, url, wait: "complete" }),
    evaluate: async (expression) => { const r = await send("script.evaluate", { expression, target: { context }, awaitPromise: true, resultOwnership: "none", serializationOptions: { maxObjectDepth: 4 } }); if (r.type === "exception") throw new Error(r.exceptionDetails.text); return r.result; },
    setFiles: async (selector, files) => { const n = (await send("browsingContext.locateNodes", { context, locator: { type: "css", value: selector } })).nodes[0]; await send("input.setFiles", { context, element: { sharedId: n.sharedId }, files }); },
    pid: p.pid,   // ai: its process group's (detached): its own processes, not another Firefox's
    close: () => { try { process.kill(-p.pid, "SIGKILL"); } catch {} setTimeout(() => rmSync(prof, { recursive: true, force: true }), 500).unref(); },
  };
}
// ai: a BiDi RemoteValue (an evaluate's result) as plain JS
export const plain = (v) => v.type === "object" ? Object.fromEntries(v.value.map(([k, x]) => [k, plain(x)])) : v.type === "array" ? v.value.map(plain) : v.value;
