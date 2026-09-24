/**
 * Headless end-to-end check: serve the built app, open it in Chrome with `?check`,
 * which replays every held-out stream through the real browser path (LiteRT.js WASM +
 * the TS decoder), and report how many match Python exactly.
 *
 *   npm run build && node scripts/browser-check.ts [path-to-chrome]
 */
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const chrome = process.argv[2] ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 4173, DEBUG = 9333;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const server = spawn("npx", ["vite", "preview", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"], { shell: true, stdio: "ignore" });
const browser = spawn(chrome, ["--headless=new", `--remote-debugging-port=${DEBUG}`, "--no-first-run",
  `--user-data-dir=${mkdtempSync(join(tmpdir(), "sb-chrome-"))}`, "about:blank"], { stdio: "ignore" });
const cleanup = () => { browser.kill(); server.kill(); if (process.platform === "win32") spawn("taskkill", ["/pid", String(server.pid), "/t", "/f"]); };

try {
  let targets: { type: string; webSocketDebuggerUrl: string }[] = [];
  for (let i = 0; i < 50 && !targets.some((t) => t.type === "page"); i++) {
    await sleep(300);
    targets = await fetch(`http://127.0.0.1:${DEBUG}/json`).then((r) => r.json()).catch(() => []);
  }
  const ws = new WebSocket(targets.find((t) => t.type === "page")!.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0;
  const pending = new Map<number, (v: any) => void>();
  ws.addEventListener("message", (m) => {
    const msg = JSON.parse(String(m.data));
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)!(msg.result); pending.delete(msg.id); }
    if (msg.method === "Runtime.consoleAPICalled") console.log("  [page]", msg.params.args.map((a: any) => a.value).join(" "));
    if (msg.method === "Runtime.exceptionThrown") console.log("  [page error]", msg.params.exceptionDetails.text, msg.params.exceptionDetails.exception?.description ?? "");
  });
  const send = (method: string, params: object = {}) => new Promise<any>((res) => {
    pending.set(++id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
  await send("Runtime.enable");
  await sleep(1500); // vite preview start-up
  await send("Page.navigate", { url: `http://127.0.0.1:${PORT}/?check` });
  const t0 = Date.now();
  let result: any = null, status = "";
  while (Date.now() - t0 < 300_000) {
    await sleep(2000);
    const r = await send("Runtime.evaluate", { expression: "JSON.stringify({c: window.__replayCheck ?? null, s: document.getElementById('status')?.textContent})", returnByValue: true });
    const v = JSON.parse(r.result.value ?? "{}");
    if (v.s !== status) { status = v.s; console.log("  status:", status); }
    if (v.c) { result = v.c; break; }
    if (/Failed|Error|error/.test(status ?? "")) break;
    if (status === undefined && Date.now() - t0 > 20_000) break; // the page never loaded
  }
  const rows = await send("Runtime.evaluate", { expression: "[...document.querySelectorAll('#check-rows tr')].map(tr => [...tr.children].map(td => td.textContent).join(' | ')).join('\n')", returnByValue: true });
  console.log(rows.result.value);
  console.log(result ? `RESULT ${JSON.stringify(result)} in ${((Date.now() - t0) / 1000).toFixed(0)} s` : "no result");
  if (process.env.SCREENSHOT) {
    await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 1500, deviceScaleFactor: 1, mobile: false });
    const shot = await send("Page.captureScreenshot", { format: "png" });
    const { writeFileSync } = await import("node:fs");
    writeFileSync(process.env.SCREENSHOT, Buffer.from(shot.data, "base64"));
    console.log(`screenshot -> ${process.env.SCREENSHOT}`);
  }
  // MediaPipe Holistic loads and runs (synthetic video, no camera)
  await send("Page.navigate", { url: `http://127.0.0.1:${PORT}/?smoke-holistic` });
  let smoke: any = null;
  for (let i = 0; i < 90 && !smoke; i++) {
    await sleep(2000);
    const r = await send("Runtime.evaluate", { expression: "JSON.stringify(window.__holisticSmoke ?? null)", returnByValue: true });
    smoke = JSON.parse(r.result.value ?? "null");
  }
  console.log(`HOLISTIC ${JSON.stringify(smoke)}`);
  process.exitCode = result && result.same === result.done && smoke?.ok ? 0 : 1;
} finally {
  cleanup();
}
