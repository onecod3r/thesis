/**
 * The SignBridge Worker (TODO §12.5 build step 4, reduced to what sign -> speech needs now).
 *
 * - Everything under `/` is the browser app (Workers static assets, free). Recognition,
 *   the decoder, the prior, rule-based English and speech all run in the browser.
 * - `POST /api/english {"glosses": [...]}` -> `{"english", "model", "prompt_sha256"}`:
 *   gloss -> English on Workers AI with `prompts/v1/gloss2en.txt`, the prompt the
 *   gloss-to-English notebook scores, so a deployed answer maps to an evaluated one.
 *   About a few neurons per sentence on the Free plan's 10k/day.
 * - `GET /api/health` tells the app whether the LLM is available; if not, it keeps
 *   using its offline rules.
 *
 * - `POST /api/asr` (body: base64 of a 16 kHz mono WAV, `text/plain`) -> `{"text", "model"}`:
 *   speech -> English on Workers AI `whisper-large-v3-turbo` (TODO §13 Phase 4, the model
 *   `speech.2.asr.eval.ipynb` measured). The browser encodes the base64 so the Worker only
 *   passes a string through (Free plan: 10 ms CPU). The speech -> gloss page (`speech.html`)
 *   glosses the text in the browser.
 *
 * No session state yet: the app sends one finished sentence per request.
 */

import prompt from "../../../packages/sb-rescore/src/sb/rescore/prompts/v1/gloss2en.txt";

interface Env {
  ASSETS: Fetcher;
  AI?: Ai;
  LLM_MODEL: string;
  ASR_MODEL: string;
}

const MAX_GLOSSES = 40;
const MAX_AUDIO_B64 = 3_000_000; // ~70 s of 16 kHz mono 16-bit WAV

async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text.replace(/\r\n/g, "\n")));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** `sb.rescore.client.clean_sentence`: the first line, without quotes or a leading label. */
function cleanSentence(text: string): string {
  let line = text.trim().split(/\r?\n/)[0] ?? "";
  for (const prefix of ["English:", "Translation:", "Output:"]) {
    if (line.startsWith(prefix)) line = line.slice(prefix.length).trim();
  }
  return line.replace(/^["'“”]+|["'“”]+$/g, "").trim();
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/health") {
      return json({ llm: Boolean(env.AI), model: env.LLM_MODEL, prompt_sha256: await sha256(prompt),
        asr: Boolean(env.AI) && env.ASR_MODEL !== "none", asr_model: env.ASR_MODEL });
    }
    if (url.pathname === "/api/english" && request.method === "POST") {
      if (!env.AI) return json({ error: "no AI binding" }, 503);
      const body = (await request.json().catch(() => null)) as { glosses?: unknown } | null;
      const glosses = body?.glosses;
      if (!Array.isArray(glosses) || !glosses.length || glosses.length > MAX_GLOSSES
          || !glosses.every((g) => typeof g === "string" && /^[A-Za-z]{1,20}$/.test(g))) {
        return json({ error: `glosses: 1-${MAX_GLOSSES} alphabetic labels` }, 400);
      }
      const out = (await env.AI.run(env.LLM_MODEL as Parameters<Ai["run"]>[0], {
        messages: [{ role: "system", content: prompt }, { role: "user", content: glosses.join(" ") }],
        max_tokens: 80,
        temperature: 0,
      } as never)) as { response?: string };
      return json({ english: cleanSentence(out.response ?? ""), model: env.LLM_MODEL, prompt_sha256: await sha256(prompt) });
    }
    if (url.pathname === "/api/asr" && request.method === "POST") {
      if (!env.AI || env.ASR_MODEL === "none") return json({ error: "no AI binding" }, 503);
      const audio = (await request.text()).trim();
      if (!audio || audio.length > MAX_AUDIO_B64 || !/^[A-Za-z0-9+/]+=*$/.test(audio.slice(-64))) {
        return json({ error: `body: base64 WAV, at most ${MAX_AUDIO_B64} characters` }, 400);
      }
      const out = (await env.AI.run(env.ASR_MODEL as Parameters<Ai["run"]>[0], {
        audio, task: "transcribe", language: "en", vad_filter: true, condition_on_previous_text: false,
      } as never)) as { text?: string; transcription_info?: { text?: string } };
      return json({ text: (out.text ?? out.transcription_info?.text ?? "").trim(), model: env.ASR_MODEL });
    }
    if (url.pathname.startsWith("/api/")) return json({ error: "not found" }, 404);
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
