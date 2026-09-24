# apps/edge — Cloudflare Worker

Serves `apps/web` as **Workers static assets** (Free plan, user decision 2026-09-24):
- the page and JS;
- the 3.5 MB step model and the 14 MB Holistic model;
- the WASM runtimes.

All inference runs in the browser, so the Worker does almost nothing per request.

| route | what |
|---|---|
| `/` … | the built app (`../web/dist`) |
| `GET /api/health` | `{llm, model, prompt_sha256}`. The app shows its "English by LLM" option only when `llm` is true |
| `POST /api/english` `{"glosses": [...]}` | gloss → English on Workers AI with `packages/sb-rescore/.../prompts/v1/gloss2en.txt`. That is the same prompt, and the same SHA-256, that `gislr.4.downstream.gloss-to-english.ipynb` scores |

```bash
cd apps/web && npm run build          # the Worker serves web/dist
cd ../edge && npm install
npm run dev:offline                   # no Cloudflare account needed; no LLM (the app uses its rules)
npm run dev                           # with the Workers AI binding: needs `wrangler login` or CLOUDFLARE_API_TOKEN
npm run deploy                        # deploys to your account (not done yet)
```

**Verified locally (2026-09-24, `dev:offline`):**
- the page, the model (3,457,948 B) and the Holistic model (13,683,609 B) are served;
- `/api/health` reports `llm: false`, with the prompt hash equal to Python's `load_prompt`.

**Not yet verified:** the LLM route itself, which needs your Cloudflare credentials, and a
deployment.

**Not here yet** (the deployment research's plan): a Durable Object session, Workers AI TTS
(the app uses the browser's voices; Aura costs about 120 neurons per sentence on Free),
custom-sign storage (§12.7).

No secrets in this tree: use `wrangler secret` or bindings.
