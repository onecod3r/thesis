# apps/edge — Cloudflare Worker(s)

Session state, the downstream LLM and TTS (TODO §12.6), deployed on Cloudflare
Workers.

**Open until §12.5 picks them:** the Workers AI models vs an external API for
the LLM and TTS; the transport (WebSocket through a Durable Object vs HTTP);
where custom-sign prototypes are stored (KV / D1 / R2 — R2 was not activated on
this account as of 2026-09-04); the CPU-time and memory limits that constrain
all of this.

Responsibilities:
- **Session** (likely a Durable Object): accepted-gloss history, which is the LLM's
  context, plus the user's custom glosses and their English meanings.
- **Fused acceptance**: LLM next-gloss prior, constrained to the vocabulary,
  × the recognizer's confidence → accept or wait (§12.6). The offline notebook
  version comes first. This one ports its tuned weights.
- **Gloss → English**: prompts versioned and hashed the way `sb-rescore`'s
  `prompts/v1/` are, so a deployed prompt maps to an evaluated one.
- **TTS**: English → audio, streamed back to `web/`.

No secrets in this tree: use `wrangler secret` / bindings, never committed
values.
