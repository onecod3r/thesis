"""Workers AI client for the downstream LLM (TODO §12.6), offline-evaluation
side.

The deployed pipeline calls Workers AI through the Worker's ``env.AI``
binding (``apps/edge``). This client calls the **same models through the
REST API** so the notebooks score exactly the model and prompt the edge
will use. Credentials come from the environment or the repo ``.env``:

- ``CLOUDFLARE_ACCOUNT_ID``
- ``CLOUDFLARE_API_TOKEN`` (a token with the *Workers AI: Read* permission)

The account is on the Workers **Free** plan (user, 2026-09-24): 10,000
neurons per day. At the pricing read on 2026-09-24, one gloss sentence is
about 400 input and 20 output tokens, well under 10 neurons on the 3B/8B
Llama models. The 132-sentence eval set costs a few hundred neurons.

Every call is cached in a JSONL file keyed by ``(model, prompt sha256,
input)``, so a notebook re-run never pays twice and a prompt edit can never
reuse an answer to the old prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from sb.core.paths import env_value

PROMPTS_DIR = Path(__file__).parent / "prompts"
API = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"


def load_prompt(version: str, name: str) -> tuple[str, str]:
    """``(text, sha256)`` of ``prompts/<version>/<name>.txt``. The hash goes
    into every result, so a score always names the prompt it came from."""
    text = (PROMPTS_DIR / version / f"{name}.txt").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()


def credentials() -> tuple[str, str] | None:
    account, token = env_value("CLOUDFLARE_ACCOUNT_ID"), env_value("CLOUDFLARE_API_TOKEN")
    return (account, token) if account and token else None


def _extract(result: dict) -> str:
    r = result.get("result", result)
    if isinstance(r, dict):
        if isinstance(r.get("response"), str):
            return r["response"]
        choices = r.get("choices")
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content") or choices[0].get("text", "")
    raise ValueError(f"unrecognized Workers AI response: {str(result)[:300]}")


class WorkersAI:
    """Chat completions against one Workers AI model, cached on disk."""

    def __init__(self, model: str, cache_path: Path, max_tokens: int = 80,
                 temperature: float = 0.0, timeout: float = 60.0):
        creds = credentials()
        if creds is None:
            raise RuntimeError("set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (env or repo .env)")
        self.account, self.token = creds
        self.model, self.cache_path = model, Path(cache_path)
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row

    def _key(self, system_sha: str, user: str) -> str:
        return hashlib.sha256(f"{self.model}\n{system_sha}\n{user}".encode()).hexdigest()

    def chat(self, system: str, system_sha: str, user: str, retries: int = 3) -> dict:
        """One completion -> ``{"text", "cached", "latency_s", ...}``."""
        key = self._key(system_sha, user)
        if key in self.cache:
            return {**self.cache[key], "cached": True}
        body = json.dumps({
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens, "temperature": self.temperature,
        }).encode()
        req = urllib.request.Request(API.format(account=self.account, model=self.model), data=body,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        err: Exception | None = None
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                text = _extract(payload).strip()
                row = {"key": key, "model": self.model, "prompt_sha256": system_sha, "input": user,
                       "text": text, "latency_s": round(time.perf_counter() - t0, 3)}
                self.cache[key] = row
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                return {**row, "cached": False}
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Workers AI call failed after {retries} tries: {err}")


def clean_sentence(text: str) -> str:
    """First line of a completion, stripped of quotes and a leading label."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    for prefix in ("English:", "Translation:", "Output:"):
        if line.startswith(prefix):
            line = line[len(prefix):]
    return line.strip().strip('"').strip()


class WorkersAITTS:
    """Text -> speech on a Workers AI TTS model, cached as audio files.

    The request body differs per model family (read 2026-09-24): Deepgram
    Aura takes ``{"text", "speaker", "encoding", ...}`` and answers with raw
    MPEG audio; MeloTTS takes ``{"prompt", "lang"}`` and may answer with
    JSON carrying base64 audio. Both are handled. The format is sniffed from
    the bytes. One file per (model, params, text) under ``cache_dir``.
    """

    def __init__(self, model: str, cache_dir: Path, timeout: float = 60.0, **params):
        creds = credentials()
        if creds is None:
            raise RuntimeError("set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (env or repo .env)")
        self.account, self.token = creds
        self.model, self.cache_dir, self.timeout, self.params = model, Path(cache_dir), timeout, params
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _body(self, text: str) -> dict:
        if "melotts" in self.model:
            return {"prompt": text, "lang": self.params.get("lang", "en")}
        return {"text": text, **self.params}

    def speak(self, text: str, retries: int = 3) -> dict:
        """-> ``{"path", "cached", "latency_s", "bytes"}``. ``latency_s`` is
        the time to the complete audio (the REST call is not streamed)."""
        import base64

        body = self._body(text)
        key = hashlib.sha256(json.dumps([self.model, body], sort_keys=True).encode()).hexdigest()[:24]
        hit = next((p for ext in (".mp3", ".wav", ".ogg") if (p := self.cache_dir / f"{key}{ext}").exists()), None)
        if hit is not None:
            meta = json.loads(hit.with_suffix(".json").read_text()) if hit.with_suffix(".json").exists() else {}
            return {"path": str(hit), "cached": True, "latency_s": meta.get("latency_s"), "bytes": hit.stat().st_size}
        req = urllib.request.Request(API.format(account=self.account, model=self.model),
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        err: Exception | None = None
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    data = resp.read()
                latency = time.perf_counter() - t0
                if "json" in ctype or data[:1] == b"{":
                    payload = json.loads(data)
                    r = payload.get("result", payload)
                    audio = r.get("audio") if isinstance(r, dict) else None
                    if not audio:
                        raise ValueError(f"no audio in Workers AI TTS response: {str(payload)[:300]}")
                    data = base64.b64decode(audio)
                ext = ".wav" if data[:4] == b"RIFF" else ".ogg" if data[:4] == b"OggS" else ".mp3"
                path = self.cache_dir / f"{key}{ext}"
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
                path.with_suffix(".json").write_text(json.dumps({"model": self.model, "body": body,
                                                                 "latency_s": round(latency, 3)}))
                return {"path": str(path), "cached": False, "latency_s": round(latency, 3), "bytes": len(data)}
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Workers AI TTS call failed after {retries} tries: {err}")
