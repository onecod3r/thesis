"""Speech generated on this machine, for when there is no microphone (TODO
§13, and the TTS stage of sign -> speech, TODO §12.6).

Two uses:
- **ASR evaluation without a recording** (``speech.2.asr.eval.ipynb``): speak
  the 30 sentences with Windows' built-in SAPI voices, optionally with added
  noise, and transcribe that. Synthetic speech is cleaner and more regular
  than a person reading, so its WER is an **optimistic** bound. It says whether
  the pipeline works and how the models compare, not what a user will get.
- **An offline TTS baseline** for the sign -> speech output stage
  (``gislr.4.downstream.tts.ipynb``), next to Workers AI's hosted TTS
  (``sb.rescore.client.WorkersAITTS``).

SAPI is driven through PowerShell's ``System.Speech``, so nothing needs to be
installed, but it is Windows only. The text reaches PowerShell through a UTF-8
temp file, never the command line, so quotes and apostrophes are safe.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

_PS_VOICES = ("Add-Type -AssemblyName System.Speech; "
              "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
              "Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }")

_PS_SPEAK = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice($env:SB_TTS_VOICE)
$s.Rate = [int]$env:SB_TTS_RATE
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($env:SB_TTS_OUT, $fmt)
$s.Speak([System.IO.File]::ReadAllText($env:SB_TTS_TEXT, [System.Text.Encoding]::UTF8))
$s.Dispose()
"""


def sapi_voices() -> list[str]:
    """Installed, enabled SAPI voices (e.g. ``Microsoft David Desktop``);
    empty off Windows."""
    if os.name != "nt":
        return []
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", _PS_VOICES],
                         capture_output=True, text=True, timeout=60)
    return [v.strip() for v in out.stdout.splitlines() if v.strip()]


def sapi_speak(text: str, path: Path, voice: str, rate: int = 0) -> dict:
    """Speak ``text`` with a SAPI voice into a 16 kHz mono 16-bit WAV at
    ``path`` (written atomically). ``rate`` runs from -10 (slow) to 10 (fast).
    Returns ``{"path", "seconds" (wall time), "audio_seconds"}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        text_file = f.name
    env = {**os.environ, "SB_TTS_VOICE": voice, "SB_TTS_RATE": str(rate), "SB_TTS_OUT": str(tmp),
           "SB_TTS_TEXT": text_file}
    t0 = time.perf_counter()
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", _PS_SPEAK], env=env,
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"SAPI synthesis failed ({voice}): {out.stderr.strip()[:400]}")
    finally:
        os.unlink(text_file)
    seconds = time.perf_counter() - t0
    os.replace(tmp, path)
    import soundfile as sf

    info = sf.info(str(path))
    return {"path": str(path), "seconds": seconds, "audio_seconds": info.frames / info.samplerate}


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White noise at ``snr_db`` relative to the signal's power over its
    non-silent samples (so leading/trailing silence doesn't dilute it)."""
    voiced = audio[np.abs(audio) > 0.01 * (np.abs(audio).max() or 1.0)]
    p_sig = float(np.mean(voiced ** 2)) if voiced.size else float(np.mean(audio ** 2))
    noise = rng.normal(0.0, np.sqrt(p_sig / 10 ** (snr_db / 10)), audio.shape)
    return (audio + noise).astype(np.float32)
