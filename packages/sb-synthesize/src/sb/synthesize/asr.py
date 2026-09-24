"""Speech -> English with faster-whisper (TODO §13).

The team notebook's settings (cell 6, "STEP 7"): ``language="en"``, beam 5,
``temperature=0``, ``condition_on_previous_text``, Silero VAD with 500 ms min
silence and 400 ms speech pad; ``large-v3`` in fp16 on GPU / int8 on CPU.
Two fixes from the audit (report §3):
- its peak normalization was dead code (it normalized an array, then
  transcribed the *file path*). Here the normalized array is what Whisper gets;
- ``best_of`` is dropped: it only applies when sampling, and temperature is 0.

Model size comes from the config: ``large-v3`` locally, ``large-v3-turbo`` is
what Workers AI hosts (§12.5), so both are worth measuring.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def _add_cuda12_dlls() -> None:
    """CTranslate2 needs CUDA 12 cuBLAS; torch here ships only CUDA 13's. The
    ``nvidia-cublas-cu12`` wheel provides it, but Windows only finds DLLs in
    registered directories."""
    if os.name != "nt":
        return
    import importlib.util

    spec = importlib.util.find_spec("nvidia.cublas")
    for loc in (spec.submodule_search_locations or []) if spec else []:
        bin_dir = Path(loc) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def load_audio(path: str | Path) -> np.ndarray:
    """Any file ffmpeg/libsndfile can read -> mono float32 at 16 kHz."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio / m).astype(np.float32) if m > 0 else audio.astype(np.float32)


def microphone() -> str | None:
    """The default input device's name, or ``None`` when there is none (a
    remote or headless machine: PortAudio reports the default input as -1,
    and ``sd.rec`` then fails with ``Error querying device -1``)."""
    try:
        import sounddevice as sd

        idx = sd.default.device[0]
        if idx is None or idx < 0:
            return None
        return str(sd.query_devices(idx)["name"])
    except Exception:
        return None


def record(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record from the default microphone (local Jupyter; replaces the
    notebook's Colab ``eval_js`` recorder). Raises a clear error when the
    machine has no default input device; check :func:`microphone` first."""
    import sounddevice as sd

    if microphone() is None:
        raise RuntimeError("no default input device (remote/headless machine?). Use an audio file, "
                           "or synthesized speech from sb.synthesize.tts.")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def model_dir(model: str) -> str:
    """A faster-whisper model name -> a local directory holding it, downloaded
    once into ``data/external/whisper/<model>``.

    The default Hugging Face cache stores files as symlinks. On Windows
    without Developer Mode, creating one fails with ``WinError 1314`` for some
    repos (seen 2026-09-24 with ``large-v3-turbo``). A plain ``local_dir``
    download writes real files. A path that already exists is used as is.
    """
    if Path(model).exists():
        return model
    from faster_whisper.utils import download_model

    from sb.core.paths import EXTERNAL_DIR

    out = EXTERNAL_DIR / "whisper" / model
    if not (out / "model.bin").exists():
        download_model(model, output_dir=str(out))
    return str(out)


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _words(n % 100))
    if n < 1_000_000:
        return _words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _words(n % 1000))
    return str(n)


def spell_numbers(text: str) -> str:
    """Whole numbers written as digits -> words (``3 o'clock`` -> ``three
    o'clock``, ``2,000`` -> ``two thousand``). Whisper writes most numbers as
    digits, while the gloss engines expect words (``TIME THREE``, not
    ``TIME 3``), and a digit can't be matched to a sign in the lexicon. Found
    2026-09-24: on synthesized speech, the only errors left on clean audio were
    digits. Times like ``3:30`` become ``three thirty``. Decimals are left as
    they are."""
    import re

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: _words(int(m[1])) + ("" if m[2] == "00" else " " + _words(int(m[2]))), text)
    return re.sub(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(?![\d.]|\.\d)",
                  lambda m: _words(int(m[1].replace(",", ""))), text)


@dataclass
class Transcript:
    text: str
    seconds: float  # wall time of transcription
    audio_seconds: float
    language_probability: float | None = None


class Transcriber:
    def __init__(self, model: str = "large-v3", device: str | None = None, compute_type: str | None = None,
                 transcribe_kwargs: dict | None = None):
        import torch

        _add_cuda12_dlls()
        from faster_whisper import WhisperModel

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self.model = WhisperModel(model_dir(model), device=device, compute_type=compute_type)
        self.kwargs = dict(transcribe_kwargs or {})

    @classmethod
    def from_config(cls, cfg: dict, model: str | None = None) -> "Transcriber":
        """``cfg`` is the ``asr`` block of ``speech.pipeline.json``."""
        return cls(model or cfg["model"], transcribe_kwargs=cfg["transcribe"])

    def __call__(self, audio: np.ndarray | str | Path, normalize: bool = True) -> Transcript:
        if not isinstance(audio, np.ndarray):
            audio = load_audio(audio)
        if normalize:
            audio = peak_normalize(audio)
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(audio, **self.kwargs)
        text = " ".join(s.text.strip() for s in segments).strip()  # segments is lazy: this runs the model
        return Transcript(text, time.perf_counter() - t0, len(audio) / SAMPLE_RATE,
                          getattr(info, "language_probability", None))
