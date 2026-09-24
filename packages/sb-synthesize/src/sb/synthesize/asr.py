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


def record(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record from the default microphone (local Jupyter; replaces the
    notebook's Colab ``eval_js`` recorder)."""
    import sounddevice as sd

    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


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
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
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
