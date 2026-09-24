"""Speech -> English -> ASL gloss, end to end (TODO §13).

The team notebook's STEP 9 main loop without Colab: record (or a file) ->
:class:`~sb.synthesize.asr.Transcriber` -> :class:`~sb.synthesize.gloss.GlossEngine`,
with per-stage wall times. Rendering (gloss -> signs) is deferred; the demo
notebook shows the gloss with each unit's lexicon status, plus WLASL clips
when their videos are on disk (:mod:`sb.synthesize.display`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sb.synthesize.asr import Transcriber
from sb.synthesize.gloss import GlossEngine, GlossResult


@dataclass
class PipelineResult:
    english: str
    gloss: GlossResult
    asr_seconds: float
    gloss_seconds: float
    audio_seconds: float


class SpeechToGloss:
    def __init__(self, asr: Transcriber, engine: GlossEngine):
        self.asr, self.engine = asr, engine

    @classmethod
    def from_configs(cls, speech_cfg: dict, text2gloss_cfg: dict, engine: str | None = None,
                     asr_model: str | None = None) -> "SpeechToGloss":
        return cls(Transcriber.from_config(speech_cfg["asr"], model=asr_model),
                   GlossEngine(engine or speech_cfg["engine"], text2gloss_cfg["t5"]))

    def __call__(self, audio: np.ndarray | str | Path) -> PipelineResult:
        tr = self.asr(audio)
        t0 = time.perf_counter()
        g = self.engine(tr.text) if tr.text else GlossResult("", "", self.engine.name)
        return PipelineResult(tr.text, g, tr.seconds, time.perf_counter() - t0, tr.audio_seconds)
