"""The live decoder, one frame at a time: D3 segmentation + fused acceptance
(TODO §12.5/§12.6).

:mod:`sb.recognize.continuous.fuse` decodes a whole stream at once, which is
what the sweeps need. The browser never has the whole stream. It gets one
probability vector per frame, and must say *at that frame* whether a sign was
just accepted. This class is that loop. It is the reference the TypeScript
client (``apps/web``) ports. Stepping it over a stream must give exactly
``fuse.decode_fused(fuse.segments_d3(...))``'s emissions (the parity check
in ``gislr.5.pipeline.sign-to-speech.demo.ipynb``).

State is O(glosses): the running vote of the current non-null run, its mass
and length, and the accepted history (the prior's context). A run ends on
the first frame whose null probability reaches ``nu``. That frame is when the
decision is made, one frame after the sign's last frame, which is the
"+1 frame" median commit latency ``continuous-models.md`` reports for D3.
"""

from __future__ import annotations

import numpy as np

from sb.recognize.continuous.fuse import Emission, Prior, Rule, Segment, decide


class OnlineDecoder:
    def __init__(self, null_index: int, nu: float, min_len: int, rule: Rule = Rule(),
                 prior: Prior | None = None, collapse: bool = True):
        self.null, self.nu, self.min_len = null_index, nu, min_len
        self.rule, self.prior, self.collapse = rule, prior, collapse
        self.reset()

    def reset(self) -> None:
        """A new sentence: empty history, no run in progress."""
        self.history: tuple[int, ...] = ()
        self._score = np.zeros(self.null, np.float64)
        self._mass = 0.0
        self._start: int | None = None
        self._last: int | None = None  # last accepted class, for collapse

    def _close(self, end: int) -> Emission | None:
        start, self._start = self._start, None
        score, mass = self._score, self._mass
        self._score, self._mass = np.zeros(self.null, np.float64), 0.0
        if start is None or end - start < self.min_len:
            return None
        seg = Segment(start, end, score / max(float(score.sum()), 1e-12), mass)
        d = decide(seg, self.history, self.prior, self.rule)
        if d is None:
            return None
        c, conf = d
        self.history = self.history + (c,)
        if self.collapse and self._last == c:
            return None
        self._last = c
        return c, end - 1, conf

    def step(self, p: np.ndarray, t: int) -> Emission | None:
        """Frame ``t``'s probabilities over glosses + null -> the sign
        accepted at this frame, if any (committed at the run's last frame,
        ``t - 1``)."""
        if p[self.null] < self.nu:
            w = 1.0 - float(p[self.null])
            self._score += w * p[:self.null]
            self._mass += w
            if self._start is None:
                self._start = t
            return None
        return self._close(t) if self._start is not None else None

    def flush(self, t: int) -> Emission | None:
        """End of input at frame ``t`` (exclusive): close a run in progress."""
        return self._close(t) if self._start is not None else None
