"""The live decoder, one frame at a time: D3 segmentation + fused acceptance
(TODO §12.5/§12.6).

:mod:`sb.recognize.continuous.fuse` decodes a whole stream at once, which is
what the sweeps need. The browser never has the whole stream. It gets one
probability vector per frame, and must say *at that frame* whether a sign was
just accepted. This class is that loop. It is the reference the TypeScript
client (``apps/web``) ports. Stepping it over a stream must give exactly
``fuse.decode_fused(fuse.segments_d3(...))``'s emissions (the parity check
in ``gislr.5.pipeline.sign-to-speech.demo.ipynb``).

State is O(glosses): the running vote of the current non-null run, its mass,
length and per-gloss peak, and the accepted history (the prior's context).
``rule`` is a :class:`~sb.recognize.continuous.fuse.Rule`, any other
``Decider`` (``select.Acceptor``), or a ``select.Lattice``, which holds a
segment back until ``lag`` more have closed. Its emissions then arrive later
than the frame they are stamped with (the sign's own last frame). A run ends on
the first frame whose null probability reaches ``nu``. That frame is when the
decision is made, one frame after the sign's last frame, which is the
"+1 frame" median commit latency ``continuous-models.md`` reports for D3.
"""

from __future__ import annotations

import numpy as np

from sb.recognize.continuous.fuse import Decider, Emission, Prior, Rule, Segment
from sb.recognize.continuous.select import Lattice


class OnlineDecoder:
    def __init__(self, null_index: int, nu: float, min_len: int, rule: Decider | Lattice = Rule(),
                 prior: Prior | None = None, collapse: bool = True):
        self.null, self.nu, self.min_len = null_index, nu, min_len
        self.rule, self.prior, self.collapse = rule, prior, collapse
        self.reset()

    def reset(self) -> None:
        """A new sentence: empty history, no run in progress."""
        self.history: tuple[int, ...] = ()
        self._runner = self.rule.runner(self.prior) if isinstance(self.rule, Lattice) else None
        self._score = np.zeros(self.null, np.float64)
        self._peak = np.zeros(self.null, np.float64)
        self._mass = 0.0
        self._start: int | None = None
        self._last: int | None = None  # last accepted class, for collapse

    def _emit(self, e: Emission) -> Emission | None:
        if self.collapse and self._last == e[0]:
            return None
        self._last = e[0]
        return e

    def _close(self, end: int) -> Emission | None:
        start, self._start = self._start, None
        score, mass, peak = self._score, self._mass, self._peak
        self._score, self._mass = np.zeros(self.null, np.float64), 0.0
        self._peak = np.zeros(self.null, np.float64)
        if start is None or end - start < self.min_len:
            return None
        seg = Segment(start, end, score / max(float(score.sum()), 1e-12), mass, peak)
        if self._runner is not None:
            got = self._runner.push(seg)  # at most one: the oldest held segment
            self.history = self._runner.history
            return self._emit(got[0]) if got else None
        assert not isinstance(self.rule, Lattice)
        d = self.rule.decide(seg, self.history, self.prior)
        if d is None:
            return None
        c, conf = d
        self.history = self.history + (c,)
        return self._emit((c, end - 1, conf))

    def step(self, p: np.ndarray, t: int) -> Emission | None:
        """Frame ``t``'s probabilities over glosses + null -> the sign
        accepted at this frame, if any (committed at the run's last frame,
        ``t - 1``)."""
        if p[self.null] < self.nu:
            w = 1.0 - float(p[self.null])
            self._score += w * p[:self.null]
            np.maximum(self._peak, p[:self.null], out=self._peak)
            self._mass += w
            if self._start is None:
                self._start = t
            return None
        return self._close(t) if self._start is not None else None

    def flush(self, t: int) -> Emission | None:
        """End of input at frame ``t`` (exclusive): close a run in progress.
        A ``Lattice`` rule can still hold several segments; use
        :meth:`flush_all` for it."""
        out = self.flush_all(t)
        assert len(out) <= 1, "several held segments: call flush_all"
        return out[0] if out else None

    def flush_all(self, t: int) -> list[Emission]:
        """End of input (end of the sentence): close a run in progress and
        decide every segment still held."""
        out = []
        if self._start is not None and (e := self._close(t)) is not None:
            out.append(e)
        if self._runner is not None:
            for e in self._runner.flush():
                if (e := self._emit(e)) is not None:
                    out.append(e)
            self.history = self._runner.history
        return out
