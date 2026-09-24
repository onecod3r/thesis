"""English -> ASL gloss engines (TODO §13).

| engine | what |
|---|---|
| ``rules_v1`` | the team's rule engine, frozen (T5's training input) |
| ``rules_v2`` | the fixed rule engine |
| ``hybrid_team`` | rules_v1 -> T5, the team's exact ``generate()``; T5 is trusted |
| ``hybrid_guarded`` | rules_v1 -> T5 with repetition control -> :func:`hybrid.guard`; rejected -> rules_v2 |

Without a T5 checkpoint both hybrids degrade the way the team notebook does:
``hybrid_team`` -> rules_v1, ``hybrid_guarded`` -> rules_v2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ENGINES = ("rules_v1", "rules_v2", "hybrid_team", "hybrid_guarded")


def split_sentences(text: str) -> list[str]:
    """The team's ``apply_rules_multi`` split: on ``.!?`` runs."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


@dataclass
class GlossResult:
    english: str
    gloss: str
    engine: str
    per_sentence: list[dict] = field(default_factory=list)


class GlossEngine:
    """One engine by name. ``t5_cfg`` is the ``t5`` block of
    ``aslg.text2gloss.json``; the checkpoint is loaded only by the hybrids,
    and only if it exists."""

    def __init__(self, name: str, t5_cfg: dict | None = None, model_dir=None):
        if name not in ENGINES:
            raise ValueError(f"unknown engine {name!r}; one of {ENGINES}")
        self.name = name
        self.t5 = None
        self.min_content_recall = (t5_cfg or {}).get("guard", {}).get("min_content_recall", 0.6)
        if name.startswith("hybrid") and t5_cfg is not None:
            from sb.synthesize.gloss import t5 as T5

            path = model_dir or t5_cfg["model_dir"]
            if T5.available(path):
                preset = "team" if name == "hybrid_team" else "guarded"
                self.t5 = T5.T5Refiner.from_config(t5_cfg, preset=preset, model_dir=path)

    @property
    def uses_t5(self) -> bool:
        return self.t5 is not None

    def gloss_sentences(self, sentences: list[str]) -> list[dict]:
        """One dict per sentence: ``english, gloss, source, rule_v1, t5, reasons``."""
        from sb.synthesize.gloss import rules_v2

        if self.name == "rules_v2":
            return [{"english": s, "gloss": rules_v2.convert(s), "source": "rules_v2"} for s in sentences]
        from sb.synthesize.gloss import rules_v1

        v1 = [rules_v1.convert(s) for s in sentences]
        if self.name == "rules_v1":
            return [{"english": s, "gloss": g, "source": "rules_v1"} for s, g in zip(sentences, v1)]
        t5 = self.t5.refine_batch(list(zip(sentences, v1))) if self.t5 is not None else [None] * len(sentences)
        if self.name == "hybrid_team":
            return [{"english": s, "gloss": t or g, "source": "t5" if t else "rules_v1", "rule_v1": g, "t5": t}
                    for s, g, t in zip(sentences, v1, t5)]
        from sb.synthesize.gloss import hybrid

        out = []
        for s, g, t in zip(sentences, v1, t5):
            o = hybrid.combine(s, g, t, rules_v2.convert(s), min_content_recall=self.min_content_recall)
            out.append({"english": s, "gloss": o.gloss, "source": o.source, "rule_v1": g, "t5": t,
                        "reasons": o.reasons})
        return out

    def __call__(self, text: str) -> GlossResult:
        """Free text (possibly several sentences) -> gloss, sentences joined
        with `` | `` as in the team notebook."""
        rows = self.gloss_sentences(split_sentences(text))
        return GlossResult(text, " | ".join(r["gloss"] for r in rows), self.name, rows)
