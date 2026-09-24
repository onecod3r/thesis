"""The team's fine-tuned T5 gloss refiner (TODO §13).

Input format and the three training-time limits are copied from the team
notebook (cell 3, "STEP 4B"), which says they must match ``run_inference.py``
/ ``training_metadata.json`` exactly: ``max_input_length`` 128,
``max_target_length`` 64, beam 4, and the prompt
``"English: … Rule gloss: … Produce ASL gloss:"`` with the **rules_v1** gloss.

Generation has two presets (``experiments/synthesis/configs/aslg.text2gloss.json``):
- ``team``: exactly the notebook's ``generate()`` call (no repetition control),
  so its outputs reproduce the recorded ones;
- ``guarded``: adds ``no_repeat_ngram_size`` and ``repetition_penalty``, the
  audit's fix for ``HE HE HE HE HE HE HE`` (report §3).

The checkpoint (``thesis_hybrid_dataset1/model/``) has **not been received**
yet. :func:`available` says whether it is on disk, and every caller falls back
to rules when it isn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def build_input(english: str, rule_gloss: str) -> str:
    # Must match run_inference.py exactly (team notebook comment) -- no wording changes.
    return "English: " + english + " Rule gloss: " + rule_gloss + " Produce ASL gloss:"


def available(model_dir: str | Path | None) -> bool:
    """A HF model folder (``config.json`` + weights) exists at ``model_dir``."""
    return model_dir is not None and (Path(model_dir) / "config.json").is_file()


@dataclass
class T5Refiner:
    model_dir: Path
    max_input_len: int = 128
    generate_kwargs: dict = field(default_factory=lambda: {"max_length": 64, "num_beams": 4})
    device: str | None = None

    def __post_init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_dir = Path(self.model_dir)
        if not available(self.model_dir):
            raise FileNotFoundError(f"no T5 checkpoint at {self.model_dir} (TODO §13: not yet received)")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(self.model_dir)
        assert tok is not None, f"no tokenizer in {self.model_dir}"
        self.tokenizer = tok
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to(self.device).eval()

    @classmethod
    def from_config(cls, cfg: dict, preset: str = "guarded", model_dir: str | Path | None = None) -> "T5Refiner":
        """``cfg`` is the ``t5`` block of ``aslg.text2gloss.json``."""
        return cls(Path(model_dir or cfg["model_dir"]), max_input_len=cfg["max_input_len"],
                   generate_kwargs=dict(cfg["generate"][preset]))

    def refine_batch(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[str]:
        """``[(english, rules_v1 gloss), ...]`` -> refined glosses (uppercased,
        stripped; empty output falls back to the rule gloss, as the team's
        ``apply_rules`` does)."""
        import torch

        out: list[str] = []
        for s in range(0, len(pairs), batch_size):
            chunk = pairs[s:s + batch_size]
            enc = self.tokenizer([build_input(e, g) for e, g in chunk], return_tensors="pt", padding=True,
                                 max_length=self.max_input_len, truncation=True).to(self.device)
            with torch.no_grad():
                ids = self.model.generate(**enc, **self.generate_kwargs)
            dec = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            out += [d.strip().upper() if d.strip() else g for d, (_, g) in zip(dec, chunk)]
        return out

    def refine(self, english: str, rule_gloss: str) -> str:
        return self.refine_batch([(english, rule_gloss)])[0]
