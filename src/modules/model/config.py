"""Training configuration: one file is the source of truth for every architecture.

**Why this exists.** The four `gislr.1.model.*.ipynb` notebooks each carried
their own `HYP` dict, so keeping batch size / LR / epochs identical across them
was manual work — and "all else identical" is the entire premise of the
architecture comparison (TODO §4) and the subset ablations (§3.1). A run whose
hyperparameters silently differ from its comparators is not a benchmark, it is
noise. The four notebooks are now one (`gislr.1.models.training.ipynb`) and the
shared parameters live in `src/config/gislr.training.json`, versioned alongside
the code and read at run time.

**Shared values, explicit overrides.** Every architecture inherits `shared`
wholesale. An architecture may override a key only by naming it in its own
block, and an override is a deliberate, visible deviation:
:meth:`TrainingConfig.overrides_for` reports them, and the notebook prints them
next to the resolved values — so a divergence is something you can *see* rather
than something you have to diff four files to find.

Overriding a key that does not exist in `shared` is an error, not a new
parameter: that check is what stops a typo (`lr_patiance`) from silently
becoming a no-op that looks like a tuned setting.

Usage:

    from modules.model.config import load_config
    cfg = load_config()
    hyp = cfg.hyp_for("gru")            # shared + gru's overrides
    cfg.subsets_for("gru")              # per-arch subset list, or the global one
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.paths import SRC_DIR

CONFIG_DIR = SRC_DIR / "config"
DEFAULT_CONFIG = CONFIG_DIR / "gislr.training.json"
SCHEMA_VERSION = 1

# keys modules.model.train.train_run requires in the resolved HYP; the config is
# rejected up front if any is missing, rather than failing an hour into a run
REQUIRED_HYP_KEYS = (
    "batch_size",
    "lr",
    "hidden_size",
    "num_layers",
    "dropout",
    "weight_decay",
    "epochs",
    "grad_clip",
    "lr_factor",
    "lr_patience",
    "es_patience",
    "es_min_delta",
)

TOP_LEVEL_KEYS = (
    "schema_version",
    "dataset",
    "regime",
    "source",
    "coords",
    "subsets",
    "shared",
    "architectures",
)


@dataclass(frozen=True)
class TrainingConfig:
    path: Path
    raw: dict

    # ---- identity -------------------------------------------------------
    @property
    def dataset(self) -> str:
        return self.raw["dataset"]

    @property
    def regime(self) -> str:
        return self.raw["regime"]

    @property
    def source(self) -> str:
        """Recorded in every run's meta.json as `training.source`."""
        return self.raw["source"]

    @property
    def coords(self) -> str:
        return self.raw["coords"]

    @property
    def architectures(self) -> list[str]:
        return list(self.raw["architectures"])

    @property
    def shared(self) -> dict:
        return dict(self.raw["shared"])

    # ---- per-architecture resolution ------------------------------------
    def _arch_block(self, arch: str) -> dict:
        blocks = self.raw["architectures"]
        if arch not in blocks:
            raise KeyError(
                f"architecture {arch!r} is not in {self.path.name}; "
                f"have {sorted(blocks)}")
        return blocks[arch] or {}

    def overrides_for(self, arch: str) -> dict:
        """Hyperparameter deviations this architecture declares — usually empty.

        Kept separate from :meth:`hyp_for` so callers can *show* the deviation
        rather than silently absorbing it.
        """
        return dict(self._arch_block(arch).get("overrides", {}))

    def hyp_for(self, arch: str) -> dict:
        """Resolved HYP for one architecture: shared values + its overrides."""
        return {**self.shared, **self.overrides_for(arch)}

    def subsets_for(self, arch: str) -> list[str]:
        """Landmark subsets to train for this architecture (global list unless
        the architecture names its own)."""
        return list(self._arch_block(arch).get("subsets", self.raw["subsets"]))

    def coords_for(self, arch: str) -> str:
        return self._arch_block(arch).get("coords", self.coords)

    def enabled(self, arch: str) -> bool:
        """False parks an architecture without deleting its settings."""
        return bool(self._arch_block(arch).get("enabled", True))

    def notes_for(self, arch: str) -> str:
        return self._arch_block(arch).get("notes", "")

    # ---- reporting ------------------------------------------------------
    def summary(self) -> list[dict[str, Any]]:
        """One row per architecture — what the notebook displays so the whole
        grid (and any deviation in it) is visible before anything trains."""
        rows = []
        for arch in self.architectures:
            ov = self.overrides_for(arch)
            rows.append({
                "architecture": arch,
                "enabled": self.enabled(arch),
                "coords": self.coords_for(arch),
                "subsets": ", ".join(self.subsets_for(arch)),
                "overrides": ", ".join(f"{k}={v}" for k, v in ov.items()) or "—",
                **self.hyp_for(arch),
            })
        return rows


def validate(raw: dict, path: Path) -> None:
    """Fail fast and specifically — a bad config must not surface as a weird
    error mid-training."""
    missing = [k for k in TOP_LEVEL_KEYS if k not in raw]
    assert not missing, f"{path}: missing top-level key(s) {missing}"
    assert raw["schema_version"] == SCHEMA_VERSION, (
        f"{path}: schema_version {raw['schema_version']} != {SCHEMA_VERSION}")

    shared = raw["shared"]
    missing_hyp = [k for k in REQUIRED_HYP_KEYS if k not in shared]
    assert not missing_hyp, f"{path}: shared block missing {missing_hyp}"

    from modules.model.architectures import ARCHS

    for arch, block in raw["architectures"].items():
        assert arch in ARCHS, (
            f"{path}: unknown architecture {arch!r}; have {sorted(ARCHS)}")
        block = block or {}
        unknown = set(block) - {"overrides", "subsets", "coords", "enabled", "notes"}
        assert not unknown, f"{path}: architecture {arch!r} has unknown key(s) {sorted(unknown)}"
        # an override must refer to a real shared parameter, so a typo can't
        # masquerade as a tuned setting that quietly does nothing
        bad = set(block.get("overrides", {})) - set(shared)
        assert not bad, (
            f"{path}: architecture {arch!r} overrides key(s) not in `shared`: "
            f"{sorted(bad)} — add them to `shared` first, or fix the spelling")

    assert raw["coords"] in ("xy", "xyz"), f"{path}: coords must be 'xy' or 'xyz'"
    assert raw["subsets"], f"{path}: `subsets` is empty"


def load_config(path: Path | str = DEFAULT_CONFIG) -> TrainingConfig:
    """Read + validate the training config. Cheap and side-effect free, so every
    cell can call it instead of depending on another cell's live variables."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"training config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    validate(raw, path)
    return TrainingConfig(path=path, raw=raw)
