"""Run provenance: what actually *ran*, not just what was configured.

`meta.json` recorded hyperparameters, the regime and the driver notebook, but
nothing that pins a run to a state of the world — no commit, no environment, no
dataset version, no link to the feature cache that produced the inputs. A run
recorded that way cannot be rebuilt and its number cannot be defended; this
module is what closes that gap (TODO §9.1, meta.json schema v4).

**Why `git_dirty` is not the alarm.** Training in this repo is started *by
editing and re-running a notebook*, so the working tree is dirty for
essentially every run and a blanket dirty warning would be ignored within a
day. What matters for reproducibility is whether the *code that executes* —
`src/modules/` and `src/config/` — was clean, which is `code_dirty` and the
only thing :func:`warn_if_dirty` shouts about. Notebook dirtiness is recorded
as information, not as an alarm.

**Dataset version.** GISLR is a Kaggle *competition* download, which carries no
version number, so `source.version` is null by construction. The fingerprint
that actually distinguishes one copy from another is `manifest_sha256` — the
hash of the split manifest (`train.csv`) every run's canonical split is derived
from.

Import-cheap and side-effect free by construction: no torch/mediapipe import
(versions come from package metadata, the GPU name only if torch is *already*
loaded), and no network. Safe to import from any notebook cell or CLI.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from modules.paths import SRC_DIR

REPO_ROOT = SRC_DIR.parent

# Paths whose dirtiness invalidates a run: the code and the parameters that
# actually execute. Notebooks are deliberately NOT here (see module docstring).
CODE_PATHS = ("src/modules", "src/config")

# Recorded per run. scikit-learn is in the list because `train_test_split`
# defines the canonical split — a change there moves the val set itself.
ENV_PACKAGES = ("torch", "numpy", "pandas", "pyarrow", "scikit-learn", "mediapipe")

# Feature-pipeline identities. These name the *code path* that produced the
# model inputs; a run is comparable to another on inputs only if this and the
# feature_cache_key both match (TODO §9.2).
PIPELINE_BASE = "base_v1"  # modules.model.data — NaN->0 at cache-build time
PIPELINE_FIRSTPLACE = "firstplace_v1"  # modules.model.features — NaN-preserving


def _git(*args: str) -> str | None:
    """One git command against the repo root; None when git or the repo isn't
    there (provenance must degrade to unknown, never raise into a training run)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict:
    """Commit, branch, and the two dirtiness flags (see module docstring)."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
            "code_dirty": None,
            "dirty_code_paths": [],
        }
    tree = _git("status", "--porcelain")
    # Plain path lists rather than porcelain status lines: no two-character
    # status prefix to slice off (and nothing to get wrong on a rename, or on
    # the first line once the output has been stripped).
    changed = _git("diff", "--name-only", "HEAD", "--", *CODE_PATHS)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS)
    lines = (changed or "").splitlines() + (untracked or "").splitlines()
    dirty_paths = sorted({ln.strip() for ln in lines if ln.strip()})
    return {
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool((tree or "").strip()),
        "code_dirty": bool(dirty_paths),
        "dirty_code_paths": dirty_paths,
    }


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str | None:
    """`sha256:<hex>` of a file's bytes, or None when it isn't there."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def sha256_obj(obj) -> str:
    """`sha256:<hex>` of a JSON-serializable object in canonical form.

    Used for configs so the hash tracks *the values that ran*: a config edited
    in a notebook cell before being passed in hashes differently from the file
    on disk, which is the honest answer.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _pkg(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def env_state() -> dict:
    """Interpreter, platform, package versions, GPU.

    The GPU name is read only when torch is *already* imported — provenance
    capture must not pull a heavyweight import into a process that didn't want
    one (e.g. a notebook cell that only reads meta.json).
    """
    env = {
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "gpu": None,
        "cuda": None,
    }
    env.update({name: _pkg(name) for name in ENV_PACKAGES})
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["cuda"] = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                env["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # a broken CUDA install must not fail a run record
            pass
    return env


def source_ref(
    dataset: str,
    data_dir: Path | str | None,
    *,
    kaggle_ref: str | None = None,
    manifest: str = "train.csv",
    n_videos: int | None = None,
) -> dict:
    """Upstream dataset reference — the pointer, never the bytes.

    Raw data is immutable upstream (Kaggle releases) and enormous, so what a run
    records is which release it read and a fingerprint of the manifest its split
    came from.
    """
    if kaggle_ref is None:  # caller didn't pass a DatasetSource's ref
        from modules.paths import DATASET_IDS

        kaggle_ref = DATASET_IDS.get("GISLR") if dataset == "gislr" else None
    ref = {
        "name": dataset,
        "kaggle_ref": kaggle_ref,
        "version": None,  # competition downloads carry no version (see docstring)
        "n_videos": int(n_videos) if n_videos is not None else None,
        "manifest": manifest,
        "manifest_sha256": None,
        "resolved_dir": None,
    }
    if data_dir is not None:
        data_dir = Path(data_dir)
        ref["resolved_dir"] = str(data_dir)
        ref["manifest_sha256"] = sha256_file(data_dir / manifest)
    return ref


def build(
    *,
    dataset: str,
    data_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    config_obj=None,
    feature_pipeline: str | None = None,
    feature_cache_key: str | None = None,
    kaggle_ref: str | None = None,
    manifest: str = "train.csv",
    n_videos: int | None = None,
) -> dict:
    """The `provenance` block for one run's meta.json (schema v4).

    Every field degrades to None rather than raising: a run record with unknown
    provenance is bad, but a training run that dies while recording provenance
    is worse.
    """
    rel_config = None
    if config_path is not None:
        config_path = Path(config_path)
        try:
            rel_config = config_path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel_config = str(config_path)
    return {
        **git_state(),
        "config_path": rel_config,
        "config_sha256": (
            sha256_obj(_public(config_obj)) if config_obj is not None else None
        ),
        "feature_pipeline": feature_pipeline,
        "feature_cache_key": feature_cache_key,
        "source": source_ref(dataset, data_dir, kaggle_ref=kaggle_ref,
                             manifest=manifest, n_videos=n_videos),
        "env": env_state(),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }


def _public(obj):
    """Drop private bookkeeping keys (``_config_path``) so a config's hash
    depends only on the values that steer the run."""
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if not str(k).startswith("_")}
    return obj


def warn_if_dirty(prov: dict, *, label: str = "") -> None:
    """Shout once, at run start, when the executing code is uncommitted.

    Only `code_dirty` warrants this — see the module docstring on why
    `git_dirty` is information rather than an alarm.
    """
    if not prov.get("code_dirty"):
        return
    paths = prov.get("dirty_code_paths") or []
    head = f"[provenance] {label} " if label else "[provenance] "
    print(
        f"\n{'=' * 72}\n"
        f"{head}UNCOMMITTED CODE -- this run will not be reproducible.\n"
        f"  dirty: {', '.join(paths[:8])}"
        + (f" (+{len(paths) - 8} more)" if len(paths) > 8 else "")
        + f"\n  commit on record: {prov.get('git_commit')}\n"
        f"  Commit src/modules + src/config before a run whose number you\n"
        f"  intend to cite.\n{'=' * 72}\n"
    )
