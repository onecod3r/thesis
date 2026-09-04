"""Off-machine copy of the registry's trained weights (TODO §9.3).

`best.pt`/`last.pt` are gitignored, so every trained weight in this project
exists on exactly one Windows machine. The 2026-07-18 registry reset already
destroyed 8 runs' checkpoints — including the ME-126 result the README still
cites — and those canonical evals can never be completed. This is the copy that
stops it happening twice, and the one a deployment would fetch from.

**Weights only.** Not the ~30 GB of feature caches (derivable, and §9.2 made
that checkable) and emphatically not POPSIGN's ~870 GB of raw video: that is an
immutable upstream Kaggle release, so a run records the *reference*, never the
bytes. Current registry: 42 `best.pt`, ~707 MB total.

Backends
--------
Pick one with ``SB_ARTIFACT_BACKEND`` in the repo ``.env``. They differ only in
where bytes land; the manifest, the hashing and the verification are identical.

``kaggle`` (default, and what this repo uses)
    A **Kaggle Model**, one variation per configuration
    (``<owner>/signbridge-gislr/pyTorch/<arch>-<subset><-coords>/<version>``).
    Costs nothing extra: this repo already authenticates to Kaggle for the GISLR
    competition data, so there is no new account, no card on file and no new
    secret. Models rather than Datasets, for two reasons that matter in daily
    use — a push uploads **only what is new** instead of re-sending everything,
    and a restore fetches **one file** instead of the whole bundle. And a Kaggle
    *inference kernel can attach a model directly*, so the backup and the
    artifact a submission run loads (TODO §6.3) are the same object.

``local``
    Any filesystem path: an external drive, a NAS share, or a folder that
    OneDrive/Drive/Dropbox syncs. Zero dependencies, works immediately.
    **A second folder on the same physical disk is not a backup** — point it at
    something that survives this machine dying.

``s3``
    Any S3-compatible endpoint via boto3 — Backblaze B2, Wasabi, MinIO, Storj,
    or Cloudflare R2 if it is ever enabled. Needs ``uv sync --group ops``.

**The manifest is the point.** `registry/checkpoints.manifest.json` is committed
and records, per object, its size, sha256, backend and upload time. It answers
"is this run backed up, and is the copy still the file I trained?" without any
credentials, and it is what `pull` verifies a restored file against. If the
remote is ever lost or re-created, the manifest still says what was supposed to
be in it.

Configuration (repo-root ``.env``, never committed)::

    SB_ARTIFACT_BACKEND=kaggle         # kaggle | local | s3

    # kaggle: all optional — owner defaults to the authenticated user
    KAGGLE_ARTIFACT_OWNER=bracu23101281
    KAGGLE_ARTIFACT_MODEL=signbridge-gislr
    KAGGLE_ARTIFACT_FRAMEWORK=pyTorch     # tfLite for an exported model
    KAGGLE_ARTIFACT_LICENSE=              # only if Kaggle rejects the upload

    # local:
    SB_ARTIFACT_DIR=D:/backup/signbridge

    # s3 / R2 / B2 / MinIO:
    S3_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com
    S3_BUCKET=signbridge-models
    S3_ACCESS_KEY_ID=...
    S3_SECRET_ACCESS_KEY=...
    S3_PREFIX=models                   # optional

The lifecycle, and the reason `best.pt` is usually **not** on this disk::

    train -> sb-evaluate -> sb-sync push --apply -> sb-sync prune --apply

`push` is a dry run unless `--apply`, and nothing is ever deleted *remotely* by
this script. `prune` is the only sanctioned way to delete a checkpoint locally:
it downloads the remote copy and unlinks the local file only when the manifest
hash, the remote hash and the local hash all agree.

    sb-sync status
    sb-sync push [--apply] [--limit N]
    sb-sync pull 1784447175 | --all
    sb-sync prune [--apply]        # verify against the remote, then delete
    sb-sync drop-resume [--apply]  # last.pt of FINISHED runs only
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from sb.core.paths import REGISTRY_DIR, MODELS_DIR, TEMP_DIR, env_value

MANIFEST = REGISTRY_DIR / "checkpoints.manifest.json"
MANIFEST_SCHEMA = 2  # v2: backend-agnostic (v1 assumed R2)
CHECKPOINTS = ("best.pt", "last.pt")
BACKENDS = ("kaggle", "local", "s3")
DEFAULT_BACKEND = "kaggle"


# ---------------------------------------------------------------- local state
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def local_checkpoints(include_last: bool = False) -> dict[str, Path]:
    """``"<run_id>/best.pt" -> path`` for every checkpoint on this machine."""
    wanted = CHECKPOINTS if include_last else CHECKPOINTS[:1]
    found = {}
    for run_dir in sorted(p for p in MODELS_DIR.iterdir()
                          if p.is_dir() and p.name.isdigit()):
        for name in wanted:
            ckpt = run_dir / name
            if ckpt.is_file():
                found[f"{run_dir.name}/{name}"] = ckpt
    return found


def load_manifest() -> dict:
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"schema_version": MANIFEST_SCHEMA, "remote": {}, "updated": None,
            "objects": {}}


def save_manifest(manifest: dict) -> Path:
    manifest["schema_version"] = MANIFEST_SCHEMA
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, MANIFEST)
    return MANIFEST


# -------------------------------------------------------------------- backends
class Backend:
    """Move one checkpoint out and back. Subclasses own only that."""

    name = "?"

    def describe(self) -> dict:
        raise NotImplementedError

    def put(self, key: str, path: Path) -> str:
        """Store one file; returns its remote location string."""
        raise NotImplementedError

    def get(self, key: str, rec: dict, dest: Path) -> None:
        """Fetch one file to ``dest`` (a temp path the caller then verifies)."""
        raise NotImplementedError

    def put_group(self, run_id: str, files: dict[str, Path]) -> dict[str, str]:
        """Store every file of one run; returns ``{key: location}``.

        The unit is a run because some backends commit per run rather than per
        file — Kaggle uploads one model variation. Grouping here, instead of
        letting the caller record a per-file success the backend has not
        committed yet, is what stops the manifest claiming an upload that never
        happened.
        """
        return {key: self.put(key, path) for key, path in files.items()}

    def finish(self, pushed: list[str]) -> None:
        """Called once after a push batch — for backends that commit in bulk."""

    def download_cache(self) -> Path | None:
        """Where this backend leaves downloaded copies, if anywhere.

        `prune` fetches every object purely to verify it. Freeing space in the
        registry while an equal number of bytes piles up in a cache would defeat
        the exercise, so a backend that caches says where.
        """
        return None


class LocalBackend(Backend):
    """A directory somewhere that is not this disk."""

    name = "local"

    def __init__(self):
        root = env_value("SB_ARTIFACT_DIR")
        if not root:
            raise SystemExit(
                "SB_ARTIFACT_BACKEND=local needs SB_ARTIFACT_DIR in .env — the "
                "directory to copy checkpoints into (an external drive, a NAS "
                "share, or a synced folder). A path on this same disk is not a "
                "backup.")
        self.root = Path(root)

    def describe(self) -> dict:
        return {"backend": self.name, "location": str(self.root)}

    def put(self, key: str, path: Path) -> str:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".pt.tmp")
        shutil.copy2(path, tmp)
        os.replace(tmp, dest)
        return str(dest)

    def get(self, key: str, rec: dict, dest: Path) -> None:
        src = self.root / key
        if not src.is_file():
            raise SystemExit(f"{key} is not in {self.root}")
        shutil.copy2(src, dest)


class KaggleBackend(Backend):
    """A **Kaggle Model**, using the credentials this repo already has.

    Naming follows Kaggle's own convention — a *model* is a family, a
    *variation* is one configuration of it, and *versions* are that
    configuration retrained::

        bracu23101281/signbridge-gislr/pyTorch/gru-me126-xy/2
        └─ owner ──┘ └─── model ────┘ └─fw─┘ └ variation ┘ └ver┘

    - **model** = the family: recognizers trained on GISLR. POPSIGN models later
      become `signbridge-popsign`, not a variation here — a different label space
      is a different model.
    - **variation** = `<architecture>-<subset><-coords>`, the thing that makes
      two runs incomparable if it differs.
    - **version** = a re-run of the *same* configuration, uploaded in
      chronological run-id order, so the version history reads as the training
      history and an improvement shows up as a later version.

    Versions are chronological, **not** ranked: the newest version is the most
    recent run, not necessarily the best one. "Which run should be deployed" is a
    separate question with a separate answer — `sb-promote` aliases.

    Using Models rather than Datasets is what makes this practical: a Dataset
    versions as one directory, so every push would re-send every checkpoint and
    every restore would pull the whole bundle. Model variations version
    independently, so a push uploads only what is new and
    ``model_download(handle, path=...)`` restores exactly one file.

    Each version carries its run's ``meta.json`` beside the weights, so an
    uploaded checkpoint is self-describing: provenance, hyperparameters and the
    canonical metrics travel with the bytes.
    """

    name = "kaggle"

    def __init__(self):
        # str(), not the bare env_value: these are path and handle components,
        # and every one of them has a non-None fallback
        self.owner = str(env_value("KAGGLE_ARTIFACT_OWNER") or self._user())
        self.model = str(env_value("KAGGLE_ARTIFACT_MODEL", "signbridge-gislr"))
        self.framework = str(env_value("KAGGLE_ARTIFACT_FRAMEWORK", "pyTorch"))
        # Kaggle applies its own default when this is omitted; set
        # KAGGLE_ARTIFACT_LICENSE if an upload is rejected for the want of one.
        self.license = env_value("KAGGLE_ARTIFACT_LICENSE")
        self.stage = TEMP_DIR / "artifact_stage"

    @staticmethod
    def _user() -> str:
        try:
            from kagglehub.auth import whoami

            return whoami()["username"]
        except Exception as exc:  # no credentials, no network, API change
            raise SystemExit(
                "could not determine the Kaggle username — set "
                "KAGGLE_ARTIFACT_OWNER in .env "
                f"({type(exc).__name__})") from None

    @staticmethod
    def variation(meta: dict) -> str:
        """`gru-me126-xy` — architecture plus the subset/coords run tag."""
        from sb.core.subsets import subset_tag

        arch = str(meta["architecture"]).replace("_", "-")
        return f"{arch}-{subset_tag(meta['subset'], meta['coords'])}"

    def handle(self, variation: str, version: int | None = None) -> str:
        base = f"{self.owner}/{self.model}/{self.framework}/{variation}"
        return f"{base}/{version}" if version else base

    def describe(self) -> dict:
        return {"backend": self.name,
                "location": f"kaggle://models/{self.owner}/{self.model}/{self.framework}"}

    @staticmethod
    def _notes(meta: dict) -> str:
        m = meta.get("metrics", {})
        acc = m.get("overall_accuracy")
        scored = (f"canonical val acc {acc:.4f}" if acc is not None
                  else f"train-loop val acc {m.get('train_val_acc')} (eval pending)")
        tr = meta.get("training", {})
        return (f"run {meta['run_id']} · {scored} · {meta['n_params']:,} params · "
                f"regime {tr.get('regime')} · {tr.get('epochs_trained')} epochs "
                f"(best {tr.get('best_epoch')})")

    def put_group(self, run_id: str, files: dict[str, Path]) -> dict[str, str]:
        """Upload one run as one new version of its variation."""
        import kagglehub

        meta_path = MODELS_DIR / run_id / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(
                f"run {run_id} has no meta.json — refusing to upload a checkpoint "
                "whose configuration is unknown, because the variation slug is "
                "derived from it")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        variation = self.variation(meta)

        run_stage = self.stage / run_id
        if run_stage.exists():
            shutil.rmtree(run_stage)
        run_stage.mkdir(parents=True)
        for key, path in files.items():
            dest = run_stage / key.split("/")[1]
            try:
                os.link(path, dest)  # NTFS hard link: no second copy of the bytes
            except OSError:
                shutil.copy2(path, dest)
        shutil.copy2(meta_path, run_stage / "meta.json")

        kwargs = {"license_name": self.license} if self.license else {}
        kagglehub.model_upload(
            handle=self.handle(variation),
            local_model_dir=str(run_stage),
            version_notes=self._notes(meta),
            **kwargs,
        )
        shutil.rmtree(run_stage, ignore_errors=True)

        # The version number is not returned by the API. It is the count of runs
        # of this variation already uploaded, plus this one — which holds because
        # uploads happen in chronological run-id order and each call adds exactly
        # one version. `prune` re-downloads the pinned handle and compares
        # sha256 before deleting anything, so a wrong guess is caught there
        # rather than becoming silent data loss.
        version = self._version_for(variation)
        return {key: f"{self.handle(variation, version)}/{key.split('/')[1]}"
                for key in files}

    def _version_for(self, variation: str) -> int:
        prefix = self.handle(variation) + "/"
        seen = {rec["remote_key"].split("/")[4]
                for rec in load_manifest().get("objects", {}).values()
                if str(rec.get("remote_key", "")).startswith(prefix)}
        return len(seen) + 1

    def download_cache(self) -> Path | None:
        cache = Path.home() / ".cache" / "kagglehub" / "models" / self.owner / self.model
        return cache if cache.exists() else None

    def get(self, key: str, rec: dict, dest: Path) -> None:
        import kagglehub

        remote = rec.get("remote_key")
        if not remote:
            raise SystemExit(f"{key}: the manifest records no remote location")
        handle, filename = remote.rsplit("/", 1)
        src = Path(kagglehub.model_download(handle, path=filename))
        shutil.copy2(src, dest)


class S3Backend(Backend):
    """Any S3-compatible endpoint: Backblaze B2, Wasabi, MinIO, Storj, R2."""

    name = "s3"

    def __init__(self):
        self.cfg = {
            "endpoint_url": env_value("S3_ENDPOINT_URL"),
            "bucket": env_value("S3_BUCKET"),
            "access_key_id": env_value("S3_ACCESS_KEY_ID"),
            "secret_access_key": env_value("S3_SECRET_ACCESS_KEY"),
        }
        missing = [k for k, v in self.cfg.items() if not v]
        if missing:
            raise SystemExit(
                "missing S3 settings in the environment or repo .env: "
                + ", ".join(f"S3_{k.upper()}" for k in missing))
        self.prefix = env_value("S3_PREFIX", "models")
        self.client = self._client()

    def _client(self):
        try:
            import boto3  # ty: ignore[unresolved-import]  # optional: uv sync --group ops
        except ImportError:
            raise SystemExit(
                "boto3 is not installed — it is an optional dependency group so "
                "the default environment stays lean.\n  uv sync --group ops"
            ) from None
        return boto3.client(
            "s3",
            endpoint_url=self.cfg["endpoint_url"],
            aws_access_key_id=self.cfg["access_key_id"],
            aws_secret_access_key=self.cfg["secret_access_key"],
            region_name="auto",
        )

    def describe(self) -> dict:
        return {"backend": self.name,
                "location": f"{self.cfg['endpoint_url']}/{self.cfg['bucket']}/{self.prefix}"}

    def put(self, key: str, path: Path) -> str:
        remote_key = f"{self.prefix}/{key}"
        self.client.upload_file(str(path), self.cfg["bucket"], remote_key)
        return remote_key

    def get(self, key: str, rec: dict, dest: Path) -> None:
        remote_key = rec.get("remote_key") or f"{self.prefix}/{key}"
        self.client.download_file(self.cfg["bucket"], remote_key, str(dest))


def get_backend(name: str | None = None) -> Backend:
    name = name or env_value("SB_ARTIFACT_BACKEND", DEFAULT_BACKEND)
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; have {', '.join(BACKENDS)}")
    return {"local": LocalBackend, "kaggle": KaggleBackend, "s3": S3Backend}[name]()


# ------------------------------------------------------------------- commands
def cmd_status(args) -> None:
    manifest = load_manifest()
    objects = manifest.get("objects", {})
    local = local_checkpoints(include_last=args.include_last)

    backed, stale, absent = [], [], []
    for key, path in local.items():
        rec = objects.get(key)
        if rec is None:
            absent.append(key)
        elif rec.get("bytes") != path.stat().st_size:
            stale.append(key)
        else:
            backed.append(key)
    orphans = [k for k in objects if k not in local]

    size = sum(p.stat().st_size for p in local.values())
    print(f"local checkpoints : {len(local)} ({size / 1e6:.0f} MB)")
    print(f"in the manifest   : {len(backed)} matching, {len(stale)} changed since upload")
    print(f"never uploaded    : {len(absent)}")
    print(f"remote-only       : {len(orphans)} (uploaded, no local copy — restorable)")
    r = manifest.get("remote") or {}
    configured = env_value("SB_ARTIFACT_BACKEND", DEFAULT_BACKEND)
    print(f"backend           : configured {configured}"
          + (f" · manifest says {r.get('backend')} at {r.get('location')}"
             if r else " · nothing pushed yet")
          + (f" · updated {manifest.get('updated')}" if manifest.get("updated") else ""))
    for label, keys in (("never uploaded", absent), ("changed", stale),
                        ("remote-only", orphans)):
        for k in keys[:10]:
            print(f"  {label:>14}: {k}")
        if len(keys) > 10:
            print(f"  {'':>14}  (+{len(keys) - 10} more)")


def cmd_push(args) -> None:
    manifest = load_manifest()
    objects = manifest.setdefault("objects", {})
    local = local_checkpoints(include_last=args.include_last)

    todo = []
    for key, path in sorted(local.items()):
        rec = objects.get(key)
        if rec and rec.get("bytes") == path.stat().st_size:
            continue  # size match — hashing 707 MB to re-confirm every run is waste
        todo.append((key, path))

    if not todo:
        print(f"nothing to upload — {len(local)} checkpoint(s) already in the manifest")
        return
    total = sum(p.stat().st_size for _, p in todo)
    for key, path in todo:
        print(f"  upload {key:<28} {path.stat().st_size / 1e6:7.1f} MB")
    # report the backend that will actually be used, not just the configured one
    chosen = args.backend or env_value("SB_ARTIFACT_BACKEND", DEFAULT_BACKEND)
    print(f"\n{len(todo)} object(s), {total / 1e6:.0f} MB -> backend {chosen}")

    if not args.apply:
        print("\ndry run — pass --apply to upload")
        return

    backend = get_backend(args.backend)
    manifest["remote"] = backend.describe()

    groups: dict[str, dict[str, Path]] = {}
    for key, path in todo:
        groups.setdefault(key.split("/")[0], {})[key] = path
    if getattr(args, "limit", None):
        groups = dict(list(groups.items())[: args.limit])
        print(f"  (--limit {args.limit}: uploading {len(groups)} run(s) this pass)")

    for run_id, files in groups.items():
        locations = backend.put_group(run_id, files)
        for key, path in files.items():
            objects[key] = {
                "run_id": int(run_id),
                "file": key.split("/")[1],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "backend": backend.name,
                "remote_key": locations[key],
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            }
        # saved per RUN, not per file: an interrupt leaves the manifest
        # describing exactly the runs that actually landed
        save_manifest(manifest)
        print(f"  pushed run {run_id} ({len(files)} file(s)) "
              f"-> {locations[next(iter(files))]}")
    backend.finish([k for k, _ in todo])
    # count what was actually pushed this pass, not every candidate — with
    # --limit those differ, and the old message overstated it
    pushed_n = sum(len(f) for f in groups.values())
    print(f"\n{pushed_n} object(s) in {len(groups)} run(s) pushed; "
          f"manifest at {MANIFEST}")


def cmd_pull(args) -> None:
    manifest = load_manifest()
    objects = manifest.get("objects", {})
    if not objects:
        raise SystemExit(f"{MANIFEST} lists no objects — nothing to restore")

    if args.all:
        keys = sorted(objects)
    else:
        wanted = set(args.run_id)
        keys = sorted(k for k in objects if k.split("/")[0] in wanted)
        unknown = wanted - {k.split("/")[0] for k in objects}
        if unknown:
            print(f"WARNING: not in the manifest: {', '.join(sorted(unknown))}")
    if not keys:
        raise SystemExit("no matching objects in the manifest")

    backend = get_backend(args.backend or (manifest.get("remote") or {}).get("backend"))
    restored = 0
    for key in keys:
        rec = objects[key]
        dest = MODELS_DIR / key
        if dest.is_file() and dest.stat().st_size == rec["bytes"] and not args.force:
            print(f"  have {key} (size matches) — skipping, --force to overwrite")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".pt.tmp")
        backend.get(key, rec, tmp)
        digest = sha256_file(tmp)
        if digest != rec["sha256"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"{key}: sha256 mismatch — manifest {rec['sha256'][:16]}…, "
                f"downloaded {digest[:16]}…. Refusing to install a checkpoint "
                "that is not the file that was trained.")
        os.replace(tmp, dest)  # atomic: a half-downloaded .pt never appears
        restored += 1
        print(f"  restored {key} ({rec['bytes'] / 1e6:.1f} MB, sha256 verified)")
    print(f"\n{restored} checkpoint(s) restored")


def cmd_prune(args) -> None:
    """Delete local checkpoints that are *verifiably* in the remote.

    The verification is the whole point, and it is deliberately paranoid: for
    every candidate this **downloads the remote copy and hashes it**, rather
    than trusting that a push returned without raising. Three things must agree
    before a byte is deleted — the manifest's recorded sha256, the sha256 of
    what the remote actually serves, and the sha256 of the local file. A
    disagreement on any of them skips that file and says why.

    That third check matters as much as the second: if the local file has
    diverged from what was uploaded, deleting it destroys something the remote
    does *not* have.
    """
    manifest = load_manifest()
    objects = manifest.get("objects", {})
    if not objects:
        raise SystemExit(f"{MANIFEST} lists no objects — nothing is backed up, "
                         "so there is nothing safe to delete")
    backend = get_backend(args.backend or (manifest.get("remote") or {}).get("backend"))

    local = local_checkpoints(include_last=True)
    candidates = {k: p for k, p in local.items() if k in objects}
    skipped = [k for k in local if k not in objects]

    print(f"{len(candidates)} local file(s) claimed by the manifest, "
          f"{len(skipped)} not backed up (never deletable)")
    if not args.apply:
        for k in list(candidates)[:5]:
            print(f"  would verify+delete {k}")
        if len(candidates) > 5:
            print(f"  … and {len(candidates) - 5} more")
        for k in skipped[:5]:
            print(f"  KEEP (not in the manifest) {k}")
        print("\ndry run — pass --apply to verify and delete")
        return

    freed, deleted, refused = 0, 0, []
    tmp_dir = TEMP_DIR / "prune_verify"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for key, path in candidates.items():
            rec = objects[key]
            tmp = tmp_dir / "candidate.pt"
            try:
                backend.get(key, rec, tmp)
            except SystemExit as exc:
                refused.append((key, f"download failed: {exc}"))
                continue
            remote_digest = sha256_file(tmp)
            local_digest = sha256_file(path)
            tmp.unlink(missing_ok=True)
            if remote_digest != rec["sha256"]:
                refused.append((key, "remote copy does not match the manifest"))
                continue
            if local_digest != rec["sha256"]:
                refused.append((key, "local file has diverged from what was uploaded"))
                continue
            size = path.stat().st_size
            path.unlink()
            freed += size
            deleted += 1
            print(f"  verified + deleted {key} ({size / 1e6:.1f} MB)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not getattr(args, "keep_cache", False):
        cache = backend.download_cache()
        if cache is not None:
            cached = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
            shutil.rmtree(cache, ignore_errors=True)
            freed += cached
            print(f"  cleared {cached / 1e6:.0f} MB of verification downloads "
                  f"from {cache}")

    print(f"\n{deleted} file(s) deleted, {freed / 1e6:.0f} MB freed")
    for key, why in refused:
        print(f"  REFUSED {key}: {why}")
    if refused:
        raise SystemExit(f"{len(refused)} file(s) kept — see above")


def cmd_drop_resume(args) -> None:
    """Delete `last.pt` for FINISHED runs — resume state the policy never reuses.

    `last.pt` is auto-resume state. The registry's rule is that a *finished* run
    (early-stopped or epoch cap reached) is never resumed — a new training always
    gets a fresh folder — so for those runs the file is dead weight. For an
    unfinished run it is the only way to continue, so it is kept.

    This deletes **without** a remote copy, which is only defensible because of
    that policy; `best.pt` is the artifact that gets backed up.
    """
    from sb.mlops import registry as R

    kept, target = [], []
    for run_dir in sorted(p for p in MODELS_DIR.iterdir()
                          if p.is_dir() and p.name.isdigit()):
        last = run_dir / "last.pt"
        if not last.is_file():
            continue
        meta = R.load_meta(run_dir) if (run_dir / "meta.json").is_file() else {}
        if meta.get("training", {}).get("finished"):
            target.append(last)
        else:
            kept.append(run_dir.name)

    size = sum(p.stat().st_size for p in target)
    print(f"{len(target)} finished run(s) hold last.pt ({size / 1e6:.0f} MB); "
          f"{len(kept)} unfinished run(s) keep theirs: {', '.join(kept) or '—'}")
    if not args.apply:
        print("\ndry run — pass --apply to delete")
        return
    for p in target:
        p.unlink()
    print(f"deleted {len(target)} last.pt, {size / 1e6:.0f} MB freed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--include-last", action="store_true",
                    help="also handle last.pt (resume state); default is best.pt only")
    sub = ap.add_subparsers(dest="command")

    def backend_flag(parser):
        parser.add_argument("--backend", choices=BACKENDS,
                            help="override SB_ARTIFACT_BACKEND for this invocation")

    status = sub.add_parser("status", help="local checkpoints vs the manifest")
    backend_flag(status)

    push = sub.add_parser("push", help="upload checkpoints missing from the manifest")
    push.add_argument("--apply", action="store_true",
                      help="actually upload (default: print the plan only)")
    push.add_argument("--limit", type=int, metavar="N",
                      help="upload at most N runs — trial one before committing "
                           "to a long batch; the manifest makes it resumable")
    backend_flag(push)

    pull = sub.add_parser("pull", help="restore checkpoints from the remote")
    backend_flag(pull)

    prune = sub.add_parser(
        "prune", help="delete local checkpoints that verify against the remote")
    prune.add_argument("--apply", action="store_true",
                       help="actually verify and delete (default: plan only)")
    prune.add_argument("--include-last", dest="include_last", action="store_true",
                       help=argparse.SUPPRESS)
    backend_flag(prune)

    drop = sub.add_parser(
        "drop-resume",
        help="delete last.pt for FINISHED runs (resume state the policy never reuses)")
    drop.add_argument("--apply", action="store_true",
                      help="actually delete (default: plan only)")
    pull.add_argument("run_id", nargs="*", help="run ids to restore")
    pull.add_argument("--all", action="store_true", help="restore every manifest object")
    pull.add_argument("--force", action="store_true",
                      help="overwrite a local checkpoint that already matches on size")

    args = ap.parse_args()
    if not getattr(args, "backend", None):
        args.backend = None
    if args.command == "push":
        cmd_push(args)
    elif args.command == "prune":
        cmd_prune(args)
    elif args.command == "drop-resume":
        cmd_drop_resume(args)
    elif args.command == "pull":
        if not args.run_id and not args.all:
            raise SystemExit("pull needs run ids or --all")
        cmd_pull(args)
    else:
        cmd_status(args)


if __name__ == "__main__":
    main()
