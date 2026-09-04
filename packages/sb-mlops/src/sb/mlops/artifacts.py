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

``kaggle`` (default, recommended here)
    A **private Kaggle Dataset**. Costs nothing extra: this repo already
    authenticates to Kaggle for the GISLR competition data, so there is no new
    account, no card on file and no new secret. Kaggle versions the dataset for
    you, and — the part that matters beyond backup — a Kaggle *inference kernel*
    can attach the dataset directly, so the same artifact that is the backup is
    also what a submission run loads (TODO §6.3). Push uploads the whole set
    (~707 MB, minutes); that is the price of Kaggle's whole-dataset versioning.

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

    # kaggle: the dataset slug. Defaults to <your-username>/signbridge-checkpoints
    KAGGLE_ARTIFACT_DATASET=bracu23101281/signbridge-checkpoints

    # local:
    SB_ARTIFACT_DIR=D:/backup/signbridge

    # s3 / R2 / B2 / MinIO:
    S3_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com
    S3_BUCKET=signbridge-models
    S3_ACCESS_KEY_ID=...
    S3_SECRET_ACCESS_KEY=...
    S3_PREFIX=models                   # optional

Runs from any CWD. `push` is a dry run unless `--apply`; nothing is ever deleted
remotely by this script.

    sb-sync status
    sb-sync push
    sb-sync push --apply
    sb-sync pull 1784447175
    sb-sync pull --all
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
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

    def finish(self, pushed: list[str]) -> None:
        """Called once after a push batch — for backends that commit in bulk."""


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
    """A private Kaggle Dataset, using the credentials this repo already has.

    Kaggle versions a dataset as a whole directory, so `push` stages *every*
    checkpoint and uploads one new version. Staging uses hard links where the
    filesystem allows it (NTFS does), so the 707 MB is not duplicated on disk.
    """

    name = "kaggle"

    def __init__(self):
        self.slug = env_value("KAGGLE_ARTIFACT_DATASET") or f"{self._user()}/signbridge-checkpoints"
        self.stage = TEMP_DIR / "artifact_stage"

    @staticmethod
    def _user() -> str:
        try:
            from kagglehub.auth import whoami

            return whoami()["username"]
        except Exception as exc:  # no credentials, no network, API change
            raise SystemExit(
                "could not determine the Kaggle username — set "
                "KAGGLE_ARTIFACT_DATASET=<user>/<slug> in .env "
                f"({type(exc).__name__})") from None

    def describe(self) -> dict:
        return {"backend": self.name, "location": f"kaggle://{self.slug}"}

    def put(self, key: str, path: Path) -> str:
        dest = self.stage / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        try:
            os.link(path, dest)  # NTFS hard link: no second copy of the bytes
        except OSError:
            shutil.copy2(path, dest)
        return f"kaggle://{self.slug}/{key}"

    def stage_all(self, objects: dict) -> None:
        """Kaggle replaces the dataset with what you upload, so everything the
        manifest claims must be present — not just what changed this run."""
        for key in objects:
            local = MODELS_DIR / key
            if local.is_file():
                self.put(key, local)

    def finish(self, pushed: list[str]) -> None:
        meta = {
            "title": "signbridge model checkpoints",
            "id": self.slug,
            "licenses": [{"name": "unknown"}],
        }
        (self.stage / "dataset-metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        exists = self._exists()
        note = f"{len(pushed)} checkpoint(s) added {datetime.now():%Y-%m-%d %H:%M}"
        cmd = (["datasets", "version", "-p", str(self.stage), "-m", note, "-r", "zip"]
               if exists else
               ["datasets", "create", "-p", str(self.stage), "-r", "zip"])
        print(f"  kaggle {' '.join(cmd)}")
        self._kaggle(*cmd)
        if not exists:
            print(f"\nCreated {self.slug} as PRIVATE. Keep it private: these are "
                  "trained weights, and the dataset page is otherwise world-readable.")

    def _exists(self) -> bool:
        out = self._kaggle("datasets", "list", "-m", "-s", self.slug.split("/")[-1],
                           check=False)
        return self.slug in (out or "")

    @staticmethod
    def _kaggle(*args: str, check: bool = True) -> str | None:
        exe = Path(sys.executable).parent / "kaggle.exe"
        cmd = [str(exe) if exe.exists() else "kaggle", *args]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SystemExit(f"kaggle CLI failed to start: {exc}") from None
        if check and r.returncode != 0:
            raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stderr.strip()}")
        return r.stdout

    def get(self, key: str, rec: dict, dest: Path) -> None:
        import kagglehub

        root = Path(kagglehub.dataset_download(self.slug))
        src = root / key
        if not src.is_file():
            raise SystemExit(f"{key} is not in the downloaded dataset at {root}")
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
    if isinstance(backend, KaggleBackend):
        # Kaggle versions the whole directory: stage what is already claimed too
        backend.stage_all(objects)
    for key, path in todo:
        digest = sha256_file(path)
        location = backend.put(key, path)
        objects[key] = {
            "run_id": int(key.split("/")[0]),
            "file": key.split("/")[1],
            "bytes": path.stat().st_size,
            "sha256": digest,
            "backend": backend.name,
            "remote_key": location,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_manifest(manifest)  # after each object: an interrupt loses nothing
        print(f"  staged {key} -> {location}")
    backend.finish([k for k, _ in todo])
    print(f"\n{len(todo)} object(s) pushed; manifest at {MANIFEST}")


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
    backend_flag(push)

    pull = sub.add_parser("pull", help="restore checkpoints from the remote")
    backend_flag(pull)
    pull.add_argument("run_id", nargs="*", help="run ids to restore")
    pull.add_argument("--all", action="store_true", help="restore every manifest object")
    pull.add_argument("--force", action="store_true",
                      help="overwrite a local checkpoint that already matches on size")

    args = ap.parse_args()
    if not getattr(args, "backend", None):
        args.backend = None
    if args.command == "push":
        cmd_push(args)
    elif args.command == "pull":
        if not args.run_id and not args.all:
            raise SystemExit("pull needs run ids or --all")
        cmd_pull(args)
    else:
        cmd_status(args)


if __name__ == "__main__":
    main()
