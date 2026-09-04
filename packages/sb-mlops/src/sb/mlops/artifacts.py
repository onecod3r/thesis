"""Off-machine copy of the registry's trained weights (TODO §9.3).

`best.pt`/`last.pt` are gitignored, so every trained weight in this project
exists on exactly one Windows machine. The 2026-07-18 registry reset already
destroyed 8 runs' checkpoints — including the ME-126 result the README still
cites — and those canonical evals can never be completed. This is the copy that
stops it happening twice, and the one a deployment would fetch from.

**Weights only.** Not the ~30 GB of feature caches (derivable, and §9.2 made
that checkable) and emphatically not POPSIGN's ~870 GB of raw video: that is an
immutable upstream Kaggle release, so a run records the *reference*, never the
bytes. Current registry: 42 `best.pt`, ~707 MB total — inside R2's free tier.

**The manifest is the point.** `data/models/checkpoints.manifest.json` is
committed and records, per object, its size, sha256 and upload time. It answers
"is this run backed up, and is the copy still the file I trained?" without any
credentials, and it is what `pull` verifies a restored file against. If the
bucket is ever re-created, the manifest still says what was supposed to be in it.

**Credentials** (repo-root `.env`, never committed) — R2 speaks the S3 API::

    R2_ACCOUNT_ID=...
    R2_BUCKET=signbridge-models
    R2_ACCESS_KEY_ID=...
    R2_SECRET_ACCESS_KEY=...
    R2_PREFIX=models            # optional, defaults to "models"

Uploading needs boto3, which is an optional dependency group so the default
environment stays lean::

    uv sync --group ops

Runs from any CWD. `push` is a dry run unless `--apply` is given; nothing is
ever deleted remotely by this script.

    .venv/Scripts/python.exe src/modules/scripts/sync_models.py status
    .venv/Scripts/python.exe src/modules/scripts/sync_models.py push
    .venv/Scripts/python.exe src/modules/scripts/sync_models.py push --apply
    .venv/Scripts/python.exe src/modules/scripts/sync_models.py pull 1784447175
    .venv/Scripts/python.exe src/modules/scripts/sync_models.py pull --all
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from sb.core.paths import MODELS_DIR, env_value

MANIFEST = MODELS_DIR / "checkpoints.manifest.json"
MANIFEST_SCHEMA = 1
DEFAULT_PREFIX = "models"
CHECKPOINTS = ("best.pt", "last.pt")


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
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    os.replace(tmp, MANIFEST)
    return MANIFEST


# --------------------------------------------------------------------- remote
def remote_config() -> dict:
    """R2 settings from the environment / repo `.env`; raises with the exact
    missing key rather than failing mid-upload."""
    cfg = {
        "account_id": env_value("R2_ACCOUNT_ID"),
        "bucket": env_value("R2_BUCKET"),
        "access_key_id": env_value("R2_ACCESS_KEY_ID"),
        "secret_access_key": env_value("R2_SECRET_ACCESS_KEY"),
        "prefix": env_value("R2_PREFIX", DEFAULT_PREFIX),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise SystemExit(
            "missing R2 settings in the environment or repo .env: "
            + ", ".join(f"R2_{k.upper()}" for k in missing)
            + "\nSee this script's module docstring for the full list."
        )
    return cfg


def client(cfg: dict):
    try:
        import boto3  # ty: ignore[unresolved-import]  # optional: uv sync --group ops
    except ImportError:
        raise SystemExit(
            "boto3 is not installed — it is an optional dependency group so the "
            "default environment stays lean.\n  uv sync --group ops"
        ) from None
    return boto3.client(
        "s3",
        endpoint_url=f"https://{cfg['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        region_name="auto",
    )


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
    if manifest.get("remote"):
        r = manifest["remote"]
        print(f"remote            : {r.get('provider')} {r.get('bucket')}/{r.get('prefix')}"
              f" · manifest updated {manifest.get('updated')}")
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
            continue  # size match — hashing 675 MB to re-confirm every run is waste
        todo.append((key, path))

    if not todo:
        print(f"nothing to upload — {len(local)} checkpoint(s) already in the manifest")
        return
    total = sum(p.stat().st_size for _, p in todo)
    for key, path in todo:
        print(f"  upload {key:<28} {path.stat().st_size / 1e6:7.1f} MB")
    print(f"\n{len(todo)} object(s), {total / 1e6:.0f} MB")

    if not args.apply:
        print("\ndry run — pass --apply to upload")
        return

    cfg = remote_config()
    s3 = client(cfg)
    manifest["remote"] = {"provider": "r2", "bucket": cfg["bucket"],
                          "prefix": cfg["prefix"]}
    for key, path in todo:
        remote_key = f"{cfg['prefix']}/{key}"
        digest = sha256_file(path)
        s3.upload_file(str(path), cfg["bucket"], remote_key)
        objects[key] = {
            "run_id": int(key.split("/")[0]),
            "file": key.split("/")[1],
            "bytes": path.stat().st_size,
            "sha256": digest,
            "remote_key": remote_key,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_manifest(manifest)  # after each object: an interrupt loses nothing
        print(f"  uploaded {key} -> {cfg['bucket']}/{remote_key}")
    print(f"\n{len(todo)} object(s) uploaded; manifest at {MANIFEST}")


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

    cfg = remote_config()
    s3 = client(cfg)
    restored = 0
    for key in keys:
        rec = objects[key]
        dest = MODELS_DIR / key
        if dest.is_file() and dest.stat().st_size == rec["bytes"] and not args.force:
            print(f"  have {key} (size matches) — skipping, --force to overwrite")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".pt.tmp")
        s3.download_file(cfg["bucket"], rec["remote_key"], str(tmp))
        digest = sha256_file(tmp)
        if digest != rec["sha256"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"{key}: sha256 mismatch — manifest {rec['sha256'][:16]}…, "
                f"downloaded {digest[:16]}…. Refusing to install a checkpoint "
                "that is not the file that was trained."
            )
        os.replace(tmp, dest)  # atomic: a half-downloaded .pt never appears
        restored += 1
        print(f"  restored {key} ({rec['bytes'] / 1e6:.1f} MB, sha256 verified)")
    print(f"\n{restored} checkpoint(s) restored")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--include-last", action="store_true",
                    help="also handle last.pt (resume state); default is best.pt only")
    sub = ap.add_subparsers(dest="command")

    sub.add_parser("status", help="local checkpoints vs the manifest")

    push = sub.add_parser("push", help="upload checkpoints missing from the manifest")
    push.add_argument("--apply", action="store_true",
                      help="actually upload (default: print the plan only)")

    pull = sub.add_parser("pull", help="restore checkpoints from the remote")
    pull.add_argument("run_id", nargs="*", help="run ids to restore")
    pull.add_argument("--all", action="store_true", help="restore every manifest object")
    pull.add_argument("--force", action="store_true",
                      help="overwrite a local checkpoint that already matches on size")

    args = ap.parse_args()
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
