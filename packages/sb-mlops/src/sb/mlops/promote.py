"""Aliases: which run is champion, and what a deployment fetches.

Deployment must never reference a run id. `registry/aliases.json` maps a stable
name — `champion.recognize.streaming` — to a run, so swapping the deployed model
is a one-line JSON edit plus a redeploy, and every consumer keeps asking for the
same thing.

Alias names are `<role>.<stage>.<qualifier>`:

    champion.recognize.streaming    the deployable model (must be streaming)
    champion.recognize.offline      the accuracy reference (may be offline)
    candidate.recognize.streaming   next up, not yet promoted

**What promotion checks.** A run cannot become `champion.recognize.streaming`
unless it is streaming-viable, canonically evaluated, and its weights are in the
off-machine store — the three ways a promotion has silently gone wrong here
before. Export to TFLite stays in `sb.recognize.export`: `sb-mlops` does not
import the recognizer, so the flow is export first, then promote.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from sb.core.paths import ALIASES, MODELS_DIR
from sb.mlops import registry as R

ROLES = ("champion", "candidate")


def load_aliases() -> dict:
    if ALIASES.is_file():
        return json.loads(ALIASES.read_text(encoding="utf-8"))
    return {"schema_version": 1, "updated": None, "aliases": {}}


def save_aliases(doc: dict) -> Path:
    doc["updated"] = datetime.now().isoformat(timespec="seconds")
    ALIASES.parent.mkdir(parents=True, exist_ok=True)
    tmp = ALIASES.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, ALIASES)
    return ALIASES


def resolve(alias: str) -> Path:
    """The run folder an alias points at."""
    entry = load_aliases()["aliases"].get(alias)
    if entry is None:
        raise KeyError(f"no such alias: {alias!r}; have {sorted(load_aliases()['aliases'])}")
    return MODELS_DIR / str(entry["run_id"])


def check(run_dir: Path, alias: str) -> list[str]:
    """Reasons this run must not take this alias; empty means it may."""
    meta = R.load_meta(run_dir)
    problems = []
    if alias.endswith(".streaming") and not meta.get("streaming"):
        problems.append(
            f"{meta.get('model_name')} is offline-only (streaming=false) and can "
            "never be the deployment candidate")
    if meta.get("metrics", {}).get("eval_status") != "canonical":
        problems.append(
            "not canonically evaluated — promote only what has been measured on "
            "the canonical split (sb-evaluate <run_dir>)")
    if R.is_legacy_gislr_split(meta):
        problems.append(
            "trained on the retired 9,448-val GISLR split (pre-2026-09-16 "
            "canonical-split reset) — not comparable to the current 18,896-val "
            "split; promote a current-split run instead")
    if not _backed_up(run_dir):
        problems.append(
            "weights are not in the off-machine store — promoting a checkpoint "
            "that exists on one machine is how the 2026-07-18 reset lost 8 runs "
            "(sb-sync push --apply)")
    return problems


def _backed_up(run_dir: Path) -> bool:
    from sb.mlops.artifacts import load_manifest

    return f"{run_dir.name}/best.pt" in load_manifest().get("objects", {})


def promote(alias: str, run_id: int, *, force: bool = False, notes: str = "") -> dict:
    run_dir = MODELS_DIR / str(run_id)
    if not (run_dir / "meta.json").is_file():
        raise SystemExit(f"not a registry run: {run_dir}")
    problems = check(run_dir, alias)
    if problems and not force:
        raise SystemExit(
            f"refusing to point {alias} at {run_id}:\n  - "
            + "\n  - ".join(problems)
            + "\nPass --force only if you mean to deploy despite this.")

    meta = R.load_meta(run_dir)
    doc = load_aliases()
    doc["aliases"][alias] = {
        "run_id": int(run_id),
        "architecture": meta.get("architecture"),
        "subset": meta.get("subset"),
        "accuracy": meta.get("metrics", {}).get("overall_accuracy"),
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "waived": problems if force else [],
    }
    save_aliases(doc)
    return doc["aliases"][alias]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("list", help="show every alias")
    st = sub.add_parser("set", help="point an alias at a run")
    st.add_argument("alias")
    st.add_argument("run_id", type=int)
    st.add_argument("--force", action="store_true",
                    help="promote despite failed checks (recorded as `waived`)")
    st.add_argument("--notes", default="")
    rs = sub.add_parser("resolve", help="print the run folder an alias points at")
    rs.add_argument("alias")
    args = ap.parse_args()

    if args.command == "set":
        entry = promote(args.alias, args.run_id, force=args.force, notes=args.notes)
        print(f"{args.alias} -> {entry['run_id']} "
              f"({entry['architecture']}/{entry['subset']}, acc {entry['accuracy']})")
        if entry["waived"]:
            print("WAIVED: " + "; ".join(entry["waived"]))
    elif args.command == "resolve":
        print(resolve(args.alias))
    else:
        doc = load_aliases()
        if not doc["aliases"]:
            print(f"no aliases yet ({ALIASES})")
        for name, e in sorted(doc["aliases"].items()):
            print(f"{name:<32} {e['run_id']}  {e['architecture']}/{e['subset']}  "
                  f"acc {e['accuracy']}")


if __name__ == "__main__":
    main()
