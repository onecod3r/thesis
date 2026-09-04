"""Regenerate everything in the docs that is derived from code or the registry.

The README said its meta.json table "is the source of truth for the schema"
while ``registry.py::REQUIRED_KEYS`` was what actually enforced it — two sources
of truth, so one of them was already wrong. Observed drift on 2026-09-04:
``index.csv`` held 38 runs against 43 run folders, ``CLAUDE.md`` claimed 36, and
the daily-log table listed 07-19 before 07-18 (TODO §9.6).

The fix is to have one definition and render the rest. ``registry.FIELDS`` is
the schema; this script writes ``schemas/meta.v<N>.json`` from it and replaces
the marked blocks in ``README.md`` with tables rendered from the same data.

What is **not** generated: the weekly summaries, daily log index and report
tables. Those are narrative — generating them would produce a worse changelog,
not a drift fix.

Generated blocks are delimited in the markdown by::

    <!-- generated:<name> --> ... <!-- /generated:<name> -->

Runs from any CWD::

    .venv/Scripts/python.exe src/modules/scripts/gen_docs.py            # regenerate
    .venv/Scripts/python.exe src/modules/scripts/gen_docs.py --check    # CI-style: fail on drift
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> src/

from modules.model import registry as R
from modules.paths import MODEL_INDEX, SRC_DIR

REPO_ROOT = SRC_DIR.parent
README = REPO_ROOT / "README.md"
SCHEMA_DIR = REPO_ROOT / "schemas"

# JSON Schema types are written straight into FIELDS; "object|null" is the one
# union the schema uses (provenance, which is null for every pre-v4 run).
def _json_type(spec: str) -> list[str] | str:
    return spec.split("|") if "|" in spec else spec


def meta_json_schema() -> dict:
    """JSON Schema for one meta.json, generated from registry.FIELDS."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://github.com/sign2speech/schemas/meta.v{R.SCHEMA_VERSION}.json",
        "title": f"sign2speech model-registry run record (schema v{R.SCHEMA_VERSION})",
        "description": (
            "One training run's meta.json. GENERATED from "
            "modules/model/registry.py::FIELDS by modules/scripts/gen_docs.py — "
            "edit the code, not this file."
        ),
        "type": "object",
        "required": list(R.REQUIRED_KEYS),
        "additionalProperties": False,
        "properties": {
            key: {"type": _json_type(spec), "description": _plain(doc)}
            for key, (spec, doc) in R.FIELDS.items()
        },
    }


def _plain(markdown: str) -> str:
    """Strip the markdown a README cell wants but a JSON Schema description
    should not carry."""
    return markdown.replace("`", "").replace("\\|", "|")


def schema_table() -> str:
    rows = ["| key | type | content |", "|---|---|---|"]
    rows += [f"| `{k}` | {spec.replace('|', ' \\| ')} | {doc} |"
             for k, (spec, doc) in R.FIELDS.items()]
    return "\n".join(rows)


def registry_table(top: int = 5) -> str:
    """Run counts + the current leaderboard, straight from the run records."""
    from modules.scripts.build_model_index import load_runs, markdown_summary

    return markdown_summary(load_runs(), top=top)


def blocks() -> dict[str, str]:
    return {
        "meta-schema": schema_table(),
        "registry-summary": registry_table(),
    }


def render(text: str, name: str, body: str) -> str:
    open_tag, close_tag = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
    pattern = re.compile(
        re.escape(open_tag) + r".*?" + re.escape(close_tag), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(
            f"{README.name}: no `{open_tag} ... {close_tag}` block to fill. "
            "Generated content needs its markers in the markdown."
        )
    return pattern.sub(f"{open_tag}\n{body}\n{close_tag}", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if anything would change (no writes)")
    ap.add_argument("--top", type=int, default=5, help="leaderboard rows (default 5)")
    args = ap.parse_args()

    # the index is regenerated as part of this: it lagging the run folders is
    # the drift that keeps coming back
    from modules.scripts.build_model_index import load_runs

    df = load_runs()
    index_csv = df.to_csv(index=False, lineterminator="\n")

    schema_path = SCHEMA_DIR / f"meta.v{R.SCHEMA_VERSION}.json"
    schema_text = json.dumps(meta_json_schema(), indent=2) + "\n"
    readme_text = README.read_text(encoding="utf-8")
    for name, body in blocks().items():
        readme_text = render(readme_text, name, body)

    targets = [
        (schema_path, schema_text),
        (README, readme_text),
        (MODEL_INDEX, index_csv),
    ]
    stale = [p for p, want in targets
             if not p.is_file() or p.read_text(encoding="utf-8") != want]

    if args.check:
        for p in stale:
            print(f"STALE: {p.relative_to(REPO_ROOT)}")
        if stale:
            raise SystemExit(
                f"{len(stale)} generated file(s) out of date — run gen_docs.py")
        print(f"up to date: {', '.join(p.name for p, _ in targets)}")
        return

    SCHEMA_DIR.mkdir(exist_ok=True)
    for p, want in targets:
        p.write_text(want, encoding="utf-8")
    print(f"wrote {schema_path.relative_to(REPO_ROOT)} "
          f"({len(R.FIELDS)} fields, schema v{R.SCHEMA_VERSION})")
    print(f"wrote {MODEL_INDEX.relative_to(REPO_ROOT)} ({len(df)} runs)")
    print(f"wrote {README.name} blocks: {', '.join(blocks())}")


if __name__ == "__main__":
    main()
