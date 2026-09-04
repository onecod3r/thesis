"""Per-dataset extraction adapters.

Each dataset differs in exactly three ways — where its videos live, how a video
maps to a (label, id) pair, and where its landmarks are written — and nothing
else. Those three live here, one module per dataset, so `holistic.py` stays a
single code path instead of growing a branch per corpus.

An adapter is a :class:`VideoSource`; ``SOURCES`` maps its name to it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class VideoSource:
    """One corpus of raw video, as the extractor sees it."""

    name: str
    #: build/read the manifest of every video to extract, as a DataFrame with
    #: at least the columns the extractor needs (path, label, id)
    manifest: Callable[..., pd.DataFrame]
    #: where extracted landmarks land for a split
    landmarks_dir: Callable[[str], Path]
    #: artifact path for one manifest row
    artifact_path: Callable[[Path, pd.Series], Path]


def get_source(name: str) -> VideoSource:
    from sb.extract.sources import popsign

    sources = {popsign.POPSIGN.name: popsign.POPSIGN}
    if name not in sources:
        raise KeyError(
            f"unknown video source {name!r}; registered: {sorted(sources)}. "
            "Adding one means a VideoSource entry, not a branch in holistic.py."
        )
    return sources[name]
