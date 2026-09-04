"""Top-k gloss hypotheses -> sentence (TODO §8, scope question still open).

Skeleton only: no client is wired up yet, and the §8 decision — whether this
repo extends to continuous/sentence-level signing at all, or whether this is
n-best re-ranking within a single prediction — has not been made.

What is fixed regardless of that answer, and why the package exists now:
prompts live as versioned files under ``prompts/<version>/`` and are hashed into
the run record, and ``evalset/`` holds a frozen evaluation set. A prompt change
must produce a measured delta, not a vibe.
"""
