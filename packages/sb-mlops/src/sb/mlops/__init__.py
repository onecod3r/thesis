"""Run records, the registry, and what happens to a model after training.

Deliberately independent of `sb-recognize`: the registry has to be readable
without importing torch, so the leaderboard, the index and the artifact sync all
work in a bare environment.

- ``registry``   — run folders, meta.json schema + writing, asset registration
- ``run``        — provenance capture: commit, config hash, cache key, env
- ``index``      — every meta.json flattened into registry/index.csv
- ``query`` — DuckDB queries over the run records (leaderboard)
- ``artifacts``  — off-machine checkpoint sync (R2) + the manifest
- ``promote``    — aliases: which run is champion, and what deployment fetches
- ``docs``       — regenerate the schema, index and README generated blocks
"""
