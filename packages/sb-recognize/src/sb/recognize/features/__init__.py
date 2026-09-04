"""Named, versioned feature pipelines.

A pipeline is identified by name AND version, and the bytes it caches are
content-addressed over both (``cache.feature_cache_key``). That is what makes
"were these two runs trained on the same inputs?" a field comparison rather than
a promise — see TODO §9.2.

- ``cache``          — cache addressing shared by every pipeline
- ``base_v1``        — row-select, NaN -> 0 at build time, uniform subsample
- ``firstplace_v1``  — NaN-preserving, reference-point normalization,
  lag-1/lag-2 differences, six augmentations (the Kaggle 1st-place input side)
"""
