"""Speech -> sign (future direction): text -> gloss -> pose sequence.

Skeleton only — nothing is implemented. It exists to fix one thing early: the
output of synthesis is the SAME tensor `sb.core.schema` defines for the input of
recognition. Getting that wrong later would mean two incompatible landmark
layouts in one repo, which is exactly what `sb-core` exists to prevent.
"""
