# Frozen evaluation set

The fixed set of cases every prompt version is scored against. Frozen means
append-only: removing or editing a case silently changes what "better" means and
makes two prompt versions incomparable — the same reason the recognizer's
canonical split is seeded and its size asserted.

Empty until TODO §8 settles what this layer does; when it is filled, the cases
should come from the semantic near-synonym pairs §7.1 identified, since those
are exactly what a rescoring layer is supposed to fix.
