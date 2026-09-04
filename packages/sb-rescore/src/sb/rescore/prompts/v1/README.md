# Prompt version 1

One file per prompt, plain text, never edited in place once used by a recorded
run — a change means a **new version directory** (`v2/`), because the whole
point is that a prompt change produces a measured delta against
`../../evalset/` rather than an unattributable shift in output.

The prompt file's sha256 is recorded in the run record alongside the model id,
the same way `provenance.config_sha256` pins hyperparameters.

Empty until TODO §8 settles what this layer actually does.
