# Prompt version 1

One file per prompt, plain text, never edited in place once a recorded run has
used it. A change means a **new version directory** (`v2/`). The point is that
a prompt change produces a measured delta against `../../evalset/`, not an
unattributable shift in output.

The prompt file's sha256 (`sb.rescore.client.load_prompt`) is recorded in every
result next to the model id, the same way `provenance.config_sha256` pins
hyperparameters.

| file | used by |
|---|---|
| `gloss2en.txt` | gloss → English, Workers AI arm (`gislr.4.downstream.gloss-to-english.ipynb`) |
