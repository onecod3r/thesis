# SignBridge — GISLR sign recognition

Isolated and continuous American Sign Language (ASL) recognition from MediaPipe Holistic
landmarks, 250 signs. Part of SignBridge, a sign ↔ speech thesis project.

## Variations

| variation | framework | what it is |
|---|---|---|
| `gru`, `lstm`, `gru-deep`, `bilstm`, `cnn1d`, `conv1d-transformer` | PyTorch | isolated-sign classifiers, one version per training run |
| `gru-phono`, `gru-phono-raw`, `bilstm-phono` | PyTorch | the same models behind a phonology front-end (handshape, palm orientation, location) |
| `gru-phono130`, `lstm-phono130`, `cnn1d-phono130`, `gru-continuous-phono130` | PyTorch | phonology-only models: their only input is 130 per-frame phonological features (`sb.recognize.phonology`), no raw coordinates |
| `gru-continuous`, `lstm-continuous` | PyTorch | continuous-signing models: per-frame gloss + "no sign" + sign-boundary outputs |
| `c1-web` | TFLite | the browser bundle of the deployed continuous model (see below) |

Every PyTorch version carries its run's `meta.json` beside `best.pt`: hyperparameters, landmark
subset, training history summary, provenance and the canonical score. The version notes say
which run it is. **Versions are chronological, not ranked**: the newest version is the latest
run, not the best one.

## Best models (canonical split, 18,896 held-out videos)

| run | handle (`pyTorch/…`) | accuracy | streams? | params |
|---|---|---|---|---|
| 1790355555 | `gru-phono-raw/2` (ME_134) | **0.7632** | yes | 929k |
| 1789559734 | `gru/20` (ME_132, xy) | 0.7517 | yes | 861k |
| 1790347646 | `bilstm/8` (ME_126, xy) | 0.7502 | no | 2.75M |

Phonology-only: `gru-phono130` 0.7487 (streams, 758k params). Averaging the probabilities of
streaming models that read different inputs is better than any single one: `gru` + `gru-phono130`
0.7912; + `gru-phono-raw` 0.8048.

Counting a synonym as correct (e.g. `wake`/`awake`, `see`/`look`), the best model reaches 0.7774.

Runs trained before 2026-09-16 used an older 90/10 split (9,448 validation videos). Their
scores are not comparable with the table above; see each version's `meta.json`.

## Loading a PyTorch checkpoint

```python
import kagglehub, torch
path = kagglehub.model_download("bracu23101281/signbridge-gislr/pyTorch/gru-phono-raw/2")
ckpt = torch.load(f"{path}/best.pt", map_location="cpu", weights_only=False)
# ckpt: arch, hyp, feature_dim, landmarks, coords, model_state
# Build the model with sb.recognize.architectures.build_model(ckpt["arch"], ckpt["feature_dim"], 250, ckpt["hyp"])
```

A `*-phono130` checkpoint (its `features` is `phono130_v1`) takes
`sb.recognize.phonology.model_features(clip)` of the full 543 × 3 clip instead of a landmark subset.

Input: GISLR landmark frames (543 × 3: face 0–467, left hand 468–488, pose 489–521, right hand
522–542), the run's `landmarks` subset selected, NaN → 0, up to 128 frames.

## `c1-web`: the browser bundle

What `apps/web` (sign → speech) serves under `/assets/`, in the same layout:
`model/model.tflite` (the per-frame step model: one 543 × 3 landmark frame + the 2 × 256
recurrent state in; a 256-d embedding + the new state out), `model/classes.f32` (the 251 × 256
class matrix: gloss probabilities are `softmax(16 · W · embedding)` over the 250 glosses + "no
sign", computed outside the graph so a new sign is one appended row), `model/manifest.json`
(landmarks used, state shape, gloss list), `pipeline.json` (decoder settings), `prior.json`
(trigram next-sign prior) and `lexicon.json` (gloss → English rules). MediaPipe's
`holistic_landmarker.task` is not included; get it from Google (URL in `pipeline.json`).

Continuous model C1 (run 1790143122): isolated accuracy 0.7188; on continuous test streams from
16 held-out signers, gloss error rate 0.29 with its decoder. It was trained on sentences
composed from isolated clips, not real continuous signing, so expect it to be wrong often on a
live camera.

## Data

GISLR — Google's Isolated Sign Language Recognition landmarks (Kaggle `asl-signs`), 94,477
clips from 21 signers, repackaged as `bracu23101281/gislr-stratified` with a fixed stratified
80/20 split.

## License

The weights are released under MIT. The training data has its own terms; check GISLR's before
commercial use.
