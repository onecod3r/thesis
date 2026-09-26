# SignBridge — English → ASL gloss (T5)

A T5-base model fine-tuned to turn English sentences into American Sign Language (ASL) gloss,
e.g. "I will visit my mother tomorrow." → `ME FUTURE VISIT MY MOTHER TOMORROW`. Part of
SignBridge, a sign ↔ speech thesis project; in the app it refines the output of a rule-based
converter, and a guard keeps the rule output whenever T5 changes the meaning.

## Variations

| variation | framework | what it is |
|---|---|---|
| `t5-hybrid` | Transformers | the fine-tuned checkpoint: `model.safetensors`, `config.json`, `generation_config.json`, tokenizer |
| `t5-web` | ONNX | the browser export used by the SignBridge web app: int8 encoder + fp32 decoder, each cut into < 25 MiB parts, with `manifest.json` (part order, sizes, sha256) and `tokenizer.json` |

## Use

```python
import kagglehub
from transformers import AutoTokenizer, T5ForConditionalGeneration
path = kagglehub.model_download("bracu23101281/signbridge-speech-to-gloss/transformers/t5-hybrid")
tok = AutoTokenizer.from_pretrained(path)
model = T5ForConditionalGeneration.from_pretrained(path)
# The input is the English sentence plus a rule-based gloss of it, in exactly this format:
english, rule_gloss = "I will visit my mother tomorrow.", "I FUTURE VISIT MY MOTHER TOMORROW"
text = "English: " + english + " Rule gloss: " + rule_gloss + " Produce ASL gloss:"
ids = tok(text, return_tensors="pt", max_length=128, truncation=True).input_ids
out = model.generate(ids, max_length=56, num_beams=2, no_repeat_ngram_size=2, repetition_penalty=1.3)
print(tok.decode(out[0], skip_special_tokens=True))  # ME FUTURE VISIT MY MOTHER TOMORROW
```

The rule gloss comes from the team's rule engine (`rules_v1`, a spaCy pipeline: part-of-speech
filtering, copula deletion, lemmatization, a `FUTURE` marker, negation, WH-movement and yes/no
questions). T5 was trained on that input, so it expects it.

For `t5-web`, concatenate each model's parts in the order `manifest.json` lists and check the
sha256; the web app does this in the browser before creating the ONNX Runtime Web sessions.

## How good it is

Measured on 30 test sentences written by the team, against draft reference glosses: BLEU-4 36.0
on its own, no better than the rule-based converter it refines (39.3). It is better at some
constructions (time fronting, IF/BEFORE clauses, phrasal verbs) and worse at others: it can
change grammatical person (you → HE), drop subjects, and repeat tokens. Used alone it is not
reliable; the app uses it behind a meaning guard, with the rules as the fallback.

## Data

Fine-tuned by the SignBridge team. The training corpus is not fully confirmed: its outputs have
none of the `DESC-`/`X-` markers of raw ASLG-PC12, so it was trained on NCSLGR or on ASLG-PC12
with those markers removed. Those corpora have their own terms of use.

## License

The weights are released under MIT. The base model (T5) is Apache-2.0, and the training data has
its own terms; check both before commercial use.
