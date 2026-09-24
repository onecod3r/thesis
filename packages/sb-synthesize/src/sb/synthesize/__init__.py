"""Speech -> sign (TODO §13): speech -> English -> ASL gloss, then (deferred) gloss -> pose.

Integrates the team's merged Colab pipeline (Maimuna/Raiyan,
``chosen_merged_asl_pipeline_hybrid_1.ipynb``) as library code, with the
audit's fixes (``docs/reports/speech-to-sign-audit.md``):

- :mod:`.asr`: faster-whisper, the team's settings, normalization bug fixed;
- :mod:`.gloss`: ``rules_v1`` (frozen, T5's training input), ``rules_v2``
  (fixed), the T5 refiner, and the guarded hybrid;
- :mod:`.metrics`: BLEU-4, chrF, ROUGE-L, METEOR, gloss WER, ASR WER;
- :mod:`.lexicon`: GISLR/WLASL coverage and trimmed WLASL clip lookup (no
  scraping); :mod:`.display` renders it in notebooks;
- :mod:`.data`: ASLG-PC12 download + leak-free splits, NCSLGR loader;
- :mod:`.evalsets`: the team's 30 sentences with recorded outputs and draft
  references;
- :mod:`.pipeline`: speech -> gloss end to end.

Rendering's contract is unchanged: whatever produces poses emits the tensor
``sb.core.schema`` defines (:mod:`.pose`).
"""
