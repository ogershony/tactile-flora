# v4 results copied off the Colab runtime (2026-10-24)

The runtime's files are lost when Colab ends it, so the results were copied out as text.

- `summary_*.txt`: `scripts/v4/summarize_results.py data/eval/results.json <bench>` output, transcribed
  from the terminal (wrapped lines rejoined). These are the reported numbers (full-precision E[s]).
- `export_packed.txt`: `scripts/v4/export_scores.py data/eval <benches> --pack`, per-clip E[s] (to
  0.01) and hard score of every test and DEV cell. Byte-identical to the runtime file (277 lines,
  19,962 bytes, md5 cf9090766d72c270b98c9c13405fa357; per-12-line block md5s in
  `export_packed.blockmd5`). Rows follow the sorted window ids of each manifest; the `B` line's sha
  matches the local manifests (DEV 6d8995b2efbe, TEST 6ebce3c8fcc3 in hrdex_local/out_full/manifest).
- `per_clip_scores.csv`: `scripts/v4/decode_export.py` output (bench, row, truth, checkpoint, mode,
  es, score). AUROCs recomputed from it match results.json within 0.001-0.004 (E[s] rounding to 0.01
  creates ties), except v4_control on PHANTOM (0.680 vs 0.662; 40 clips). Use it for figures and
  error analysis, and results.json for reported numbers.
