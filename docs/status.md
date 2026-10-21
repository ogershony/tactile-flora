# Status report (overnight run, 2026-10-21)

For the project owner. Everything here was built and checked on a CPU laptop with synthetic
fixtures. No real dataset, no GPU, no Baxter hardware and no released checkpoint was touched.

## Numbers
- Tests: 288 passed, 3 skipped (`uv run pytest -q -o addopts=""` prints the count; plain `-q` shows only dots). CPU, under a minute, no datasets or GPU.
- Code: 10938 lines of Python in `tactile_flora/` (plus a Teensy `.ino` firmware sketch),
  5361 lines in `tests/` (unit tests, synthetic fixtures for every source, and
  `tests/test_pipeline_smoke.py`, which runs verify, manifest, dataset build and one eval cell
  with a stub model).
- ruff, ruff format and pyright are clean.

## What was built, per milestone

1. Skeleton and schema: package layout, pydantic schema (`schema.py`), YAML config models
   that forbid extra keys, typer CLI with every subcommand, CI checks. Built, tested.
2. Force channel: resampler (200 ms median frames at 4 Hz, 0.5 s groups, raw slices kept),
   per-source normalization constants with a hashed `norm.json`, byte-exact text renderer
   with a token-budget check, and the ablation transforms (omitted, zero, real, shuffled,
   video_blank). Built, tested.
3. Prompts and parsers: released FLORA dense and outcome prompts reproduced verbatim plus the
   force block and sentence in one place, parsers with the paper's Worst==min rejection, window
   geometry (anchors, 1 Hz stitching). Built, tested, round-trip tested.
4. Labels: per-source native label mapping, six programmatic detectors (grasp, slip, contact
   loss, regrasp, crush, off-axis), precedence merge. Built, tested, and wired into the
   pipeline (the first review found them unwired; that is fixed).
5. Adapters: phantom, sharpadex, rh20t, reassemble, airoa (training) and hrdexdb, softvtbench,
   baxter (eval-only), each with a committed synthetic fixture in the real on-disk layout.
   Built, tested on fixtures only.
6. Verification and manifest: all spec checks, perceptual frame hashing and overlap detection,
   per-source report, manifest writer, one-shot holdout freeze, sync shift applied.
   Built, tested.
7. Released-model inference (`eval/inference.py`, backends stub|hf|vllm): code complete, only
   the stub backend has run.
8. Pseudo-labeling (`labels/pseudo.py`, `tf pseudo`): code complete, stub only; refuses
   eval-only sources and held-out ids.
9. Training (`train/dataset.py`, `train/sft.py`): code complete; dry run and a tiny smoke run
   on fixtures only. No real tokenizer histogram or GPU run exists.
10. Eval harness (`eval/`): harness, metrics (hand-computed unit tests), grid with resume,
    report with shuffle-delta column, roboreward_500 builder. Code complete, stub only.
11. Baxter stack (firmware, serial protocol parser, ROS node, calibration, recorder,
    converter, labeler): code complete, host-side parser and recorder tests with fakes only.
    Untested on hardware.

`docs/review1.md` is a code review of an earlier state; its wiring bugs (labels, shift,
drop_channel, budget fallback, caps, RH20T -1 rating, REASSEMBLE spans) are addressed and listed
in `docs/deviations.md`.

## NOT verified

- Anything needing a GPU: released-model loading, the ROCm / Gated DeltaNet / flash-linear-
  attention path, vllm, FSDP, real throughput (the 9 s/window estimate is a default), loss
  decreasing on real data, regex-constrained decoding against the real model.
- Anything needing a dataset: every adapter has only seen its synthetic fixture. The list of
  adapter facts guessed from dataset cards is in `docs/deviations.md` section 5. `discover()`
  counts have not been compared with the dataset cards.
- The token budget with the real tokenizer: `render.tokenizer_json` is null, so counts use the
  0.75 tokens/char estimate. Pad line cost (40 tokens) was measured once.
- Anything needing hardware: the Baxter firmware, serial link, ROS topics, calibration, the
  force-video sync (under 100 ms gate), and the FSR part itself.
- Scores: no benchmark number exists. roboreward_500 (about 40.0 MAE expected), phantom_val and
  every ablation row are empty.
- `[train]` dependency versions: the released FLORA code wants vllm 0.24.0, transformers 5.8.1,
  qwen-vl-utils 0.0.14; `pyproject.toml` does not pin them.

## Open questions from the spec

| Question | Status |
|----------|--------|
| PHANTOM license and manifest fields | Resolved: HapticWAM, CC-BY-4.0; fields in `sources/phantom.py` docstring. Not run on a real episode |
| RH20T port fidelity | Partly resolved: no fingertip tactile in any port (wrist only), licenses per scene. Still open: whether `robot_ft` is tared |
| REASSEMBLE anomaly format | Resolved: there is none; per-segment `success` plus known-issues table. Dense spans come from failed segments |
| SoftVTBench contact fields and DSR tolerance | Contact resolved (`obs/finger_force`). Tolerance still open: unset in config (`dsr_tolerance_pct: null`) |
| HRDexDB tactile layout and hands | Resolved for allegro_v5 and inspire_dftp (included); inspire_f1, robotiq, human raise NotImplementedError |
| Sensor identity (ASIN B0CZ6L5NMM) | Open. `sensor_confirmed: false`; `r_ref_ohm` 4700 and saturation counts are placeholders |
| Baxter gripper force command range and per-episode variation | Open; blocks the recorder's `meta.json` fields |
| AIRoA wrench key and per-primitive labels | Open (gated dataset; heuristics in the adapter) |
| Force block token budget | Open decision: 3,000 now, spec said 1,500 (see deviations) |
| NC license implications | Open decision (below) |

## What to do next (prioritized)

1. Free disk space. SharpaDex alone is about 7.7 TB in full, AIRoA about 190 GB (the 23k-episode airoa-moma release; the 5k-hour release is 4.4 TB), and the
   training mix only needs a few seasons / a subset; plan the subset first and download
   only that (the caps are in `configs/sources/*.yaml`).
2. Accept the gated AIRoA MoMa dataset terms on Hugging Face (auto-approve) so the adapter can
   be checked at all.
3. Download one real episode per source and run the adapter acceptance (`discover()` count vs
   the dataset card, `load()` on the episode clock, `tf verify`). Start with PHANTOM
   (the headline benchmark) and fix the UNVERIFIED items in `docs/deviations.md` section 5 as
   you go.
4. Run milestone 7 on the MI210s: load the released checkpoint, one PHANTOM window, dense and
   outcome prompts, check for a parsable output. If ROCm blocks it, move to the cloud GPU.
   Record the working torch / ROCm / transformers / FLA versions in the README placeholders and
   pin the `[train]` extra. Set `render.tokenizer_json` and rerun `tf train --dry-run`.
5. Run the released no-force baselines on phantom_val and roboreward_500 (gate 4), then the
   pseudo-labeling, `tf manifest`, `tf manifest freeze-holdout`, and the force and control runs.
6. Confirm the sensor part (B0CZ6L5NMM): two-wire thin-film FSR, about 10 mm, at least 50 N.
   Measure no-load resistance and resistance at 10 N, pick `r_ref_ohm`, redo calibration, then
   set `sensor_confirmed: true` and measure force-video sync before recording.
7. Decide the force block token budget (3,000 now; options in `docs/deviations.md`).
8. Decide the licensing implication of NC data in training: SharpaDex (CC-BY-NC-4.0), RH20T
   scenes 6-10 (CC-BY-NC-4.0), AIRoA MoMa (CC-BY-NC-SA-4.0, share-alike on top) all go into the
   training mix, and HRDexDB (CC-BY-NC-4.0) is eval-only. If the adapter is to be released or
   used commercially, drop NC sources (config `license_allowed` and per-row `license` make this a
   filter, not a rewrite) or keep the weights internal.
9. Set the SoftVTBench `dsr_tolerance_pct` and decide the other "decision needed" items in
   `docs/deviations.md` (severity vs precedence merge, outcome parse-failure score, RH20T
   "moved toward goal").
