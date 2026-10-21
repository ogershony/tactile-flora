# Status report (2026-10-21)

The original implementation was checked with synthetic fixtures on CPU. Continuation work now
uses a revision-pinned three-episode PHANTOM subset, a fresh Slurm GPU smoke and the released
checkpoint's real processor/tokenizer. No released checkpoint generation or Baxter hardware
acceptance has passed.

## Continuation acceptance

- PHANTOM download provenance records revision `ff0f0cda62f6c8ecbb7c384520952035d17998f5`,
  790 catalog episodes, and a three-episode subset totaling 72,859,054 bytes. The subset has one
  training success, one training failure, and one validation success. These are subset counts,
  not full-dataset adapter coverage.
- Real stores use `*.zarr` directories. Their JPEG arrays also contain a legacy numeric bytes
  fill value that the installed Zarr reader rejects. The adapter and video reader now share a
  read-only metadata compatibility layer; raw files retain their recorded checksums.
  All 186 downloaded files matched the provenance hashes. First/middle/last video frames
  decoded for each episode. Scene-based durations exceed right-pad sensor spans by 0.25–0.36 s;
  the adapter holds the last force reading across that tail.
- The existing `data/cluster/smoke/result.json` records 20 tiny random Qwen3.5 LoRA steps on an
  RTX 2080 Ti, with loss 4.895 to 3.561, torch 2.14.1+cu126, CUDA 12.6 and transformers 5.8.1.
  A fresh Slurm run reproduced these losses in `data/acceptance/cluster-gpu-smoke`, validating
  the saved adapter (40 tensors, SHA-256 recorded). The original interactive job's 1 GB host RAM
  caused OOM; a separate five-minute allocation with 8 GB host RAM succeeded.
- The revised smoke tool also completed two CPU steps (loss 4.854 to 4.351) under
  `data/acceptance/cluster-smoke`, including saved-adapter validation.
- Source median sync failure now blocks manifest construction as well as verification. Reports
  derive this gate from persisted per-episode sync checks.
- PHANTOM saturation now uses native observations before nearest-sampling duplicates maxima.
  The failure demo has one maximum per pad among 110/109 native samples (<1%), duplicated into
  two of 195 grid samples (>1%). Regression tests cover native clipping and configured ADC rails.
- Real verification completed: all three episodes pass integrity, sensor health, prompt
  difference and license checks. Pooled normalization passes. Absolute sync lags are 50, 225 and 100 ms,
  so the 100 ms median fails the strict under-100 ms source gate. Eleven spot strips await
  review. The source is **not accepted**, and no training manifest or holdout was frozen.
  Report: `data/acceptance/phantom/verified/phantom/report.md`.
- Further acceptance investigation: all 11 spot strips were inspected by Codex and all 704
  frames across the 11 MP4s decoded with increasing timestamps. This checks readability, not
  human sign-off or label correctness (`docs/spec.md` requires a human spot review).
  See `verified/phantom/spot-decode-review.json` under the acceptance data root.
  Native pad median intervals are 160–176 ms. Correlation peaks are broad: the failure demo's
  75 ms candidate scores 99.87% of its winning 100 ms candidate. Thus the strict source gate
  still fails, but the 25 ms search grid does not establish 25 ms measurement precision.
  `data/acceptance/phantom/timing-diagnostics.json` records native timing and nearby peaks;
  no threshold or timestamp correction was changed to manufacture a pass.
- Released processor preparation passed for one real episode in dense/outcome, omitted/real
  conditions at revision `7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`. Force blocks use 2,594/2,615
  tokens; total inputs use 1,056/3,697 dense and 693/3,355 outcome tokens. All clips are
  64x96x96 RGB frames. See `data/acceptance/released-inference/result.json` and the generated
  `force.real-tokenizer.yaml`. The processor required the missing torchvision dependency;
  it also exposed an invalid `do_sample_frames` metadata field, now removed (the option stays
  in processor kwargs). A separate real-tokenizer training prefix probe also passed.
- Released generation is blocked: both MI210 nodes (`matthew`, `monster`) are down. Slurm
  preflight on the RTX 2080 Ti records about 11.2 GB free versus 55.6 GB checkpoint weights in
  `data/acceptance/released-gate-status/result.json`. No released weights were downloaded.
- Downloader and smoke launch instructions are in the README and `scripts/cluster/README.md`.
  Data and generated acceptance artifacts remain gitignored.

## Numbers
- Tests: 324 passed, no skips, including the installed torch CPU checks. Original implementation:
  288 passed, 3 skipped before the train extra was installed.
- The full continuation suite passed using `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
  and a temporary `/tmp` selector polling workaround for this managed shell's missed cross-thread
  wakeups. The workaround is not part of the repository or production I/O. Without it,
  even a standalone `asyncio.to_thread` file read can hang here.
- `tests/test_pipeline_smoke.py` runs verify, manifest, dataset build and one eval cell
  with a stub model.
- Ruff, Ruff format and Pyright are clean. Pyright needed an explicit temporary config pointing
  `venvPath` at this checkout and `venv` at `.venv`; automatic interpreter discovery was
  unreliable in the managed shell. The temporary config kept the project's standard type mode
  and checked both `tactile_flora` and `tests`.

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
   Built, tested on fixtures; PHANTOM also loads the three real subset episodes.
6. Verification and manifest: all spec checks, perceptual frame hashing and overlap detection,
   per-source report, manifest writer, one-shot holdout freeze, sync shift applied.
   Built, tested.
7. Released-model inference (`eval/inference.py`, backends stub|hf|vllm): released processor
   preparation passes via `scripts/cluster/smoke_inference.py`; generation remains blocked.
8. Pseudo-labeling (`labels/pseudo.py`, `tf pseudo`): code complete, stub only; refuses
   eval-only sources and held-out ids.
9. Training (`train/dataset.py`, `train/sft.py`): code complete; dry run and a tiny smoke run
   on fixtures; tiny random-model GPU smoke recorded, plus a fresh CPU smoke. No released-model
   tokenizer histogram or production SFT run exists.
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
- Other real datasets: adapters besides PHANTOM have only seen synthetic fixtures. The list of
  adapter facts guessed from dataset cards is in `docs/deviations.md` section 5. `discover()`
  counts have not been compared with the dataset cards for the complete releases.
- The token budget across the full training mix: one PHANTOM episode's dense/outcome inputs
  pass with the released tokenizer; the default config still uses the fallback estimate.
  The generated acceptance force config uses the revision-pinned real tokenizer.
- Anything needing hardware: the Baxter firmware, serial link, ROS topics, calibration, the
  force-video sync (under 100 ms gate), and the FSR part itself.
- Scores: no benchmark number exists. roboreward_500 (about 40.0 MAE expected), phantom_val and
  every ablation row are empty.
- `[train]` dependency versions: the released FLORA code wants vllm 0.24.0, transformers 5.8.1,
  qwen-vl-utils 0.0.14; `pyproject.toml` does not pin them.

## Open questions from the spec

| Question | Status |
|----------|--------|
| PHANTOM license and manifest fields | Resolved: HapticWAM, CC-BY-4.0; three real subset episodes loaded and video decoded; source sync gate still fails |
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
3. Resolve PHANTOM's source sync failure and review the generated spot strips. The false
   saturation failure is fixed. Then download one real episode for each remaining
   source and run adapter acceptance (`discover()` count vs the dataset card, `load()` on the
   episode clock, `tf verify`). Fix the UNVERIFIED items in `docs/deviations.md` section 5 as
   you go. Do not treat the three-episode subset as full benchmark coverage.
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
