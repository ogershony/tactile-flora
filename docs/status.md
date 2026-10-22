# Status report (2026-10-21)

Matched four-epoch cleaned-pilot training and evaluation completed (Slurm jobs 6321/6325).
The control adapter matches released no-force accuracy at 84.6% (11/13); the force adapter
matches released real-force accuracy at 61.5% (8/13). Shuffled force scores 69.2% (9/13).
There is no demonstrated training or force benefit on this post hoc development subset.
See [the fixed training experiment](experiments/hrdexdb-pilot/clean-training.md).

Expansion to all 618 Inspire DFTP clips is downloaded and mechanically audited with a pinned
source revision and frozen object assignments. Seventeen exclusions leave **350 train /
136 development / 115 new test** clips; all retained videos decode completely.
Recorded video timestamps omitted from the pilot download reveal both start offsets and
capture/encoded clock drift. A new explicit source-config option maps sensor time to video
time; legacy runs retain their previous clock behavior. The aligned released real-force
diagnostic scores 69.2% (9/13), including refitted normalization, versus 61.5% previously.
Expanded training preflights pass. Matched force/control training is running as Slurm 6329
on two MI210s, with development evaluation 6330 and new-test evaluation 6331 queued on
success dependencies. Each run plans four epochs / 88 updates. No expanded trained-model
score is available yet.
See [the expanded experiment](experiments/hrdexdb-expanded-v2.md).
The first expanded force optimizer update is finite (loss 2.594008). Report job 6332 is
queued after test evaluation to validate artifacts and write the final comparison.

Active HRDexDB working data: **25 train / 13 validation** episodes under
`data/acceptance/hrdexdb-clean-v1/`, after reviewing all 48 local pilot clips and excluding
seven unresolved outcomes plus three no-attempt/incomplete captures. No label flips or split
moves were made. The SFT launcher defaults to this cleaned root; original data and holdout
assignments remain intact. See [cleaned dataset details](experiments/hrdexdb-pilot/cleaned-dataset.md).

Resumed HRDexDB diagnosis checked all 48 pilot episodes: manifest, raw and catalog labels
agree; no exact duplicate videos or train/validation object overlap was found. Force/video
coverage differs by 0.24–3.14 seconds, so clock alignment remains unverified. A fresh released
no-force baseline on the current manifest reaches 75% accuracy / 0.859375 AUROC.
Explicit stable-grasp task wording reaches 68.75% / 0.796875, with two corrected and three
regressed classifications. Both conditions have zero parse failures; no labels were changed.
See [the quality diagnosis](experiments/hrdexdb-pilot/data-quality.md) for the review packet,
failure cases and the recommendation to establish human gold labels before VLM expansion.
An AI first-pass review is now complete for all 16 validation episodes: eight stable-lift
and five failure labels are visually supported; three remain uncertain. Evidence and event
intervals are saved separately in the pilot's `ai-review.json` / `ai-review.csv`; original
labels and human-review status are unchanged.

Later exploratory HRDexDB SFT and evaluation results are recorded in
[`experiments/hrdexdb-pilot/`](experiments/hrdexdb-pilot/README.md), including a portable
prediction/config snapshot for diagnosis. The signed-force pilot has 56.25% validation
accuracy against 62.5% for released FLORA with real force on the same current manifest.
The historical acceptance report below predates those experiments; its statements about
missing scores and production SFT do not describe this later, limited pilot.

A subsequent hackathon CPU video classifier reached **81.25% validation accuracy (13/16)**
and **0.90625 AUROC** on the frozen pilot split. This is a validation-tuned development
result, selected among 56 settings; all classifiers fit only the training episodes.
The saved predictor, usage and full candidate scores are documented in
[`experiments/hrdexdb-pilot/video-predictor.md`](experiments/hrdexdb-pilot/video-predictor.md).

The original implementation was checked with synthetic fixtures on CPU. Continuation work has
passed released-model HF inference and tiny two-GPU FSDP training on two MI210s. Bounded real
acceptance now covers PHANTOM, RH20T, SharpaDex and REASSEMBLE; data gates and human reviews still
block manifest creation and production SFT. Baxter hardware and full-model training remain
unverified.

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
- Expanded PHANTOM acceptance selected eight episodes per train-success, train-failure and
  validation-success stratum from the same pinned catalog, balanced across four task families.
  The downloader fetched 24 episodes / 1,365 files / 522,267,311 bytes. Episode checks report
  integrity 23/24, sensor health 24/24 and sync 22/24; source median absolute lag is 150 ms,
  above the strict 100 ms gate. Pooled normalization passes. Twenty random review windows are
  ready at `data/acceptance/phantom-expanded/verified/phantom/spot/review.html`; no human verdict
  has been recorded. The report remains unaccepted.
- Released processor preparation passed for one real episode in dense/outcome, omitted/real
  conditions at revision `7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`. Force blocks use 2,594/2,615
  tokens; total inputs use 1,056/3,697 dense and 693/3,355 outcome tokens. All clips are
  64x96x96 RGB frames. See `data/acceptance/released-inference/result.json` and the generated
  `force.real-tokenizer.yaml`. The processor required the missing torchvision dependency;
  it also exposed an invalid `do_sample_frames` metadata field, now removed (the option stays
  in processor kwargs). A separate real-tokenizer training prefix probe also passed.
- Initial released generation was blocked: both MI210 nodes (`matthew`, `monster`) were down. Slurm
  preflight on the RTX 2080 Ti records about 11.2 GB free versus 55.6 GB checkpoint weights in
  `data/acceptance/released-gate-status/result.json`. No released weights were downloaded.
- `matthew` returned with two MI210s; allocation 6274 reserved both with 64 GB host RAM.
  Both cards passed bf16 matrix multiplication and report gfx90a, 104 CUs, ~64 GiB VRAM.
  The requested 128 GB host allocation exceeded the account's memory limit; 64 GB succeeded.
- Released inference **passed** on both cards in `data/acceptance/released-inference/result.json`:
  dense omitted/real took 14.03/11.22 s; outcome omitted/real took 5.21/8.03 s. All four outputs
  parsed. GPU peak allocations were 35.80/40.68 GiB. Device placement uses both GPUs with no
  CPU offload. Runtime: torch 2.12.1+rocm7.1, torchvision 0.27.1+rocm7.1, HIP 7.1.52802,
  transformers 5.8.1, Triton ROCm 3.7.1. The native PyTorch DeltaNet fallback was used, so
  flash-linear-attention / causal-conv1d and their performance are still unverified.
- The real run exposed two HF inference bugs: mixed fp32 DeltaNet/bf16 projections needed
  autocast, and the checkpoint's generation EOS differs from its tokenizer's chat-end EOS.
  Both are fixed and regression-tested. The first failed raw output is preserved separately;
  the successful run generates one complete answer without truncating or relaxing parsing.
- The ROCm LoRA smoke also passed 20 steps (loss 4.895 to 3.561) with a validated 40-tensor
  adapter under `data/acceptance/mi210-training-smoke`. This is still a tiny random-model test,
  not production FSDP training. The isolated environment is `data/cluster/rocm-venv`;
  hardware, package versions, Slurm allocation and logs accompany the inference artifact.
- The production HFBackend also passed two-GPU FSDP smoke runs on a deterministic tiny random
  Qwen3.5 checkpoint: eight language steps (12.53 to 11.21 loss) and eight multimodal steps
  (12.46 to 11.25), both with exact adapter reload logits and 32 saved tensors. The multimodal
  run used synthetic video through the real processor, vision encoder and answer mask. These
  checks exercise wrapping, mixed precision, distributed batching and save/reload; they do not
  measure full 27B checkpoint host/GPU memory or train on accepted data. Artifacts are in
  `data/acceptance/fsdp-smoke/`.
- Real-data adapter checks: REASSEMBLE's first recording passes sensor health, sync,
  normalization and license; integrity flags wrist-right video at 97.80 s for a 102.03 s
  episode, and 12 spot windows await review. RH20T episode `rh20t_cfg3/0` passes integrity,
  sensor health and license, but has +675 ms sync lag, cannot fit normalization from this one
  episode, and has one spot window
  awaiting review. SharpaDex episode 0 passes integrity, sensor health, sync (+ source median),
  label sanity, prompt difference and license; one-episode normalization could not be fit and no
  human spot review is recorded. RH20T and SharpaDex were fetched from pinned revisions using
  one-row episode indexes; their chunk files include other demonstrations and do not establish
  full-release coverage. `PyAVReader` now caps frame counts at each episode boundary, including
  the first episode in a concatenated MP4 chunk.
- Downloader and smoke launch instructions are in the README and `scripts/cluster/README.md`.
  Data and generated acceptance artifacts remain gitignored.

## Numbers
- Tests: 332 passed, no skips; original implementation:
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
7. Released-model inference (`eval/inference.py`, backends stub|hf|vllm): milestone 7 passes
   for HF on two MI210s via `scripts/cluster/smoke_inference.py`; vLLM remains untested.
8. Pseudo-labeling (`labels/pseudo.py`, `tf pseudo`): code complete, stub only; refuses
   eval-only sources and held-out ids.
9. Training (`train/dataset.py`, `train/sft.py`): code complete; dry run and a tiny smoke run
   on fixtures; tiny random-model GPU smoke recorded. The two-GPU tiny FSDP language and
   multimodal save/reload paths pass. No accepted-manifest dry run, full 27B memory check or
   production SFT run exists.
10. Eval harness (`eval/`): harness, metrics (hand-computed unit tests), grid with resume,
    report with shuffle-delta column, roboreward_500 builder. Code complete, stub only.
11. Baxter stack (firmware, serial protocol parser, ROS node, calibration, recorder,
    converter, labeler): code complete, host-side parser and recorder tests with fakes only.
    Untested on hardware.

`docs/review1.md` is a code review of an earlier state; its wiring bugs (labels, shift,
drop_channel, budget fallback, caps, RH20T -1 rating, REASSEMBLE spans) are addressed and listed
in `docs/deviations.md`.

## NOT verified

- Production GPU paths: flash-linear-attention / causal-conv1d, vllm, full-model FSDP memory,
  throughput over
  the full input distribution, loss decreasing on real data, regex-constrained decoding.
  HF loading and unconstrained greedy outputs validated by parsing now pass on two MI210s.
- Other real datasets: one REASSEMBLE recording, one curated RH20T episode and one curated
  SharpaDex episode have passed through adapter loading and verification. None establishes
  full-release `discover()` counts, benchmark coverage or a training-ready source. The list of
  unresolved adapter facts is in `docs/deviations.md` section 5.
- The token budget across the full training mix: one PHANTOM episode's dense/outcome inputs
  pass with the released tokenizer; the default config still uses the fallback estimate.
  The generated acceptance force config uses the revision-pinned real tokenizer.
- Anything needing hardware: the Baxter firmware, serial link, ROS topics, calibration, the
  force-video sync (under 100 ms gate), and the FSR part itself.
- Scores: no benchmark number exists. roboreward_500 (about 40.0 MAE expected), phantom_val and
  every ablation row are empty.
- The optional vLLM stack: the released FLORA code wants vllm 0.24.0 and qwen-vl-utils 0.0.14;
  neither was needed for the HF gate. The training extra now pins transformers 5.8.1,
  peft 0.21.2 and accelerate 1.15.0. Tokenizers 0.22.2 matches Transformers' declared range;
  the former tokenizers>=0.23.3 requirement made a clean install unsatisfiable and was relaxed.

## Open questions from the spec

| Question | Status |
|----------|--------|
| PHANTOM license and manifest fields | Resolved: HapticWAM, CC-BY-4.0; three real subset episodes loaded and video decoded; source sync gate still fails |
| RH20T port fidelity | Partly resolved: no fingertip tactile in any port (wrist only), licenses per scene; one real cfg3 episode loads. Still open: whether `robot_ft` is tared; sampled episode fails sync and normalization gates |
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
3. Review the 20-window PHANTOM packet and record the pass count (the spec requires 17/20).
   Investigate the 150 ms source median against the actual acquisition clocks; keep the current
   sync gate unchanged. REASSEMBLE's 12 windows and RH20T's one window also need human review
   before either source can be accepted. SharpaDex needs a larger fixed sample to fit
   normalization and produce review windows.
4. Resolve the PHANTOM and RH20T sync failures or exclude those sources. Request AIRoA dataset
   access if it belongs in the training mix. Decide whether the model may use NC data before
   including SharpaDex, RH20T scenes 6-10 or AIRoA.
5. After at least one training source passes its source gates, build the manifest, freeze the
   holdout, run the real-tokenizer training dry run, and qualify full 27B FSDP memory on the
   available GPUs before starting force/control SFT.
6. Run the released no-force baselines on phantom_val and roboreward_500 (gate 4), then the
   pseudo-labeling, `tf manifest`, `tf manifest freeze-holdout`, and the force and control runs.
7. Confirm the sensor part (B0CZ6L5NMM): two-wire thin-film FSR, about 10 mm, at least 50 N.
   Measure no-load resistance and resistance at 10 N, pick `r_ref_ohm`, redo calibration, then
   set `sensor_confirmed: true` and measure force-video sync before recording.
8. Decide the force block token budget (3,000 now; options in `docs/deviations.md`).
9. Decide the licensing implication of NC data in training: SharpaDex (CC-BY-NC-4.0), RH20T
   scenes 6-10 (CC-BY-NC-4.0), AIRoA MoMa (CC-BY-NC-SA-4.0, share-alike on top) all go into the
   training mix, and HRDexDB (CC-BY-NC-4.0) is eval-only. If the adapter is to be released or
   used commercially, drop NC sources (config `license_allowed` and per-row `license` make this a
   filter, not a rewrite) or keep the weights internal.
10. Set the SoftVTBench `dsr_tolerance_pct` and decide the other "decision needed" items in
   `docs/deviations.md` (severity vs precedence merge, outcome parse-failure score, RH20T
   "moved toward goal").
