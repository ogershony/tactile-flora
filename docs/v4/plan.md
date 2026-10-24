# Tactile FLORA v4 plan (2026-10-23)

Response to "Tactile FLORA Training Plan" (HRDexDB grasp outcome, 2026-10-23). Verdict: the plan's
diagnosis is right and its measurement discipline is right; its visual fix does not go far enough,
its force decision throws away the one mechanism that has worked, and its compute plan does not
fit two working GPUs. This document is the plan we are executing.

## 1. What the original plan gets right (kept)

- Fix the measurement first: object-disjoint splits, a larger untouched test set, thresholds set
  off-test, AUROC beside accuracy, paired tests against released FLORA. Kept, made stricter (s.4).
- The visual input is the bottleneck. Confirmed, and worse than stated (s.2).
- Score-only targets: the "Rationale: no rationale" placeholder trains nothing and the evaluation
  reads a different prompt. Kept: train and evaluate on the released score-only prompt.
- Leave out the human trials (all successes, a "human hand means success" shortcut). Kept; we also
  leave out allegro_v5, whose 453 trials are all successes in the catalog (the plan says 90%).
- Use a cheap sensor model. Kept, but used differently (s.3).
- Skip GRPO, learned force encoders, unfrozen vision tower. Agreed.

## 2. What it gets wrong or misses, and the fix

1. **Resolution: 96 px is not "FLORA's ~105 px".** The released `batch_utils/core.py` plans every
   view as a ~100,000-pixel cell (`TARGET_PIXELS_PER_VIEW = 100_000`), 384x288 for HRDexDB's
   2048x1536 cameras, up to 4 views in 14,336 tokens. Our 96x96 padded cell gives a 4:3 camera
   96x72 pixels: 1/19 of the released input, off the model's training distribution. The plan's
   192-256 px target would still be 2-4x short. Fix: reproduce the released planner exactly
   (`tactile_flora/video/collage.py`, parity-tested against the shipped code), no crop or padding.
2. **The released baseline was handicapped too.** Every repo-harness FLORA number (HRDexDB test
   0.593 AUROC, REASSEMBLE 0.477) used 96 px cells. On REASSEMBLE, NVIDIA's own pipeline gave
   0.630. A paper must compare against FLORA at its best: same native collage, same cameras, the
   released score-only prompt, thresholds set the same way. Our harness's omitted mode now equals
   `batch_utils` and is checked clip by clip (gate in Phase 2).
3. **Removing force from the prompt is the wrong call.** On REASSEMBLE (399 held-out clips) the
   judge that reads a force-only classifier's probability inside its force block reaches AUROC
   0.828 / balanced accuracy 0.744 against 0.727 / 0.684 for the best no-force judge, and drops to
   0.488 without force and 0.619 with shuffled force. That is early fusion that needs no
   out-of-fold VLM scores, which a late stacker would (five cross-fit 27 B runs). We keep the
   classifier line and drop only the per-frame lines, which are what clock offsets corrupt. The
   compact block (code summary + classifier probability) is ~70 tokens, insensitive to offsets of
   seconds, and the classifier uses trial-level statistics computed on the sensor clock.
4. **More data exists.** inspire_f1 (592 trials, 165 failures) is usable since origin/main
   70f47ac. Both Inspire hands give 1,205 labelled trials (395 failures). REASSEMBLE (1,146
   segments) and PHANTOM (47) train the same adapter, which serves the "more general datasets"
   goal and multiplies the failures the VLM sees.
5. **Test size.** The object-hash split below gives 430 test trials (151 failures, 33 objects) and
   a 149-trial dev set, frozen before any v4 model ran. At AUROC ~0.8 the standard error is ~0.02.
6. **Compute.** The plan's 110 MI210-hours does not exist this week: both MI210s run a teammate's
   v5 chain, and Colab allows one A100 session. The plan below fits one A100 (~32 h) with the
   MI210s as a bonus.
7. **Operating point.** "Failure recall" alone is gamed by predicting failure everywhere. We
   report failure recall at the model's own hard score (no tuning), failure recall at 80% and 90%
   success recall (points on the ROC curve), and balanced accuracy at a dev-set threshold.

## 3. Phases, gates and what can go wrong

| Phase | Work | Gate | Anticipated failure -> mitigation |
|---|---|---|---|
| 0 | Frozen object split (`configs/v4/hrdexdb_split.json`, sha 8a66e91f); exclusion rule = adapter cannot establish the sensor/video clock (label-blind, logged) | done | catalog/grasp_result label mismatch -> trial excluded and logged |
| 1 | Data: sensor files + 6 candidate cameras rendered at 384x288 (`hrdexdb_fetch_render.py`, Colab and SDSC khai), per-hand normalization on TRAIN, force classifier with OBJECT-grouped folds (`prepare_hrdexdb_v4.py`) | classifier OOF AUROC on TRAIN, AUROC on DEV reported; TEST untouched | Hub anonymous limit (3,000 resolves / 5 min / IP) -> waits out the window, resumable; two hands' force units differ -> one normalization group per hand |
| 2 | Zero-shot camera study, released FLORA via `batch_utils`, 300 TRAIN trials: legacy 96 px vs each native single view, then the top-2/3/4 collages (`choose_views.py`, rule fixed in advance) | native > legacy by >= 0.03 AUROC; chosen collage; harness-vs-batch_utils E[s] parity on 20 clips: mean abs delta <= 0.02 and max <= 0.10 (re-specified 2026-10-23 from the measured engine noise floor, see the status log; was max <= 0.02) | 4-view sequences (~14k tokens) may not fit training memory -> smoke test; fall back to the best <= 2-view collage |
| 3 | Smoke: one optimizer step on the 8 longest clips, full recipe | peak memory < 78 GB, s/sample measured | OOM -> fewer views, never fewer frames (the released sampling is kept) |
| 4 | Train two matched adapters on HRDexDB train + REASSEMBLE + PHANTOM: force (compact block) and control (no force); LoRA r16, lr 1e-4, 1 epoch, score-only, failure weight = success/failure ratio | dev AUROC of each, no other tuning on dev | Colab front-end disconnects -> jobs run under setsid/nohup, status files; runtime limit -> one fresh runtime covers the deadline |
| 5 | Final evaluation once: test sets of HRDexDB (430), REASSEMBLE (399), PHANTOM (40) for released (omitted, real), control (omitted), force (real, omitted, shuffled) | pre-registered comparison (s.4) | none expected; the test sets are scored once |
| 6 (time permitting) | Second seed of the force arm; RoboRewardBench regression check (camera-only, released vs ours) | | |

## 4. Pre-registered evaluation

- Primary: AUROC (failure = positive, risk = -E[s]) of the force adapter vs released FLORA (native
  collage, same cameras, released score-only prompt), per dataset, paired cluster bootstrap
  (objects for HRDexDB, recordings for REASSEMBLE, episodes for PHANTOM, 10,000 resamples) and
  DeLong; Holm over the three datasets.
- Secondary: balanced accuracy and failure recall at the hard score (score <= 2 is failure, no
  tuning), failure recall at 80% / 90% success recall, AUPRC (failure), McNemar at the hard score;
  per hand on HRDexDB; force vs control (what force adds) and control vs released (what
  fine-tuning adds); shuffled and omitted ablations of the force adapter.
- Thresholds, if any, are set on DEV. TEST is scored once per final model.
- Label audit: two people label 100+ TEST trials blind (team task; never with model outputs).
- Clip coverage: every inspire_f1 video ends 9-37 frames (0.3-1.2 s) before its sensor timeline
  (origin/main 70f47ac). Excluding them would remove a hand, so the gap is recorded per trial
  (``clock_alignment.frame_ids_beyond_video``) and the primary comparison is repeated on trials
  with a gap under 0.5 s as a sensitivity analysis, decided before any test result.
- Baselines reported beside FLORA: hand prior (training success rate of the hand), the force-only
  classifier alone, and the per-hand AUROC of every model.
- Late-fusion baselines (added 2026-10-23 17:23 UTC, before any TEST cell or force-adapter DEV
  cell was read): the mean normalized risk rank of a force-free video model (released FLORA, or
  the control adapter) and the force-only classifier on the same clips. No fitted parameter, so
  nothing is tuned; threshold-free metrics, plus the DEV-quantile cut on HRDexDB. The force adapter
  is compared with both: joint training on force text should beat combining the two signals after
  the fact.
- REASSEMBLE and PHANTOM test sets guided the earlier (96 px) development, so they are secondary;
  the locked HRDexDB v4 test is the primary evidence.

## 5. Anticipated failures and the planned response

| Stage | What can go wrong | Response already in place |
|---|---|---|
| Data | Hub 429s (anonymous 3,000 resolves / 5 min / IP) | waits out the window; resumable; three IPs (Colab, SDSC, laptop) |
| Data | A trial's clock cannot be established | label-blind exclusion, logged in audit.json (2 of 618 DFTP so far) |
| Video | Training resizes clips differently from evaluation | the released budget (29,360,128 px per clip) is passed to every processor call; video_preprocessor_config.json's 25,165,824 would shrink 4-view clips by ~6% |
| Video | Our harness differs from NVIDIA's pipeline | parity gate on 20 clips before any training |
| Train | 14k-token 4-view sequences exceed 80 GB | smoke run first; fall back to the best 2-view collage, never to fewer frames |
| Train | Colab front-end disconnects; runtime 24 h limit | setsid/nohup jobs, status files, resumable stages; one fresh runtime covers the deadline |
| Eval | vLLM engine left holding the GPU after a kill | pkill EngineCore after every grid |
| Stats | Small effect on a small set | 430-trial test; cluster bootstrap by object; Holm across datasets |

## 6. Status log

- 2026-10-22 23:39 PT: split frozen (430 / 149 / 626 trials).
- 2026-10-23 00:10 PT: merged origin/main (HRDexDB inspire_f1, tared force, classifier) with the
  REASSEMBLE work on branch plan-v4; collage planner parity-tested; all unit tests pass.
- 00:05 PT: Phase 1 partial (DFTP, 351 train trials): classifier OOF AUROC 0.723, DEV 0.69;
  six end-of-trial features add +0.018 (within CV noise) and are not adopted.
- 00:07 PT: 4-view clips (28.3 M px) exceed video_preprocessor_config.json's 25.2 M budget; the
  released 29.4 M budget is now pinned in training and HF inference too.
- 00:25 PT: Phase 1 complete (both hands, local CPU). 1,195 trials kept (train 618, dev 149, test
  428); 10 excluded by label-blind rules (8 unrecoverable video timelines, 1 missing tactile file,
  1 missing reference camera). Force-only classifier (31 features, fit_cv defaults, folds by
  object): TRAIN out-of-fold AUROC 0.732 (accuracy 72.7%); DEV AUROC 0.791 (DFTP 0.719, F1 0.829).
  Below the plan's 80% stacking gate, so the VLM stays the main path with the classifier line as
  an input. inspire_f1 shear (tangential force, 8 trial-level features) raises F1 TRAIN OOF AUROC
  0.721 -> 0.767 (+0.046): under the +0.05 bar set before the probe, not adopted now (v5 item:
  shear must travel with the rendered features so the shuffled control stays valid).
  Coverage covariate: 190 of 579 F1 trials end > 0.5 s of sensor timeline after the last frame.
- Training cost estimate (A100, 1 epoch): HRDexDB 618 clips at ~5 / 10 / 20 s for 1 / 2 / 4 views
  plus REASSEMBLE 1,146 two-view segments at ~7 s: about 3.1 / 3.9 / 5.7 h per arm.

- 01:26 PT (08:26 UTC): fresh A100 runtime; `scripts/v4/colab/orchestrate.sh` restarted with the
  render split into passes (A: 300 camera-study TRAIN trials x 6 cameras; B: remaining 326 TRAIN
  trials x 6; C: 579 DEV+TEST trials x the chosen collage), so the GPU study starts after pass A
  instead of after the whole render. Pass A renders ~4.4 trials/min (12 vCPU, decode-bound).
- 01:35 PT: GPU otherwise idle during pass A, so `side_released.sh` scores released FLORA on
  REASSEMBLE test (omitted, real) and PHANTOM test (omitted) through the stage-4 harness and config
  (cells are reused by stage 4 through their config hash; PHANTOM real must wait for the HRDexDB
  scorer file, which `render.classifier_line` opens for non-REASSEMBLE sources). A watchdog stops
  it at pass A 275/300. RoboRewardBench 500 is fetched at the same time.
- 01:40 PT: analysis brought in line with s.4 before any v4 result exists: Holm family = the three
  TEST sets only (DEV and coverage subsets were in it), point estimates for paired hard-score
  deltas, HRDexDB TEST at the DEV-chosen E[s] cut, and the force-only classifier as eval cells
  (`scripts/v4/classifier_cells.py`, the scorer the prompt quotes, unrounded; reproduces the
  REASSEMBLE classifier-alone AUROC 0.815 exactly). `tf eval grid` writes to data/eval whatever
  the eval config's output_dir says, so the analysis reads data/eval.
- Unattended chain on the runtime: `chain3.sh` starts stage 3 when the orchestrator ends with
  parity_ok and smoke_exit_0 (else holds); `chain4.sh` then checks DEV for collapse (E[s] spread,
  parse failures; no metric is looked at), writes the classifier cells, scores the locked test
  sets (stage 4) and runs the analysis into /content/results_v4.md.
- 01:45 PT: the harness's vLLM path sent one clip per `LLM.generate` call, so frame decoding and
  the processor alternated with the GPU (released on REASSEMBLE test: ~5.5 s/clip, GPU ~30%
  busy; stage 3 + 4 hold ~6,800 clip scorings). `run_eval` now builds every prompt first
  (shuffled pairings unchanged), and `VLLMBackend.generate_many` sends chunks of 8 while 6 threads
  decode the current and next chunk (`TF_VLLM_BATCH=1` restores one call per clip). Unit test:
  answers stay in request order across chunks. GPU check `scripts/v4/batch_parity.py` (PHANTOM
  test, omitted, both paths on one engine) gates its use; the side job was stopped for it.
- 01:50 PT: pass A had stalled on Hub 429s: setup's PHANTOM fetch (10,513 files, 3 threads)
  used most of the IP's 3,000 resolves / 5 min; it finished at 01:53 PT with no failures and the
  render went from ~4.4 to ~6 trials/min. The render now resolves each camera URL once and
  decodes from the CDN address (frames byte-identical on a test trial: 64/64 JPEGs, same
  video.json); passes B and C use it.
- 02:04 PT: batched vs per-clip on one engine (PHANTOM test, omitted, 40 clips): 1.36 vs 4.68 s
  per clip (3.4x), 0/40 hard scores changed, |delta E[s]| mean 0.0067, max 0.056, signs mixed
  (largest on clips whose digit distribution is spread, E[s] 2.6-4.7). That is bf16 batch
  nondeterminism, and it exceeds the parity gate's max <= 0.02, which would then fail on noise
  against batch_utils (itself batched by 16). Gate re-specified before it ran: mean <= 0.02 and
  max <= 0.10 (a preprocessing or prompt mismatch shifts most clips one way, which the mean
  bound catches). Stage 3/4 evaluation uses the batched path (TF_VLLM_BATCH=8).
- 02:18 PT: pass A done (299/300). Its last trial, inspire_f1/blue_vase/2, has no timestamp files
  upstream (404; excluded by the label-blind rule in the local prep too), and the sensor fetch
  retried it with exponential backoff (~19 min); the worker was killed so the study could start.
  The render now raises on a 404 at once (sensor files and cameras). The study job builder skips
  trials with an absent candidate camera (a `{"missing": true}` entry would have crashed the
  batch_utils clip builder); none of the 299 has one.
- 02:21 PT: `chain5.sh` queued after chain4: phase 6 (second seed) for both arms, force first,
  as copies of the final seed-0 configs with only `seed` changed (seed drives the LoRA init via
  torch.manual_seed and the batch order), DEV, the locked test sets and the analysis again. Seed-1
  comparisons are reported as supplementary rows, outside the pre-registered Holm family. Stage
  scripts now pick force modes by pattern (`v4_force*`) and stage 4 takes checkpoints as arguments
  (defaults unchanged).
- 02:33 PT: parity camera. Of 444 rendered trials, 8 have one camera a frame longer than another
  (e.g. 913 vs 914 frames), and the scene camera (first of `hrdexdb_cameras`) sets an episode's
  duration and video clock. The parity gate therefore runs on the chosen collage's first camera
  (phase 1 has batch_utils scores for every single camera), so the manifest windows reproduce;
  with 22641005 hard-coded it could fail on a non-parity reason (~30% chance over 20 clips).
- 02:35 PT: RoboRewardBench clip check (CPU, before any GPU time): the first clip failed with
  `IndexError: frame 134 not found`. That h264 file's header counts 135 frames, 134 decode (96x96
  at 10 fps). PyAVReader now gives wanted frames at most 2 past the last decoded frame that frame,
  and still raises beyond; reads that used to succeed are unchanged. A full 500-clip decode sweep
  (`/content/v4/rr_sweep.py`) checks the rest before stage 4.
- 02:42 PT: RoboRewardBench sweep with the reader fix: 500/500 clips decode through the eval path
  (collages 256x448 x186, 320x320 x151, 288x384 x124, 288x352 x20, 256x416 x19), 309 s on CPU.
- 02:55 PT: training built each optimizer step's 16 micro-batches (clip decode plus the processor,
  run twice per sample) before any forward pass, so CPU and GPU alternated. `_prefetched` now
  builds batches in a thread up to `accum + 2` ahead (TF_TRAIN_PREFETCH; 0 restores the old
  loop): same batches, same order, so training is numerically unchanged; ~6 GB host RAM at most.
- 03:04 PT: Phase 2a done (released FLORA zero-shot via batch_utils, 299 TRAIN study trials, 50
  objects, failure positive). Legacy 96 px scene camera: AUROC 0.642, hard-score BA 0.532,
  failure recall 0.065 (700 tokens). Native single views (3,868 tokens): 22641005 0.723 (vs legacy
  +0.083, object-bootstrap 95% CI [-0.010, +0.179]; BA 0.608, FR 0.215), 22684737 0.691,
  22645029 0.684, 23263780 0.684, 25452061 0.682, 22684755 0.676. Gate (native >= legacy + 0.03)
  met on the point estimate; the interval includes 0 at this size. Ranked order for the collage
  candidates: 22641005, 22684737, 22645029, 23263780. Pass B: 325/326 (banana/2: no tactile file).
- 03:12 PT: phase 2b crashed after 32/897 clips: batch_utils' clip builder applied the first
  camera's frame indices to every view, and inspire_dftp/bamboo_box/5's 22684737 stream is one
  frame shorter (762 vs 763), so frame 00762.jpg was never rendered. The study loader now reads
  the nearest rendered index within one frame, the rule FrameCacheReader already applies in
  training and evaluation (where the gap is never more than one frame). The orchestrator was
  restarted (phase 1 kept, phase 2 resumes from its output) and the chains re-armed; the
  STUDY2_FAILED status is archived in v4.status.run1.
- 03:25 PT: the first Colab prep runs before DEV/TEST are rendered, so DEV is empty there and the
  classifier's DEV report would have called roc_auc_score on empty arrays (PREP1_FAILED). The
  report now records n = 0 when DEV does not load; the second prep (after pass C) writes the DEV
  report and the DEV/TEST manifests. Checked locally under the same conditions: train 618, dev 0,
  test 0, manifests written, 13 s.
- 04:09 PT: Phase 2b done (same 299 trials, released FLORA zero-shot): native collages of the
  ranked cameras: 1 view 0.723 (BA 0.608), 2 views 22641005+22684737 0.745 (BA 0.686), 3 views
  +22645029 0.756 (BA 0.673), 4 views +23263780 0.739 (BA 0.667). The pre-registered rule (a larger
  collage must beat every smaller one by >= 0.01 AUROC) chose the 3-view collage (0.756 vs 0.745:
  +0.011, a near tie with 2 views, kept as fixed in advance): 1152x288 per frame, ~10.4k visual
  tokens per 64-frame clip. Legacy 96 px -> chosen native collage: 0.642 -> 0.756 zero-shot.
- 04:24 PT: Phase 3 smoke on the 3-view collage (the 8 longest HRDexDB clips, ~14.66k tokens each,
  full recipe): CUDA OOM, peak 81,150 MiB of 81,920 (failed allocating 130 MiB). Weights are ~54 GB
  in bf16 plus ~8 GB because the Gated-DeltaNet projections are held in fp32 for the torch
  fallback path (no causal-conv1d build on this image); gradient checkpoints for 14.7k tokens take
  the rest. The pre-registered response applies: "OOM -> fewer views, never fewer frames; fall
  back to the best <= 2-view collage". By the same rule restricted to <= 2 views, 22641005 +
  22684737 (0.745 zero-shot vs 0.723 single view) is the collage for training, the released
  baseline and every evaluation. `fallback_2view.sh` re-preps TRAIN for that camera set, reruns
  the smoke with memory sampled and re-arms chain3/4/5 on success. Parity (run on 22641005, the
  first camera of both collages) carries over; pass C had rendered all three cameras for DEV/TEST.
- 04:34 PT: 2-view smoke passed: peak 76,854 MiB of 81,920 (sampled every 2 s), step-1 loss
  0.764, cameras [22641005, 22684737]. TRAIN 618 trials unchanged by the camera change (no TRAIN
  trial lacked 22645029 or 22684737). Chains re-armed; v4_force training started 11:33:47 UTC
  (114 optimizer steps of 16). Headroom ~5 GB on the longest clips; log.jsonl reports peak torch
  memory every 10 steps.
- 05:13 PT: v4_force at step 11/114: losses 0.54, 0.67, 0.39, -, 0.23, 0.08, 0.27, 0.08, 0.09,
  0.15, 0.16; step-10 record: 849 tokens/s (~141 s per optimizer step of 16), peak allocated
  73.7 GiB (nvidia-smi 77,858 MiB incl. allocator cache). Projected: force done ~15:50 UTC, control
  ~20:30 UTC, locked tests ~01:00 UTC 2026-10-24, seed 1 ~14:00 UTC 2026-10-24.
- 05:51 PT (12:51 UTC): v4_force crashed at step 29: `FileNotFoundError: .../inspire_dftp/blue_alarm/0/
  22684737: frame 563 was not rendered`. Each camera is rendered at its own outcome indices, but
  a collage reads every view at the first camera's indices; that trial's 22684737 stream is 2
  frames longer than 22641005 (937 vs 935), so mid-clip the nearest rendered frame was 2 away
  and FrameCacheReader allowed 1. Exact check over all 1,203 rendered trials (cameras 22641005,
  22684737, the eval/training mapping round(t * fps)): worst gap 0 frames in 1,192 trials, 1 in
  10, 2 in 1 (this one). FrameCacheReader now allows 3 frames (100 ms); training and evaluation
  read through the same reader, so they stay identical; no trial is dropped. The control arm (same
  seed, same batch order, would have crashed at the same sample) was stopped while loading. The
  failed run is kept as data/runs/v4_force.failed1; v4_force restarted 13:02:50 UTC, with a CPU
  sweep of all 1,811 training clips through the same clip path running beside it.
- 06:27 PT: CPU sweep of the training clips through the shared clip path: 1,811/1,811 read, 0
  failures (1,441 s). The restarted v4_force reproduces the first run's losses step for step
  (e.g. step 6: 0.08313 vs 0.08305), so the restart changed nothing but the reader tolerance.
  REASSEMBLE and PHANTOM test manifests are being swept the same way.
- 10:09 PT (17:09 UTC): v4_force training done, 114/114 optimizer steps in 4 h 06 min (~2.16 min
  per step, ~919 tokens/s, peak allocated 73.75 GiB). Loss of the last five steps 0.054, 0.026,
  0.054, 0.039, 0.019 (each sample is seen once, so these are losses on clips the adapter had not
  yet trained on). Merge 3 min. prep2 (DEV/TEST manifests on the 2-view collage) done 17:12:35 UTC:
  train 618, dev 149, test 428 trials. DEV scoring (real, omitted, shuffled) started 17:18 UTC.
- 10:17 PT: schedule against Colab's runtime limit. This runtime started 08:26 UTC and paid
  runtimes are reported to end at 24 h (08:26 UTC 2026-10-24; not confirmed for Pro, so treated as
  a risk). chain4 should end ~02:00 UTC. The old chain5 trained both seed-1 arms before testing
  either, so a reclaim at 24 h would have left no seed-1 test result. chain5 now finishes each arm
  (train, DEV, locked tests, analysis) before starting the next, force first: the seed-1 force
  result lands ~07:45 UTC, before the limit; seed-1 control follows if the runtime survives.
  chain4 scores the force adapter first, so a failure in its test cells shows ~2 h sooner.
  analyze.py gains a supplementary cross-seed row (force seed 1 vs control seed 0). Compute:
  272.21 units left at 6.77 units/h (~40 h). Everything on the runtime is lost when it ends, so
  results are copied out as text as each stage completes.
- 10:34 PT (17:34 UTC): force adapter on DEV (149 trials, 48 failures; DEV sets thresholds only),
  ~2.0 s per clip batched. AUROC (DFTP / F1), BA, failure / success recall at the hard score:
  real force 0.973 (0.919 / 0.997), BA 0.922, FR 0.854 / SR 0.990; shuffled force 0.968
  (0.921 / 0.994), BA 0.901, FR 0.812; force omitted 0.960 (0.900 / 0.996), BA 0.933, FR 0.875.
  FR at 80% / 90% success recall 0.938 in all three. Answers are 1 or 5 only (E[s] sd ~1.5), no
  parse failures. References on DEV: force-only classifier 0.791; released FLORA zero-shot on the
  2-view collage 0.745 on the TRAIN study (its DEV cells come at the end of stage 3).
  Reading: after fine-tuning, the video carries nearly all of the HRDexDB signal; the correct
  force block adds ~+0.005 AUROC over a shuffled one and nothing at the hard score. Checked for
  leakage before believing it: objects are disjoint across splits (95 / 22 / 59 objects, 0
  shared); the task text names only the object; trial duration predicts failure weakly (AUROC
  0.44-0.65 on TRAIN/DEV by hand, every trial gets 64 frames). The likely reason is that a lift
  is plainly visible at native resolution once the model learns what HRDexDB calls success.
  What force adds will rest on the control arm, REASSEMBLE and the DFTP hand.
- 10:35 PT: v4_control training started 17:33:56 UTC (ETA ~21:40 UTC). Added before reading any
  TEST cell: (1) late-fusion baselines (s.4), coded at 17:23 UTC before the first force-adapter DEV
  cell was read; (2) exploratory, after the DEV cells: the seed-0 force adapter with mid-gray
  frames and the real force block (`video_blank`) on HRDexDB and REASSEMBLE test
  (`stage_blank.sh`, run by chain5 after the seed-1 force result), to ask how far touch alone
  goes against the classifier line it carries. The blank collage needed stored frame sizes, which
  REASSEMBLE camera refs lack; it now takes the size from the first decoded frame of each view
  (non-blank clips unchanged; unit test added; full suite passes). GPU-free checks: the stub
  backend built every test prompt in real and shuffled mode (HRDexDB 428, REASSEMBLE 399,
  PHANTOM 40) and the blank HRDexDB test clips (428/428).
- 10:48 PT: v4_control step 1 at ~17:44 UTC (loss 0.615; force arm's first steps 0.54-0.67), GPU
  76.5 GB at 100%; weight loading took ~10 min of the start (CPU-side materialization, as before).
  classifier_cells.py on the new DEV manifest reproduces the prep report exactly (AUROC 0.791,
  DFTP 0.719, F1 0.829; at its own P >= 0.5: BA 0.652, FR 0.354, SR 0.950), so chain4's baseline
  step works on v4 manifests. DEV errors of the force adapter (real force): 7 missed failures +
  1 false alarm (inspire_f1/brown_ramen/4). The misses sit at intermediate E[s] (3.1-4.6, not 5),
  cluster on three objects (coffee_tin 4 and 6, wood_organizer 2/3/5 across both hands,
  plastic_elephant_jug 3, circular_frying_pan 1), and the force classifier also called most of
  them successes (P 0.48-0.88), so the force line could not have rescued them. Candidates for the
  blind label audit.
- 11:38 PT (18:38 UTC): CONTAINER LOST. While re-attaching from a new agent session, a second Colab
  tab was opened on the notebook (the first tab, still connected, belonged to the old session),
  force-reloaded, the keepalive cell interrupted and Reconnect clicked; Colab handed back a fresh
  container on the same host. Lost: /content (code, both venvs, the base weights, logs, status, chain
  scripts) and v4_control at step ~27/114 (killed run kept as data/runs/v4_control.killed1 with its
  loss.csv). Survived on /mnt/local-scratch: frames, raw data, manifests, scorers, every eval cell
  and the trained force adapter with its merged model. Rebuild (`data/colab/setup_resume.sh`, started
  18:44 UTC): the same pinned bootstrap and weights download, repo at 533e4e6 installed first and the
  plan-v4 code tar (147 files) extracted over it as before, the two edits stage 2 had made (2-view
  collage, local base model), status markers carried over, then chains 3-5; no Hub fetchers. The
  force arm's DEV cells are reused (cells go stale only on manifest or config hash). Check that the
  rebuilt environment matches: the restarted control run must reproduce the killed run's losses
  step for step. Rule from now on: one Colab tab, no reload, interrupt or Reconnect while jobs run.
- 12:08 PT (19:08 UTC): rebuild verified. Weights re-downloaded in 2 min 48 s; vLLM venv done; chains
  re-armed 18:48:08 UTC; the force arm's DEV cells were reused (no re-scoring); v4_control restarted
  18:48:12 UTC. Its config, git hash (533e4e6), manifest hash and dry run are byte-identical to the
  killed run's, and its config differs from the force arm's only in force_mode. Step 1 loss equals
  the killed run's exactly (0.615455); step 2 is 0.537555 vs 0.534578 (GPU nondeterminism after the
  first update, as in the force-arm restart). ETA ~22:52 UTC. chain5 reordered for the possible VM
  limit at 08:26 UTC 2026-10-24 (the rebuild kept the host): exploratory blank-video cells first
  (~30 min after chain4), then seed-1 force with DEV in real mode only and its real-force cells on
  the three Holm test sets before the rest of its stage 4 (`stage4_first.sh`), then seed-1 control.
  Projected: chain4 done ~02:30 UTC, seed-1 force primary ~07:50 UTC.
- 12:42 PT (19:42 UTC): the rebuilt container has 4 vCPUs (12 before), so training runs ~2.27 min
  per step (770 tokens/s at step 10 vs 919): the batch prefetch is CPU-bound; GPU clocks, power and
  temperature are normal. v4_control step 18 at 19:40 UTC, ETA ~23:20 UTC. chain5 reordered again
  so the seed-1 force result on HRDexDB test can still land before 08:26 UTC: seed-1 force (DEV real
  only) -> its real-force HRDexDB test cell -> REASSEMBLE and PHANTOM real -> blank-video cells ->
  rest of its stage 4 -> seed-1 control. Projected: chain4 done ~03:00 UTC, seed-1 HRDexDB ~08:00.
- 13:12 PT (20:12 UTC): the slowdown was warm-up, not the 4 vCPUs: v4_control step 30 logs 926
  tokens/s and steps 18-34 took 1.94 min each (cold page cache after the rebuild, the first ~15
  steps). ETA ~22:50 UTC; with chain5's new order, seed-1 force HRDexDB ~07:05 UTC and the
  blank-video cells ~07:50 UTC, both before 08:26.
- 15:51 PT (22:51 UTC): v4_control training done (114 steps; last three losses 0.113, 0.039, 0.144);
  merged in 155 s (54.74 GB); its DEV cell (omitted) started, then released FLORA's DEV cells.
- 16:09 PT (23:09 UTC): control adapter on DEV (force omitted, as trained): AUROC 0.983 (DFTP 0.959
  / F1 0.996), BA 0.943, FR 0.896 / SR 0.990, FR@SR80 0.979, FR@SR90 0.938, MAE 0.16; answers 1 or 5
  only, no parse failures. Force adapter with real force: 0.973 (0.919 / 0.997), BA 0.922, FR 0.854.
  On HRDexDB DEV the matched control is level with or slightly ahead of the force arm (+0.010 AUROC,
  +0.042 FR = 2 of 48 failures; within DEV noise), so the in-context force block shows no benefit
  here; the gain over released FLORA is the fine-tuning on native video. Nothing changes in the
  pre-registered analysis; the locked test (and REASSEMBLE / PHANTOM, where vision is less decisive)
  decides the force question.
- 19:38 PT (02:38 UTC 2026-10-24): stage 3 done 23:22 UTC (chain3_exit_0); DEV gate passed 23:24
  (E[s] sd: force real 1.507, control 1.590, released 0.526; no parse failures). Released FLORA on
  DEV (same 2-view collage and harness): omitted AUROC 0.747 (DFTP 0.586 / F1 0.896), BA 0.677,
  FR 0.354 / SR 1.000, FR@SR90 0.604; with force text, no training: 0.791 (0.633 / 0.949), BA 0.708,
  FR 0.417. Locked test cells: force adapter done 01:03 UTC, control 01:42 UTC, released running.
- 20:05 PT (03:05 UTC 2026-10-24): LOCKED TEST RESULTS (chain4_done_0 02:54:59 UTC; analysis
  02:49-02:55; transcribed to docs/v4/results/*.txt; page https://claude.ai/artifact/RvbrQrfDvfnUnPvXyuxjxb).
  HRDexDB test (428 / 151 failures): released 0.783 [0.739, 0.827] (hard FR 0.344, FR@SR90 0.636);
  force adapter real 0.956 [0.932, 0.975] (FR 0.841 / SR 0.982, FR@SR90 0.881); control 0.964
  [0.943, 0.980] (FR 0.848). Primary +0.173 [+0.131, +0.215], Holm p 1.5e-11; force vs control
  -0.008 [-0.023, +0.004]; real vs shuffled -0.001; vs classifier alone +0.206; vs late fusion
  control+classifier +0.040. REASSEMBLE (399 / 133): released 0.648, force 0.919, control 0.940;
  primary +0.271 [+0.212, +0.329], Holm p 1.5e-20; force vs control -0.021 [-0.042, -0.002]; real
  vs shuffled +0.028 [+0.000, +0.059]. PHANTOM (40 / 20): primary +0.073 [-0.038, +0.188], Holm p
  0.20. RoboRewardBench 500: MAE 0.40 (released; = paper's 40.0) -> 0.96-0.98 (adapters answer only
  1 or 5), AUROC 0.934 -> 0.916-0.920. Coverage subset unchanged (primary +0.174). Reading:
  native-resolution fine-tuning is the result; in-context force is a controlled null (the force
  adapter reads the block on REASSEMBLE but the no-force control learned the video better).
- 20:09 PT (03:09 UTC): per-clip scores of every test and DEV cell copied off the runtime
  (docs/v4/results/export_packed.txt, byte-identical by md5; decoded to per_clip_scores.csv; AUROCs
  recomputed within 0.004 of results.json). Seed-1 force training since 02:55 UTC.
- 21:30 PT (04:30 UTC 2026-10-24): RUNTIME ENDED. The terminal answered at 04:00:04 UTC (seed-1
  force at step 26/114, GPU 100%); by 04:30 the tab showed Reconnect and Runtime > Manage sessions
  listed no active sessions. Cause unknown (not the 24 h mark; ~190 compute units were left). Lost
  with the VM unless a new runtime lands on the same host: the trained seed-0 force and control
  adapters and merged models, the seed-1 run, frames and raw data on the scratch disk. Kept
  locally: every seed-0 result (docs/v4/results: summaries, byte-identical per-clip export, CSV),
  the code (plan-v4 worktree), the plan and results pages. Not done: seed-1 replicate and the
  blank-video ablation. No new runtime started pending a decision.
- 21:54 PT (04:54 UTC): rebuild for the seed-1 replicate of both arms (user's choice). Fresh A100
  runtime on a new host (12 vCPUs, empty scratch). `data/colab/setup_rebuild.sh`: same pinned
  environment, repo 533e4e6 + plan-v4 code (152 files), shipped artifacts; HRDexDB frames for the
  2-view collage only (TRAIN first, DEV/TEST at low priority during training), the REASSEMBLE port,
  the 87 PHANTOM episodes the manifests use (5,070 of 10,513 files); prep1 and manifests, held unless
  the training manifest equals the first runtime's (bytes 64f9a590..., or content: 1,811 rows, sorted
  window-id sha 6c1573a2...); then `chain_s1.sh`: seed-1 force (DEV real) -> its real-force test
  cells and released FLORA's omitted test cells (rescored; seed-0 cells exist only as the export)
  -> analysis -> its omitted/shuffled cells -> seed-1 control -> tests -> analysis. No camera study,
  parity or smoke (unchanged code, data, collage). Projected: force_s1 ~09:50 UTC, its tests ~11:30,
  control_s1 tests ~16:30 UTC.
- 22:40 PT (05:40 UTC): rebuild check. TRAIN render done 05:35 (626 trials x 2 cameras, ~38 min);
  prep1 and the training manifest are BYTE-IDENTICAL to the first runtime's (manifest hash
  64f9a590...). Both seed-1 trainings then failed within a minute: KeyError on a PHANTOM episode not
  fetched. The PHANTOM force inputs take per-object statistics over the whole success set
  (verify/windowing.build_source_inputs), not just the 87 manifest episodes, so the subset fetch was
  wrong. Fix: fetch the whole plan (`fetch_phantom_needed.py --all`, the remaining 5,443 files); the
  chain re-arms itself when the fetch ends with 0 failures; failed run dirs kept as *.failed_phantom.
  DEV/TEST render at low priority (~6.5 trials/min, done ~07:00). Side note: a mis-aimed click at
  05:36 typed a status command into notebook cell 2 (not executed); the cell was restored exactly
  (SHA-256 of its source equals the original notebook's).
- 23:17 PT (06:17 UTC): full PHANTOM plan fetched (0 failures, 05:56); seed-1 force training
  restarted 05:56:33 UTC. Its step-1 loss is 0.517989, identical to the seed-1 run on the first
  runtime, so the rebuilt environment and data reproduce it exactly. Projected: force_s1 trained
  ~10:05, seed-1 HRDexDB comparison ~11:00, all force_s1 cells ~12:30, control_s1 tested ~17:30 UTC.
