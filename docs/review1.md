# Review 1: tactile-flora vs docs/spec.md

Test run: `uv run pytest -q` gives about 241 passed, 3 skipped (final run). The tree changed between my runs. The first run showed 235 passing, one run had 10 collection errors, and later runs showed `tests/unit/test_video.py::test_auto_reader_dispatches_zarr_and_mp4` failing with `PIL.UnidentifiedImageError`. The suite was not stable during the review, so I did not trust a single count. I ran the suite more than once because of this. No files were edited.

Line numbers are from the files as read. Verified empirically (python snippets) where marked [verified].

## A. Wiring gaps (highest impact)

1. BUG (blocking) - labels/mapping.py and labels/events.py are never called by the pipeline.
   - Spec: the Labels section merges source-native labels, programmatic events and pseudo labels into the targets.
   - Code: every training adapter returns `labels=[]` from `load()` (phantom.py:191, sharpadex.py:185, rh20t.py:192, reassemble.py:229, airoa.py:185). `verify/windowing.py:episode_windows` runs `merge_labels(episode.labels)` on that empty list. `W.make_windows` then emits no dense window and no outcome window.
   - A grep for `map_native_labels` and `detect_all` outside `labels/` finds no callers. The only references are comments in the adapters.
   - Pseudo labels: `cli.py:176` writes `data/verified/<source>/pseudo.parquet`. Nothing reads that file back into `Episode.labels`.
   - Failing input: `tf verify` then `tf manifest` on phantom, rh20t, airoa or sharpadex gives a manifest with 0 rows. Reassemble gets no labels through `load()`, although `native_labels()` exists. `episode_windows` and `_pairs` in `verify/__init__.py` use `ep.labels` from `load()`, never `native_labels()`.
   - Consequence: the training pipeline cannot produce a non-empty manifest from real adapters. The unit tests pass only because they inject labels or windows.

2. BUG - sync "shift" is never applied.
   - Spec: "shift records a clock correction that the manifest stores and the resampler applies".
   - Code: `manifest.py:191` writes `<name>.shifts.json`. A grep finds no reader. `resample()` has no shift parameter, and the harness and dataset call it without one.
   - Effect: an episode with a 200 ms lag passes the check as "shift" and is used unshifted.

3. BUG - `drop_channel` has no effect.
   - `checks.py:174` returns it, and `manifest._accepted` (manifest.py:81) only looks at `drop_episode`. Nothing zeroes or removes the failed channel.
   - Worse for wrist streams (`check_sensor_health`, checks.py:147-174). `dead` needs all 6 channels to fail. Torque channels are not "required", so fx, fy and fz all flat while the torques are fine gives only `drop_channel`. The episode is kept with a dead force channel and renders F=0 everywhere.

## B. force/

4. BUG - `resample.py:79-83`, raw per-group slices are off by one frame [verified].
   - Spec: group g covers 1-based positions 2g-1 and 2g, which is 0-based (2k, 2k+1).
   - Code: `pos = (2k-1, 2k)`. Group 0 holds only frame 0, group 1 holds frames 1 and 2, and so on. Frame 63 is in no group.
   - Measured: group 1 spans t 7.88 to 8.12 (one frame), the last group spans 23.13 to 23.62, and frame 63 at 23.75 is uncovered.
   - Not consumed downstream yet, so the effect is latent. It is wrong for any later encoder.

5. SPEC-DEVIATION - `resample.py:69-72`, the first frame's dF is hard-coded to 0 [verified: dF[0]=0.0 where the true value is 1.0].
   - Spec: dF = (F_t - F_{t-0.25})/0.25 with F_{t-0.25} taken from the raw median.
   - Code: it differences against the previous frame in the list. Every window's first frame gets dF=0 even though data exists, and a window with non-0.25 spacing is divided by the actual dt.
   - Same in train and eval, so it is consistent, but it is not the spec.

6. SPEC-DEVIATION - `normalize.py:85-91`, the fit uses frame-level medians.
   - The fit takes the median of the 4 Hz frame medians, so sigma is the std of medians. That underestimates raw 100 Hz free-space noise, so tau = baseline + 3 sigma is too low.
   - It also depends on the caller passing frames at t < 0.5 s. `verify/__init__.py` does (frame times from 0), but anything fitting from window frames only sees padded or NaN frames.
   - It pools std across episodes, so between-episode baseline spread is included in sigma.
   - Severity: spec-deviation, medium.

7. nit - the wrist F in `resample.py:33` uses only the force xyz norm (spec says "wrench magnitude"). That is defensible, because mixed units are meaningless, but it is undocumented in the spec.

8. nit - `render.py:51-55`, a missing gripper renders `g=0.00` (open) and NaN F renders as 0.00 with c=0, which is indistinguishable from free space. Frame times like `-0.00` are possible in `t=` for tiny negatives (not hit with the current anchors).

9. BUG - `render.budget_check` fallback vs `max_tokens` [verified].
   - `render.py:25,94` use 0.8 tokens per char. A one-arm pad line is 55 chars, so 64 lines plus the header estimate as 2912 tokens. `configs/force.yaml` max_tokens is 2800, so `check_prompt_diff` (checks.py:404, no tokenizer passed) fails every single-arm pad window with `TokenBudgetError`.
   - That fails the prompt_diff source-level check for phantom and sharpadex. The wrist one-arm case gives 2656 and passes.
   - The spec also says the budget uses the Qwen tokenizer, and no caller supplies one.
   - The documented deviation (1500 to 2800) therefore does not do what the doc says.
   - `train/dataset.py` never calls `budget_check`.

10. Confirmed OK:
    - Hash round trip: `version()` is a sha256 of canonical JSON, `save` and `load` verify it, and floats round-trip.
    - Normalization formula: clip((F-b)/(g-b), 0, 3).
    - Scale: Delta and dF are divided by the same scale.
    - Grasp definition: g>0.6 and F>tau.
    - Line format matches the spec example.
    - L-then-R order and `none` for an omitted arm.
    - Header and control (empty block, no header).
    - Median window is [t-0.125, t+0.125] inclusive.
    - Nearest gripper sample, with ties resolved to the earlier sample.

## C. labels/windows.py

Checked and correct:
- Anchors: `dense_anchors` gives 10k while 10k < t_last-9.75, then the final window at t_last-9.75. For example 19.75 gives [0, 10], 25 gives [0, 10, 15.25].
- Frame times alpha+(i-9)/4 for i=1..64.
- Group mapping (positions 2g-1 and 2g, the more severe frame wins).
- Padded groups: times are clamped to [0, duration], so they take the boundary label.
- Stitching: non-final windows keep groups 1-24, and the final window keeps only groups whose centre is at or after the previous kept end.
- Outcome indices: m=min(64,max(2,floor(10N/fps))), rounded down to even, indices round-half-up of i(N-1)/(m-1). N=300 at 30 fps gives m=64, and N=1 gives m=2 with indices [0,0].
- Targets: layout `Worst/Quality/Rationales`, with runs of equal non-normal digit and a rationale taken from the best-overlapping same-quality span.

Findings:
11. nit - `windows.py:94-104`, the closing (t >= duration) branch means the last real frame at duration-1/fps is never "inside" a span ending at duration. This is harmless. Padding after the end copies the label at t=duration, which the code does, but note that the label at exactly `duration` is the closed-interval one.

12. SPEC-DEVIATION (minor) - `make_windows` windows:
    - Dense windows exist only when a dense or event label exists, so "no labels" and "all normal" differ (documented). Because of finding 1, this means no windows at all.
    - Outcome frame times are `i/cam.fps` and ignore `meta["clock_offset_s"]`. This is fine if the adapters already moved to the episode clock.
    - `to_flora_outcome_target` takes `spans[0].rationale` only.

13. nit - the final window may extend past `duration`. The target digits are padded, but the video frames are clamped by the harness (`_frame_indices`). That is consistent with the spec's padding rule.

## D. prompts/parse.py

Grammar, Worst==min rejection and the all-normal replacement are correct. `parse_dense` uses fullmatch on stripped text, rejects Worst != min digit, and returns worst=3, 32 threes, `parse_failed=True`.

14. SPEC-DEVIATION (minor) - `parse.py:84-87` also rejects `last < first` in a rationale range. The spec rejects only on Worst mismatch. It is defensible, but it is stricter than "exactly as the paper".

15. nit - `[^\n]{1,240}` admits `\r` and whitespace-only text such as `1-2  `, which strips to an empty rationale. Group ranges are not validated against the digit runs.

16. nit - `parsed_to_label` maps digit 2 to quality 2 and digit 1 to quality 1, correct. It picks a rationale only when the run is fully inside a rationale range, otherwise "".

## E. labels/events.py

All six detectors exist (grasp_acquired, slip, contact_loss, regrasp, crush, off_axis) with thresholds from config, and the rationales match the spec strings.

17. BUG-RISK / SPEC-DEVIATION - `events.py:208`, crush uses `np.diff(g) > 0` with no tolerance and no minimum duration.
    - Any positive jitter in g while F' > 2.0 triggers quality 1, and a single frame is enough.
    - It also does not require a hold.
    - It can never fire once the gripper has finished closing (g constant), even at F'=3. The spec says "while g is still closing", so this is literal, but it is noisy. Compare slip, which at least requires a steady g.

18. SPEC-DEVIATION - `events.py:162-163`, contact_loss quality is 2 when `task_complete=True`. The spec only defines "1 if the task was not complete".
    - `task_complete` is never passed by any caller (see finding 1), so it is always the default False.

19. SPEC-DEVIATION - `events.py:35-40`, "over 0.3 s" is mapped to 2 frames at 4 Hz, which is 0.25 s of samples but a 0.5 s span. "At least 0.1 s" is 1 frame. At 4 Hz the 0.1 s slip bound is vacuous: one frame with dF' < -2 (a 0.5 normalized drop in 0.25 s) is enough. Documented, but worth a decision.

20. nit - slip (`events.py:120-142`) requires `held[i-1]` and steady g over only the neighbouring frames. A normal release where F falls before g moves more than 0.02 in a frame is flagged as a slip. The span starts at the frame after the drop, not at its start.

21. nit - `detect_all` returns one label per arm, and the merge does not distinguish arms. Left and right spans of the same quality and rationale text differ only through the arm word, so coalescing is fine.

## F. labels/merge.py

22. SPEC-AMBIGUITY / possible deviation - `merge.py:28-34`, precedence is applied first, severity second.
    - A human or source span of quality 2 overrides a programmatic quality-1 span at the same time.
    - The spec says "Precedence when spans overlap: human > source > programmatic > pseudo; within a frame, the more severe quality wins", which can also be read as severity across origins when they overlap within one frame.
    - The code's reading also means a source "all normal" label (empty spans) cannot mask programmatic events, because only explicit quality-3 spans override.
    - The merged dense label's `origin` is the top origin present, even when every span is programmatic, so `ManifestRow.label_origin` can be mislabelled (`manifest._label_origin` takes the min rank over the episode's labels).

Outcome merge (highest origin wins, ties take the lowest score) is fine.

## G. labels/mapping.py

23. BUG - `mapping.py:26-34`, `rh20t_outcome` raises `ValueError` for rating -1.
    - `docs/deviations.md` says RH20T `meta.rating` ranges -1 to 9, with -1 presumably unrated.
    - Failing input: any unrated RH20T episode crashes the mapper instead of yielding no label.

24. Spec match: the RH20T rule is 0 to 1, 1 to 2 (3 if `object_moved_toward_goal`), 2-9 to 5, and `object_moved_toward_goal` defaults to False. No adapter ever sets `meta["object_moved_toward_goal"]`, so the "3" branch is dead, and nothing derives it. The unit-testable rule exists, but the spec's "decided by a single rule" has no data feed.

25. SPEC-DEVIATION - `map_phantom` (`mapping.py:62-70`) applies the p99 failure span to every episode, including successful demos, using normalized F' against a p99 that comes from the caller or `meta["peak_force_p99"]`.
    - Nothing computes the per-object success-set p99 (a grep finds no producer).
    - Without p99, no dense label is produced.
    - The span uses `features.frame_times`, so it depends on the caller supplying features covering the whole episode.

26. SPEC-DEVIATION - `map_reassemble` reads `meta["anomalies"]`, which the adapter never sets (it sets `meta["segments"]` and `meta["action_success"]`). The outcome mapping works through `action_success`. No dense failure spans are produced from failed segments, although `docs/deviations.md` says they should be. `_failure_spans` raises KeyError if `t0` or `t1` are missing.

## H. verify/manifest.py

27. SPEC-DEVIATION / BUG - `manifest.py:140-157`, caps apply across all splits.
    - `max_episodes` and `max_windows` sample over val and test episodes and windows as well as train. Eval splits (PHANTOM val, REASSEMBLE held-out sessions) get randomly thinned. The cap is also a single `max_windows` over both prompt kinds, so per-kind caps (~300 dense, ~100 outcome episodes) cannot be expressed.
    - Failing input: `max_windows=300` on a manifest that includes the val split drops val windows.

28. nit - `manifest.py:140-143`, the episode cap is drawn before the frozen-split check, and the seeded sample depends on the candidate set. Adding episodes later reshuffles which episodes survive. Frozen splits are not violated (the check follows), but a frozen train episode can silently disappear.

29. Freeze semantics: `freeze_holdout` writes once (`HoldoutExistsError` thereafter) and records every episode's split, not just held-out ones. `build_manifest` raises `SplitFrozenError` if `assign_split` differs from the frozen split.
    - Episodes absent at freeze time are unconstrained, so a new episode can enter any split.
    - Only episodes that pass checks and the cap are compared.
    - `window_id` is a sha1 over (source, episode_id, kind, anchor to 6 decimals), deterministic and split-free. Duplicates are silently de-duplicated.

30. nit - `shifts.json` (finding 2) is written beside the manifest but is not part of the schema or the hash.

## I. verify/checks.py

31. BUG-ish - `checks.py:151-155`, saturation counts the minimum value as saturation.
    - `bottom = mean(x == x.min())`. A pad sensor that reads exactly 0 (or a quantized floor) for more than 1% of an episode is flagged "saturated".
    - Unloaded FSR or DM-Tac channels often clamp at zero, so healthy channels can fail and the episode is dropped or the channel dropped.
    - A channel with one peak is fine.

32. SPEC-DEVIATION - `check_normalization` (checks.py:227-257) runs per episode, but the spec says to fail the source. `report.source_verdicts` requires every episode to pass, so one episode with a light grasp (median far from 1.0) fails the whole source.
    - The 99th percentile is taken over F' that `apply` already clipped at 3.0, so the check can only see "at least 1% of frames at the clip".

33. SPEC-DEVIATION - sync (`checks.py:192-221`).
    - Sign convention is correct: positive lag means force trails the gripper (np.correlate(f, g)).
    - Thresholds: per-episode drop at |lag| >= 0.25 s (`sync_max_s`), and a median under 0.1 s evaluated in `verify/__init__.py` (`sync_median`) over |lag| including failed episodes.
    - Search window 1.0 s, grid from config.
    - Whole-episode cross-correlation of F with g is biased by the release edge (both fall), but this is fine in the usual case.
    - Passes with action "shift" when |lag| >= grid step, but the shift is never applied (finding 2).
    - Without a gripper stream it returns "ok, not checkable". Baxter's stricter "force onset vs gripper command crossing under 100 ms per episode" rule is not implemented separately.

34. SPEC-DEVIATION - spot check (`write_spot_strips`) writes PNG strips, not "video plus label strip". The pass threshold (17 of 20, scaled when n differs) is right.

35. SPEC-DEVIATION - holdout (`checks.py:415-427`) drops only if the id is in the frozen set and the episode's split is train.
    - The spec: "no episode id in holdout_ids.json". The code is looser, and `held_out_ids` already excludes the train splits. That is a reasonable reading.
    - But `ctx.split_of` defaults to train when it is absent, so an eval-split episode is flagged if `split_of` is not populated. `verify/__init__.py:185` does fill `split_of` per episode.

36. Confirmed OK:
    - integrity rules (decode, frame count, monotonic, rate);
    - license;
    - overlap (strict `< 6` on all three hashes; an empty index passes silently);
    - label_sanity (observed in [exp/k, exp*k]);
    - check order matches the spec table.

## J. eval/metrics.py

Definitions match the spec: bad-action F1 is F1 of non-3; failure recall is recall of class 1; MAE is |score-truth| mean x100; AUROC is a rank statistic with average ties; E[s] (`inference.expected_score`) keeps digits 1-5 and renormalizes. Dense pooling over (episode, t) is done in the harness.

37. nit/SPEC-DEVIATION - unparsable outcomes get a NaN score and are dropped from MAE, accuracy and AUROC (documented). The spec says parse failures are "replaced ... and counted". Dropping inflates MAE if the model fails on hard windows. Dense is replaced by all-normal, as specified.

38. nit - `_align` is an inner join, so predictions that are missing (crashes or missing windows) are silently dropped rather than counted as errors.

39. nit - MAE uses the integer parsed `score`, not E[s]. Accuracy uses `E[s] >= 3.0`, an arbitrary threshold not in the spec. The harness defines truth success as outcome >= 4, a different cut from the 3.0 prediction threshold.

40. Not implemented: SoftVTBench "MAE vs DSR" and "failure recall on over-compression spans". A grep for DSR and over-compression in `eval/` finds nothing. HRDexDB and SoftVTBench therefore run through the generic outcome and dense paths only.

## K. eval/ablations.py

Five modes present: real, omitted, zero, shuffled, video_blank.

41. SPEC-DEVIATION - `ablations._shuffle` and `harness.py:264-296`, the donor is "a different episode" by `episode_ref` but not "of the same source".
    - The pool is every window of the manifest, filtered only by `key != ekey`. There is no source or prompt_kind filter, only channel and length (`ablations.py:46-55`).
    - A dense target can receive an outcome-window donor (long episode timescale) or a donor from another source if the benchmark manifest has several.
    - Seed is `seed + k` per window, which is deterministic.
    - In `train/dataset.py:137-141` the pool is also all windows with no same-episode exclusion. Shuffled training could use another window of the same episode.

42. nit - `_zero` leaves `g` None if the source had no gripper. It renders as 0.00 anyway, so the result is equivalent.

## L. eval/harness.py

43. BUG - `harness.py:468`, the grid skip keys on `predictions.parquet` existing, but `metrics.json` is written last (`:435`).
    - A crash during metric computation leaves a cell that is skipped forever and absent from the report, since `write_report` globs `metrics.json`.
    - The skip also ignores a changed manifest, seed or config. `write_report` will then raise `ManifestMismatchError`, which is intended.

44. BUG-RISK - `checkpoint_tag` (`harness.py:95-98`) uses the last path component.
    - `data/runs/force/adapter` and `data/runs/control/adapter` (the spec's per-run layout) both tag as `adapter`. They collide in `cell_dir`, so the control cell is skipped because the force cell exists, or they overwrite each other. `by` in `write_report` collides the same way.

45. BUG-RISK - `harness.py:277-337`, `dense_pred`, `dense_truth` and `obj_of` are keyed by `episode_id` alone, not by source/episode_id. A benchmark manifest mixing sources with overlapping ids (for example sequential ints) stitches windows from different episodes together.

46. nit - `prov()` is called up to three times (sources for constants, the shuffle pool, the main loop), so every episode is reloaded and re-windowed three times. The shuffle pool holds all window features in memory.

47. nit - when `score_token_logprobs` is missing, `expected_score` silently falls back to the integer score, so AUROC can be computed on integer scores without notice.

Confirmed OK:
- `roboreward_500` is omitted-only (`OMITTED_ONLY` plus `check_mode`). It is hard-coded by name rather than driven by config, and the grid raises rather than skips if the grid file lists another mode.
- `write_report` raises `ManifestMismatchError` when the (manifest_hash, windows_cache_hash) set for a benchmark has more than one element, including across checkpoints.
- The shuffle-delta column is real minus shuffled on the headline metric (failure_recall if present, else the first metric), shown on the `real` row. For MAE (lower is better) the sign is inverted relative to the other metrics.
- The baxter per-object table is present.

## M. train/dataset.py

Answer-only loss mask: `answer_labels` is -100 on the prompt prefix and token ids on the answer. It requires full_ids to extend prompt_ids and raises otherwise. Prompt/control diff is asserted via `prompts.flora.prompt_diff` for every sample with a block.

48. RISK - `QwenTokenizerAdapter` (`dataset.py:241-288`).
    - It tokenizes prompt and prompt+answer separately. The Qwen3 or 3.5 chat template can insert an empty think block or strip it on historical assistant turns, so the prefix check can fail at train time (the docstring admits "Unverified against the real checkpoint").
    - Every sample gets exactly `VISUAL_TOKEN_CAP` video pad tokens regardless of frame count. The length sort and wall-clock estimate are not meaningful for the visual part.
    - The trailing `<|im_end|>\n` is part of the supervised answer tokens, including the newline.

49. SPEC-DEVIATION - `dataset.py:149-154` calls `prompt_diff` only when a block exists, and it compares text only. Control and force samples are not compared at token level. `video_blank` differences are invisible to the diff.

50. nit - `adapter_windows_source` constructs the adapter and calls `load` without `discover`. At least phantom's `_ensure()` discovers lazily from `self._root`, so this probably works. Not verified for every adapter.

51. Confirmed OK: only `split in splits` (default train) rows are used, and samples are sorted by (n_tokens, window_id).

## N. Spec requirements not implemented, or not wired (grep over the tree)

- Per finding 1: native label mapping, programmatic events and pseudo labels are not attached to episodes in the verify or manifest path.
- Per finding 2: applying the sync shift.
- Per finding 3: dropping or masking a failed channel.
- PHANTOM per-object success-set 99th-percentile computation: no producer. `peak_force_p99` is only read from meta.
- RH20T `object_moved_toward_goal`: nothing sets it.
- Token budget with the real Qwen tokenizer: only the fallback is used in verification and training (`budget_check(block, max_tokens)` has a single call site, with no tokenizer).
- SoftVTBench DSR metric and over-compression recall.
- Spot check as video: strips are PNG only.
- Baxter's per-episode "force onset vs gripper command under 100 ms" sync check as a separate rule.
- `eval_only` enforcement in build and train: the manifest does not stop an eval-only source from entering a train split. `pseudo.py` refuses, but `build_samples` does not check the source's `eval_only` flag.
- Gated DeltaNet fp32 cast is described in `sft.py` (it targets parameter names containing `linear_attn`, which is unverified against real checkpoint names).

## O. Top priorities

1. Finding 1: wire mapping, events and pseudo into `Episode.labels` (or `episode_windows`), otherwise nothing trains.
2. Finding 9: the budget check cannot pass for one-arm pad windows with the fallback estimator.
3. Findings 2 and 3: shift and drop_channel are no-ops.
4. Findings 23 and 26: RH20T -1 crash and the REASSEMBLE mapper contract.
5. Findings 43 and 44: grid skip and checkpoint tag collisions.
6. Finding 4: raw-slice off-by-one.
7. Finding 27: caps hitting eval splits.
8. Finding 31: min-value saturation false positives.
