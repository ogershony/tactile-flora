# Deviations from the spec

Decisions taken during implementation where the spec was inconsistent with reality or with
itself, or where it left a choice open. Each entry says what the spec asked for, what was
done, and why. Entries marked **decision needed** require a human choice. Current as of
2026-10-21; the code is the source of truth (module docstrings carry the same facts).

## 1. Force channel and prompt

### Force block token budget (decision needed)

Spec: force block under 1,500 tokens per 64-frame window, one line per sampled frame in the
exact format given. Measured with the Qwen3 tokenizer (digits tokenize one per character):
one-arm pad line 40 tokens, one-arm wrist line 34, two-arm pad line 68. Sixty-four lines cost
2,560 / 2,176 / 4,352 tokens. The line format is kept byte-exact. `render.max_tokens` in
`configs/force.yaml` is now 3,000 (it was 2,800), which admits every single-arm window (the
evaluation case) and rejects two-arm windows.

Budget checks count tokens with the real tokenizer when `render.tokenizer_json` points at the
Qwen3.5 `tokenizer.json`. With it unset (tests, CI) the estimate is
`FALLBACK_TOKENS_PER_CHAR = 0.75` tokens per character, measured on the exact line format
(it was 0.8, which over-estimated and made every one-arm pad window fail at 2,800).
`tokenizer_json` is null in the committed config and must be set on GPU machines.

Decision needed: keep 3,000 (costs about 2.6k of the 16k-token window per sample), or cut the
block. Options: one line per 0.5 s group instead of per frame (halves it), or drop `D` and `g`
from the line. Either changes the byte-exact format and requires re-running training.

### FLORA answer grammar follows the released code, not the spec sketch

Spec sketch: `Worst: <1-3> then 32 digits then Rationales:` and `Rationale: ... Score: <1-5>`.
Released `batch_utils/core.py` and paper Appendix G (see `docs/research/flora_prompts.md`):
dense answer is `Worst: d\nQuality:\n<32 digits>\nRationales:` followed by
`\n<first>-<last> <text 1..240 chars>` per run; outcome rationale answer is
`Rationale: <text>\n\nScore: <d>` with a blank line. The verbatim grammar wins because the
spec says to reproduce the released prompts verbatim. The parser additionally rejects a
rationale range with `last < first` (stricter than the paper).

### Outcome parse failures are replaced by score 3

Spec: parse failures are replaced by "all-normal" and counted. For dense output that is exact
(Worst 3, 32 threes, `parse_failed`). An outcome answer has no all-normal equivalent, so an
unparsable outcome answer becomes score 3 (`metrics.OUTCOME_PARSE_FAIL_SCORE`, middle of the
1-5 scale), stays in MAE, accuracy and AUROC (with E[s] = 3), and is counted in
`metrics.json` under `parse_failed`. Binary labels and the predicted-success cut share one
constant, `SUCCESS_THRESHOLD` (>= 4.0). Decision needed only if the owner prefers dropping
such rows from the metrics instead.

## 2. Labels

- Precedence is applied first, severity second: a human or source quality-2 span overrides a
  programmatic quality-1 span at the same time. The spec sentence can also be read as severity
  across origins. Decision needed if the other reading is wanted.
- Programmatic detectors (`labels/events.py`), all on the 4 Hz frame clock:
  - `contact_loss` quality is 2 when the task is complete, 1 otherwise (the spec defines only
    the incomplete case).
  - `slip` requires a net F' drop of more than 0.5 (normalized) across the run, measured from
    the frame before the run. At 4 Hz the spec's "at least 0.1 s" is one frame, so a lone noisy
    dF' sample would otherwise trigger it.
  - `crush` requires g to rise by more than `slip_g_tol` between consecutive frames while
    F' > crush_f, so quantisation jitter in g is not "closing". It still cannot fire after the
    gripper has stopped closing (literal reading of "while g is still closing").
  - "over 0.3 s" maps to 2 frames (a 0.5 s span), "at least 0.1 s" to 1 frame.
- Label sanity (verification check): skipped, with an "only N labeled episodes; rate not
  assessed" note, when fewer than 10 episodes of the source are labeled.
- PHANTOM dense failure spans: the p99 of normalized F' is computed per object from the
  success set (`labels.mapping.phantom_success_p99`, once per source in verify), and the span
  starting at the first F' above it is emitted only for failure demos (`meta["failure_demo"]`),
  not for every episode.
- REASSEMBLE has no anomaly annotation. Dense failure spans are built from failed segments
  (`segments_info/<i>/success == false`, the segment's start..end), outcome from the per-segment
  success flag. No spans are invented.
- RH20T: `meta.rating` -1 (unrated) produces no label instead of raising. Rating mapping:
  0 to 1, 1 to 2 (3 if `object_moved_toward_goal`), 2-9 to 5. No adapter sets
  `object_moved_toward_goal`, so the 3 branch is unreachable (decision needed: derive it or
  drop the branch). Rating semantics are undocumented in the port.
- SoftVTBench DSR derivation: there is no DSR column. The adapter takes
  `fem_kabsch_max_pct` and a per-object tolerance from `split_rules["dsr_tolerance_pct"]` in
  `configs/sources/softvtbench.yaml` (one float or a mapping `{asset_name | default: float}`);
  contiguous over-tolerance runs become quality-1 event spans ("excessive force on the object").
  The tolerance values are not in the HDF5 and the config is currently unset, so no event labels
  are produced (decision needed: set the tolerance; the paper calibrates it per object).
  Rigid subsets have no FEM fields at all; published subsets keep only successes, so outcome is
  nearly constant.

## 3. Manifest, splits and verification

- Sampling caps (`max_episodes`, `max_windows`, plus optional per-kind `max_dense_windows` and
  `max_outcome_windows`) apply to the train split only; val, test and benchmark are never
  thinned, and episodes frozen as train in `holdout_ids.json` are never dropped by a cap.
- Eval-only sources (`eval_only: true`: baxter, hrdexdb, softvtbench) refuse train rows
  (`EvalOnlyTrainError`), and `tf pseudo` refuses them (`PseudoRefused`).
- `ManifestRow.shift_s` (the sync check's clock correction, `-lag`) is applied as a shift of
  the force clock when the force is resampled: the force at frame time t is the raw force at
  `t - shift_s`. The gripper and video are not shifted. Raw lags also go to
  `<name>.shifts.json`.
- `drop_channel` removes the failed arm from the stream (`verify/windowing.prepare_episode`).
  A `drop_channel` that names no arm, or that removes every arm, is treated as `drop_episode`.
- Shuffled ablation donors come from a different episode of the same source and the same
  `prompt_kind`, with the same channel and at least as many frames; no donor raises an error.
- Checkpoint tag rule for `data/eval/<tag>/...`: the tag is the last path component, except
  that a path ending in `adapter` uses its run directory name
  (`data/runs/force/adapter` becomes `force`); `released` stays as is. Two checkpoints mapping
  to one tag raise `CheckpointTagCollisionError` when the grid is expanded.
- Resampling and normalization deviations from the spec that remain (see `docs/review1.md`
  items 5, 6, 7): the first frame's dF is differenced against the previous frame rather than
  the raw median 0.25 s earlier, the baseline sigma is fitted from frame medians rather than
  raw 100 Hz samples, and the wrist channel's F uses the force xyz norm only (no torques).
  Decision needed only if raw-sample fidelity matters for the later learned encoder.

## 4. Dataset facts that differ from the spec (see docs/research/datasets.md)

- PHANTOM is HapticWAM (`armteam/hapticwam-teleop-raw`, CC-BY-4.0). Layout is per-episode
  zarr plus JSON manifests, not LeRobot. Scene video is per-frame JPEG bytes in zarr at
  15 Hz. Fingertip force is a 6-D wrench per pad at about 6 Hz, not DM-Tac raw; it is
  nearest-sampled to a 10 Hz grid. Failure demos are forced into the train split; the
  `failure_demo` manifest flag is reliable, `success` is not for some `*_fail` batches.
- REASSEMBLE: one episode is a whole recording (many actions); see Labels above.
- RH20T community ports carry no fingertip tactile in any config, so RH20T feeds the wrist
  channel only; `meta.rating` ranges -1 to 9. Licenses are per scene (1-5 CC-BY-SA-4.0,
  6-10 CC-BY-NC-4.0).
- AIRoA MoMa is gated (auto-approve) and LeRobot v3-style, license CC-BY-NC-SA-4.0.
- SharpaDex is CC-BY-NC-4.0, not CC-BY-4.0 as the spec lists; `configs/sources/sharpadex.yaml`
  has been updated to match. **Decision needed**: whether NC-licensed data (SharpaDex, RH20T
  scenes 6-10, AIRoA, HRDexDB) may be in the training mix for released weights.
- RoboRewardBench is the test split of `teetone/RoboReward` (2,831 rows), not a FLORA release.
- Per-row `license` is stored in the manifest and `tf verify` checks it against
  `license_allowed` in each source config.

## 5. Adapter facts that remain UNVERIFIED

Nothing below has been checked against a real episode. Each is a place a first real-data run
can fail or silently mislead. Source: grep for unverified/guess in `tactile_flora/sources/*.py`.

| Module | Unverified fact |
|--------|-----------------|
| `sources/phantom.py` | Whether newer exports use zarr v2 or v3 stores (zarr>=3 reads both). |
| `sources/sharpadex.py` | Index of the right-index MCP joint when `info.json` has no names (`RIGHT_INDEX_MCP_FALLBACK_INDEX = 40`, a placeholder); polarity of that joint (larger = more closed?); exact schema of `meta/subtasks.parquet`; force units (not documented). |
| `sources/rh20t.py` | Whether `observation.robot_ft` is tared or raw (cfg3 is clipped to +-300, cfg7 Fz min -571 N); whether a larger gripper value means closed; semantics of `meta.rating`. |
| `sources/reassemble.py` | Exact names of the per-sensor timestamp keys beyond "one per sensor"; units and open/closed polarity of `gripper_positions` (closure = 1 - w / max w assumes larger width = more open). |
| `sources/airoa.py` | Wrist wrench key (heuristic: `observation.wrist.wrench`, or first `ft`/`wrench` feature); order of the 100-sample wrench history (`WRENCH_HISTORY_ROW = -1`); gripper polarity; where the session date comes from (uuid, then version, else `undated`); per-primitive O/X and failure-cause schema (`native_labels()` is empty and outcome comes from `task_success`). |
| `sources/hrdexdb.py` | Which camera serial is a sensible scene view (first sorted serial used); video-to-sensor clock sync (assumes first video frame = earliest sensor sample); `inspire_f1` tactile layout (load raises NotImplementedError, as do robotiq_2f85 and human); per-finger semantics and units; the hand-closure proxy from `position.npy` assumes larger = more closed. |
| `sources/softvtbench.py` | DSR tolerance per object (not in the HDF5); video/HDF5 frame-count equality; Safety Success vs DSR naming across sources. |
| `sources/baxter.py` | Real FSR units and range (sensor identity is an open spec question); whether the writer can provide `video_t0_s`. |

Decision needed for the owner: none of these can be resolved without data. The acceptance
step is one real episode per source (spec milestone 5).
