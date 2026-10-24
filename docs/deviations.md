# Deviations from the spec

Decisions taken during implementation where the spec was inconsistent with reality or with
itself, or where it left a choice open. Each entry says what the spec asked for, what was
done, and why. Entries marked **decision needed** require a human choice. Current as of
2026-10-22; the code is the source of truth (module docstrings carry the same facts).

## 1. Force channel and prompt

### Force block token budget (resolved 2026-10-21 by a second, token-cheap line format)

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

The options considered were: keep 3,000 (costs about 2.6k of the 16k-token window per sample),
one line per 0.5 s group instead of per frame (halves it), or drop `D` and `g` from the line.
Either of the last two changes the byte-exact format and requires re-running training.

Resolution: a second line format, `render.format: code`, selectable per config
(`configs/force.code.yaml`; `configs/force.yaml` keeps `decimal`, so every existing artifact,
test and prompt is byte-identical). The code format keeps one line per frame and every quantity
of the decimal line, but as single-character levels instead of two-decimal numbers, because the
Qwen tokenizer emits one token per digit and the decimals were 60% of the line's tokens. Per arm
`F<0-4> [A<a|b|=>] T<+|0|-|!> C<0|1> G<o|m|c>` (force level by `render.levels` on F', pad
asymmetry by `asym_frac`, trend by `trend_up`/`trend_down`/`sharp_drop` on dF', contact flag,
gripper closure by `closure`); thresholds live in the config and the block starts with a legend
header (`render.header_code`) and, with `render.summary`, one per-arm summary line (first
contact, peak level, contact lost while closed, sharp-drop count, final level) derived from the
lines themselves (`force/render.py`). The system prompt announces it with
`FORCE_SENTENCE_CODE` instead of `FORCE_SENTENCE`; `prompt_diff` still asserts the force and
control prompts differ only by the block and that sentence. Ablations act on the features before
rendering and are unchanged.

Measured 2026-10-21 with the Qwen3.5 tokenizer.json (Qwen3.5-9B snapshot, assumed to share the
FLORA base's vocabulary; confirm by hashing the released checkpoint's tokenizer.json) on 64-line
fixture windows, legend header and summary included: one-arm pad 19-21 tokens per line, 1,473
per window (decimal: 40 per line, 2,602); one-arm wrist 17-19 per line, 1,345 (decimal 34,
2,218); two-arm pad 27-30 per line, 2,077 (decimal 68, 4,394). The worst realistic two-arm pad
window (four-digit frame times, every field at its two-token spelling) is 2,200. The summary
line costs 41-72 tokens of those totals. So one-arm windows meet the spec's 1,500 (for frame
times under 100 s; three-digit times add 64 tokens) and two-arm windows fit for the first time.
`render.max_tokens` is 2,500 in `configs/force.code.yaml`; the no-tokenizer estimate is
`FALLBACK_TOKENS_PER_CHAR_CODE = 0.72` tokens per character, the measured per-line maximum
(two-arm wrist at `t=1000.00`), which gives 1,751 / 1,613 / 2,267 for the three windows above
and 2,397 for the worst case. The `decimal` budget and fallback are unchanged.

What stays with the owner: which format the production training and eval runs use (switching
changes the prompt, so force and control must both be re-run), and the four threshold sets,
which are first guesses in normalized units (levels at 0.5 / 1.5 / 2.5 F', asymmetry at 30% of
F', trend at +/-1.0 F' per second, sharp drop at the slip detector's -2.0, closure at g 0.3 /
0.6); a per-source histogram of F', dF' and |delta'|/F' on the verified data would show whether
the levels split the grasp regime usefully. Gating `A` on contact (`c`) is a possible refinement:
below `tau'` the pad difference is noise and can still print `a`/`b`.

### Normalization per object on PHANTOM (added 2026-10-22; measured in data/colab/e0_local)

Spec: one set of normalization constants per source, with grasp frames defined by gripper
closure g > 0.6. Measured on the local 180-episode PHANTOM subset (E0-local report,
`data/colab/e0_local/REPORT.md`): the train-success median peak force spans 10.4 N (waffles)
to 54 N (whiteboard), so one pooled scale puts a normal whiteboard grasp at crush level and a
Carton over-squeeze (26.7 N failure vs 12.0 N success median) at a normal level; a rule on the
pooled block reaches balanced accuracy 0.57, and the same rule on per-object-relative peaks
0.75. whiteboard_fail is an under-grasp mode (5.0 N failures vs 33.8 N successes), the opposite
sign. The median maximum closure per object is 0.49-0.61, so only waffles contributed grasp
frames to the pooled fit (the "median grasp" constant was the waffles grasp) and a per-object
fit on Carton raised NormError. Deviation: constants are fitted and applied per
(source, `norm_group`), where `force.normalize.norm_group(episode)` is the PHANTOM task folder
with `_fail` stripped (Carton and Carton_fail share constants) and "all" for every other source;
`norm.json` holds `groups: {group: {arm: ArmNorm}}` and `norm_version` hashes the whole mapping,
the manifest row records `norm_group`, and every renderer (verify windows, checks, train
dataset, eval harness) applies the episode's group. Single-group sources are unchanged
(a bare-features fit is the group "all", which applies to every episode). The grasp-frame cut is
now `normalize.grasp_g` in `configs/force.yaml` (default 0.6, the old constant); when fewer than
`normalize.min_group_frac` (default 0.25) of a group's fitted episodes have a frame above it,
the fit uses the frames at or above the group's 75th percentile of g and records
`grasp_rule: p75` with the percentile in `grasp_g` (the normalization check reads that rule
back), and NormError is only raised when no frame has F > tau at all. The fraction rule replaced
a plain "no frame above the cut" trigger after the first per-object fit on the subset: egg
passed g > 0.6 in 6 of 47 episodes, 5 of them over-squeeze demos, and its "median grasp" came
out at 41.6 N against a 25.2 N success median peak. For the same reason the fit now skips
episodes whose adapter set `meta["exclude_from_norm_fit"]` (verify pass 1; the constants are
still applied to every episode); the PHANTOM adapter sets it for failure demos, whose
over-squeeze (egg_fail frames at g > 0.6 pulled the pooled median up) or under-grasp
(whiteboard_fail at 5.7 N pulled it from about 54 N to 42.6 N) must not define the typical
grasp. A group whose every episode is excluded raises NormError (the source fails
normalization) rather than being fitted on its failure demos, and the pooled normalization
verdict is taken over the fit episodes only (the excluded ones are marked in checks.parquet and
still reported): on the subset the success-only pool gives grasp median 0.996, free-space
0.000 and p99 1.04, whereas including the failure demos (egg_fail grasp median 2.05,
waffles_fail 1.43, off-scale by design) put the pooled grasp median at 1.215 and failed the
source. Fitted this way the subset gives Carton 13.6 N, egg 27.4 N (p75 rule, g >= 0.525),
waffles 15.3 N and whiteboard 42.6 N (p75 rule, g >= 0.50); the whiteboard value is below its
53.9 N success median peak because the constant is the median grasp frame, not the peak (the
success median peak renders at F' 1.28, the other objects' at 0.84-0.96). The code-format
legend gains one clause stating that levels are relative to this object's typical grasp
("F2 = this object's median grasp"); for non-PHANTOM sources "this object" still means the
source's single group, and the clause costs about 20 tokens on the measured header totals above.
Expected effect on the block: a normal grasp of any object sits at F2 and an over-squeeze at
F3-F4 regardless of the object's absolute stiffness. Not yet measured: whether a per-object
scale survives on the full dataset's other objects (the subset has four), and whether the 75th
percentile fallback picks the hold phase on objects with a slow closure; the per-group
`grasp` / `grasp_rule` values in `data/verified/phantom/norm.json` are the thing to inspect.

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
  success set (`labels.mapping.phantom_success_stats`, once per source in verify), and the span
  starting at the first F' above it ("excessive force on the <object>") is emitted only for
  failure demos (`meta["failure_demo"]`), not for every episode. Added 2026-10-22, the symmetric
  under-grasp rule (motivation: whiteboard_fail peaks at 5.0 N against 33.8 N for successes in
  the E0-local report, so the crush rule never fires on it and the folder got no dense span): a
  failure demo whose peak F' is below the success set's 5th percentile of per-episode peak F'
  (`phantom_peak_p5`, same stats pass) gets a quality-1 span from its first contact frame
  (c = 1 on any arm; the whole episode if it never made contact) to the end with rationale
  "insufficient grip force on the <object>" (the too-weak class). The crush rule is checked
  first and is unchanged; a failure demo whose peak lies between the two thresholds gets the
  outcome label only. The same span is attached to the outcome label, so the episode-level
  rationale names the failure class (it was the "no rationale" placeholder before, for both
  classes). Decision needed only if the owner wants the under-grasp span to start at gripper
  closure rather than at first contact.
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
- Source-level sync gate (`sync_median_s` in `configs/sources/<name>.yaml`, default the spec's
  0.1 s; read by verify, the report and `build_manifest`): PHANTOM uses 0.2 s because the dense
  frame grid is 0.25 s and the local 180-episode subset measures a median |lag| of 0.15 s
  (p90 0.275 s), so 0.1 s rejected the source on a lag below one frame period; the per-episode
  0.25 s drop rule is unchanged. reassemble_lerobot (2026-10-22, full port, 149 recordings) uses
  0.25 s, the frame period itself: median |lag| 0.225 s, p75 0.325, p90 0.50, max 1.0 (7
  recordings pinned at the 1.0 s search bound), and 138 of 149 lags negative (median -0.225 s:
  the wrist force moves before the reported gripper width does), i.e. a systematic port offset
  rather than jitter. The per-episode shift corrects it for the 86 recordings under the 0.25 s
  drop rule; 54 are dropped by that rule regardless of this gate. The 0.025 s margin to the
  measured median is one estimator grid step, so this is the largest defensible value.
  **Decision needed**: a constant clock correction in the adapter (about +0.2 s on the force
  clock, or a check of which stream the port delays) would bring the median under 0.1 s and
  recover most of the 54 dropped recordings; it needs a video-based confirmation of the sign.
- Source-level pooled grasp band (`norm_grasp_range` in `configs/sources/<name>.yaml`, default
  the spec's [0.8, 1.2]; read by verify, the report and `build_manifest` like `norm_p99_max`):
  reassemble_lerobot uses [0.8, 1.3]. Measured 2026-10-22: the pooled verdict takes the median
  over recordings of each recording's grasp median, 1.233, while the fit's constant is the
  median over all grasp frames (1.0 by construction), and the per-recording medians spread from
  0.29 (p10) to 2.13 (p90) because a recording is 5-10 min of different assembly actions on a
  wrist sensor (19 of 149 recordings sit inside [0.8, 1.2], 58 inside [0.5, 1.5]). The typical
  recording's grasp still renders at level F2 (0.5 to 1.5), so the 1.233 does not move the
  rendered levels; the band is widened by one tenth rather than re-weighting the fit, which
  would change every source's constants. **Decision needed**: whether the fit (or the pooled
  check) should be episode-weighted for long heterogeneous recordings.
- Source-level pooled-p99 gate (`norm_p99_max` in `configs/sources/<name>.yaml`, default the
  spec's 3; read by verify, the report and `build_manifest`; the per-group fit-time
  `p99_fprime` in `norm.json` and the report's constants table show the actual tail):
  reassemble and reassemble_lerobot use 60 because their gravity-compensated grasp scale on
  light parts is about 0.26 N (p10 0.08 N), so an ordinary 11.8 N peak is F' of about 44 and
  the 3 gate rejected the source; `render.clip_max` stays global and still bounds the block.
- Multi-source manifests (added 2026-10-22): `tf manifest --name X` builds ONE source and
  overwrites `X.parquet`, so calling it per source with `--name train` kept only the last source.
  The spec's single accepted manifest is now built per source (`--name <source>`, every split,
  train capped by that source's config) and concatenated with `tf manifest merge --out train
  --split train phantom reassemble_lerobot rh20t` (`verify/manifest.merge_manifests`): rows are
  copied unchanged, identical duplicates collapse, a differing duplicate window_id refuses
  (`MergeConflictError`), and an empty selection refuses so a misnamed split cannot produce an
  empty benchmark. Benchmark manifests are the same merge over one source and split
  (`phantom_val` = phantom val, `reassemble_test` = reassemble_lerobot test). The merged
  `.shifts.json` keys lags by "source/episode_id".
- Spot review recording (added 2026-10-22): `tf verify --spot-pass N --record-only` records the
  reviewer's count and re-scores only the spot gate from the saved `checks.parquet`
  (`verify.record_spot_pass`); the plain `--spot-pass N` re-runs every check, which on RH20T
  means decoding about 2 M AV1 frames again for an integrity result that cannot change.
  **Decision needed / flag**: the spot reviews recorded on 2026-10-22 for phantom (20/20),
  reassemble_lerobot and rh20t were done by the agent from the strips plus the force lines in
  `spot/review.json` (the `reviewer` note in each `spot/index.json` says so), not by a human.
  They unblock the dry run only; re-record with your own count before any reported run.
- PHANTOM split on the local subset (2026-10-22): the subset was fetched by
  `data/colab/fetch_phantom_plan.py`, which writes `manifests/plan.json` but not the catalog
  `manifests/all.jsonl` that `scripts/download_phantom.py` writes and the adapter reads, so every
  episode fell to `train`, `session` was "unknown" for all 180 (one pooled baseline group for the
  sensor-health drift check) and `phantom_val` would have been empty. `all.jsonl` was written from
  the plan's episode rows (the catalog rows at revision ff0f0cda, catalog sha256 e2319f31...).
  With it the split is HapticWAM's own: 60 val successes (15 per object) and 120 train (60
  successes, 60 failure demos, which the adapter forces into train). Consequences: `phantom_val`
  holds successes only, so there is no PHANTOM-fail evaluation set yet (decision needed: hold out
  one object family's failure demos, e.g. whiteboard_fail, or a per-object fraction); and the
  drift check now runs per real session, where 38 of 57 sessions have fewer than 3 episodes and
  get no baseline statistics, so sensor_health drops fell from 13 to 6 episodes (136 of 180
  accepted instead of 129). The per-session rule is the spec's; the sparse subset is why it bites.
- RH20T fails verification (2026-10-22, cfg3 + cfg5, 2,023 episodes, 216 accepted; 16 min with
  the two downloaded cameras): `sync_median` 1.0 s (1,117 of 1,763 lags sit at the +-1.0 s search
  bound, 703 positive and 414 negative, 260 episodes not checkable), pooled p99 of F' 14.9
  (per-episode p75 70, p90 103) against the spec's 3, `label_sanity` 0.624 against the 0.3
  placeholder, 956 sensor-health drops for baseline drift (fx 472, fy 291, fz 193) and 86
  integrity drops (the five undownloaded cfg5 scene files). The cause is the channel, not the
  gates: one "all" normalization group pools cfg3's untared 30 N wrench with cfg5's 4.7 N one
  (baseline 5.31 N, grasp 6.09 N, scale 0.78 N), so cfg3 free space renders at F' 18-39 and any
  arm motion at level 4, the lag estimator finds no force/closure structure (correlation ~0),
  and the crush detector stamps "excessive force" on 62% of episodes with the gripper open
  (that is the label-rate failure). No per-source override is justified: the lag median is the
  estimator's bound rather than a lag, and a p99 gate above 100 would admit a channel whose
  levels do not track grasping. Spot review (agent, 8 of 20 strips viewed): 12/20, the eight
  windows at constant F4 under an open gripper fail. **Decision needed** before RH20T can
  train: per-config normalization groups (`normalize.norm_group` keyed by `meta["cfg"]`),
  taring or gravity compensation for cfg3 (the port carries `observation.state.ee_pose`),
  `expected_label_rate` measured after that, and a sync check that is skipped or replaced for
  wrist channels without force/closure structure. Until then `train` is phantom +
  reassemble_lerobot.
- Resampling and normalization deviations from the spec that remain (see `docs/review1.md`
  items 5, 6, 7): the first frame's dF is differenced against the previous frame rather than
  the raw median 0.25 s earlier, the baseline sigma is fitted from frame medians rather than
  raw 100 Hz samples, and the wrist channel's F uses the force xyz norm only (no torques).
  Decision needed only if raw-sample fidelity matters for the later learned encoder.

## 4. Dataset facts that differ from the spec (see docs/research/datasets.md)

- PHANTOM is HapticWAM (`armteam/hapticwam-teleop-raw`, CC-BY-4.0). Layout is per-episode
  zarr plus JSON manifests, not LeRobot. Scene video is per-frame JPEG bytes in zarr at
  15 Hz. Fingertip force is a 6-D wrench per pad at about 6 Hz, not DM-Tac raw; it is
  nearest-sampled to a 10 Hz grid. Sensor saturation fractions are measured before this
  upsampling, which otherwise duplicates maxima and falsely rejects short episodes.
  Failure demos are forced into the train split; the
  `failure_demo` manifest flag is reliable, `success` is not for some `*_fail` batches.
- REASSEMBLE: one episode is a whole recording (many actions); see Labels above.
- REASSEMBLE LeRobot port (`sources/reassemble_lerobot.py`, `robot-lev/reassemble`, the ~20 GB
  alternative to the raw HDF5 archive): `observation.force` is the uncompensated sensor reading.
  Over all 149 recordings the mean force on idle "No action." frames has |mean| 13.6 N (median;
  IQR 12.9-14.3 N; the tool weight along sensor -z), while `observation.force.base` has 0.43 N
  (IQR 0.33-0.56 N; 7 recordings above 1 N, 5 of them only in the last idle segment and 3 of
  those the README's "ft_invalid_last_action" files; 2025-01-13-18-06-08 and
  2025-01-13-18-16-38 show the same signature and are not in the README table). Neither key
  separates failed from successful segments by force magnitude (per-segment p99 |F| AUROC 0.60
  raw vs 0.58 compensated, p50 0.52 vs 0.55; pooled p99 22.8 vs 23.6 N raw, 11.4 vs 10.7 N
  compensated), so the adapter reads force from `.base`: |F| then measures contact instead of
  |gravity + contact| (a 5 N lateral contact changes the raw magnitude by 0.9 N, an axial one by
  5 N) and the normalization scale is not degenerate: grasp-frame median |F| minus free-space
  median |F| is -0.08 N (median; <= 0 in 77 of 149 recordings) with the raw key and 0.26 N
  (<= 0 in 7 of 149) with the compensated one, so `normalize.fit` cannot use the raw key at
  all. This is the opposite preference to the raw adapter (`measured_force` first). Torque stays on
  `observation.torque`: `observation.torque.base` carries standing offsets of 5-7.4 N m in 26
  recordings (21 of them 2025-01-09) where the sensor torque is 0.15 N m, so it is not a
  compensated wrist torque; torque feeds no feature, only the sensor-health check. Measured
  2026-10-22 with `scripts/measure_reassemble_force_keys.py`; `meta["force_keys"]` records the
  columns read.
- RH20T community ports carry no fingertip tactile in any config, so RH20T feeds the wrist
  channel only; `meta.rating` ranges -1 to 9. Licenses are per scene (1-5 CC-BY-SA-4.0,
  6-10 CC-BY-NC-4.0). The local download (`data/colab/fetch_rh20t.py`) holds meta, data and two
  of the 9 (cfg3) or 10 (cfg5) cameras per config, so `configs/sources/rh20t.yaml` sets
  `cameras:` (a `SourceConfig` field added 2026-10-22) to those four keys in role order; the
  first one a config has becomes the scene view (cfg3: cam_104122062295, external; cfg5:
  cam_036422060215, external and the port's first declared camera) and the in-hand camera is
  `extra_<serial>`. Without it the adapter referenced 7-8 missing mp4 streams per episode and
  every episode failed integrity. The 5 cfg5 scene files the fetcher skipped
  (`cam_036422060215` file-008/011/036/039/048) make their episodes fail integrity by design.
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
| `sources/phantom.py` | Three real revision-pinned episodes now checked: published streams use `*.zarr` v2 groups and JPEG VLenBytes metadata needs a read-only numeric-fill compatibility shim. Newer v3 exports remain untested. All three pass sensor health after fixing duplicated-max saturation; source verification fails median sync and awaits spot review. |
| `sources/sharpadex.py` | Index of the right-index MCP joint when `info.json` has no names (`RIGHT_INDEX_MCP_FALLBACK_INDEX = 40`, a placeholder); polarity of that joint (larger = more closed?); exact schema of `meta/subtasks.parquet`; force units (not documented). |
| `sources/rh20t.py` | Measured 2026-10-22 on the cfg3 (798 episodes) and cfg5 (1,225) ports: `observation.robot_ft` is RAW on cfg3 (per-episode first-second median \|F\| 30 N, p10-p90 20-43 N; peak median 102 N; within-episode std 14 N: tool weight plus arm dynamics, untared) and a ~4.7 N estimated wrench on cfg5 (peak median 8 N, std 1.3 N). Force/closure correlation is ~0 in both (median -0.03 and +0.02), so force cannot confirm the gripper polarity; the gripper value starts at ~0 of the stats max in every episode (cfg3 0.013, cfg5 0.000) and rises (max median cfg3 0.47, cfg5 1.0), which is consistent with larger = closed only if episodes start open (assumed, unverified from video). cfg3 never exceeds 0.48, so `grasp_g` 0.6 selects no cfg3 frame. Ratings cluster at 7-9 (cfg3 627 of 798, cfg5 1,022 of 1,225; rating 1: 103 and 31); their semantics remain undocumented. |
| `sources/reassemble.py` | Exact names of the per-sensor timestamp keys beyond "one per sensor"; units and open/closed polarity of `gripper_positions` (closure = 1 - w / max w assumes larger width = more open). |
| `sources/reassemble_lerobot.py` | Frame of `observation.force.base` (base or sensor; only the magnitude is used, which is frame-invariant); meaning of `observation.torque.base` (5-7.4 N m standing offsets in 26 recordings); gripper polarity (closure = 1 - w / stats max assumes larger width = more open); why `meta/splits.json` leaves 2025-01-10-16-17-40 `unassigned`; which hand-camera frames the port black-filled. |
| `sources/airoa.py` | Wrist wrench key (heuristic: `observation.wrist.wrench`, or first `ft`/`wrench` feature); order of the 100-sample wrench history (`WRENCH_HISTORY_ROW = -1`); gripper polarity; where the session date comes from (uuid, then version, else `undated`); per-primitive O/X and failure-cause schema (`native_labels()` is empty and outcome comes from `task_success`). |
| `sources/hrdexdb.py` | Which camera serial is a sensible scene view (first sorted serial used); video-to-sensor clock sync (assumes first video frame = earliest sensor sample); `inspire_f1` tactile layout (load raises NotImplementedError, as do robotiq_2f85 and human); per-finger semantics and units; the hand-closure proxy from `position.npy` assumes larger = more closed. |
| `sources/softvtbench.py` | DSR tolerance per object (not in the HDF5); video/HDF5 frame-count equality; Safety Success vs DSR naming across sources. |
| `sources/baxter.py` | Real FSR units and range (sensor identity is an open spec question); whether the writer can provide `video_t0_s`. |

Decision needed for the owner: none of these can be resolved without data. The acceptance
step is one real episode per source (spec milestone 5).

## 6. Evaluation grid

### Released checkpoint runs the force modes too (changed 2026-10-22)

Spec and the first `configs/eval/grid.yaml`: the released checkpoint and the control run are
force-free models, so only their `omitted` row is meaningful (`checkpoint_modes` capped both at
`[omitted]`). The 2026-10-22 frozen-model probe on Colab (released Flora-Rewarder, code force
format, REASSEMBLE sample) measured balanced accuracy 0.613 with the block omitted, 0.645 with
the real block and 0.585 with a shuffled block (McNemar p = 0.014 for real vs omitted), so the
released model already reads the block. Its `real` and `shuffled` rows on `phantom_val` and
`reassemble_test` are therefore the no-training baseline the force adapter must beat, and
`checkpoint_modes.released` is now `[omitted, real, shuffled]`; the control run stays at
`[omitted]`, and omitted-only benchmarks (`roboreward_500`) still cap every checkpoint at
`omitted` through `harness.OMITTED_ONLY`. `tests/unit/test_config.py` pins the expanded grid.
The sanity expectation in the README ("the force model's failure recall must drop under
shuffled") now has a released-model counterpart: its shuffled row should sit below its real row,
as in the probe. Provenance of the probe numbers: the coordinator's note of 2026-10-22; the
Colab run artifacts are not in this repository.

## 7. Segment-level REASSEMBLE episodes for the failure judge (added 2026-10-22)

Spec and the recording-level adapter: one REASSEMBLE episode is a whole recording (5-10 min,
many actions) with one outcome label. 131 of 149 recordings contain a failed segment, so the
recording-level outcome is "failure" almost everywhere (the 2026-10-22 `train` manifest holds 6
REASSEMBLE outcome rows, all score 1), which cannot train or evaluate a success/failure judge.
Deviation: `SourceConfig.episode_unit: recording | segment` (default `recording`; the
`reassemble_lerobot` adapter honours it, and `configs/sources/reassemble_seg.yaml` registers the
segment view as its own source `reassemble_seg`, same root). In segment mode every qualifying
segment is an Episode on its own clock: id `<episode_index>_seg<segment.index:03d>`, task = the
segment text, cameras = the recording's mp4 with `video_from_s` advanced by the segment's first
frame and `n_frames` = its frame count (the readers already index concatenated files that way),
force / gripper / times = the recording's slices, one outcome label 5 (success) / 1 (failure)
from `segment.success` (constant within every segment of the port, checked over all 1,433,406
frames), `meta["prompt_kinds"] = ["outcome"]` so neither dense windows nor programmatic event
labels are built (`verify.windowing.prompt_kinds`). Splits follow the port directly (test
recordings -> test, every other recording -> train; no val hashing), so no recording is train
in one view and held out in the other.

Selection is the E0b protocol of `data/colab/e0b_reassemble_force.py` (`SegmentSelect` in the
config): a segment qualifies when its task is not "No action." and its last frame is at least
1.0 s after its first; per split every failed segment plus up to 2x as many successes after a
`random.Random(0)` shuffle of the successes in recording-then-segment order, the test split
capped at 450 in total. Measured on the local port 2026-10-22: test 37 recordings, 1,110
qualifying segments, 133 failures, 399 clips; the 399 ids equal the E0b Colab set in the same
order (the script's `groupby("segment.index")` and the adapter's frame-order runs agree: the
only out-of-order run, a one-frame segment in train recording 78, is excluded by both). Train:
111 recordings, 3,432 segments, 382 failures, 1,146 episodes. Differences from E0b that remain:
(1) the force block is the repo's code format rendered with the recording-level pooled
constants (`data/verified/reassemble_lerobot/norm.json`, group "all": baseline 0.46 N, grasp
2.50 N) and the per-recording sync shift, where E0b normalised per recording by 5th/60th
percentiles of the raw `observation.force` magnitude with no shift; (2) the prompt is the repo's
FLORA outcome prompt with rationale, where E0b used NVIDIA's `terminal_reward` batch prompt.
The request to make `norm_group` the recording id was NOT followed: the recording-level fit is a
single pooled group, not per recording, and segments cannot be fitted on their own (their first
0.5 s is mid-task, not free space), so `norm_group` stays "all" and `meta["recording"]` is
there for a later per-recording grouping. **Decision needed** if a per-recording scale is
wanted: it needs a recording-level fit with 149 single-episode groups, 7 of which have a
non-positive grasp-minus-baseline scale with the compensated key (sources docstring).

Verification cost: `SourceConfig.verify_from: <parent source>` makes `run_verify` reuse the
parent's `checks.parquet` and `norm.json` instead of decoding 14 GB of mp4 again: integrity,
sensor_health, sync, normalization and overlap are copied per parent recording
(`verify.INHERITED_CHECKS`, `inherit_results`; the normalization result every child gets is the
parent's pooled stats, so the pooled verdict is the parent's whatever the number of segments per
recording), label_sanity, spot_check, prompt_diff, holdout and license run on the segments, no
frame hashes are computed (`frame_hashes` is empty in the rows), and the spot strips use a
reader that trusts the declared frame count. Measured 2026-10-22 on the 1,545 segments:
31.06 s wall (46.1 s user) against 47 min for the 149 recordings. `inherited_drops: apply |
record` (default `apply`) decides whether a parent `drop_episode` / `drop_channel` drops the
child; `reassemble_seg` uses `record`, which keeps the child and stores the parent failure with
action `keep` (visible in `report.md` and `checks.parquet`, 593 segments with an inherited sync
drop and 146 with an inherited sensor-health drop; their force is rendered unshifted, as E0b
did), because applying the drops would leave 185 of the 399 test clips (62 failures) and the
judge could not be compared with the E0b baseline. Segment-weighted median |lag| is 0.200 s,
under the 0.25 s gate. **Decision needed / flag**: the spot review of the 20 segment strips
(20/20) was done by the agent at coarse level (`spot/index.json`, `spot/review.json`), not by a
human; and whether a training set that keeps the 54 lag-dropped recordings is acceptable for
the force-conditioned judge (`configs/train/judge.yaml`, real block, 3 epochs) against its
control (`judge_control.yaml`, omitted).

Manifests: `tf manifest merge` gained `--prompt-kind` (`merge_manifests(prompt_kinds=...)`).
`train_judge` = phantom train outcome rows (87, all score 5 by origin "source") + reassemble_seg
train (1,146: 382 score 1, 764 score 5), 1,233 rows; `reassemble_seg_test` = the 399 test
segments (133 score 1, 266 score 5), benchmark `reassemble_seg_test` (prompts `[outcome]`,
modes omitted / real / shuffled; `data/runs/judge` and `data/runs/judge_control` added to the
grid, the control capped at omitted). The frozen holdout, `train.parquet`, `phantom_val` and
`reassemble_test` are untouched.

## 8. Held-out PHANTOM failure split for the judge (added 2026-10-22)

Spec: the judge is evaluated on the sources' own held-out splits. HapticWAM's split, which
`phantom.parquet` follows, puts every failure demo (the 48 episode ids containing `_fail_`,
outcome target `Score: 1`) in train, and `phantom_val` (40 episodes) holds successes only, so
there is no held-out PHANTOM failure on which a success/failure judge can be scored;
`reassemble_seg_test` was the only failure benchmark.
Deviation: `scripts/make_phantom_fail_split.py` (seed 0, `--n-fail 20 --n-succ 20`) removes
20 failure and 20 success PHANTOM episodes from the judge training set and makes their outcome
rows a test benchmark. Candidates are the train-split PHANTOM episodes whose outcome row is in
`train_judge` (43 failures, 44 successes; 5 failure demos and 4 train successes have no
outcome row and stay where they are). The choice is `random.Random(0)` over id-sorted candidates,
stratified by the object named in the episode id with the quota spread as evenly as the
candidates allow: failures Carton 6, egg 6, waffles 6, whiteboard 2 (whiteboard has only two
failure demos with an outcome row), successes 5 per object. Outputs, both with a
`.shifts.json` sidecar copied from the inputs and keyed `source/episode_id` like
`tf manifest merge`:

* `data/manifest/phantom_fail_test.parquet`: the 40 outcome rows with `split` = `test`; window
  ids and every other column unchanged, same parquet schema. Benchmark `phantom_fail_test`
  (prompts `[outcome]`, metrics balanced_acc / failure_recall / macro_f1 / bad_action_f1 / mae,
  modes omitted / real / shuffled). The eval harness rebuilt all 40 windows from the ids
  (20 `Score: 1`, 20 `Score: 5`; 32 episodes carry a sync shift).
* `data/manifest/train_judge_pf.parquet`: `train_judge` minus every row of the 40 episodes,
  1,193 rows (phantom 47 = 23 failures + 24 successes, reassemble_seg 1,146). The dry run
  counts 405 failed outcome samples (382 reassemble_seg + 23 phantom) and 788 successes, so
  `configs/train/judge_pf.yaml` / `judge_pf_control.yaml` (judge / judge_control on this
  manifest) set `failure_weight` 1.95 = 788 / 405; `data/runs/judge_pf` and
  `data/runs/judge_pf_control` are in the grid, the control capped at omitted.

The rows of the 40 episodes stay `train` in `holdout_ids.json`, `phantom.parquet`,
`train_judge.parquet` and `train.parquet`, which are untouched: a model trained on
`train_judge` or `train` has seen the `phantom_fail_test` episodes, so only `judge_pf` /
`judge_pf_control` (and the released checkpoint) are comparable on it. The split column is not
a training filter for the test file: `train.dataset.build_samples` keeps only rows whose split
is in its `splits` argument (default `("train",)`, which `sft.train` does not override), and the
eval harness filters rows by prompt kind only, so `phantom_fail_test` cannot leak into a run
that names it by mistake, and a benchmark manifest needs no split filtering. The script is
deterministic: the same inputs and arguments give byte-identical parquet and sidecar files.

## 9. Force-only classifier line in the force block (judge_fp, 2026-10-22)

The spec's force block holds only per-frame lines (plus our summary line). judge_fp adds one more
line under the summary, `Force-only classifier P(success)=0.xx`, from a gradient-boosted tree
ensemble over 31 label-free force and gripper statistics (`tactile_flora/force/classifier.py`,
taken unchanged from origin/main a72f6a3, where it was built for HRDexDB). It is fitted by
`scripts/fit_force_classifier.py` on the 1,146 reassemble_seg rows of train_judge_pf with
5 folds grouped by recording; training episodes see their out-of-fold score (OOF AUROC 0.869),
test episodes the full model (reassemble_seg_test AUROC 0.815 for failure, balanced accuracy
0.705 at P(success) 0.5). Hyperparameters are fit_cv's defaults, fixed before the test set was
scored. The probability is computed from the features being rendered, so under the shuffled
ablation it describes the borrowed clip, and the line is absent under omitted. Only episodes of
the scorer's source (reassemble_seg) get the line; PHANTOM blocks are unchanged.
Measured before training (same 399 test clips, 133 failures): the gripper statistics alone reach
AUROC 0.728 and the force statistics alone 0.739 [0.676, 0.789]; failed segments have about 3x
the normalized peak force of successful ones (median 2.25 vs 0.72). Configs:
configs/force.code_fp.yaml, configs/train/judge_fp.yaml.

## 10. ESP32 boards over WiFi instead of one Teensy over USB (Baxter readout, 2026-10-22)

The spec reads all four pads with one Teensy 4.0 on USB serial. The club has Adafruit HUZZAH32
(ESP32) Feathers, so `tactile_flora/baxter/firmware/esp32_fsr` reads them instead: one board per
gripper, mounted after the last wrist joint, battery powered, streaming over UDP. Over USB it
speaks the Teensy line protocol unchanged; over UDP `tactile_flora/baxter/udp_boards.py` merges
the boards into the same four-channel lines, so `fsr_node`, `calibrate` and the recorder are
unchanged apart from three additions. `FsrNode` takes stamps from a source that provides
`to_host_time`, so each sample is stamped by its board's clock fit only. `ClockFit` fits the
lower envelope of arrival times (the earliest pair per 0.25 s bin) instead of least squares
over every pair, and accepts `min_span_s`: in simulation, least squares followed bursts of late
WiFi datagrams to a ~1e-3 slope error. `calibrate.read_counts` flushes buffered lines before
each weight. ESP32 counts are made ratiometric in firmware (pin voltage over the measured 3V3
rail), so `r_ref_ohm` and `adc_max_counts` mean what they meant for the Teensy. Oversampling is
8 instead of 16 (the ESP32 ADC read is slower). Sensor: ASIN B0FGD267GK, rated 20 g to 2 kg,
below the spec's 50 N class.
