# Deviations from the spec

Decisions taken during implementation where the spec was inconsistent with reality or
with itself. Each entry says what the spec asked for, what was done, and why. Revisit
the ones marked **decision needed**.

## Force block token budget (decision needed)

Spec: force block under 1,500 tokens per 64-frame window, one line per sampled frame in
the exact format given. Measured with the Qwen3 tokenizer (digits tokenize one per
character): one-arm pad line 40 tokens, one-arm wrist line 34, two-arm pad line 68.
Sixty-four lines therefore cost 2,560 / 2,176 / 4,352 tokens. The line format is kept
byte-exact and `render.max_tokens` in `configs/force.yaml` is 2,800, which admits every
single-arm window (the evaluation case) and rejects two-arm windows. Options if the
budget matters: one line per 0.5 s group instead of per frame (halves it), or drop `D`
and `g` from the line.

## FLORA answer grammar follows the released code, not the spec sketch

Spec sketch: `Worst: <1-3> then 32 digits then Rationales:` and `Rationale: ... Score: <1-5>`.
Released `batch_utils/core.py` and paper Appendix G (see `docs/research/flora_prompts.md`):
dense answer is `Worst: d\nQuality:\n<32 digits>\nRationales:` followed by
`\n<first>-<last> <text 1..240 chars>` per run; outcome rationale answer is
`Rationale: <text>\n\nScore: <d>` with a blank line. The verbatim grammar wins because the
spec says to reproduce the released prompts verbatim.

## Dataset facts that differ from the spec (see docs/research/datasets.md)

- PHANTOM is HapticWAM (`armteam/hapticwam-teleop-raw`, CC-BY-4.0). Layout is per-episode
  zarr plus JSON manifests, not LeRobot. Scene video is per-frame JPEG bytes in zarr at
  15 Hz. Fingertip force is a 6-D wrench per pad at about 6 Hz, not DM-Tac raw.
- REASSEMBLE has no anomaly spans; only a per-segment `success` flag and a known-issues
  table. Dense failure spans come from failed segments, not annotation text.
- RH20T community ports carry no fingertip tactile in any config; `meta.rating` ranges
  -1 to 9 (-1 presumably unrated). Whether `observation.robot_ft` is tared is unverified.
- AIRoA MoMa is gated (auto-approve) and LeRobot v3-style; per-primitive O/X and failure
  cause fields are unverified. The wrist wrench key name is unverified.
- RoboRewardBench is the test split of `teetone/RoboReward` (2,831 rows), not a FLORA
  release.
- SoftVTBench has no DSR column; it must be derived from `fem_*` channels and a
  per-object tolerance that is not in the HDF5.
