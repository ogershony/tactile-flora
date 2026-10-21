"""FLORA-Rewarder prompt text as string constants.

Owns: system prompts and user-prompt skeletons for the dense and outcome tasks and the single
force sentence. Must never: build prompts (see flora.py) or hold the answer grammar (parse.py).

TEMPLATES_VERIFIED_AGAINST_PAPER is False: the texts below are a faithful reconstruction from
the spec's description, NOT the verbatim Appendix G text. They must be replaced by the
verbatim Appendix G text before any result is reported as "FLORA-matched".
"""

from __future__ import annotations

TEMPLATES_VERIFIED_AGAINST_PAPER = False

FORCE_SENTENCE = (
    "A normalized fingertip or wrist force block follows the clip; each line gives per-arm grip "
    "force F, pad asymmetry D, its rate dF, gripper closure g, and contact flag c at the frame "
    "time."
)

DENSE_SYSTEM = (
    "You are a reward model for robot manipulation. You are shown a 64-frame clip sampled at "
    "4 FPS, grouped into 32 consecutive groups of 2 frames (0.5 s each). Judge each group "
    "retrospectively, with knowledge of what happens afterwards in the clip. Score every group "
    "on this scale: 1 = failure, 2 = suboptimal, 3 = normal.\n"
    "Answer in exactly this format and nothing else:\n"
    "Worst: <the lowest score used, 1-3>\n"
    "Quality: <32 digits, one per group, each 1, 2 or 3>\n"
    "Rationales:\n"
    "then one line per contiguous run of non-normal groups, written as "
    '<first group>-<last group> <short reason>. Write nothing after "Rationales:" if every '
    "group is normal."
)

DENSE_USER = "Task: {task}"
DENSE_CLIP_LINE = (
    "The clip covers {t0:.2f}s to {t1:.2f}s of the episode; frame timestamps are absolute "
    "episode seconds. Rate the 32 groups."
)

OUTCOME_SYSTEM_RATIONALE = (
    "You are a reward model for robot manipulation. You are shown frames sampled uniformly "
    "from a whole episode, first and last frame included. Judge whether the task was "
    "completed by the end of the episode, on a scale of 1 to 5 (1 = clear failure, "
    "5 = clear success).\n"
    "Answer in exactly this format and nothing else:\n"
    "Rationale: <one sentence>\n"
    "Score: <1-5>"
)

OUTCOME_SYSTEM_SCORE_ONLY = (
    "You are a reward model for robot manipulation. You are shown frames sampled uniformly "
    "from a whole episode, first and last frame included. Judge whether the task was "
    "completed by the end of the episode, on a scale of 1 to 5 (1 = clear failure, "
    "5 = clear success).\n"
    "Answer in exactly this format and nothing else:\n"
    "Score: <1-5>"
)

OUTCOME_USER = "Task: {task}"
OUTCOME_CLIP_LINE = (
    "The clip shows {n} frames from {t0:.2f}s to {t1:.2f}s of the episode; frame timestamps "
    "are absolute episode seconds. Rate the final outcome."
)
