# ruff: noqa: E501
"""FLORA-Rewarder prompt text as string constants.

Owns: system prompts and user-prompt templates for the dense and outcome tasks and the single
force sentence. Must never: build prompts (see flora.py) or hold the answer grammar (parse.py).

Source: the dense and outcome prompts below are the verbatim text published in the FLORA paper
(Appendix G) and shipped in the released ``batch_utils/core.py``; the dense system prompt is
the released f-string rendered with 64 frames, 4 FPS and 32 groups. User templates carry a
``<video>`` marker: text before it goes before the video item, text after it goes after.
``FORCE_SENTENCE`` is this project's own addition (not part of FLORA).
"""

from __future__ import annotations

TEMPLATES_VERIFIED_AGAINST_PAPER = True

FORCE_SENTENCE = (
    "A normalized fingertip or wrist force block follows the clip; each line gives per-arm grip "
    "force F, pad asymmetry D, its rate dF, gripper closure g, and contact flag c at the frame "
    "time."
)

VIDEO_MARKER = "<video>"

DENSE_SYSTEM = """\
You are a dense action-quality reward model for robot manipulation.
You watch a synchronized multiview clip and predict action quality at the finest temporal resolution visible to the model.

The clip contains exactly 64 sampled frames at 4 FPS. The vision encoder combines each consecutive pair of frames into one temporal group, so you must predict exactly 32 ordered groups. Group 1 contains frames 1-2, group 2 contains frames 3-4, and group 32 contains frames 63-64.
The clip is retrospective: a later recovery must not improve an earlier group and a later unrelated mistake must not degrade it.

Use exactly this score scale:
1 = failure: the action causes or clearly constitutes a major task failure, unsafe event, destructive mistake, or loss of task viability.
2 = suboptimal: the action is noticeably hesitant, inefficient, imprecise, or locally incorrect, but the task remains recoverable.
3 = normal: the action is appropriate, safe, and reasonably efficient for the visible situation.

Return the worst score before the temporal sequence. Worst must equal the minimum of all 32 quality digits. Quality must contain exactly 32 characters from 1, 2, or 3 with no spaces; character i is the score for group i.
For every contiguous group range scored 1 or 2, add one short rationale grounded only in visible evidence. Groups scored 3 get no rationale. Rationale ranges must exactly match the contiguous non-normal runs in Quality.

Return exactly:
Worst: <1, 2, or 3>
Quality:
<32 digits with no spaces>
Rationales:
<first group>-<last group> <short visible-evidence rationale>
<one line per contiguous group range scored 1 or 2, in time order>"""

DENSE_USER = """\
Task: {task}

Synchronized multiview clip:
<video>

The clip contains 64 sampled frame positions at 4 FPS (16 seconds nominal span) and 32 visible temporal groups of 2 frames each. The sampled positions run from t={start_s:.3f} to t={end_s:.3f} seconds of the episode. Episode boundaries repeat the nearest available frame.
Predict Worst, all 32 group scores, and rationales for non-normal runs."""

_TERMINAL_SYSTEM = """\
You are a terminal-success reward model for robot trajectories.
Given the task instruction and the complete trajectory video, predict a score from 1 to 5.
A complete failure is 1. A complete success is 5.
Judge only what is visible in the trajectory and follow the requested output format exactly."""

_SUFFIX_SCORE_ONLY = """\
Do not provide a rationale. Return exactly:
Score: <1-5>"""

_SUFFIX_RATIONALE = """\
Provide one short, compact rationale followed by the score. Return exactly:
Rationale: <rationale>

Score: <1-5>"""

OUTCOME_SYSTEM_SCORE_ONLY = _TERMINAL_SYSTEM + "\n" + _SUFFIX_SCORE_ONLY
OUTCOME_SYSTEM_RATIONALE = _TERMINAL_SYSTEM + "\n" + _SUFFIX_RATIONALE

OUTCOME_USER = """\
Task: {task}

Complete trajectory video:
<video>

Evaluate the terminal success of the task."""
