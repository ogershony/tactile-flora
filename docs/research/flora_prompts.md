# FLORA Rewarder prompts (verbatim from the released code)

Source: https://huggingface.co/nvidia/Flora-Rewarder/blob/main/batch_utils/core.py (and README in batch_utils/). No arXiv id/HTML exists that I could find; the paper PDF (https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf, Appendix G.2-G.4) contains the same prompts and they match this code exactly (paper shows the dense user prompt example with t=353.000 to t=368.733 and target e.g. `Worst: 1 / Quality: 11111111111133333322222222222233 / Rationales: 1-12 ... / 19-30 ...`; outcome example target `Rationale: ...\n\nScore: 2`). The Flora-Rewarder model card itself contains no prompt text. These are the actual prompts used by the shipped inference scripts. Dense system prompt below is the f-string rendered with DENSE_FRAMES=64, DENSE_FPS=4, DENSE_GROUPS=32.

Message layout: [system: <system text>] [user: content list = text-before-<video>, {"type":"video"}, text-after]. Video = one collage video (views tiled: 1 as is, 2/3 side by side, 4 as 2x2; order top, front/bottom, left, right for 4; left,center,right for 3).

## Outcome ("terminal") mode

System = `_TERMINAL_SYSTEM + "\n" + suffix`

_TERMINAL_SYSTEM:
```
You are a terminal-success reward model for robot trajectories.
Given the task instruction and the complete trajectory video, predict a score from 1 to 5.
A complete failure is 1. A complete success is 5.
Judge only what is visible in the trajectory and follow the requested output format exactly.
```
suffix (no rationale, default):
```
Do not provide a rationale. Return exactly:
Score: <1-5>
```
suffix (--with-rationale):
```
Provide one short, compact rationale followed by the score. Return exactly:
Rationale: <rationale>

Score: <1-5>
```
User template (task whitespace-collapsed; `<video>` marks where the video content item goes):
```
Task: {task}

Complete trajectory video:
<video>

Evaluate the terminal success of the task.
```
Decoding: greedy, temperature 0, structured-output regex `Score: [1-5]` (or `Rationale: [^\n]{1,512}\n\nScore: [1-5]`), max_tokens 12 (160 with rationale), top-20 logprobs; expected_score = prob-weighted mean of P(1..5); progress=(E-1)/4. Frames: min(64, floor(duration_s*10)) rounded down to even, >=2, uniformly sampled over whole episode; thinking disabled in chat template.

## Dense ("process") mode

System:
```
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
<one line per contiguous group range scored 1 or 2, in time order>
```
User template (window.start_s / end_s are the absolute episode times of the first/last sampled frame, 3 decimals):
```
Task: {task}

Synchronized multiview clip:
<video>

The clip contains 64 sampled frame positions at 4 FPS (16 seconds nominal span) and 32 visible temporal groups of 2 frames each. The sampled positions run from t={start_s:.3f} to t={end_s:.3f} seconds of the episode. Episode boundaries repeat the nearest available frame.
Predict Worst, all 32 group scores, and rationales for non-normal runs.
```
Decoding: greedy, regex `Worst: [123]\nQuality:\n[123]{32}\nRationales:(?:\n<g>-<g> [^\n]{1,240})*`, max_tokens 640. 16 s windows (64 frames @4 FPS, 8 history frames before anchor), stride 10 s, last 4 s of each non-final window dropped when stitching.

## Rubric mode (bonus)
System: `You are a helpful assistant.` User: one video item per camera (fps 2, max 166 frames) then text. Header: "You are an expert evaluator of robot manipulation. You are shown {one video|two time-synchronized videos|...} of a {robot} performing one task: Video 1 is {cam1}, Video 2 is ..., and Video N is {camN}.\n\n" + instructions "Score how well the robot performed, using the rubric below. The rubric lists the task's steps; each step has its own maximum number of points and an explicit list of the point values you may award for it. For every step, award exactly one of the values listed for that step -- not a value in between." then `Task: ...`, `Instruction to the robot: "..."`, numbered rubric lines with `award one of: label = 0.00 | ...`, footer "Respond with your scores as a JSON array ... ANSWER: [..]". Thinking enabled; 9 samples, medoid. See core.py `rubric_prompt` for exact text.
