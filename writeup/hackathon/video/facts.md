# Tactile FLORA hackathon: fact sheet (single source for video, writeup, deck)

Every number below has a source. Do not add numbers that are not here.

## Event
- GPT-6 Astra Robotics Hardware Hackathon (Saturday Robotics x Robotics Center, SF). Demo 2026-10-24 14:00-17:00 PT.
- Rubric: technical depth 30, novelty 25, impact 25, demo 20.
- Repo: https://github.com/ogershony/tactile-flora (public). Branches: main (v3 + writeup/comparison.md + 26 rendered clips), plan-v4 (v4 locked test results), baxter-fsr (Baxter FSR drivers).

## Problem
- Learned robot policies fail silently; success is still judged by people watching video.
- Video-only judges cannot see contact: the gripper hides the object at the moment that matters.
- Our own Baxter ACT policy (OG7407/baxter-act-cube-left, 30k steps): 10 live rollouts on 2026-10-24 (dataset OG7407/baxter-cube-pick-left-rollouts): 1 success, 9 failures. 8 of the 9 failures were safety aborts: "chunk N starts 0.25-0.30 rad from the arm (> --max-jump 0.25)". 1 hit the 20 s time limit.

## Hardware (branch baxter-fsr, tactile_flora/baxter)
- Baxter (ROS1), left arm 7 joints (s0 s1 e0 e1 w0 w1 w2) + gripper. Head camera + left wrist camera, 640x360.
- 4 force-sensitive resistor (FSR) pads on the gripper fingers (channels L_a, L_b, R_a, R_b), read at 200 Hz by an ESP32 (WiFi UDP) or over USB serial (CP2104).
- Rollout tactile, raw ADC counts: no-contact baseline about 92; the successful grasp (ep 0) peaks at 425 / 482 on the two left-finger pads.
- Pads are not yet force-calibrated (meta: "calibrated": false); numbers are counts, not newtons.

## Demonstrations (OG7407/baxter-cube-pick-left-v0)
- 18 kinesthetic demos (a person guides the arm), task "pick up the red cube", 1,622 frames at 10 Hz (2.7 min total), 59-178 frames per demo.
- State and action: 8-D (7 joint positions in rad + gripper 0.24-0.96). Action is the next joint target.

## Reward model: Tactile FLORA (branch plan-v4, writeup/v4-results.md)
- Base: NVIDIA FLORA-Rewarder (Qwen3.5 27B video reward model, released 1-5 success score).
- Our fine-tune: LoRA r16, 1 epoch, 1,811 clips (618 HRDexDB dexterous-hand trials + 1,146 REASSEMBLE segments + 47 PHANTOM episodes, 598 failures), at FLORA's native input (2 views x 384x288, 64 frames). Matched control trained identically without force.
- Locked HRDexDB test (428 trials, 151 failures, 59 objects never trained on):
  - AUROC: released 0.783 -> force adapter 0.956, control 0.964. Primary +0.173 [+0.131, +0.215], Holm p 1.5e-11.
  - Failure recall at the model's own threshold: 0.344 -> 0.841 (force) / 0.848 (control), success recall 0.982 / 0.978.
  - Failure recall at 90% success recall: 0.636 -> 0.881.
- REASSEMBLE test (399 segments, 133 failures): 0.648 -> 0.919 (force) / 0.940 (control), +0.271, Holm p 1.5e-20.
- PHANTOM (40 episodes): not significant (+0.073, Holm 0.20).
- RoboRewardBench (500 general clips, no force): ranking kept (AUROC 0.934 -> 0.92) but graded 1-5 scale lost (MAE 0.40 -> 0.97).
- Force finding: once the video is native resolution, force text adds nothing (-0.008 [-0.023, +0.004] HRDexDB). In v3 at 96 px thumbnails, force added +0.043 AUROC (p about 0.02). Interpretation: touch matters where vision is starved.
- Force-only classifier (no video, runs on the 200 Hz signal): AUROC 0.751 HRDexDB, 0.815 REASSEMBLE.
- Released FLORA + force text, no training: 0.783 -> 0.838 (+0.055, p 5.5e-4).
- Compute: Colab A100 80 GB for v4; seed-0 results; a seed-1 replicate was running.

## Policy: Pi0.5 on Baxter (trained 2026-10-24, code in vla/pi05_baxter)
- Base: Physical Intelligence pi0.5 (lerobot/pi05_base; PaliGemma 3B VLM + 300M flow-matching action expert, trained with knowledge insulation).
- Fine-tune: LeRobot 0.6.1, laptop RTX 5090 (24 GB, 95 W cap). VLM frozen; LoRA r32 on every linear layer of the action expert + action/time projections. Relative joint targets (action - current state, gripper absolute), 10-step chunks (1.0 s at 10 Hz), quantile normalization recomputed on relative chunks.
- Why relative: 8 of 9 ACT failures were chunks starting > 0.25 rad from the arm; a relative chunk starts at the arm by construction.
- Split: 16 demos train, demos 5 and 14 held out.
- Results: PENDING (fill from runs/*/eval.jsonl): held-out joint MAE, step-0 MAE, 1 s-ahead MAE, jump rate, vs hold baseline.

## Team
- Oren Gershony, Kenan Blair, with the robotics club.
