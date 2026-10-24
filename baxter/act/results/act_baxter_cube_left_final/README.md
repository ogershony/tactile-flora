---
license: other
library_name: lerobot
pipeline_tag: robotics
tags: [robotics, lerobot, act, baxter]
datasets: [OG7407/baxter-cube-pick-left]
---
# ACT policy: Baxter left-arm cube pick (checkpoint step 50000, tag act_baxter_cube_left_final)
Trained with lerobot 0.6.1 on OG7407/baxter-cube-pick-left (50 eps, 15 fps; episodes 45-49 held out).
chunk_size=30, n_action_steps=15, resnet18, VAE on, global batch 65 (5x RTX 2080 Ti, fp16), lr 1e-5.
Inputs at 15 Hz: observation.state [8], observation.images.head and .wrist (360x640 RGB). Output: action [8]
(left_s0..left_w2 rad, left_gripper 0-1). Normalization stats live in the processor files next to the weights.
Load: ACTPolicy.from_pretrained('OG7407/baxter-act-cube-left') (or the pretrained_model/ folder) + make_pre_post_processors(..., pretrained_path=...).
Held-out action L1 per checkpoint: see heldout_l1.json; curves: loss_curves.png.
