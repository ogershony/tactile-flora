# Fine-tuned FLORA vs fine-tuned tactile FLORA on HRDexDB grasp outcomes

Date: 2026-10-22. Branch `hrdexdb-v3-force-representation`. Everything here is read from the
finished evaluation cells under `data/acceptance/hrdexdb-expanded-{v2,v3}/eval`; no model was
re-run to produce this write-up. Videos and poster frames are in `videos/`, per-clip numbers in
`clips.json`, and `make_videos.py` (in this folder) regenerates all of it.

## What was compared

Three models answer the same question on the same clips: "did this grasp succeed?" scored 1 to 5
with 4.0 as the success threshold. The task is HRDexDB's Inspire DFTP hand, 23-camera rig, one
fixed scene camera, success defined as lifted and held for 10 s.

| Model | Inputs | Training |
|---|---|---|
| Released FLORA (benchmark) | 64 video frames | none |
| Fine-tuned FLORA (control) | 64 video frames | LoRA on 350 HRDexDB clips, 4 epochs |
| Fine-tuned tactile FLORA (v3) | 64 video frames + a text block of tared fingertip force, a label-free summary line, and a force-only classifier probability | same recipe, same clips, same seed |

The control is the v2 control adapter; its training inputs are identical to what a v3 control
would have used (the v3 control retrain was preempted). Held-out sets are object-disjoint from
training: 136 dev clips over 16 objects, 115 test clips over 16 different objects.

## Headline numbers

Accuracy at the 4.0 threshold. AUROC is the probability a random success is scored above a
random failure.

| Split | Model | Accuracy | AUROC | MAE (0-100) |
|---|---|---:|---:|---:|
| dev (136) | released, video only | 66.2% | 0.614 | 150.0 |
| dev | released, video + force | 66.2% | 0.648 | 148.5 |
| dev | fine-tuned, video only | 82.4% | 0.899 | 70.6 |
| dev | **fine-tuned tactile** | **85.3%** | **0.926** | **52.9** |
| test (115) | released, video only | 67.0% | 0.593 | |
| test | released, video + force | 70.4% | 0.647 | |
| test | fine-tuned, video only | 69.6% | 0.733 | 121.7 |
| test | **fine-tuned tactile** | **72.2%** | **0.800** | **97.4** |

Pooled over all 251 held-out clips:

| Model | Accuracy | vs benchmark | AUROC |
|---|---:|---:|---:|
| majority class | 57.0% | | 0.500 |
| released FLORA | 66.5% | | |
| fine-tuned FLORA | 76.5% | +10.0 pts | 0.830 |
| fine-tuned tactile FLORA | 79.3% | +12.8 pts | 0.873 |

## Is it significant?

- **Ranking (AUROC) gain is significant when pooled.** Tactile minus control is +0.043 over 251
  clips, object-level bootstrap 95% interval +0.001 to +0.080, p about 0.02. Per split it is
  +0.026 on dev (not significant) and +0.067 on test (borderline, p about 0.04).
- **Accuracy gain is not.** The tactile model fixes 26 clips the control gets wrong and breaks 19
  the control gets right (exact McNemar p = 0.37). Three to four clips per split.
- **The tactile model really reads the force.** Feeding it another clip's force (shuffled donors)
  drops dev accuracy from 85.3% to 73.5% and it loses 25 paired clips while winning 9 (p = 0.009).
  In v2 that gap was three clips; the v3 representation fix is what made the block legible.
- **Force alone is weak.** With the video blanked, the tactile model scores 53.7% accuracy and
  0.775 AUROC on dev: better ranking than the released model, but it predicts failure almost
  everywhere. Every force-only probe tried lands at AUROC 0.76 to 0.82, which is the ceiling of
  what this sensor carries about outcome.
- **Force helps where video is unsure.** On the 101 clips where the control's expected score is
  within 0.75 of the threshold, the tactile model is 6 points more accurate; on the 150 confident
  clips, under 1 point.

## The videos

Every clip where the tactile model is right and the video-only model is wrong is rendered in
`videos/` (26 clips). Each mp4 shows the scene camera twice, once per model with that model's
prediction, above the tared fingertip-force trace the tactile model read (two most-loaded pads,
hand closure, the contact threshold as a dashed line, tick marks for the 64 frames both models
saw). The poster PNG is the frame at 60% of the clip. Clips the control wins (19) are listed in
`clips.json` under `control_wins` and can be rendered with `--losses`.

The pattern across the wins: the video-only model under-calls successes whose hold is visually
unremarkable (long, steady contact at above-median force), and over-calls failures where the hand
closes convincingly on camera but the force shows a weak grip, a re-grasp, or contact that ends
early.

### Clips where the tactile model is right and the video-only model is wrong (26)

Scores are expected score out of 5 (success at 4.0 or above). Force columns are read from the tactile model's input: peak total fingertip force in units of the median training grasp, seconds in contact, and when contact ended. Sorted by how much further the tactile model sits on the correct side.

| # | Split | Clip | Truth | Video-only FLORA | Tactile FLORA | Peak force | Contact | Last contact | Video |
|---:|---|---|---|---|---|---:|---:|---:|---|
| 1 | test | standing frame #4 | SUCCESS | 1.27 → FAILURE | **4.19 → SUCCESS** | 2.47x | 16.4 s | 23.7 s of 28.2 s | [mp4](videos/test_standing_frame_4_success.mp4) · [poster](videos/test_standing_frame_4_success.png) |
| 2 | dev | servingbowl small #3 | FAILURE | 4.66 → SUCCESS | **2.29 → FAILURE** | 1.44x | 6.7 s | 33.6 s of 39.0 s | [mp4](videos/dev_servingbowl_small_3_failure.mp4) · [poster](videos/dev_servingbowl_small_3_failure.png) |
| 3 | dev | cactus3 #6 | FAILURE | 4.19 → SUCCESS | **1.98 → FAILURE** | 1.83x | 7.3 s | 14.7 s of 20.2 s | [mp4](videos/dev_cactus3_6_failure.mp4) · [poster](videos/dev_cactus3_6_failure.png) |
| 4 | dev | hammer #11 | SUCCESS | 2.28 → FAILURE | **4.27 → SUCCESS** | 3.46x | 24.5 s | 34.9 s of 39.7 s | [mp4](videos/dev_hammer_11_success.mp4) · [poster](videos/dev_hammer_11_success.png) |
| 5 | test | black kitchen roll holder #9 | SUCCESS | 2.28 → FAILURE | **4.27 → SUCCESS** | 2.37x | 17.0 s | 31.8 s of 34.7 s | [mp4](videos/test_black_kitchen_roll_holder_9_success.mp4) · [poster](videos/test_black_kitchen_roll_holder_9_success.png) |
| 6 | dev | knife sharpener #5 | SUCCESS | 2.75 → FAILURE | **4.62 → SUCCESS** | 3.72x | 8.3 s | 23.3 s of 29.5 s | [mp4](videos/dev_knife_sharpener_5_success.mp4) · [poster](videos/dev_knife_sharpener_5_success.png) |
| 7 | test | standing frame #2 | SUCCESS | 2.75 → FAILURE | **4.47 → SUCCESS** | 2.7x | 14.6 s | 22.1 s of 25.5 s | [mp4](videos/test_standing_frame_2_success.mp4) · [poster](videos/test_standing_frame_2_success.png) |
| 8 | dev | screwdriver #3 | SUCCESS | 2.63 → FAILURE | **4.34 → SUCCESS** | 3.36x | 10.3 s | 32.0 s of 37.7 s | [mp4](videos/dev_screwdriver_3_success.mp4) · [poster](videos/dev_screwdriver_3_success.png) |
| 9 | test | balloon whisk #5 | SUCCESS | 2.75 → FAILURE | **4.40 → SUCCESS** | 1.44x | 28.9 s | 33.5 s of 37.5 s | [mp4](videos/test_balloon_whisk_5_success.mp4) · [poster](videos/test_balloon_whisk_5_success.png) |
| 10 | test | balloon whisk #7 | SUCCESS | 2.63 → FAILURE | **4.19 → SUCCESS** | 2.4x | 12.6 s | 22.1 s of 26.5 s | [mp4](videos/test_balloon_whisk_7_success.mp4) · [poster](videos/test_balloon_whisk_7_success.png) |
| 11 | test | black kitchen roll holder #4 | SUCCESS | 3.25 → FAILURE | **4.69 → SUCCESS** | 2.5x | 11.5 s | 25.1 s of 30.2 s | [mp4](videos/test_black_kitchen_roll_holder_4_success.mp4) · [poster](videos/test_black_kitchen_roll_holder_4_success.png) |
| 12 | test | organizer beige #2 | FAILURE | 4.27 → SUCCESS | **2.88 → FAILURE** | 0.55x | 8.6 s | 27.1 s of 34.4 s | [mp4](videos/test_organizer_beige_2_failure.mp4) · [poster](videos/test_organizer_beige_2_failure.png) |
| 13 | dev | potato masher #8 | FAILURE | 4.02 → SUCCESS | **2.63 → FAILURE** | 1.7x | 17.9 s | 23.6 s of 32.8 s | [mp4](videos/dev_potato_masher_8_failure.mp4) · [poster](videos/dev_potato_masher_8_failure.png) |
| 14 | dev | metal scoop small #10 | SUCCESS | 3.37 → FAILURE | **4.41 → SUCCESS** | 1.94x | 9.9 s | 24.2 s of 28.8 s | [mp4](videos/dev_metal_scoop_small_10_success.mp4) · [poster](videos/dev_metal_scoop_small_10_success.png) |
| 15 | test | bamboo box #2 | SUCCESS | 3.72 → FAILURE | **4.66 → SUCCESS** | 3.07x | 10.4 s | 20.9 s of 27.2 s | [mp4](videos/test_bamboo_box_2_success.mp4) · [poster](videos/test_bamboo_box_2_success.png) |
| 16 | test | wipe #0 | SUCCESS | 3.72 → FAILURE | **4.66 → SUCCESS** | 1.02x | 6.9 s | 13.8 s of 19.4 s | [mp4](videos/test_wipe_0_success.mp4) · [poster](videos/test_wipe_0_success.png) |
| 17 | dev | white soap dish #1 | SUCCESS | 3.49 → FAILURE | **4.40 → SUCCESS** | 3.12x | 5.4 s | 13.9 s of 19.9 s | [mp4](videos/dev_white_soap_dish_1_success.mp4) · [poster](videos/dev_white_soap_dish_1_success.png) |
| 18 | dev | cactus2 #15 | SUCCESS | 3.72 → FAILURE | **4.41 → SUCCESS** | 2.71x | 7.2 s | 15.4 s of 20.3 s | [mp4](videos/dev_cactus2_15_success.mp4) · [poster](videos/dev_cactus2_15_success.png) |
| 19 | test | organizer beige #8 | FAILURE | 4.27 → SUCCESS | **3.60 → FAILURE** | 2.25x | 5.7 s | 16.1 s of 21.1 s | [mp4](videos/test_organizer_beige_8_failure.mp4) · [poster](videos/test_organizer_beige_8_failure.png) |
| 20 | test | frog bowl #6 | SUCCESS | 3.92 → FAILURE | **4.52 → SUCCESS** | 0.75x | 11.5 s | 25.2 s of 29.3 s | [mp4](videos/test_frog_bowl_6_success.mp4) · [poster](videos/test_frog_bowl_6_success.png) |
| 21 | test | meat thermometer #0 | SUCCESS | 3.82 → FAILURE | **4.41 → SUCCESS** | 2.31x | 6.4 s | 17.3 s of 22.0 s | [mp4](videos/test_meat_thermometer_0_success.mp4) · [poster](videos/test_meat_thermometer_0_success.png) |
| 22 | test | black kitchen roll holder #11 | SUCCESS | 3.49 → FAILURE | **4.02 → SUCCESS** | 3.02x | 8.6 s | 17.5 s of 23.1 s | [mp4](videos/test_black_kitchen_roll_holder_11_success.mp4) · [poster](videos/test_black_kitchen_roll_holder_11_success.png) |
| 23 | dev | pastel blue cup #1 | SUCCESS | 3.71 → FAILURE | **4.11 → SUCCESS** | 2.9x | 8.1 s | 25.5 s of 28.8 s | [mp4](videos/dev_pastel_blue_cup_1_success.mp4) · [poster](videos/dev_pastel_blue_cup_1_success.png) |
| 24 | dev | hammer #13 | SUCCESS | 3.72 → FAILURE | **4.11 → SUCCESS** | 2.0x | 27.8 s | 41.3 s of 43.5 s | [mp4](videos/dev_hammer_13_success.mp4) · [poster](videos/dev_hammer_13_success.png) |
| 25 | dev | knife sharpener #1 | SUCCESS | 3.82 → FAILURE | **4.02 → SUCCESS** | 3.6x | 9.4 s | 19.0 s of 25.9 s | [mp4](videos/dev_knife_sharpener_1_success.mp4) · [poster](videos/dev_knife_sharpener_1_success.png) |
| 26 | test | clothes hanger #3 | FAILURE | 4.11 → SUCCESS | **3.92 → FAILURE** | 2.16x | 4.5 s | 16.6 s of 21.5 s | [mp4](videos/test_clothes_hanger_3_failure.mp4) · [poster](videos/test_clothes_hanger_3_failure.png) |

### Clips where the video-only model is right and the tactile model is wrong (19)

| # | Split | Clip | Truth | Video-only FLORA | Tactile FLORA | Peak force | Contact | Last contact |
|---:|---|---|---|---|---|---:|---:|---:|
| 1 | dev | metal scoop small #5 | FAILURE | **1.27 → FAILURE** | 4.11 → SUCCESS | 2.15x | 18.3 s | 29.2 s of 29.2 s |
| 2 | dev | screwdriver #2 | FAILURE | **2.18 → FAILURE** | 4.66 → SUCCESS | 4.1x | 12.5 s | 25.7 s of 29.4 s |
| 3 | dev | hammer #5 | FAILURE | **1.89 → FAILURE** | 4.10 → SUCCESS | 2.94x | 35.3 s | 73.3 s of 73.3 s |
| 4 | dev | icecream scoop #7 | SUCCESS | **4.47 → SUCCESS** | 2.88 → FAILURE | 1.92x | 6.5 s | 20.4 s of 25.5 s |
| 5 | test | frog bowl #5 | FAILURE | **3.12 → FAILURE** | 4.52 → SUCCESS | 0.76x | 12.8 s | 20.1 s of 24.4 s |
| 6 | test | brass pot #0 | SUCCESS | **4.76 → SUCCESS** | 3.37 → FAILURE | 0.91x | 12.5 s | 21.2 s of 21.2 s |
| 7 | test | black kitchen roll holder #6 | SUCCESS | **4.47 → SUCCESS** | 3.12 → FAILURE | 2.25x | 11.4 s | 22.9 s of 28.2 s |
| 8 | dev | frog cup #1 | SUCCESS | **4.88 → SUCCESS** | 3.60 → FAILURE | 3.93x | 5.9 s | 19.5 s of 22.7 s |
| 9 | test | black kitchen roll holder #8 | FAILURE | **3.12 → FAILURE** | 4.40 → SUCCESS | 1.84x | 14.1 s | 22.5 s of 28.1 s |
| 10 | test | organizer beige #1 | SUCCESS | **4.94 → SUCCESS** | 3.82 → FAILURE | 3.5x | 8.3 s | 16.3 s of 16.3 s |
| 11 | dev | plant pot #7 | SUCCESS | **4.66 → SUCCESS** | 3.71 → FAILURE | 2.12x | 16.3 s | 22.6 s of 28.7 s |
| 12 | test | ceramic elephant jug #14 | SUCCESS | **4.52 → SUCCESS** | 3.60 → FAILURE | 1.81x | 19.3 s | 27.1 s of 30.0 s |
| 13 | test | metal measuring flask #2 | SUCCESS | **4.11 → SUCCESS** | 3.25 → FAILURE | 1.68x | 16.7 s | 34.7 s of 39.9 s |
| 14 | test | balloon whisk #4 | FAILURE | **3.82 → FAILURE** | 4.52 → SUCCESS | 1.77x | 12.1 s | 18.5 s of 22.4 s |
| 15 | test | jja ramen #4 | SUCCESS | **4.19 → SUCCESS** | 3.71 → FAILURE | 3.52x | 13.8 s | 30.3 s of 35.2 s |
| 16 | test | metal measuring flask #0 | FAILURE | **3.82 → FAILURE** | 4.26 → SUCCESS | 1.65x | 12.5 s | 30.0 s of 35.0 s |
| 17 | dev | icecream scoop #5 | SUCCESS | **4.19 → SUCCESS** | 3.82 → FAILURE | 2.54x | 15.1 s | 35.3 s of 35.3 s |
| 18 | dev | hammer #15 | SUCCESS | **4.19 → SUCCESS** | 3.82 → FAILURE | 3.88x | 12.3 s | 26.1 s of 31.2 s |
| 19 | test | organizer beige #13 | FAILURE | **3.92 → FAILURE** | 4.19 → SUCCESS | 1.35x | 8.6 s | 18.5 s of 23.4 s |

## Caveats

- The baseline is the v2 control adapter, not a v3 retrain; the only difference would be
  run-to-run training noise, which this experiment therefore cannot report.
- 251 held-out clips over 32 objects cannot certify a 3-point accuracy gain; at the observed
  win/loss ratio about 1,000 clips would be needed for p < 0.05.
- Part of the tactile model's gain comes through a small gradient-boosted force classifier whose
  probability is printed in the force block, not only from the VLM reading raw force lines.
- Training loss reaches its floor by update 20 of 88 because the target is a constant template
  plus one digit; both fine-tuned models are undertrained and data-limited.
