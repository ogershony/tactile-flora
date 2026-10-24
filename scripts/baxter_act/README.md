# Baxter ACT policy: the policy that produced the Baxter rollouts

Folded in from `~/baxter_teleop` on 2026-10-24. These scripts trained the LeRobot ACT policy
`OG7407/baxter-act-cube-left` on the teleop dataset `OG7407/baxter-cube-pick-left` (50 episodes,
15 fps, left arm, "pick up the red cube"); live rollouts of that policy are the benchmark
`OG7407/baxter-cube-pick-left-rollouts` evaluated in `writeup/baxter-rollouts.md`.

Only the scripts and the final checkpoint's small artefacts are in git. The working directory
(`BAXTER_ACT_HOME`, default `~/baxter_teleop`) holds the lerobot 0.6.1 venv, `outputs/<run>/`
(checkpoints, 6.4 GB) and `deploy/<tag>/` (staged checkpoints, 789 MB); it is never committed.

| File | What it does |
|---|---|
| `slurm_common.sh` | Sourced by the sbatch scripts: activates the venv in `BAXTER_ACT_HOME`, reads `HF_TOKEN` from the HF CLI cache, disables wandb. |
| `smoke.sbatch` | 300-step smoke run on one RTX 2080 Ti (`debug` partition, node zixian). |
| `train.sbatch` | The real run: 50 000 steps, 5 GPUs via Accelerate, fp16, global batch 65, `--requeue` with resume from `checkpoints/last`. Knobs `RUN`, `STEPS`, `BS`, `NGPU`. |
| `progress.sh <jobid> <run_dir>` | Progress and speed report from the Slurm log, held-out L1 for any new checkpoint (one GPU on `debug`), loss-curve PNG. |
| `eval_heldout.py <run_dir>` | Held-out action L1 per checkpoint on episodes 45 to 49, run the way the robot runs the policy (`predict_action_chunk`, no VAE latent from ground truth), raw units. |
| `plot_losses.py <run_dir> LOG...` | Loss curves from the Slurm logs plus `heldout_l1.json`. |
| `publish.sh <run_dir> <maxstep> <tag>` | Picks the best checkpoint by held-out chunk L1, stages it under `deploy/<tag>`, pushes it to the private model repo `OG7407/baxter-act-cube-left`. Token from `HF_WRITE_TOKEN` only. |

The sbatch scripts source `slurm_common.sh` from `TACTILE_FLORA_ROOT` (default `~/tactile-flora`),
so run them from this checkout on the SDSC cluster:

```sh
sbatch scripts/baxter_act/smoke.sbatch
sbatch scripts/baxter_act/train.sbatch
scripts/baxter_act/progress.sh <jobid> ~/baxter_teleop/outputs/act_baxter_cube_left
HF_WRITE_TOKEN=... scripts/baxter_act/publish.sh ~/baxter_teleop/outputs/act_baxter_cube_left 50000 act_baxter_cube_left_final
```

## The published checkpoint

`results/act_baxter_cube_left_final/` is what was pushed to the Hub as the final policy (Slurm job
6442 on zixian, 5 x RTX 2080 Ti, 2026-10-23 23:03 to 2026-10-24 03:56): the model card
(`README.md`), the training log, the loss curves and the held-out L1 of every checkpoint. The weights
themselves (`pretrained_model/`, 200 MB) are on the Hub and in `~/baxter_teleop/deploy/`.

Held-out action L1 (mean over the 30-step chunk, raw units: radians, gripper 0 to 1) fell from
0.0325 at step 5 000 to 0.0267 at step 50 000, still improving slowly at the end; the first action of
each chunk from 0.0225 to 0.0162. Step 50 000 was published. Training config: chunk_size 30,
n_action_steps 15, resnet18 backbone, VAE on, lr 1e-5, inputs `observation.state` [8] plus head and
wrist cameras at 360 x 640.
