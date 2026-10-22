# Cleaned HRDexDB training experiment

Training uses the reviewed v1 subset: 25 training clips and 13 object-disjoint validation
clips. Both adapters start from released FLORA revision
`7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`; neither resumes an earlier pilot adapter.
The matched runs differ only in force mode: real force versus omitted force.

Fixed settings: four epochs, seed 17, LoRA rank 32 / alpha 64, learning rate 2e-5,
effective batch 16, warmup fraction 0.1 and gradient clipping 1.0. Each run has eight
optimizer steps. The final epoch is evaluated without validation-based checkpoint selection.

Slurm job 6321 runs the two trainings sequentially on Matthew's two MI210s with two-rank
FSDP. Job 6322 is queued with an after-success dependency to evaluate both adapters and the
released model. Monster's three additional MI210s were configured but unavailable
(`DOWN+DRAIN+NOT_RESPONDING`) when checked.

Local artifacts under `data/acceptance/hrdexdb-clean-v1/`:

- `training-plan.json`: matched configs, input/config/code hashes and checkpoint choice.
- `training-environment.json`: observed runtime packages and GPUs.
- `runs/clean-v1-force/` and `runs/clean-v1-control/`: configs, loss CSVs and saved adapters.
- `jobs/train-6321.log` and `jobs/eval-6322.log`: scheduler output.
- `evaluation-plan.json` and `evaluation-result.json`: five fixed conditions and results.

The five planned evaluation conditions are released FLORA with omitted/real force,
the control adapter with omitted force, and the force adapter with real/shuffled force.
All use the same cleaned validation manifest, original task wording and evaluation threshold.
No original pilot score is overwritten. This remains a small, post hoc curated development
evaluation, with unresolved force/video timing; it is not an untouched benchmark.
