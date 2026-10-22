# Cleaned HRDexDB training experiment

Training uses the reviewed v1 subset: 25 training clips and 13 object-disjoint validation
clips. Both adapters start from released FLORA revision
`7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`; neither resumes an earlier pilot adapter.
The matched runs differ only in force mode: real force versus omitted force.

Fixed settings: four epochs, seed 17, LoRA rank 32 / alpha 64, learning rate 2e-5,
effective batch 16, warmup fraction 0.1 and gradient clipping 1.0. Each run completed seven
optimizer updates. The scheduler planned eight, but the current trainer accumulates across
short epoch batches: 52 microbatches per rank produce seven updates, including the final
partial accumulation. Both runs use the same behavior. The final epoch is evaluated without
validation-based checkpoint selection.

Slurm job 6321 completed both trainings sequentially in 35 minutes on Matthew's two MI210s
with two-rank FSDP. Both saved adapters contain 512 finite tensors and 256 nonzero LoRA B
matrices. Force loss decreased from 2.606 to 1.495; control loss from 2.605 to 1.744.
Job 6322 completed the released-model evaluations, then exhausted GPU memory when loading
the next checkpoint in the same interpreter. The evaluator now uses a fresh process per
checkpoint; job 6325 completed the adapter evaluations and reused the released-model cells.
Monster's three additional MI210s were configured but unavailable
(`DOWN+DRAIN+NOT_RESPONDING`) when checked.
The launcher verifies the active job's planned configuration, manifest and core-code hashes
before using its pinned report-generation version. This permits documentation commits and
pushes during the experiment without changing its learning inputs; other launches retain
the standard stale-manifest check.

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

Completed results (job 6325, nine minutes; 13 clips and zero parse failures per condition):

| Checkpoint | Force | Correct | Accuracy | AUROC |
|---|---|---:|---:|---:|
| Released FLORA | Omitted | 11/13 | 84.6% | 0.925 |
| Released FLORA | Real | 8/13 | 61.5% | 0.650 |
| Control SFT | Omitted | 11/13 | 84.6% | 0.800 |
| Force SFT | Real | 8/13 | 61.5% | 0.650 |
| Force SFT | Shuffled | 9/13 | 69.2% | 0.725 |

Training did not improve accuracy. Shuffled force outperformed real force by one clip;
this experiment provides no evidence of useful force conditioning. The tiny development
set cannot distinguish label, timing, representation and optimization limitations reliably.
All conditions use the same episode IDs, window IDs and truth targets, verified against
the saved prediction files. Both adapter hashes still match their training artifacts.
[The portable snapshot](clean-result.json) contains all five configurations, predictions,
raw answers, metrics and training provenance. Historical 16-clip results remain separate.
