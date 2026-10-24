# Documentation index

| Document | What it holds |
|---|---|
| [spec.md](spec.md) | The technical specification the repository was built to: purpose, FLORA reference facts, layout, every stage's contract. |
| [deviations.md](deviations.md) | Where the implementation departs from the specification, and why. |
| [pipeline.md](pipeline.md) | Command reference (`tf verify`, `manifest`, `train`, `eval`, `baxter`), the data flow under `data/`, the training loop, verified GPU environments and the acceptance gates. |
| [status.md](status.md) | Dated log of every run, newest first. |
| [review1.md](review1.md) | An early code review against the specification and what it found. |
| [handoff/baxter-rollouts.md](handoff/baxter-rollouts.md) | Step-by-step instructions to reproduce the Baxter rollout evaluation on another machine. |

## Experiment records

Each experiment keeps its frozen plan, job scripts, splits and audits beside its page, so a result
can be traced to the exact code hash, manifest hash and Slurm job that produced it.

| Experiment | Page | What changed |
|---|---|---|
| HRDexDB pilot (32 / 16 clips) | [experiments/hrdexdb-pilot/](experiments/hrdexdb-pilot/README.md) | First LoRA runs, data-quality audit, cleaned subset, video-only classifier baseline. |
| Expanded v2 (618 clips) | [experiments/hrdexdb-expanded-v2.md](experiments/hrdexdb-expanded-v2.md) | Recorded video clocks, frozen object-disjoint splits (350 / 136 / 115), matched force and control training. No force benefit; the force block was found to be nearly uninformative. |
| v3, corrected representation | [experiments/hrdexdb-expanded-v3.md](experiments/hrdexdb-expanded-v3.md) | Per-trial tare, sign-corrected closure, label-free summary line, force-only classifier probability. The primary tactile adapter. |
| Baxter rollouts | [handoff/baxter-rollouts.md](handoff/baxter-rollouts.md) and [../writeup/baxter-rollouts/](../writeup/baxter-rollouts/README.md) | Zero-shot transfer of the v3 adapter to the Baxter cube-pick rollouts. |

## Research notes

- [research/datasets.md](research/datasets.md): what each candidate tactile dataset actually contains (layouts, labels, sensors, licences), verified by inspection.
- [research/flora_prompts.md](research/flora_prompts.md): the released FLORA prompts and how the force block fits into them.

## Figures and archive

- `figures/flora_architecture.png`: the architecture diagram used in the README.
- `archive/`: superseded material kept for the record (a plain-text export of the specification, an unexecuted later plan).
