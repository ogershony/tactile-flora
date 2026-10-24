### Headline

Accuracy at the 4.0 threshold on E[s]; AUROC over the 2 successes x 8 failures pairs; MAE on the 0-100 scale.

| Model | Accuracy | AUROC | MAE | Successes right (of 2) | Failures right (of 8) | Hard scores given |
|---|---:|---:|---:|---:|---:|---|
| Baseline FLORA, no force | 0.80 | 0.375 | 140 | 0 | 8 | 2x10 |
| Baseline FLORA + force block | 0.80 | 1.000 | 100 | 0 | 8 | 1x4, 2x6 |
| Fine-tuned FLORA (video only) | 0.80 | 0.312 | 80 | 0 | 8 | 1x10 |
| Fine-tuned tactile FLORA, real force | 0.80 | 1.000 | 80 | 0 | 8 | 1x10 |
| Fine-tuned tactile FLORA, shuffled force | 0.80 | 0.250 | 90 | 0 | 8 | 1x9, 2x1 |
| Fine-tuned tactile FLORA, force removed | 0.80 | 0.438 | 80 | 0 | 8 | 1x10 |
| Force-only classifier (from the block) | 0.80 | 0.500 | | 0 | 8 | P(success) |

### Per-episode expected score E[s] (success threshold 4.0)

| Episode | Label | baseline | baseline+force | fine-tuned | tactile | tactile, shuffled | tactile, no force | force P(success) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ep0000 | success | 1.91 | 2.41 | 1.34 | 2.41 | 1.03 | 1.43 | 0.40 |
| ep0001 | success | 2.00 | 1.92 | 1.05 | 1.60 | 1.03 | 1.12 | 0.04 |
| ep0002 | failure | 2.01 ok | 1.78 ok | 1.40 ok | 1.05 ok | 1.05 ok | 1.52 ok | 0.09 |
| ep0003 | failure | 2.03 ok | 1.81 ok | 1.99 ok | 1.09 ok | 1.08 ok | 2.97 ok | 0.11 |
| ep0004 | failure | 1.91 ok | 1.06 ok | 1.60 ok | 1.02 ok | 1.02 ok | 1.32 ok | 0.05 |
| ep0005 | failure | 1.82 ok | 1.05 ok | 1.05 ok | 1.03 ok | 1.03 ok | 1.05 ok | 0.09 |
| ep0006 | failure | 1.99 ok | 1.00 ok | 1.05 ok | 1.02 ok | 1.62 ok | 1.11 ok | 0.09 |
| ep0007 | failure | 1.85 ok | 1.48 ok | 1.03 ok | 1.03 ok | 1.03 ok | 1.06 ok | 0.09 |
| ep0008 | failure | 2.04 ok | 1.70 ok | 1.75 ok | 1.04 ok | 1.06 ok | 2.31 ok | 0.05 |
| ep0009 | failure | 2.02 ok | 1.74 ok | 1.55 ok | 1.07 ok | 1.07 ok | 2.02 ok | 0.05 |

'ok' marks a correct call. Bold: tactile FLORA right where both baseline and fine-tuned FLORA are wrong.

Episodes where tactile FLORA alone is right: none.

### What each model said (rationale, hard score)

**ep0000** (success)

- baseline: score 2 (E[s] 1.91). The robot approached and contacted the red cube but never lifted it. The cube remained on the table in the final frames.
- baseline+force: score 2 (E[s] 2.41). The robot approached and contacted the red cube, but the cube remained on the table and was not lifted or removed.
- fine-tuned: score 1 (E[s] 1.34). no rationale
- tactile: score 1 (E[s] 2.41). no rationale

**ep0001** (success)

- baseline: score 2 (E[s] 2.00). The robot approached and contacted the red cube but did not lift it from the surface. The cube remained on the table with only a small displacement.
- baseline+force: score 2 (E[s] 1.92). The robot made a brief contact with the red cube and exerted a small force, but the cube remained on the support surface and was not lifted or held.
- fine-tuned: score 1 (E[s] 1.05). no rationale
- tactile: score 1 (E[s] 1.60). no rationale

**ep0002** (failure)

- baseline: score 2 (E[s] 2.01). The robot approached and contacted the red cube but never lifted it from the surface. The cube remained on the table in the final frames.
- baseline+force: score 2 (E[s] 1.78). The robot approached and contacted the red cube but never lifted it. The cube remained on the support surface in the final frames.
- fine-tuned: score 1 (E[s] 1.40). no rationale
- tactile: score 1 (E[s] 1.05). no rationale

**ep0003** (failure)

- baseline: score 2 (E[s] 2.03). The robot approached and contacted the red cube but never lifted it from the surface. The cube remained on the table in the final frames.
- baseline+force: score 2 (E[s] 1.81). The robot approached and contacted the red cube but never lifted it. The cube remained on the support surface in the final frames.
- fine-tuned: score 1 (E[s] 1.99). no rationale
- tactile: score 1 (E[s] 1.09). no rationale

**ep0004** (failure)

- baseline: score 2 (E[s] 1.91). The robot approached and contacted the red cube but never lifted it from the surface. The cube remained on the table in the final frames.
- baseline+force: score 1 (E[s] 1.06). The robot approached the red cube but never lifted or removed it. The cube remained on the table in the final frames.
- fine-tuned: score 1 (E[s] 1.60). no rationale
- tactile: score 1 (E[s] 1.02). no rationale

**ep0005** (failure)

- baseline: score 2 (E[s] 1.82). The robot approached and contacted the red cube but did not lift it from the surface. The cube remained on the table in the final frames.
- baseline+force: score 1 (E[s] 1.05). The robot approached the red cube but never lifted or secured it. The cube remained on the support surface, with no successful grasp or pickup.
- fine-tuned: score 1 (E[s] 1.05). no rationale
- tactile: score 1 (E[s] 1.03). no rationale

**ep0006** (failure)

- baseline: score 2 (E[s] 1.99). The robot approached and contacted the red cube but did not lift it from the surface. The cube remained on the table with only a small displacement.
- baseline+force: score 1 (E[s] 1.00). The robot approached the red cube but never grasped or lifted it. The cube remained on the support surface, with no goal-relevant change.
- fine-tuned: score 1 (E[s] 1.05). no rationale
- tactile: score 1 (E[s] 1.02). no rationale

**ep0007** (failure)

- baseline: score 2 (E[s] 1.85). The robot approached and contacted the red cube but did not lift it from the surface. The cube remained on the table in the final frames.
- baseline+force: score 1 (E[s] 1.48). The robot approached and contacted the red cube but never lifted it. The cube remained on the support surface in the final frames.
- fine-tuned: score 1 (E[s] 1.03). no rationale
- tactile: score 1 (E[s] 1.03). no rationale

**ep0008** (failure)

- baseline: score 2 (E[s] 2.04). The robot approached and contacted the red cube but never lifted it from the surface. The cube remained on the table in the final frames.
- baseline+force: score 2 (E[s] 1.70). The robot approached and contacted the red cube but never lifted it. The cube remained on the support surface in the final frames.
- fine-tuned: score 1 (E[s] 1.75). no rationale
- tactile: score 1 (E[s] 1.04). no rationale

**ep0009** (failure)

- baseline: score 2 (E[s] 2.02). The robot approached and contacted the red cube but never lifted it from the surface. The cube remained on the table in the final frames.
- baseline+force: score 2 (E[s] 1.74). The robot approached and contacted the red cube but never lifted it. The cube remained on the support surface in the final frames.
- fine-tuned: score 1 (E[s] 1.55). no rationale
- tactile: score 1 (E[s] 1.07). no rationale

