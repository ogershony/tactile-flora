# Scripts

| Path | What |
|---|---|
| `tf` | Thin wrapper around `uv run tf "$@"`, the package command line (reference in [docs/pipeline.md](../docs/pipeline.md)). |
| `download_phantom.py` | Bounded, revision-pinned download of a PHANTOM subset with checksums and provenance. |
| `diagnose_sync.py` | Force-to-video synchronization diagnostics for one episode. |
| `make_spot_review.py` | Builds the spot-check mosaics a reviewer looks at before accepting a source. |
| [cluster/](cluster/README.md) | The Slurm-side scripts behind every reported experiment: HRDexDB download, audits, split preparation, LoRA training launchers, evaluation and summaries (`*_hrdexdb_*`), the Baxter rollout benchmark (`prepare_baxter_rollouts.py`, `evaluate_baxter_rollouts.py`), and the GPU smoke tests. |

Robot-side scripts (demo collection, policy rollouts, ACT training) live under [baxter/](../baxter/README.md).
