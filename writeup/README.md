# Write-ups and results

Start with the report; the rest are the sources it cites and the two presentation formats.

| Item | What it is |
|---|---|
| [tactile-flora-report.md](tactile-flora-report.md) | The experiment report: hypothesis, what was built, the four-stage process, evidence tables, results statement, caveats, artefact index. |
| [hrdexdb/](hrdexdb/README.md) | Fine-tuned FLORA vs fine-tuned tactile FLORA on HRDexDB: headline tables, significance, ablations, the 26 clips the tactile model wins (side-by-side videos in `hrdexdb/videos/`) and the 19 it loses. `make_videos.py` regenerates the videos, `clips.json` holds the per-clip numbers. |
| [baxter-rollouts/](baxter-rollouts/README.md) | Released, fine-tuned and tactile FLORA on ten Baxter cube-pick rollouts: per-episode scores, ablations, the force-trace figure and seven side-by-side videos. `summarize.py` prints the tables, `make_videos.py` renders the videos. |
| [paper/](paper/README.md) | NeurIPS-style manuscript (`main.tex`, compiled `main.pdf`). |
| [slides/](slides/README.md) | Sixteen-slide deck (`tactile-flora.pptx`) and the script that builds it. |

All numbers come from finished evaluation cells under `data/acceptance/` (not in git); copies of
the evaluation JSON that the Baxter tables read from are in `baxter-rollouts/`.
