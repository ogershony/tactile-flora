# NeurIPS-style manuscript

| File | What |
|---|---|
| `main.tex` | The paper: "Tactile FLORA: Fingertip Force as Text Improves a Vision-Language Reward Model's Grasp Judgments". Every number comes from `../tactile-flora-report.md`, `../hrdexdb/README.md` and `../baxter-rollouts/README.md`. |
| `main.pdf` | Compiled output (9 pages: 8 body and references, 1 appendix) in the `preprint` style. |
| `references.bib` | Bibliography. Entries marked `note = {verify}` have unconfirmed fields (FLORA, HRDexDB, Qwen3.5, LeRobot). |
| `neurips_2023.sty` | NeurIPS style file. The 2024 and 2025 copies were not downloadable from media.neurips.cc at the time of writing (404), so the 2023 file is used; swap it for the current year's and update the `\usepackage` line. |
| `environ.sty`, `trimspaces.sty`, `multirow` | `environ` and `trimspaces` are dependencies of the style file, copied from CTAN because the small TeX Live image lacks them. `multirow` is not used. |
| `figures/` | `force_traces.png` (per-rollout force and closure) and `ep0000_success.png` (poster of the ep0000 side-by-side video), copied from `../baxter-rollouts/`. |

## Compile

There is no LaTeX on the login node. With Apptainer:

```sh
cd writeup/paper
export APPTAINER_CACHEDIR=/tmp/$USER/apptainer   # the image is about 600 MB
apptainer exec --bind "$PWD":/work --pwd /work docker://texlive/texlive:latest-small \
  sh -c 'pdflatex main && bibtex main && pdflatex main && pdflatex main'
```

Any full TeX Live works too (`latexmk -pdf main.tex`). The `[preprint]` option on the style
package gives the arXiv look; use `[final]` for a camera-ready or no option for anonymous review.

## Placeholders

Red `[TODO: ...]` marks in the PDF: co-author names and affiliations, the corresponding author's
email, the acknowledgements, and three citation groups in Related Work (video reward benchmarks,
classic slip-detection work, LLM time-series prompting). Fill those before circulating.
