# Tactile FLORA slide deck

`tactile-flora.pptx` is a 16-slide, 16:9 deck of the experiment: the idea (force as text in
the FLORA-Rewarder prompt, LoRA fine-tune), the controls, the four-stage process, the HRDexDB
results and statistics, the shuffled-force and video-blank diagnostics, one win clip, the Baxter
robot setup and its zero-shot result, caveats, and next steps. Every number is copied from
`writeup/tactile-flora-report.md` (which cites `hrdexdb/README.md` and `baxter-rollouts/README.md`);
nothing is recomputed in the build.

## Files

| File | What |
|---|---|
| `tactile-flora.pptx` | the deck |
| `tactile-flora.pdf` | the same deck rendered to PDF by LibreOffice 7 (headless, in a container) |
| `build_deck.py` | builds it from scratch with python-pptx; renders the two charts with matplotlib |
| `figures/hrdexdb_auroc.png` | AUROC per model and split on HRDexDB (slide 7) |
| `figures/baxter_auroc.png` | AUROC per condition on the Baxter rollouts, chance line (slide 13) |

Images reused from elsewhere in `writeup/`: `hrdexdb/videos/test_standing_frame_4_success.png` (slide 11),
`baxter-rollouts/force_traces.png` (slide 12), `baxter-rollouts/videos/ep0000_success.png`
(slide 14). The diagram (slide 3) and the timeline (slide 6) are native, editable shapes.

## Regenerate

The project venv has matplotlib but not python-pptx; use a scratch venv:

```sh
uv venv /tmp/slides-venv --python 3.11
uv pip install --python /tmp/slides-venv/bin/python python-pptx pillow matplotlib
/tmp/slides-venv/bin/python writeup/slides/build_deck.py
```

Fonts: Cambria for titles (32 pt and up), Calibri for body (12 to 18 pt), Courier New for the
force block. Palette: ink `1F2937`, blue accent `2A6FB0` (the tactile model), orange `D9480F`
(force), card tint `EEF2F6`, dark slate `17212B` for the first and last slide.

Checks run on 2026-10-24: the pptx skill's `validate.py` passes; a geometric estimate found no
text box overflowing its frame and no shape off the slide. The PDF was rendered with

```sh
apptainer exec docker://linuxserver/libreoffice:latest \
  soffice --headless --convert-to pdf --outdir writeup/slides writeup/slides/tactile-flora.pptx
```

so its fonts are LibreOffice's substitutes for Cambria and Calibri; open the pptx in PowerPoint
once before presenting from it.
