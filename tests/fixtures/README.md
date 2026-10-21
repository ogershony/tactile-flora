# Fixtures

Each source adapter ships `tests/fixtures/<name>/` holding one tiny synthetic episode so unit
tests run without the real datasets or a GPU.

Convention:
- 10 video frames per camera (tiny resolution, generated, never real footage).
- Synthetic force with a known shape (baseline, a grasp plateau, an optional slip or crush) so
  detectors and normalization can be asserted against hand-computed values.
- Laid out exactly as the real dataset is on disk, so the adapter's `discover()` and `load()` are
  exercised unchanged.
- Small enough to commit (well under 1 MB per source).

## Eval-only sources (hrdexdb, softvtbench, baxter)

`tests/fixtures/{hrdexdb,softvtbench,baxter}/make_fixture.py` each write a tiny synthetic episode
(10 frames) into a sibling `data/` directory in the real on-disk layout; the generated files are
committed (well under 200 KB each). Videos are real mp4s written with PyAV (mpeg4, 32x32), so no
`frames.npy` placeholder is needed. The shared `tests/fixtures/_video.py:write_mp4` writes the videos. Each fixture deliberately puts an outcome word in the object/language text (e.g.
`retry_mug`, `crushed soda can`) to check `strip_outcome_words`. The Baxter fixture follows the
contract documented in the `tactile_flora/sources/baxter.py` docstring.
