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
