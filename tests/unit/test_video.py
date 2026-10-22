"""Tests for video/readers.py and video/mosaic.py (the single frame + mosaic implementation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tactile_flora.schema import Episode, VideoRef
from tactile_flora.video.mosaic import (
    BLANK_GRAY,
    VISUAL_TOKEN_CAP,
    blank_frames,
    clip_frames,
    compose_mosaic,
    frame_indices,
    mosaic_layout,
    video_metadata,
    visual_token_estimate,
)
from tactile_flora.video.readers import (
    AutoReader,
    PyAVReader,
    SyntheticReader,
    ZarrJpegReader,
)
from tests.fixtures.lerobot_writer import write_mp4
from tests.fixtures.synthetic import make_pad_episode

FIX = Path(__file__).parents[1] / "fixtures"


def test_video_metadata_matches_transformers_contract() -> None:
    """Timestamp metadata must construct the real processor's metadata object."""
    pytest.importorskip("transformers")
    from transformers.video_utils import VideoMetadata

    meta = VideoMetadata(**video_metadata([0, 0, 15, 29], 15.0, 30))
    assert meta.total_num_frames == 30
    assert meta.timestamps == pytest.approx([0, 0, 1, 29 / 15])


def _ep(roles: tuple[str, ...], n: int = 10, fps: float = 4.0) -> Episode:
    ep = make_pad_episode(0, n_frames=n)
    cams = {r: VideoRef(path=f"{r}.mp4", fps=fps, n_frames=n) for r in roles}
    return ep.model_copy(update={"cameras": cams, "duration_s": n / fps})


def _solid(v: int, h: int = 48, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


@pytest.mark.parametrize(("n", "grid"), [(1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 2))])
def test_compose_shapes_and_scene_upper_left(n: int, grid: tuple[int, int]) -> None:
    roles = ["wrist_l", "scene", "top", "wrist_r"][:n]
    if n == 1:
        roles = ["scene"]
    frames = {r: _solid(200 if r == "scene" else 50) for r in roles}
    out = compose_mosaic(frames, cell_px=64)
    assert mosaic_layout(roles)[0] == grid
    assert out.shape == (grid[0] * 64, grid[1] * 64, 3) and out.dtype == np.uint8
    if "scene" in roles:
        assert out[0, 0, 0] == 200  # scene cell is upper left
    if n > 1:
        assert out[0, 64, 0] == 50
    with pytest.raises(ValueError):
        compose_mosaic(frames, (3, 3), cell_px=64)
    with pytest.raises(ValueError):
        compose_mosaic(frames, cell_px=50)


def test_blank_is_mid_gray() -> None:
    assert set(np.unique(blank_frames(_solid(7)))) == {BLANK_GRAY} and BLANK_GRAY == 128
    ep = _ep(("scene", "wrist_l"))
    clip = clip_frames(ep, SyntheticReader(), [0.0, 0.5], ["scene", "wrist_l"], blank=True)
    assert len(clip) == 2 and clip[0].shape == (96, 192, 3)
    assert all(set(np.unique(f)) == {128} for f in clip)


def test_clip_frames_repeats_boundary_frames() -> None:
    ep = _ep(("scene",), n=10, fps=4.0)  # duration 2.5 s
    assert frame_indices(ep, "scene", [-3.0, 0.0, 1.0, 99.0]) == [0, 0, 4, 9]

    class Probe(SyntheticReader):
        def read_many(self, episode: Episode, role: str, indices):  # type: ignore[no-untyped-def]
            self.seen = list(indices)
            return [np.full((8, 8, 3), i * 20, dtype=np.uint8) for i in indices]

    r = Probe()
    clip = clip_frames(ep, r, [-1.0, 0.0, 1.0, 50.0], ["scene"], cell_px=32)
    assert r.seen == [0, 0, 4, 9] and len(clip) == 4 and clip[0].shape == (32, 32, 3)
    assert np.array_equal(clip[0], clip[1]) and clip[3][0, 0, 0] == 180


def test_visual_token_estimate_cap() -> None:
    assert visual_token_estimate(64, (96, 96)) == 32 * 9
    assert visual_token_estimate(64, (192, 192)) == 32 * 36
    assert visual_token_estimate(64, (1024, 1024)) == VISUAL_TOKEN_CAP


def test_pyav_reader_offset_and_count(tmp_path: Path) -> None:
    write_mp4(tmp_path / "cat.mp4", fps=10, n_frames=20)  # two 10-frame episodes back to back
    ref = VideoRef(path="cat.mp4", fps=10.0, n_frames=10)
    first = make_pad_episode(0).model_copy(update={"cameras": {"scene": ref}})
    second = first.model_copy(update={"meta": {"video_from_s": {"scene": 1.0}}})
    r = PyAVReader(tmp_path)
    assert r.frame_count(first, "scene") == 10 and r.frame_count(second, "scene") == 10
    # write_mp4 shade of in-file frame k is 10 * k
    a = r.read_many(first, "scene", [0, 3, 3, 9])
    b = r.read_many(second, "scene", [0, 3, 9])
    assert [x.shape for x in a] == [(32, 32, 3)] * 4 and a[1].dtype == np.uint8
    for got, k in zip(a, [0, 3, 3, 9], strict=True):
        assert abs(int(got.mean()) - 10 * k) <= 6
    for got, k in zip(b, [10, 13, 19], strict=True):
        assert abs(int(got.mean()) - 10 * k) <= 6
    assert np.array_equal(a[1], a[2]) and np.array_equal(r.read(first, "scene", 3), a[1])
    with pytest.raises(IndexError):
        r.read(first, "scene", 10)


def test_auto_reader_dispatches_zarr_and_mp4(tmp_path: Path) -> None:
    from tactile_flora.config import SourceConfig
    from tactile_flora.sources.phantom import PhantomAdapter

    root = FIX / "phantom"
    cfg = SourceConfig(
        name="phantom", root=str(root), license_allowed=["CC-BY-4.0"], max_windows=300,
        max_episodes=100, eval_only=False, expected_label_rate=0.3, seed=0,
    )  # fmt: skip
    a = PhantomAdapter(cfg)
    ep = a.load(a.discover(root)[0])
    z = ZarrJpegReader()
    assert z.frame_count(ep, "scene") == 10
    f = z.read_many(ep, "scene", [0, 4, 9])
    assert [x.shape for x in f] == [(24, 32, 3)] * 3 and f[0].dtype == np.uint8
    assert abs(int(f[2][..., 0].mean()) - 180) <= 12  # fixture frame i has red 20 * i
    auto = AutoReader()
    assert auto.frame_count(ep, "scene") == 10
    assert np.array_equal(auto.read(ep, "scene", 4), f[1])
    clip = clip_frames(ep, auto, [0.0, 0.3, 99.0], ["scene"], cell_px=32)
    assert len(clip) == 3 and clip[0].shape == (32, 32, 3)

    write_mp4(tmp_path / "scene.mp4", fps=4, n_frames=10)
    mp4 = _ep(("scene",)).model_copy(update={"source": "x"})
    mp4 = mp4.model_copy(
        update={"cameras": {"scene": VideoRef(path="scene.mp4", fps=4.0, n_frames=10)}}
    )
    assert AutoReader(tmp_path).frame_count(mp4, "scene") == 10
