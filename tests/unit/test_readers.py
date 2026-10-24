"""Video path resolution for the frame readers."""

import os
from pathlib import Path

from tactile_flora.schema import Episode, VideoRef
from tactile_flora.video.readers import video_path


def _ep(path: str) -> Episode:
    return Episode.model_construct(cameras={"scene": VideoRef(path=path, fps=30.0, n_frames=1)})


def test_video_path_accepts_root_prefixed_relative_paths(tmp_path: Path) -> None:
    """LeRobot adapters store ``data/raw/<src>/...`` (relative to the CWD, root included)."""
    root = tmp_path / "data" / "raw" / "src"
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "a.mp4").write_bytes(b"x")
    absolute = root / "videos" / "a.mp4"
    assert video_path(root, _ep(str(absolute)), "scene") == absolute
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rel_root = Path("data/raw/src")
        prefixed = Path("data/raw/src/videos/a.mp4")
        assert video_path(rel_root, _ep(str(prefixed)), "scene") == prefixed
        assert video_path(rel_root, _ep("videos/a.mp4"), "scene") == prefixed
        # neither form exists: the root-joined path is returned (and later raises on open)
        assert video_path(rel_root, _ep("videos/b.mp4"), "scene") == rel_root / "videos/b.mp4"
    finally:
        os.chdir(cwd)
