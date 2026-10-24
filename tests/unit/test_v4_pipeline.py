"""v4 pipeline: frame cache, FLORA-native collage, HRDexDB frame-cache mode, score-only targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tactile_flora.config import SourceConfig
from tactile_flora.schema import Episode, VideoRef
from tactile_flora.sources.hrdexdb import HrdexdbAdapter
from tactile_flora.train.dataset import score_only_target
from tactile_flora.video.mosaic import clip_frames
from tactile_flora.video.readers import AutoReader, FrameCacheReader, is_frame_cache


def _cache(root: Path, ep: str, cams: dict[str, tuple[int, int]], indices: list[int]) -> Path:
    d = root / ep
    meta = {}
    for cam, (w, h) in cams.items():
        (d / cam).mkdir(parents=True)
        for i in indices:
            arr = np.full((288, 384, 3), i % 251, dtype=np.uint8)
            Image.fromarray(arr).save(d / cam / f"{i:05d}.jpg", quality=95, subsampling=0)
        meta[cam] = {"fps": 30.0, "n_frames": 300, "width": w, "height": h, "indices": indices}
    (d / "video.json").write_text(json.dumps(meta))
    return d


def _episode(d: Path, cams: list[str]) -> Episode:
    refs = {
        ("scene" if k == 0 else f"cam_{c}"): VideoRef(
            path=str(d / c), fps=30.0, n_frames=300, width=2048, height=1536
        )
        for k, c in enumerate(cams)
    }
    return Episode.model_construct(
        source="hrdexdb", episode_id="x", session="s", task="t", duration_s=10.0, cameras=refs,
        force=None, gripper=None, labels=[], license="CC-BY-NC-4.0", meta={},
    )  # fmt: skip


def test_frame_cache_reader_reads_nearest_rendered_frame_within_slack(tmp_path: Path) -> None:
    d = _cache(tmp_path, "inspire_dftp/mug/0", {"111": (2048, 1536)}, [0, 10, 20])
    ep = _episode(d, ["111"])
    assert is_frame_cache(d / "111") and not is_frame_cache(d)
    r = FrameCacheReader()
    a, b, c = r.read_many(ep, "scene", [10, 21, 23])
    assert a.shape == (288, 384, 3) and abs(int(a[0, 0, 0]) - 10) <= 2
    assert abs(int(b[0, 0, 0]) - 20) <= 2  # 21 was not rendered: nearest (20)
    assert abs(int(c[0, 0, 0]) - 20) <= 2  # 3 frames away: a view 2 frames longer than the first
    for far in (15, 24):
        with pytest.raises(FileNotFoundError):
            r.read_many(ep, "scene", [far])
    assert isinstance(AutoReader()._pick(ep, "scene"), FrameCacheReader)


def test_flora_collage_tiles_native_cells_without_padding(tmp_path: Path) -> None:
    d = _cache(tmp_path, "e", {"111": (2048, 1536), "222": (2048, 1536)}, [0, 30])
    ep = _episode(d, ["111", "222"])
    frames = clip_frames(ep, AutoReader(), [0.0, 1.0], ["scene", "cam_222"], collage="flora")
    assert len(frames) == 2 and frames[0].shape == (288, 768, 3)
    legacy = clip_frames(ep, AutoReader(), [0.0, 1.0], ["scene", "cam_222"])
    assert legacy[0].shape == (96, 192, 3)
    blank = clip_frames(ep, AutoReader(), [0.0], ["scene", "cam_222"], blank=True, collage="flora")
    assert blank[0].shape == (288, 768, 3) and int(blank[0].mean()) == 128


def test_blank_flora_clip_without_stored_sizes_matches_the_real_clip(tmp_path: Path) -> None:
    d = _cache(tmp_path, "e", {"111": (2048, 1536), "222": (2048, 1536)}, [0, 30])
    ep = _episode(d, ["111", "222"])
    ep = ep.model_copy(update={"cameras": {k: v.model_copy(update={"width": None, "height": None})
                                           for k, v in ep.cameras.items()}})
    roles = ["scene", "cam_222"]
    real = clip_frames(ep, AutoReader(), [0.0, 1.0], roles, collage="flora")
    blank = clip_frames(ep, AutoReader(), [0.0, 1.0], roles, blank=True, collage="flora")
    assert len(blank) == 2 and blank[0].shape == real[0].shape and int(blank[1].mean()) == 128


def test_score_only_target_reads_either_grammar() -> None:
    assert score_only_target("Rationale: no rationale\n\nScore: 1") == "Score: 1"
    assert score_only_target("Score: 5") == "Score: 5"
    with pytest.raises(ValueError):
        score_only_target("Worst: 3\nQuality:\n" + "3" * 32 + "\nRationales:")


def test_hrdexdb_frame_cache_mode_builds_multiview_refs_and_task(tmp_path: Path) -> None:
    from tests.unit.test_sources_hrdexdb import _cfg, _inspire_trial

    n = 12
    force = np.zeros((n, 6))
    force[6:, 2] = 600
    position = np.full((n, 6), 1000.0)
    position[6:] = 400.0
    eid = _inspire_trial(tmp_path / "raw", position, force)
    frames = tmp_path / "frames"
    _cache(frames, eid, {"22641005": (2048, 1536), "22645029": (2048, 1536)}, [0, 5, 11])
    meta = json.loads((frames / eid / "video.json").read_text())
    for m in meta.values():
        m["n_frames"] = n
    (frames / eid / "video.json").write_text(json.dumps(meta))
    cfg: SourceConfig = _cfg().model_copy(
        update={
            "root": str(tmp_path / "raw"),
            "hrdexdb_force_representation": "tared",
            "hrdexdb_tare_s": 0.1,
            "hrdexdb_frames_root": str(frames),
            "hrdexdb_cameras": ["22645029", "22641005"],
            "hrdexdb_task_template": "Grasp the {object} and hold it.",
        }
    )
    ep = HrdexdbAdapter(cfg).load(eid)
    assert list(ep.cameras) == ["scene", "cam_22641005"]
    assert ep.cameras["scene"].path.endswith("22645029") and ep.cameras["scene"].width == 2048
    assert ep.task == "Grasp the mug and hold it."
    assert ep.meta["camera_serials"] == ["22645029", "22641005"] and ep.meta["norm_by_hand"]
    from tactile_flora.force.normalize import norm_group

    assert norm_group(ep) == "inspire_dftp"


def test_build_samples_score_only_uses_the_released_score_only_prompt(tmp_path: Path) -> None:
    from tactile_flora.prompts import templates as T
    from tactile_flora.train.dataset import build_samples
    from tests.fixtures.train_data import StubTokenizer, windows_source, write_manifest
    from tests.fixtures.verify_ctx import FORCE_CFG

    mp = write_manifest(tmp_path / "m.parquet")

    def src(row):  # type: ignore[no-untyped-def]
        item = windows_source(row)
        target = "Rationale: no rationale\n\nScore: " + (
            "1" if row.episode_id.endswith("0") else "5"
        )
        return item.model_copy(update={"window": item.window.model_copy(update={"target": target})})

    out = build_samples(
        mp, "omitted", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
        windows_source=src, score_only=True, collage="flora",
    )  # fmt: skip
    outcome = [s for s in out if "Score:" in s.target and "Worst" not in s.target]
    assert outcome and all(s.target.startswith("Score: ") and "\n" not in s.target for s in outcome)
    for s in outcome:
        system = s.messages[0]["content"][0]["text"]
        assert system.startswith(T.OUTCOME_SYSTEM_SCORE_ONLY) and s.video.collage == "flora"
