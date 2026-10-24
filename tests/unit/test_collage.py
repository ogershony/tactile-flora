"""FLORA-native collage plan: HRDexDB cell sizes, and parity with the released planner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tactile_flora.video.collage import plan_flora_collage


def test_hrdexdb_views_are_384x288_cells() -> None:
    src = (2048, 1536)
    assert plan_flora_collage((src,)) == (((0, 0, 384, 288),), 384, 288)
    rects, w, h = plan_flora_collage((src,) * 4)
    assert (w, h) == (768, 576) and all(r[2:] == (384, 288) for r in rects)
    rects, w, h = plan_flora_collage((src,) * 2)
    assert (w, h) == (768, 288)


def _released_core():  # type: ignore[no-untyped-def]
    roots = [os.environ.get("TACTILE_FLORA_MODEL_DIR", "")]
    roots += [
        str(p)
        for p in Path.home().glob(
            ".cache/huggingface/hub/models--nvidia--Flora-Rewarder/snapshots/*"
        )
    ]
    for r in roots:
        if r and (Path(r) / "batch_utils" / "core.py").is_file():
            sys.path.insert(0, r)
            from batch_utils import core  # type: ignore[import-not-found]

            return core
    return None


@pytest.mark.parametrize(
    "sources",
    [
        ((2048, 1536),),
        ((640, 480), (640, 480)),
        ((1280, 720), (640, 480)),
        ((1280, 720), (1280, 720), (640, 480)),
        ((2048, 1536),) * 4,
        ((1280, 720), (640, 480), (848, 480), (2048, 1536)),
    ],
)
def test_matches_the_released_planner(sources) -> None:  # type: ignore[no-untyped-def]
    core = _released_core()
    if core is None:
        pytest.skip("no Flora-Rewarder snapshot with batch_utils")
    assert plan_flora_collage(tuple(sources)) == core.plan_collage(tuple(sources))
