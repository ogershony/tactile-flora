"""Shared builders for the dataset and SFT tests: tiny manifest, stub tokenizer, windows source."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tactile_flora.force import normalize
from tactile_flora.force.resample import resample
from tactile_flora.prompts.flora import VideoSpec
from tactile_flora.schema import ManifestRow
from tactile_flora.train.dataset import WindowItem
from tactile_flora.verify.manifest import window_id
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.verify_ctx import FORCE_CFG, TIMES, fit_consts, outcome_window

EPS = {f"pad-{s}": make_pad_episode(s) for s in range(4)}
CONSTS = fit_consts(list(EPS.values()))
VOCAB = 97


class StubTokenizer:
    """Whitespace tokenizer: the prompt ends in a <gen> token, the answer follows it."""

    def _text(self, messages: Sequence[dict[str, Any]]) -> list[str]:
        words: list[str] = []
        for m in messages:
            for part in m["content"]:
                words += part["text"].split() if part["type"] == "text" else ["<video>"]
        return words + ["<gen>"]

    def _ids(self, words: list[str]) -> list[int]:
        return [zlib.crc32(w.encode()) % VOCAB for w in words]

    def encode_prompt(self, messages: Sequence[dict[str, Any]], video: VideoSpec) -> list[int]:
        return self._ids(self._text(messages))

    def encode_with_answer(
        self, messages: Sequence[dict[str, Any]], answer: str, video: VideoSpec
    ) -> list[int]:
        return self._ids(self._text(messages) + answer.split() + ["<eos>"])


def make_rows(report_version: str = "rv") -> list[ManifestRow]:
    """One outcome row per episode; pad-3 is a val row."""
    rows: list[ManifestRow] = []
    for eid in EPS:
        rows.append(
            ManifestRow(
                window_id=window_id("synthetic", eid, "outcome", 0.0),
                source="synthetic",
                episode_id=eid,
                session=f"s{eid}",
                split="val" if eid == "pad-3" else "train",
                prompt_kind="outcome",
                anchor_s=0.0,
                frame_hashes=[],
                norm_version="v",
                label_origin="source",
                license="CC0",
                report_version=report_version,
            )
        )
    return rows


def write_manifest(path: Path, report_version: str = "rv") -> Path:
    """Write the tiny manifest parquet and return its path."""
    rows = make_rows(report_version)
    cols: dict[str, Any] = {f: [getattr(r, f) for r in rows] for f in ManifestRow.model_fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)
    return path


def windows_source(row: ManifestRow) -> WindowItem:
    """Resolve a row to its window with normalized features; targets vary in length."""
    ep = EPS[row.episode_id]
    w = outcome_window(ep, CONSTS, row.split)
    n = int(row.episode_id.split("-")[1])
    w = w.model_copy(update={"target": "Score: " + " ".join(["ok"] * n) + " 5"})
    feats = normalize.apply(
        resample(ep.force, ep.gripper, TIMES, FORCE_CFG.resample), CONSTS, FORCE_CFG.render
    )
    return WindowItem(window=w, task=ep.task, features=feats)
