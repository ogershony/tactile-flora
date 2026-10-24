"""Fit the force-only outcome classifier for one source and write its JSON scorer.

Usage: uv run --with scikit-learn python scripts/fit_force_classifier.py \
    [--train data/manifest/train_judge_pf.parquet] \
    [--test data/manifest/reassemble_seg_test.parquet] [--source reassemble_seg] \
    [--out data/verified/reassemble_seg/force_classifier.json]

Train rows are the outcome rows of ``--source`` in the train manifest; features come from the
training path (train.dataset.adapter_windows_source), labels are success = score >= 3, folds are
grouped by recording (the episode id before ``_seg``) so out-of-fold scores never see a sibling
segment. Hyperparameters are force/classifier.fit_cv's defaults, fixed before any test look.
The test manifest is scored through the evaluation path (eval.harness.window_features), which
also checks that both paths give the same feature vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.force import classifier as C
from tactile_flora.prompts.parse import parse_outcome
from tactile_flora.train.dataset import adapter_windows_source
from tactile_flora.verify import manifest as manifest_mod


def _rows(path: Path, source: str, split: str) -> list[manifest_mod.ManifestRow]:
    return [
        r
        for r in manifest_mod.read_manifest(path)
        if r.source == source and r.split == split and r.prompt_kind == "outcome"
    ]


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

    return float(roc_auc_score(y, s))


def main() -> None:
    """Fit, save and test-score the scorer (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/manifest/train_judge_pf.parquet")
    ap.add_argument("--test", default="data/manifest/reassemble_seg_test.parquet")
    ap.add_argument("--source", default="reassemble_seg")
    ap.add_argument("--force-config", default="configs/force.code.yaml")
    ap.add_argument("--out", default="data/verified/reassemble_seg/force_classifier.json")
    a = ap.parse_args()
    root = Path("data")
    fcfg = load_config(a.force_config, ForceConfig)
    src = adapter_windows_source(root, fcfg)

    X, y, groups, ids = [], [], [], []
    for r in _rows(Path(a.train), a.source, "train"):
        it = src(r)
        if it is None or it.features is None:
            continue
        X.append(C.extract_features(it.features))
        y.append(int((parse_outcome(it.window.target).score or 0) >= 3))
        groups.append(r.episode_id.split("_seg")[0])
        ids.append(r.episode_id)
    scorer = C.fit_cv(np.stack(X), np.asarray(y), groups, ids, source=a.source)
    scorer.save(a.out)
    print(f"train {len(y)} (success {sum(y)}), oof {json.dumps(scorer.cv_metrics)} -> {a.out}")

    # Test rows through the evaluation path; compare with the training path's features.
    from tactile_flora.eval.harness import _load_source_cfgs, adapter_provider, window_features
    from tactile_flora.force import normalize

    rows = _rows(Path(a.test), a.source, "test")
    cfgs = _load_source_cfgs({a.source}, Path("configs/sources"), None)
    consts = normalize.load(root / "verified" / a.source / "norm.json")
    by_id = {r.window_id: r for r in rows}
    p_ok, yt, worst = [], [], 0.0
    for b in adapter_provider(rows, cfgs, fcfg, root)():
        for wid, w in zip(b.window_ids, b.windows, strict=True):
            f_eval = window_features(b.episode, w, consts, fcfg, b.shift_s)
            it = src(by_id[wid])
            assert it is not None and it.features is not None
            worst = max(
                worst,
                float(
                    np.max(
                        np.abs(C.extract_features(f_eval) - C.extract_features(it.features))
                    )
                ),
            )
            p_ok.append(scorer.score(f_eval, b.episode.episode_id))
            yt.append(int((parse_outcome(w.target).score or 0) >= 3))
    yt_a, p_a = np.asarray(yt), np.asarray(p_ok)
    ba = 0.5 * (np.mean(p_a[yt_a == 1] >= 0.5) + np.mean(p_a[yt_a == 0] < 0.5))
    print(
        f"test {len(yt)} (failures {int((yt_a == 0).sum())}): AUROC(failure) "
        f"{_auroc(1 - yt_a, 1 - p_a):.3f}, balanced accuracy at P(success) 0.5 {ba:.3f}; "
        f"max |feature difference| eval vs train path {worst:.2e}"
    )


if __name__ == "__main__":
    main()
