"""The force-only classifier: features are label-free, the JSON export matches sklearn, and
training episodes are scored out-of-fold."""

from __future__ import annotations

import numpy as np
import pytest

from tactile_flora.force import classifier as C
from tactile_flora.schema import ArmFeatures, ResampledForce


def _features(seed: int, strong: bool) -> ResampledForce:
    rng = np.random.default_rng(seed)
    n = 24
    times = [0.5 * i for i in range(n)]
    F = np.zeros(n)
    hold = slice(8, 20)
    F[hold] = (1.6 if strong else 0.5) + 0.05 * rng.standard_normal(12)
    if not strong:
        F[14:20] *= 0.3  # grip fades late
    g = np.full(n, 0.1)
    g[hold] = 0.95
    c = (F > 0.05).astype(int).tolist()
    raw = [[float(v)] * 10 for v in F[::2]]
    return ResampledForce(
        channel="pad",
        frame_times=times,
        arms={
            "R": ArmFeatures(
                F=F.tolist(),
                delta=(0.1 * rng.standard_normal(n)).tolist(),
                dF=np.gradient(F, 0.25).tolist(),
                g=g.tolist(),
                c=c,
                raw_groups=raw,
            )
        },
    )


def test_extract_features_is_fixed_length_and_describes_the_hold() -> None:
    x = C.extract_features(_features(0, strong=True))
    assert x.shape == (len(C.FEATURE_NAMES),)
    named = dict(zip(C.FEATURE_NAMES, x, strict=True))
    assert named["contacts"] == 1.0
    assert named["hold_s"] == pytest.approx(6.0)
    assert 1.4 < named["hold_median"] < 1.8
    assert named["late_early"] == pytest.approx(1.0, abs=0.15)
    weak = dict(zip(C.FEATURE_NAMES, C.extract_features(_features(0, strong=False)), strict=True))
    assert weak["late_early"] < 0.5 and weak["hold_median"] < named["hold_median"]
    with pytest.raises(ValueError, match="missing"):
        C.extract_features(_features(0, strong=True), arm="L")


def test_fit_cv_exports_sklearn_exactly_and_scores_training_episodes_out_of_fold() -> None:
    pytest.importorskip("sklearn")
    from sklearn.ensemble import GradientBoostingClassifier

    feats = [_features(i, strong=i % 3 != 0) for i in range(60)]
    X = np.stack([C.extract_features(f) for f in feats])
    y = np.asarray([int(i % 3 != 0) for i in range(60)])
    groups = [f"obj{i // 4}" for i in range(60)]
    ids = [f"ep{i}" for i in range(60)]
    scorer = C.fit_cv(X, y, groups, ids, source="synthetic", params={"n_estimators": 20})
    full = GradientBoostingClassifier(**scorer.params).fit(X, y)
    for i in range(0, 60, 7):
        assert scorer.model.predict_proba(X[i]) == pytest.approx(
            full.predict_proba(X[i : i + 1])[0, 1]
        )
    assert set(scorer.fold_of_episode) == set(ids)
    assert len(scorer.fold_models) == 5
    # The out-of-fold score comes from the fold model, not the full model.
    k = scorer.fold_of_episode["ep3"]
    assert scorer.score(feats[3], "ep3") == pytest.approx(scorer.fold_models[k].predict_proba(X[3]))
    assert scorer.score(feats[3], "unseen") == pytest.approx(scorer.model.predict_proba(X[3]))
    assert scorer.cv_metrics["oof_auroc"] > 0.9
    round_trip = C.ForceScorer.model_validate_json(scorer.model_dump_json())
    assert round_trip.score(feats[5], None) == scorer.score(feats[5], None)


def test_summary_line_shows_classifier_probability(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from tactile_flora.force.render import summary_lines

    feats = [_features(i, strong=i % 2 == 0) for i in range(40)]
    X = np.stack([C.extract_features(f) for f in feats])
    y = np.asarray([int(i % 2 == 0) for i in range(40)])
    scorer = C.fit_cv(
        X, y, [f"o{i // 2}" for i in range(40)], [f"e{i}" for i in range(40)],
        source="synthetic", params={"n_estimators": 20},
    )  # fmt: skip
    path = tmp_path / "clf.json"
    scorer.save(path)
    (plain,) = summary_lines(feats[0])
    (scored,) = summary_lines(feats[0], str(path), "e0")
    assert "P(success)" not in plain
    assert scored.startswith(plain)
    assert "force-only classifier P(success)=" in scored
    value = float(scored.rsplit("=", 1)[1])
    assert 0.0 <= value <= 1.0


def test_code_block_puts_the_classifier_line_under_the_summary_for_its_source_only(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    pytest.importorskip("sklearn")
    from pathlib import Path

    from tactile_flora.config import ForceConfig, load_config
    from tactile_flora.force import render

    feats = [_features(i, strong=i % 3 != 0) for i in range(40)]
    X = np.stack([C.extract_features(f) for f in feats])
    y = np.asarray([int(i % 3 != 0) for i in range(40)])
    ids = [f"ep{i}" for i in range(40)]
    groups = [f"rec{i // 4}" for i in range(40)]
    scorer = C.fit_cv(X, y, groups, ids, source="src", params={"n_estimators": 10})
    path = tmp_path / "clf.json"
    scorer.save(path)
    root = Path(__file__).resolve().parents[2]
    code = load_config(root / "configs" / "force.code.yaml", ForceConfig).render
    cfg = code.model_copy(update={"classifier_json": str(path)})
    assert render.classifier_line(code, feats[0], "src", "ep0") is None  # no scorer configured
    assert render.classifier_line(cfg, feats[0], "other", "ep0") is None  # another source
    line = render.classifier_line(cfg, feats[3], "src", "ep3")
    k = scorer.fold_of_episode["ep3"]
    assert (
        line == f"Force-only classifier P(success)={scorer.fold_models[k].predict_proba(X[3]):.2f}"
    )
    lines = render.render_lines(feats[3], feats[3].frame_times, "pad", cfg=cfg)
    block = render.render_block(lines, cfg, feats[3], "ep3", source="src").split("\n")
    assert block[0] == cfg.header_code and block[1].startswith("Summary ") and block[2] == line
    assert block[3:] == lines
    # another source: same block without the classifier line
    other = render.render_block(lines, cfg, feats[3], "ep3", source="other").split("\n")
    assert other == block[:2] + block[3:]
    # compact: header_compact, summary, classifier line, no frame lines
    compact = cfg.model_copy(update={"frame_lines": False})
    assert render.render_block(lines, compact, feats[3], "ep3", source="src").split("\n") == [
        compact.header_compact,
        block[1],
        line,
    ]
