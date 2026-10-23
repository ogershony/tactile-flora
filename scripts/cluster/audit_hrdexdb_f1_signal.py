"""Audit how much grasp-outcome signal the inspire_f1 fingertip force carries, before training.

Reads every labeled inspire_f1 trial's sensor files under ``<base>/raw/inspire_f1`` (videos are
not needed) and the already-downloaded inspire_dftp trials, computes the same label-free
per-trial force statistics for both hands, and reports object-grouped cross-validated AUROC
of a gradient-boosted classifier per hand and pooled. Writes ``<base>/f1-signal-audit.json``.

Run from the repository root: ``.venv/bin/python scripts/cluster/audit_hrdexdb_f1_signal.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from tactile_flora.sources.hrdexdb import F1_PADS, _signed_force, load_npy, tared_excess

BASE = Path("data/acceptance/hrdexdb-expanded-v4")


def trial_stats(force: np.ndarray, t: np.ndarray, closure: np.ndarray, t_pos: np.ndarray) -> dict:
    """Label-free statistics of one trial's tared force (T, C) on its own clock."""
    excess, _ = tared_excess(force, t - t[0], 1.0)
    total = excess.sum(axis=1)
    top = np.sort(excess.max(axis=0))[::-1]
    sigma = max(float(np.std(total[: max(5, int(np.searchsorted(t - t[0], 1.0)))])), 1e-6)
    tau = max(5.0 * sigma, 0.05 * max(float(total.max()), 1e-6))
    contact = total > tau
    hold_s = float(np.trapezoid(contact.astype(float), t)) if contact.any() else 0.0
    runs = np.flatnonzero(np.diff(np.concatenate([[0], contact.astype(int), [0]])))
    segments = len(runs) // 2
    longest = max(
        (t[min(b, len(t)) - 1] - t[a] for a, b in zip(runs[::2], runs[1::2], strict=True)),
        default=0.0,
    )
    held = total[contact]
    n = len(held)
    early = held[: max(1, n // 3)].mean() if n else 0.0
    late = held[-max(1, n // 3) :].mean() if n else 0.0
    g = closure
    close_frac = float(np.mean(g > 0.5)) if len(g) else 0.0
    return {
        "peak": float(total.max()),
        "peak_top1": float(top[0]),
        "peak_top2": float(top[1]) if len(top) > 1 else 0.0,
        "hold_s": hold_s,
        "longest_s": float(longest),
        "segments": float(segments),
        "late_over_early": float(late / early) if early > 0 else 0.0,
        "hold_mean": float(held.mean()) if n else 0.0,
        "last_contact_frac": float((t[contact][-1] - t[0]) / (t[-1] - t[0]))
        if contact.any()
        else 0.0,
        "closed_frac": close_frac,
        "duration_s": float(t[-1] - t[0]),
    }


def load_f1(trial: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an inspire_f1 trial: (T, 8) normal force, its clock, a closure proxy, its clock."""
    frames = load_npy(trial / "raw/hand/right_tactile.npy")
    force = np.array([[float(f[f"{p}_normal_force"]) for p in F1_PADS] for f in frames])
    t = np.asarray(load_npy(trial / "raw/hand/right_tactile_time.npy"), dtype=float)
    joints = np.asarray(load_npy(trial / "raw/hand/right_joint_states.npy"), dtype=float)
    tj = np.asarray(load_npy(trial / "raw/hand/right_joint_states_time.npy"), dtype=float)
    closure = 1.0 - (joints.mean(axis=1) - joints.min()) / max(np.ptp(joints.mean(axis=1)), 1e-6)
    return force, t, closure, tj


def load_dftp(trial: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an inspire_dftp trial: (T, 6) signed force, its clock, a closure proxy, its clock."""
    force = _signed_force(np.load(trial / "raw/hand/force.npy")).reshape(-1, 6)
    t = np.asarray(np.load(trial / "raw/hand/time.npy"), dtype=float).reshape(-1)
    pos = np.asarray(np.load(trial / "raw/hand/position.npy"), dtype=float).reshape(len(force), -1)
    m = pos.mean(axis=1)
    closure = 1.0 - (m - m.min()) / max(np.ptp(m), 1e-6)
    return force, t, closure, t


def main() -> None:
    """Compute per-hand force statistics and grouped-CV separability; write the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    raw = args.base / "raw"
    catalog = pd.read_parquet(raw / "metadata/episodes.parquet")
    rows = []
    skipped: list[str] = []
    for r in catalog[catalog.success.notna()].itertuples():
        trial = raw / r.episode_id
        try:
            if r.embodiment == "inspire_f1":
                if not (trial / "raw/hand/right_tactile.npy").is_file():
                    continue
                force, t, closure, _ = load_f1(trial)
            elif r.embodiment == "inspire_dftp":
                if not (trial / "raw/hand/force.npy").is_file():
                    continue
                force, t, closure, _ = load_dftp(trial)
            else:
                continue
            if len(t) < 50:
                raise ValueError("too short")
            stats = trial_stats(force, t, closure, t)
        except Exception as exc:  # noqa: BLE001 - audit records, does not stop
            skipped.append(f"{r.episode_id}: {type(exc).__name__}: {exc}"[:160])
            continue
        rows.append(
            {
                "episode_id": r.episode_id,
                "hand": r.embodiment,
                "object": r.object_id,
                "success": int(bool(r.success)),
                **stats,
            }
        )
    df = pd.DataFrame(rows)
    features = [c for c in df.columns if c not in ("episode_id", "hand", "object", "success")]
    report: dict = {"n": int(len(df)), "skipped": skipped, "features": features, "per_hand": {}}

    def cv_auroc(d: pd.DataFrame, label: str) -> dict:
        X = d[features].to_numpy(dtype=float)
        y = d["success"].to_numpy(dtype=int)
        g = d["object"].to_numpy()
        preds = np.zeros(len(d))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            m = GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
            )
            m.fit(X[tr], y[tr])
            preds[te] = m.predict_proba(X[te])[:, 1]
        out = {
            "clips": int(len(d)),
            "failures": int((y == 0).sum()),
            "objects": int(d.object.nunique()),
            "majority": round(float(max(y.mean(), 1 - y.mean())), 3),
            "cv_auroc": round(float(roc_auc_score(y, preds)), 3),
            "cv_acc": round(float(((preds >= 0.5) == y).mean()), 3),
            "peak_only_auroc": round(float(roc_auc_score(y, d["peak"])), 3),
            "hold_s_only_auroc": round(float(roc_auc_score(y, d["hold_s"])), 3),
            "failures_without_contact": int(np.sum((y == 0) & (d["hold_s"].to_numpy() < 0.5))),
            "successes_without_contact": int(np.sum((y == 1) & (d["hold_s"].to_numpy() < 0.5))),
            "median_peak_success": round(float(d[d.success == 1].peak.median()), 1),
            "median_peak_failure": round(float(d[d.success == 0].peak.median()), 1),
        }
        print(f"{label:14s} " + "  ".join(f"{k}={v}" for k, v in out.items()), flush=True)
        return out

    for hand in sorted(df.hand.unique()):
        report["per_hand"][hand] = cv_auroc(df[df.hand == hand], hand)
    report["pooled"] = cv_auroc(df, "pooled")
    # Cross-hand transfer: fit on one hand, score the other (scale-free features only).
    scale_free = [f for f in features if f not in ("peak", "peak_top1", "peak_top2", "hold_mean")]
    transfer = {}
    for a, b in [("inspire_dftp", "inspire_f1"), ("inspire_f1", "inspire_dftp")]:
        da, db = df[df.hand == a], df[df.hand == b]
        if len(da) and len(db):
            m = GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
            )
            m.fit(da[scale_free].values, da.success.values)
            transfer[f"{a}->{b}"] = round(
                float(
                    roc_auc_score(db.success.values, m.predict_proba(db[scale_free].values)[:, 1])
                ),
                3,
            )
    report["cross_hand_transfer_auroc_scale_free"] = transfer
    print("transfer", transfer)
    (args.base / "f1-signal-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    df.to_parquet(args.base / "f1-signal-audit-trials.parquet")


if __name__ == "__main__":
    main()
