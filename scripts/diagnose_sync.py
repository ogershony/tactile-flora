"""Write force/gripper timing diagnostics without changing acceptance or timestamps.

Correlation peak neighborhoods describe sensitivity to the search grid, not statistical
confidence intervals or proof of a clock offset. Native timestamp summaries come from the
adapter; plots are for review alongside the original video.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force.resample import resample
from tactile_flora.schema import VerifyThresholds
from tactile_flora.sources.base import make_adapter
from tactile_flora.verify.checks import _xcorr_lag


def correlation_profile(
    force: np.ndarray, closure: np.ndarray, step: float, search_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the verifier's bounded correlation scores; empty when uncheckable."""
    expected = _xcorr_lag(force, closure, step, search_s)
    if expected is None:
        return np.array([]), np.array([])
    force = (force - force.mean()) / force.std()
    closure = (closure - closure.mean()) / closure.std()
    scores = np.correlate(force, closure, mode="full")
    lags = np.arange(-(closure.size - 1), force.size) * step
    mask = np.abs(lags) <= search_s
    lags, scores = lags[mask], scores[mask]
    if not np.isclose(lags[int(np.argmax(scores))], expected):
        raise RuntimeError("diagnostic correlation differs from the acceptance checker")
    return lags, scores


def run(source_path: Path, force_path: Path, output: Path) -> dict[str, Any]:
    """Inspect every downloaded episode and save auditable traces and plots."""
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(source_path, SourceConfig)
    force_cfg = load_config(force_path, ForceConfig)
    adapter = make_adapter(cfg.name, cfg)
    ids = sorted(adapter.discover(Path(cfg.root)))
    thresholds = VerifyThresholds()
    rows: list[dict[str, Any]] = []
    episode_lags: list[float] = []
    for eid in ids:
        ep = adapter.load(eid)
        times = np.arange(int(ep.duration_s / thresholds.sync_grid_s) + 1) * thresholds.sync_grid_s
        feats = resample(ep.force, ep.gripper, times, force_cfg.resample)
        row: dict[str, Any] = {
            "episode": eid,
            "duration_s": ep.duration_s,
            "native_timing": ep.meta.get("native_timing", {}),
            "arms": {},
        }
        for arm, feature in feats.arms.items():
            if feature.g is None:
                continue
            force = np.nan_to_num(np.asarray(feature.F))
            closure = np.asarray(feature.g)
            lags, scores = correlation_profile(
                force, closure, thresholds.sync_grid_s, thresholds.sync_search_s
            )
            if not lags.size:
                row["arms"][arm] = {"checkable": False}
                continue
            best = int(np.argmax(scores))
            denominator = max(abs(float(scores[best])), 1e-12)
            near = lags[scores >= scores[best] - denominator * 0.001]
            order = np.argsort(scores)[-5:][::-1]
            row["arms"][arm] = {
                "checkable": True,
                "lag_s": float(lags[best]),
                "near_peak_fraction": 0.999,
                "near_peak_lags_s": near.tolist(),
                "peaks": [
                    {"lag_s": float(lags[i]), "relative_score": float(scores[i] / denominator)}
                    for i in order
                ],
            }
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
            axes[0].plot(times, force, label="Force (raw units)")
            axes[0].set(xlabel="Episode time (s)", ylabel="Force")
            other = axes[0].twinx()
            other.plot(times, closure, color="tab:orange", alpha=0.7, label="Gripper closure")
            other.set(ylabel="Gripper closure", ylim=(-0.05, 1.05))
            axes[0].legend(loc="upper left")
            other.legend(loc="upper right")
            axes[1].plot(lags * 1000, scores / denominator)
            axes[1].scatter([lags[best] * 1000], [scores[best] / denominator], color="black")
            for bound in (-0.1, 0.1):
                axes[1].axvline(bound * 1000, color="tab:orange", linestyle="--")
            for bound in (-0.25, 0.25):
                axes[1].axvline(bound * 1000, color="tab:red", linestyle=":")
            axes[1].set(xlabel="Force lag behind closure (ms)", ylabel="Correlation / peak")
            fig.suptitle(f"{eid}, arm {arm}: lag {lags[best] * 1000:+.0f} ms")
            name = f"{eid.replace('/', '_')}_{arm}"
            fig.savefig(output / f"{name}.png", dpi=130)
            plt.close(fig)
            np.savez_compressed(
                output / f"{name}.npz",
                times=times,
                force=force,
                closure=closure,
                lags=lags,
                scores=scores,
            )
        found = [a["lag_s"] for a in row["arms"].values() if a["checkable"]]
        if found:
            row["worst_lag_s"] = max(found, key=abs)
            episode_lags.append(abs(row["worst_lag_s"]))
        rows.append(row)
    median = float(np.median(episode_lags)) if episode_lags else None
    result = {
        "source": cfg.name,
        "episodes": rows,
        "median_abs_lag_s": median,
        "source_median_passed": median < thresholds.sync_median_s if median is not None else None,
        "thresholds": {"median_s": thresholds.sync_median_s, "maximum_s": thresholds.sync_max_s},
        "interpretation": (
            "Peak neighborhoods describe grid sensitivity, not clock-offset confidence bounds."
        ),
    }
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "median_abs_lag_s": median, "output": str(output)}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-config", type=Path, default=Path("configs/force.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.force_config, args.output)
