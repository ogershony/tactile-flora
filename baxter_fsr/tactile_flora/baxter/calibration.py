"""Pure calibration math: monotone piecewise-linear counts-to-newtons maps and the JSON file.

Owns: ``PiecewiseLinear``, ``fit_pad`` (isotonic-adjusted monotone fit), ``PadRecord`` (the raw
points behind one pad's fit), ``CalibrationFile`` (save / load of configs/baxter_calibration.json)
and the sensor-unconfirmed warning text.
Must never: touch serial, ROS or episodes; hold sensor physics (the map is purely empirical).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from tactile_flora.config import BaxterConfig

SENSOR_UNCONFIRMED_WARNING = (
    "WARNING: sensor identity is UNCONFIRMED (sensor_confirmed: false in configs/baxter.yaml). "
    "R_ref and calibration are placeholders for ASIN B0FGD267GK (thin film, rated 20 g-2 kg) "
    "and must be redone once its resistance curve is measured."
)


def sensor_warning(cfg: BaxterConfig) -> str | None:
    """Return the startup warning while the sensor is unconfirmed, else None."""
    return None if cfg.sensor_confirmed else SENSOR_UNCONFIRMED_WARNING


class PiecewiseLinear(BaseModel):
    """Monotone non-decreasing piecewise-linear map; clamps outside the fitted counts range."""

    counts: list[float]
    newtons: list[float]

    @model_validator(mode="after")
    def _valid(self) -> PiecewiseLinear:
        if len(self.counts) != len(self.newtons) or not self.counts:
            raise ValueError("counts and newtons must be equal-length and non-empty")
        if any(b <= a for a, b in zip(self.counts, self.counts[1:], strict=False)):
            raise ValueError("counts must be strictly increasing")
        if any(b < a for a, b in zip(self.newtons, self.newtons[1:], strict=False)):
            raise ValueError("newtons must be non-decreasing")
        return self

    def apply(self, counts: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Map raw counts to newtons (always returns a float64 array)."""
        return np.interp(np.asarray(counts, dtype=np.float64), self.counts, self.newtons)


def _pav(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the non-decreasing least-squares fit of ``y``."""
    blocks: list[list[float]] = []  # [sum, count]
    for v in y:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    return np.concatenate([np.full(int(n), s / n) for s, n in blocks])


def fit_pad(counts: Sequence[float], newtons: Sequence[float]) -> PiecewiseLinear:
    """Fit a monotone piecewise-linear counts-to-newtons map for one pad.

    Points are sorted by counts, duplicate counts averaged, and newtons made non-decreasing by
    isotonic regression (pool adjacent violators).
    """
    if len(counts) != len(newtons) or len(counts) < 2:
        raise ValueError("need at least two (counts, newtons) points of equal length")
    x = np.asarray(counts, dtype=np.float64)
    y = np.asarray(newtons, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    if ux.size < 2:
        raise ValueError("need at least two distinct count values")
    return PiecewiseLinear(counts=ux.tolist(), newtons=_pav(uy).tolist())


class PadRecord(BaseModel):
    """How one pad's fit was made: date, rig and the raw points (newtons = grams / 1000 * g)."""

    date: str
    method: str
    unloaded_counts: float
    points: list[tuple[float, float]]  # (grams, median counts) used by the fit
    dropped: list[tuple[float, float, str]] = Field(default_factory=list)  # + reason


class CalibrationFile(BaseModel):
    """configs/baxter_calibration.json: one fit per pad plus provenance.

    A file may cover only some channels (one gripper fitted); ``apply`` reads the others as 0 N.
    """

    date: str
    r_ref_ohm: float
    sensor_confirmed: bool
    pads: dict[str, PiecewiseLinear]
    records: dict[str, PadRecord] = Field(default_factory=dict)

    @property
    def version(self) -> str:
        """Calibration version string recorded in every episode's meta.json."""
        return f"cal-{self.date}" + ("" if self.sensor_confirmed else "-unconfirmed")

    def missing(self, channels: Sequence[str]) -> list[str]:
        """Channels with no fit in this file."""
        return [c for c in channels if c not in self.pads]

    def apply(self, channels: Sequence[str], counts: Sequence[float]) -> list[float]:
        """Convert one tick of raw counts (ordered as ``channels``) to newtons; unfitted are 0."""
        return [
            float(self.pads[c].apply(v)) if c in self.pads else 0.0
            for c, v in zip(channels, counts, strict=True)
        ]

    def save(self, path: Path | str) -> Path:
        """Write the file as indented JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
        return p

    @classmethod
    def load(cls, path: Path | str) -> CalibrationFile:
        """Read and validate a calibration file."""
        return cls.model_validate_json(Path(path).read_text())
