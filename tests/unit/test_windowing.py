"""verify/windowing: shift, drop_channel, label assembly, pseudo read-back, label origin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tactile_flora.labels import windows as W
from tactile_flora.labels.merge import merge_labels
from tactile_flora.schema import CheckResult, Episode, Label, Span
from tactile_flora.train import dataset
from tactile_flora.verify import manifest
from tactile_flora.verify import windowing as VW
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.train_data import StubTokenizer, windows_source, write_manifest
from tests.fixtures.verify_ctx import FORCE_CFG, fit_consts


class _Adapter:
    name = "synthetic"

    def __init__(self, native: list[Label] | None = None) -> None:
        self.native = native or []

    def load(self, episode_id: str) -> Episode:
        raise NotImplementedError

    def native_labels(self, episode_id: str) -> list[Label]:
        return self.native


def test_shifted_frame_times_and_resample_shifted() -> None:
    assert VW.shifted_frame_times([1.0, 2.0], 0.25) == [0.75, 1.75]
    ep = make_pad_episode(0)
    times = [i * 0.25 for i in range(10)]
    plain = VW.resample_shifted(ep, times, FORCE_CFG.resample, 0.0)
    shifted = VW.resample_shifted(ep, times, FORCE_CFG.resample, 0.5)
    # force at t equals the unshifted force at t - 0.5, the gripper and frame times stay put
    assert shifted.frame_times == times
    assert shifted.arms["L"].F[4:] == pytest.approx(plain.arms["L"].F[2:8], nan_ok=True)
    assert shifted.arms["L"].g == plain.arms["L"].g


def test_windows_apply_shift_from_checks() -> None:
    ep = make_pad_episode(0)
    consts = fit_consts([ep])
    lab = Label(kind="outcome", origin="source", outcome=5)
    ad = _Adapter([lab])
    base = VW.SourceInputs(adapter=ad)
    shifted = VW.SourceInputs(
        adapter=ad,
        results={ep.episode_id: {"sync": CheckResult(passed=True, value=-0.5, action="shift")}},
    )
    a = VW.episode_windows(ep, FORCE_CFG, consts, "train", inputs=base)
    b = VW.episode_windows(ep, FORCE_CFG, consts, "train", inputs=shifted)
    assert shifted.shift_of(ep.episode_id) == 0.5
    assert [w.force_lines for w in a] != [w.force_lines for w in b]


def test_drop_channel_removes_arm() -> None:
    ep = make_pad_episode(0)
    consts = fit_consts([ep])
    res = {
        ep.episode_id: {
            "sensor_health": CheckResult(passed=False, action="drop_channel", detail="arm=L")
        }
    }
    inputs = VW.SourceInputs(
        adapter=_Adapter([Label(kind="outcome", origin="source", outcome=5)]), results=res
    )
    prepared = VW.prepare_episode(ep, inputs)
    assert prepared is not None and set(prepared.force.arms) == {"R"}
    (w,) = [
        w
        for w in VW.episode_windows(ep, FORCE_CFG, consts, "train", inputs=inputs)
        if w.prompt_kind == "outcome"
    ]
    assert all("pad L F=" not in ln and "pad R F=" in ln for ln in w.force_lines)
    both = {
        ep.episode_id: {
            "sensor_health": CheckResult(passed=False, action="drop_channel", detail="")
        }
    }
    gone = VW.SourceInputs(adapter=_Adapter(), results=both)
    assert VW.prepare_episode(ep, gone) is None
    assert VW.episode_windows(ep, FORCE_CFG, consts, "train", inputs=gone) == []


def test_labels_assembled_from_native_events_and_pseudo(tmp_path: Path) -> None:
    ep = make_pad_episode(0)
    consts = fit_consts([ep])
    out = Label(kind="outcome", origin="source", outcome=5)
    ps = Label(
        kind="dense", origin="pseudo", spans=[Span(t0=0.0, t1=1.0, quality=2, rationale="p")]
    )
    pq = tmp_path / "pseudo.parquet"
    pd.DataFrame(
        [{"episode_id": ep.episode_id, "anchor_s": 0.0, "raw_text": "", "parse_failed": False,
          "label_json": ps.model_dump_json(), "origin": "pseudo", "seed": 0}]
    ).to_parquet(pq, index=False)  # fmt: skip
    assert VW.read_pseudo(tmp_path / "missing.parquet") == {}
    inputs = VW.SourceInputs(adapter=_Adapter([out]), pseudo=VW.read_pseudo(pq))
    labels = VW.assemble_labels(ep, FORCE_CFG, consts, inputs)
    origins = {(lab.kind, lab.origin) for lab in labels}
    assert ("outcome", "source") in origins and ("event", "programmatic") in origins
    assert ("dense", "pseudo") in origins
    kinds = {
        w.prompt_kind for w in VW.episode_windows(ep, FORCE_CFG, consts, "train", inputs=inputs)
    }
    assert kinds == {"dense", "outcome"}


def test_eval_only_source_gets_native_labels_only() -> None:
    ep = make_pad_episode(0).model_copy(update={"source": "baxter"})
    inputs = VW.SourceInputs(adapter=_Adapter([Label(kind="outcome", origin="source", outcome=4)]))
    labels = VW.assemble_labels(ep, FORCE_CFG, fit_consts([ep]), inputs)
    assert [(lab.kind, lab.origin) for lab in labels] == [("outcome", "source")]


def test_label_origin_follows_the_producing_labels() -> None:
    ep = make_pad_episode(0)
    prog = Label(
        kind="event", origin="programmatic", spans=[Span(t0=0, t1=3, quality=1, rationale="x")]
    )
    src = Label(kind="dense", origin="source", spans=[Span(t0=0, t1=1, quality=2, rationale="y")])
    out = Label(kind="outcome", origin="human", outcome=2)
    ws = W.make_windows(ep, None, merge_labels([prog, src, out]), "train")
    by = {w.prompt_kind: w for w in ws}
    assert by["dense"].label_origin == "programmatic"  # 3 s of programmatic vs 1 s of source
    assert by["outcome"].label_origin == "human"
    allnormal = Label(kind="dense", origin="pseudo", spans=[])
    (d,) = W.make_windows(ep, None, merge_labels([allnormal]), "train")
    assert d.label_origin == "pseudo"


def test_build_samples_refuses_eval_only_train_rows(tmp_path: Path) -> None:
    mp = write_manifest(tmp_path / "m.parquet")
    with pytest.raises(manifest.EvalOnlyTrainError):
        dataset.build_samples(
            mp, "real", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
            windows_source=windows_source, eval_only_sources={"synthetic"},
        )  # fmt: skip
    ok = dataset.build_samples(
        mp, "real", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
        windows_source=windows_source, eval_only_sources=set(),
    )  # fmt: skip
    assert ok and np.all([s.n_tokens > 0 for s in ok])


def _phantom(seed: int, folder: str, *, gain: float = 1.0, success: bool = True) -> Episode:
    """A synthetic pad episode posing as PHANTOM; ``gain`` scales the force above baseline."""
    from tactile_flora.schema import PadArmForce

    ep = make_pad_episode(seed, arms=("R",))
    arm = ep.force.arms["R"]
    assert isinstance(arm, PadArmForce)
    weak = PadArmForce(
        f_a=[0.5 + (v - 0.5) * gain for v in arm.f_a], f_b=[0.5 + (v - 0.5) * gain for v in arm.f_b]
    )
    return ep.model_copy(
        update={
            "source": "phantom",
            "episode_id": f"ep_{folder}_{seed}",
            "force": ep.force.model_copy(update={"arms": {"R": weak}}),
            "meta": {
                "task_folder": folder,
                "object": folder.removesuffix("_fail"),
                "success": success,
                "failure_demo": folder.endswith("_fail"),
            },
        }
    )


def test_phantom_inputs_and_under_grasp_label_per_group(tmp_path: Path) -> None:
    from tactile_flora.force import normalize
    from tactile_flora.force.resample import resample

    eps = [_phantom(s, "cup") for s in range(3)] + [
        _phantom(9, "cup_fail", gain=0.3, success=False)
    ]
    by_id = {e.episode_id: e for e in eps}

    class _Phantom(_Adapter):
        name = "phantom"

        def load(self, episode_id: str) -> Episode:
            return by_id[episode_id]

    times = [i * 0.25 for i in range(11)]
    consts = normalize.fit(
        [
            (normalize.norm_group(e), resample(e.force, e.gripper, times, FORCE_CFG.resample))
            for e in eps
        ],
        FORCE_CFG.resample,
        source="phantom",
    )
    assert set(consts.groups) == {"cup"}
    inputs = VW.build_source_inputs(_Phantom(), FORCE_CFG, consts, tmp_path, list(by_id))
    assert set(inputs.phantom_p99) == {"cup"} and set(inputs.phantom_peak_p5) == {"cup"}
    assert 0.8 < inputs.phantom_peak_p5["cup"] < 1.1
    labels = VW.assemble_labels(eps[-1], FORCE_CFG, consts, inputs)
    dense = next(lab for lab in labels if lab.kind == "dense" and lab.origin == "source")
    assert dense.spans[0].rationale == "insufficient grip force on the cup"
    assert 0.5 < dense.spans[0].t0 < 1.5  # first contact, not the episode start
    out = next(lab for lab in labels if lab.kind == "outcome")
    assert out.outcome == 1 and out.spans[0].rationale == dense.spans[0].rationale
    good = VW.assemble_labels(eps[0], FORCE_CFG, consts, inputs)
    assert [lab.kind for lab in good if lab.origin == "source"] == ["outcome"]
    # windows render with the group constants: the weak grasp stays well under level 1.0
    (w,) = [
        w
        for w in VW.episode_windows(eps[-1], FORCE_CFG, consts, "train", inputs=inputs)
        if w.prompt_kind == "outcome"
    ]
    assert all("F=0." in ln or "F=0.0" in ln for ln in w.force_lines)
