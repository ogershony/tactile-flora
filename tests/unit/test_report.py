import json
from pathlib import Path

import pyarrow.parquet as pq

from tactile_flora.schema import CheckResult
from tactile_flora.verify import report


def _results() -> report.Results:
    ok = CheckResult(passed=True)
    return {
        "e1": {
            "integrity": ok,
            "sync": CheckResult(passed=True, value=0.05, action="shift", detail="L=+50ms"),
            "license": ok,
            "spot_check": CheckResult(passed=False, detail="awaiting human review"),
        },
        "e2": {
            "integrity": CheckResult(passed=False, action="drop_episode", detail="bad timestamps"),
            "sync": CheckResult(passed=False, value=0.4, action="drop_episode", detail="400ms"),
            "license": ok,
            "spot_check": CheckResult(passed=False, detail="awaiting human review"),
        },
    }


def test_write_report(tmp_path: Path) -> None:
    out = tmp_path / "verified" / "synthetic"
    path = report.write_report("synthetic", _results(), out)
    assert path == out / "report.md"
    text = path.read_text()
    assert "| integrity | 1 | 1 | drop_episode=1 | - |" in text
    assert "| sync | 1 | 1 | drop_episode=1 | shift=1 |" in text
    assert "### e2" in text and "bad timestamps" in text and "### e1" in text
    assert "spot_check: FAIL" in text

    rows = pq.read_table(out / "checks.parquet").to_pylist()
    assert len(rows) == 8
    assert set(rows[0]) == {"episode_id", "check", "passed", "value", "action", "detail"}
    sync = [r for r in rows if r["episode_id"] == "e2" and r["check"] == "sync"][0]
    assert sync["value"] == 0.4 and sync["passed"] is False
    assert [r["value"] for r in rows if r["check"] == "license"] == [None, None]


def test_read_checks_roundtrip(tmp_path: Path) -> None:
    res = _results()
    report.write_report("synthetic", res, tmp_path)
    assert report.read_checks(tmp_path) == res


def test_source_verdicts() -> None:
    assert report.source_verdicts(_results()) == {"spot_check": False, "sync_median": False}
    assert report.source_verdicts({"e": {"integrity": CheckResult(passed=False)}}) == {}


def test_sync_median_uses_absolute_lags_including_rejected_episodes() -> None:
    results = {
        "early": {"sync": CheckResult(passed=True, value=-0.1, action="shift")},
        "late": {"sync": CheckResult(passed=False, value=0.4, action="drop_episode")},
        "unknown": {"sync": CheckResult(passed=True, detail="not checkable")},
    }
    assert report.source_verdicts(results)["sync_median"] is False
    results["late"]["sync"] = CheckResult(passed=True, value=0.05, action="shift")
    assert report.source_verdicts(results)["sync_median"] is True


def test_all_pass_report(tmp_path: Path) -> None:
    res = {"e": {"license": CheckResult(passed=True)}}
    assert "None." in report.write_report("s", res, tmp_path).read_text()


def _norm(grasp: list[float], free: list[float], p99: list[float]) -> CheckResult:
    import json

    detail = json.dumps({"grasp": grasp, "free": free, "p99": p99, "problems": []})
    return CheckResult(passed=True, detail=detail)


def test_normalization_verdict_is_pooled() -> None:
    light = CheckResult(passed=False, detail=_norm([0.4], [0.0], [1.5]).detail)
    res = {
        "a": {"normalization": _norm([1.0], [0.0], [1.5])},
        "b": {"normalization": _norm([1.1], [0.02], [2.0])},
        "c": {"normalization": light},  # one light grasp does not fail the source
    }
    assert report.source_verdicts(res) == {"normalization": True}
    res["d"] = {"normalization": _norm([0.5], [0.0], [1.5])}
    res["e"] = {"normalization": _norm([0.5], [0.0], [1.5])}
    assert report.source_verdicts(res) == {"normalization": False}
    free = {"a": {"normalization": _norm([1.0], [0.3], [1.5])}}
    assert report.source_verdicts(free) == {"normalization": False}
    p99 = {"a": {"normalization": _norm([1.0], [0.0], [3.5])}}
    assert report.source_verdicts(p99) == {"normalization": False}
    nodetail = {"a": {"normalization": CheckResult(passed=False, detail="no constants")}}
    assert report.source_verdicts(nodetail) == {"normalization": False}


def test_sync_median_gate_is_configurable() -> None:
    results = {
        f"e{i}": {"sync": CheckResult(passed=True, value=v, action="shift")}
        for i, v in enumerate((0.15, -0.15, 0.12))
    }
    assert report.source_verdicts(results)["sync_median"] is False  # spec default 0.1 s
    assert report.source_verdicts(results, sync_median_s=0.2)["sync_median"] is True
    assert report.source_verdicts(results, sync_median_s=0.1)["sync_median"] is False


def test_pooled_normalization_skips_episodes_excluded_from_the_fit() -> None:
    def _n(grasp: float, excluded: bool = False) -> CheckResult:
        doc: dict[str, object] = {"grasp": [grasp], "free": [0.0], "p99": [grasp], "problems": []}
        if excluded:
            doc["excluded"] = True
        return CheckResult(passed=True, detail=json.dumps(doc))

    typical = {f"s{i}": {"normalization": _n(1.0)} for i in range(3)}
    squeezed = {f"f{i}": {"normalization": _n(2.0, excluded=True)} for i in range(3)}
    assert report.source_verdicts({**typical, **squeezed}) == {"normalization": True}
    # the same off-scale episodes without the marker fail the pooled verdict (median 1.5)
    counted = {f"f{i}": {"normalization": _n(2.0)} for i in range(3)}
    assert report.source_verdicts({**typical, **counted}) == {"normalization": False}
    # a pool of excluded episodes only has nothing to judge
    assert report.source_verdicts(squeezed) == {"normalization": False}


def test_pooled_grasp_band_is_configurable(tmp_path: Path) -> None:
    from tactile_flora.force.normalize import ArmNorm, NormConstants

    def _n(grasp: float) -> CheckResult:
        doc = {"grasp": [grasp], "free": [0.0], "p99": [2.0], "problems": []}
        return CheckResult(passed=True, detail=json.dumps(doc))

    heavy = {f"e{i}": {"normalization": _n(1.25)} for i in range(3)}
    assert report.source_verdicts(heavy) == {"normalization": False}
    assert report.source_verdicts(heavy, norm_grasp_range=(0.8, 1.3)) == {"normalization": True}
    assert report.source_verdicts(heavy, norm_grasp_range=(0.8, 1.2)) == {"normalization": False}
    arm = ArmNorm(baseline=0.46, grasp=2.5, sigma=0.08, tau=0.7, scale=2.04, p99_fprime=5.1)
    norm = NormConstants(source="s", channel="wrist", groups={"all": {"R": arm}})
    text = report.write_report(
        "s", heavy, tmp_path, norm=norm, norm_grasp_range=(0.8, 1.3)
    ).read_text()
    assert "- normalization: pass" in text and "pooled grasp band [0.8, 1.3]" in text
    plain = report.write_report("s", heavy, tmp_path / "b", norm=norm).read_text()
    assert "- normalization: FAIL" in plain and "pooled grasp band [0.8, 1.2]" in plain


def test_norm_p99_gate_is_configurable_and_constants_are_reported(tmp_path: Path) -> None:
    from tactile_flora.force.normalize import ArmNorm, NormConstants

    def _n(p99: float) -> CheckResult:
        doc = {"grasp": [1.0], "free": [0.0], "p99": [p99], "problems": []}
        return CheckResult(passed=True, detail=json.dumps(doc))

    heavy_tail = {f"e{i}": {"normalization": _n(44.0)} for i in range(3)}
    assert report.source_verdicts(heavy_tail) == {"normalization": False}
    assert report.source_verdicts(heavy_tail, norm_p99_max=60) == {"normalization": True}
    assert report.source_verdicts(heavy_tail, norm_p99_max=40) == {"normalization": False}
    arm = ArmNorm(baseline=0.43, grasp=0.69, sigma=0.01, tau=0.46, scale=0.26, p99_fprime=44.0)
    norm = NormConstants(source="s", channel="pad", groups={"all": {"L": arm}})
    text = report.write_report("s", heavy_tail, tmp_path, norm_p99_max=60, norm=norm).read_text()
    assert "- normalization: pass" in text and "gate 60" in text
    assert "| all | L | 0.43 | 0.69 | 0.01 | 0.46 | threshold (g 0.6) | 44.00 |" in text
    plain = report.write_report("s", heavy_tail, tmp_path / "b").read_text()
    assert "- normalization: FAIL" in plain and "Normalization constants" not in plain
