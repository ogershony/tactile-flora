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
