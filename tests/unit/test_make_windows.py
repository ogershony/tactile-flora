from tactile_flora.labels.merge import merge_labels
from tactile_flora.labels.windows import make_windows
from tactile_flora.prompts.parse import parse_dense, parse_outcome
from tactile_flora.schema import Label, Span
from tests.fixtures.synthetic import make_pad_episode


def test_35s_episode_windows() -> None:
    ep = make_pad_episode(0, n_frames=139)  # duration 35.0 s
    assert ep.duration_s == 35.0
    ep = ep.model_copy(
        update={
            "cameras": {
                "scene": ep.cameras["scene"].model_copy(update={"fps": 4.0, "n_frames": 140})
            }
        }
    )
    labels = [
        Label(kind="outcome", origin="source", outcome=1),
        Label(
            kind="event",
            origin="programmatic",
            spans=[
                Span(
                    t0=12.0, t1=14.0, quality=1, rationale="the object drops from the left gripper"
                )
            ],
        ),
    ]
    lines = {0.0: ["t=0.00 pad L F=0.00"]}
    ws = make_windows(ep, lines, merge_labels(labels), "train")
    dense = [w for w in ws if w.prompt_kind == "dense"]
    outcome = [w for w in ws if w.prompt_kind == "outcome"]
    assert [w.anchor_s for w in dense] == [0, 10, 20, 25.25]
    assert len(outcome) == 1 and len(ws) == 5
    assert dense[0].force_lines == ("t=0.00 pad L F=0.00",) and dense[1].force_lines == ()
    assert dense[1].episode_ref == f"synthetic/{ep.episode_id}"
    d = parse_dense(dense[1].target)
    assert not d.parse_failed if hasattr(d, "parse_failed") else True
    assert d.worst == 1 and d.digits[8:12] == [1, 1, 1, 1]  # t 12-14 -> groups 9-12 of anchor 10
    assert "the object drops" in dense[1].target
    o = outcome[0]
    assert o.anchor_s == 0.0 and len(o.frame_times) % 2 == 0 and o.frame_times[0] == 0.0
    assert o.frame_times[-1] == 139 / 4.0
    assert parse_outcome(o.target).score == 1


def test_no_labels_no_windows() -> None:
    assert make_windows(make_pad_episode(0), None, [], "train") == []
