"""Protect reviewed subset construction from missing decisions and split/label leakage."""

from copy import deepcopy

import pytest

from scripts.cluster.clean_hrdexdb_pilot import select_episodes


def sample():
    documents = {
        "train": {
            "episodes": [{"episode_id": "hand/apple/0", "object_id": "apple", "success": True}]
        },
        "val": {"episodes": [{"episode_id": "hand/cup/0", "object_id": "cup", "success": False}]},
    }
    review = {
        "records": [
            {
                "episode_id": "hand/apple/0",
                "split": "train",
                "source_label": True,
                "decision": "keep",
                "verdict": "stable_lift",
                "reason": "Clear stable lift",
            },
            {
                "episode_id": "hand/cup/0",
                "split": "val",
                "source_label": False,
                "decision": "keep",
                "verdict": "failed_or_unstable",
                "reason": "Visible drop",
            },
        ]
    }
    return documents, review


def test_selection_preserves_original_labels_and_objects():
    documents, review = sample()
    before = deepcopy(documents)
    selected = select_episodes(documents, review)
    assert selected["train"][0]["success"] is True
    assert selected["val"][0]["success"] is False
    assert documents == before


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "move", "flip", "uncertain", "conflict"]
)
def test_unsafe_reviews_are_rejected(mutation):
    documents, review = sample()
    row = review["records"][1]
    if mutation == "missing":
        review["records"].pop()
    elif mutation == "duplicate":
        review["records"].append(dict(row))
    elif mutation == "move":
        row["split"] = "train"
    elif mutation == "flip":
        row["source_label"] = True
    elif mutation == "uncertain":
        row["verdict"] = "cannot_determine"
    elif mutation == "conflict":
        row["verdict"] = "stable_lift"
    with pytest.raises(ValueError):
        select_episodes(documents, review)


def test_uncertain_episode_is_removed_without_moving_other_rows():
    documents, review = sample()
    documents["val"]["episodes"].append(
        {
            "episode_id": "hand/cup/1",
            "object_id": "cup",
            "success": False,
        }
    )
    review["records"].append(
        {
            "episode_id": "hand/cup/1",
            "split": "val",
            "source_label": False,
            "decision": "exclude",
            "verdict": "cannot_determine",
            "reason": "Occluded",
        }
    )
    selected = select_episodes(documents, review)
    assert [r["episode_id"] for r in selected["val"]] == ["hand/cup/0"]
    assert [r["episode_id"] for r in selected["train"]] == ["hand/apple/0"]
