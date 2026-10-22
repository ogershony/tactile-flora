"""Expanded object splits must preserve all prior exposure and reserve new test objects."""

import json

import pandas as pd
import pytest

from scripts.cluster.download_hrdexdb_expanded import freeze_splits


def catalog_and_pilot(tmp_path):
    directory = tmp_path / "manifest"
    directory.mkdir()
    for name, objects in [
        ("train-sample.json", ["object00", "object01"]),
        ("split.json", ["object02", "object03"]),
    ]:
        (directory / name).write_text(json.dumps({"episodes": [{"object_id": o} for o in objects]}))
    catalog = pd.DataFrame(
        [
            {
                "episode_id": f"inspire_dftp/object{i:02}/{trial}",
                "object_id": f"object{i:02}",
                "embodiment": "inspire_dftp",
                "success": bool(trial),
            }
            for i in range(40)
            for trial in range(2)
        ]
    )
    return catalog


def test_expansion_preserves_exposure_and_is_independent_of_catalog_order(tmp_path):
    catalog = catalog_and_pilot(tmp_path)
    rows = freeze_splits(catalog, tmp_path)
    assert rows == freeze_splits(catalog.sample(frac=1, random_state=5), tmp_path)
    objects = {
        split: {r["object_id"] for r in rows if r["split"] == split}
        for split in ["train", "val", "test"]
    }
    assert {"object00", "object01"} <= objects["train"]
    assert {"object02", "object03"} <= objects["val"]
    assert len(objects["test"]) == 16
    assert not objects["train"] & objects["val"]
    assert not objects["train"] & objects["test"]
    assert not objects["val"] & objects["test"]
    assert len(rows) == 80


@pytest.mark.parametrize("invalid", ["duplicate", "unknown_label"])
def test_expansion_refuses_ambiguous_catalog(tmp_path, invalid):
    catalog = catalog_and_pilot(tmp_path)
    if invalid == "duplicate":
        catalog = pd.concat([catalog, catalog.iloc[:1]])
    else:
        catalog["success"] = catalog["success"].astype("boolean")
        catalog.loc[0, "success"] = None
    with pytest.raises(ValueError, match="duplicate or unlabeled"):
        freeze_splits(catalog, tmp_path)
