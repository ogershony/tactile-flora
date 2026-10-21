"""Registry check: every training adapter resolves by name."""

from __future__ import annotations

import pytest

from tactile_flora.sources.base import get_source


@pytest.mark.parametrize("name", ["phantom", "sharpadex", "rh20t", "reassemble", "airoa"])
def test_training_sources_resolve(name: str) -> None:
    cls = get_source(name)
    assert cls.name == name
