"""SourceAdapter interface, the adapter registry and the shared task-text cleaner.

Owns: the SourceAdapter Protocol, @register_source / get_source, and strip_outcome_words.
Must never: know how any one dataset is laid out, filter episodes, or write to disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from tactile_flora.schema import Episode, Label

OUTCOME_WORDS_RE = re.compile(
    r"(?<![A-Za-z])(?:fail[a-z]*|succe[a-z]*|retr(?:y|ies|ied|ying)|recover[a-z]*|crush[a-z]*)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)


def strip_outcome_words(task: str) -> str:
    """Remove outcome words (failure, success, retry, recovery, crush) and tidy whitespace."""
    cleaned = OUTCOME_WORDS_RE.sub(" ", task)
    cleaned = re.sub(r"[_\s]+", " ", cleaned)
    return cleaned.strip()


@runtime_checkable
class SourceAdapter(Protocol):
    """Lazy, pure wrapper around one dataset; yields Episodes on the episode clock."""

    name: str

    def discover(self, root: Path) -> list[str]:
        """Return every episode id under ``root`` (no filtering)."""
        ...

    def load(self, episode_id: str) -> Episode:
        """Load a full episode on the episode clock."""
        ...

    def splits(self) -> dict[str, list[str]]:
        """Return the session-level split assignment."""
        ...

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return the source's own labels with origin="source"."""
        ...


_REGISTRY: dict[str, type[SourceAdapter]] = {}

A = TypeVar("A", bound=type[SourceAdapter])


def register_source(name: str):
    """Class decorator registering an adapter under ``name``."""

    def deco(cls: A) -> A:
        if name in _REGISTRY:
            raise ValueError(f"source {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return deco


def registered_sources() -> list[str]:
    """Names of all registered adapters (importing the adapter modules first)."""
    import tactile_flora.sources  # noqa: F401  (populates the registry)

    return sorted(_REGISTRY)


def get_source(name: str) -> type[SourceAdapter]:
    """Return the adapter class registered as ``name``."""
    import tactile_flora.sources  # noqa: F401  (populates the registry)

    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown source {name!r}; known: {sorted(_REGISTRY)}") from None
