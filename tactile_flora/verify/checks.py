"""One function per verification check, each returning a CheckResult.

Owns: one function per verification check, each returning a CheckResult.
Must never: mutating an episode or calling another stage.
"""

from __future__ import annotations

from tactile_flora.schema import CheckResult, Episode, VerifyContext


def run_checks(episode: Episode, ctx: VerifyContext) -> dict[str, CheckResult]:
    """Public entry point of this module (milestone 6)."""
    raise NotImplementedError("milestone 6")
