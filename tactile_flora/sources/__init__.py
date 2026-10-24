"""Dataset adapters: one SourceAdapter per dataset; importing this package registers them all."""

from tactile_flora.sources import (  # noqa: F401
    airoa,
    baxter,
    baxter_rollouts,
    hrdexdb,
    phantom,
    reassemble,
    rh20t,
    sharpadex,
    softvtbench,
)
