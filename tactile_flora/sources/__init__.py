"""Dataset adapters: one SourceAdapter per dataset; importing this package registers them all."""

from tactile_flora.sources import (  # noqa: F401
    airoa,
    baxter,
    hrdexdb,
    phantom,
    reassemble,
    reassemble_lerobot,
    rh20t,
    sharpadex,
    softvtbench,
)
