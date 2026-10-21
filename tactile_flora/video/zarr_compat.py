"""Read-only compatibility for HapticWAM's Zarr v2 JPEG arrays.

Some published stores encode the VLenBytes array's fill value as JSON ``0``. Zarr 3
expects a string fill value for this object codec and rejects the otherwise readable
metadata. Fix that one metadata field in memory while reading the store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import zarr
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.storage import LocalStore


def _normalize_metadata(
    key: str, result: Buffer | None, prototype: BufferPrototype | None
) -> Buffer | None:
    if result is None or not (key == "data/.zarray" or key.endswith("/data/.zarray")):
        return result

    try:
        metadata = json.loads(result.to_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return result
    if (
        metadata.get("zarr_format") == 2
        and metadata.get("dtype") == "|O"
        and metadata.get("fill_value") == 0
        and any(f.get("id") == "vlen-bytes" for f in metadata.get("filters", []))
    ):
        metadata["fill_value"] = ""
        buffer_type = (prototype or default_buffer_prototype()).buffer
        return buffer_type.from_bytes(json.dumps(metadata).encode())
    return result


class _VLenBytesFillStore(LocalStore):
    """A read-only local store that normalizes legacy VLenBytes fill metadata."""

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        result = await super().get(key, prototype, byte_range)
        return _normalize_metadata(key, result, prototype)

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        result = super().get_sync(key, prototype=prototype, byte_range=byte_range)
        return _normalize_metadata(key, result, prototype)


def open_group(path: Path | str) -> Any:
    """Open a Zarr group read-only, tolerating HapticWAM's legacy bytes fill value."""
    store = _VLenBytesFillStore(path, read_only=True)
    return zarr.open_group(store=store, mode="r")
