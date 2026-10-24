"""Merge a LoRA adapter into the base weights and write a standalone model directory.

Owns: merge_adapter (the ``tf eval merge-adapter`` entry point), resolve_base_dir,
processor_files, dir_size and MergeInfo.
Must never: import torch, transformers or peft at module level (the CLI help and the tests
import this module on machines without them), promote the Gated DeltaNet parameters to fp32
(the saved weights stay bf16 exactly like the base; HFBackend promotes at load time), or write
outside ``out``.

Why: vLLM loads a full model directory, not a run directory that holds only ``adapter/``.
``merge_adapter`` loads the base in bf16 (``device_map="auto"`` when CUDA is available,
otherwise CPU; transformers 5 initialises on the meta device, so host memory peaks near one
copy of the weights), applies the adapter with peft, ``merge_and_unload``s, saves safetensors
shards of at most 5 GB, and copies the processor, tokenizer, chat-template and generation
files from the base snapshot so the result loads with ``AutoProcessor`` and vLLM exactly like
the base. ``config.json`` is taken verbatim from the base (same architecture, same dtype).
The base holds ``mtp.*`` (multi-token-prediction head) and ``*._extra_state`` keys that the
transformers model has no parameters for; they are not in the merged directory. vLLM 0.24
skips ``mtp.`` on load unless MTP speculative decoding is requested, which this project never
does.

Expected disk: the base is 55.56 GB of safetensors (model.safetensors.index.json total_size
55,562,878,432 bytes); a merged directory is that minus the dropped mtp/extra-state tensors,
so budget about 55 GB per adapter plus 30 MB of tokenizer files.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tactile_flora.eval.inference import MERGED_DIRNAME, MODEL_ID

# What the released batch_utils fetch for a processor-only snapshot (``PROCESSOR_FILES``):
# every top-level json, jinja and txt file. The weight index is written by save_pretrained.
PROCESSOR_GLOBS = ("*.json", "*.jinja", "*.txt")
SKIP_FILES = frozenset({"model.safetensors.index.json"})
WEIGHT_GLOBS = ("*.safetensors", "*.json", "*.jinja", "*.txt")
MAX_SHARD_SIZE = "5GB"


class MergeInfo(BaseModel):
    """Provenance written to ``<out>/merge_info.json`` (``bytes`` excludes that file)."""

    base: str
    base_dir: str
    adapter: str
    out: str
    device_map: str | None
    bytes: int
    seconds: float
    copied: list[str]


def dir_size(path: Path) -> int:
    """Return the total size in bytes of the regular files under ``path``."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def processor_files(base_dir: Path) -> list[Path]:
    """List the top-level processor/tokenizer/template/generation files of a base snapshot."""
    found: dict[str, Path] = {}
    for pattern in PROCESSOR_GLOBS:
        for p in sorted(base_dir.glob(pattern)):
            if p.is_file() and p.name not in SKIP_FILES:
                found[p.name] = p
    return [found[k] for k in sorted(found)]


def resolve_base_dir(base: str) -> Path:
    """Return a local directory for ``base``: the path itself, else the cached Hub snapshot."""
    p = Path(base).expanduser()
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    return Path(snapshot_download(base, allow_patterns=list(WEIGHT_GLOBS)))


def resolve_adapter_dir(run: Path) -> Path:
    """Return ``<run>/adapter`` when it holds a peft adapter, else ``run`` itself; else raise."""
    for cand in (run / "adapter", run):
        if (cand / "adapter_config.json").is_file():
            return cand
    raise RuntimeError(
        f"no peft adapter under {run}: expected {run / 'adapter' / 'adapter_config.json'}"
    )


def merge_adapter(
    run: Path,
    base: str = MODEL_ID,
    out: Path | None = None,
    *,
    overwrite: bool = False,
    device_map: str | None = None,
    max_shard_size: str = MAX_SHARD_SIZE,
    echo: Callable[[str], Any] = print,
) -> Path:
    """Merge ``<run>/adapter`` into ``base`` and write a full bf16 model to ``out``.

    ``device_map`` defaults to "auto" when CUDA is available, else None (CPU). Returns ``out``
    (default ``<run>/merged``). Refuses to overwrite an existing output unless ``overwrite``.
    """
    adapter_dir = resolve_adapter_dir(Path(run))
    out_dir = Path(out) if out is not None else Path(run) / MERGED_DIRNAME
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    base_dir = resolve_base_dir(base)
    copied = processor_files(base_dir)
    if not any(p.name == "config.json" for p in copied):
        raise RuntimeError(f"{base_dir} has no config.json; is it a model snapshot?")

    import torch  # pyright: ignore[reportMissingImports]
    import transformers  # pyright: ignore[reportMissingImports]
    from peft import PeftModel  # pyright: ignore[reportMissingImports]

    if device_map is None and torch.cuda.is_available():
        device_map = "auto"
    config = transformers.AutoConfig.from_pretrained(base_dir)
    arch = (config.architectures or ["AutoModelForImageTextToText"])[0]
    model_cls = getattr(transformers, arch)
    echo(f"base:     {base_dir} ({arch}, device_map={device_map})")
    echo(f"adapter:  {adapter_dir}")
    echo(f"output:   {out_dir}")
    t0 = time.perf_counter()
    kw: dict[str, Any] = {"dtype": torch.bfloat16}
    if device_map is not None:
        kw["device_map"] = device_map
    model = model_cls.from_pretrained(base_dir, **kw)
    t_load = time.perf_counter() - t0
    echo(f"loaded base in {t_load:.0f} s")
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged.eval()
    bad = {str(p.dtype) for p in merged.parameters() if p.dtype != torch.bfloat16}
    if bad:
        raise RuntimeError(f"merged parameters are not all bf16: {sorted(bad)}")
    echo(f"merged in {time.perf_counter() - t0 - t_load:.0f} s; saving")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size=max_shard_size)
    for src in copied:  # after save_pretrained: the base config.json replaces the saved one
        shutil.copyfile(src, out_dir / src.name)
    seconds = time.perf_counter() - t0
    size = dir_size(out_dir)
    info = MergeInfo(
        base=base,
        base_dir=str(base_dir),
        adapter=str(adapter_dir),
        out=str(out_dir),
        device_map=device_map,
        bytes=size,
        seconds=round(seconds, 1),
        copied=[p.name for p in copied],
    )
    (out_dir / "merge_info.json").write_text(json.dumps(info.model_dump(), indent=2))
    echo(f"wrote {out_dir}: {size / 1e9:.2f} GB in {seconds:.0f} s")
    return out_dir
