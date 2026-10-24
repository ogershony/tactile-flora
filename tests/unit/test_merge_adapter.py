"""merge-adapter on a tiny Qwen3.5 VLM (4 text layers, 2 vision blocks, random weights).

Covered: the merged weights equal base + scaling * B @ A on the LoRA targets (bf16 rounding),
every other tensor is bit-identical to the base, every saved tensor is bf16 (no fp32 GDN
promotion), the base's processor/tokenizer/template/generation files are copied byte for
byte, config.json is the base's, the merged model's logits match the un-merged PeftModel's,
the merged directory is what ``merged_model_dir`` resolves, and the CLI command runs it.
Not covered (needs the real 56 GB snapshot and a GPU): loading the merged directory with
``AutoProcessor`` on the real tokenizer files and with vLLM, ``device_map="auto"``, and
sharding across several 5 GB files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tactile_flora.eval import inference as I
from tactile_flora.eval import merge_adapter as M

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

PROCESSOR_FILES = {
    "chat_template.jinja": "{{ messages }}",
    "merges.txt": "#version: 0.2\na b\n",
    "preprocessor_config.json": json.dumps({"processor_class": "Qwen3VLProcessor"}),
    "processor_config.json": json.dumps({"processor_class": "Qwen3VLProcessor"}),
    "tokenizer.json": json.dumps({"version": "1.0"}),
    "tokenizer_config.json": json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
    "video_preprocessor_config.json": json.dumps({"video_processor_type": "x"}),
    "vocab.json": json.dumps({"a": 0}),
}
TARGETS = r".*language_model.*\.(q_proj|gate_proj)"


def tiny_config() -> Any:
    text = dict(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]},
        eos_token_id=127, pad_token_id=127,
    )  # fmt: skip
    vision = dict(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64,
        deepstack_visual_indexes=[],
    )  # fmt: skip
    return transformers.Qwen3_5Config(
        text_config=text, vision_config=vision, image_token_id=120, video_token_id=121,
        vision_start_token_id=122, vision_end_token_id=123,
    )  # fmt: skip


@pytest.fixture(scope="module")
def base_and_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Any]:
    """A tiny bf16 base snapshot, a run dir with a non-trivial LoRA adapter, the PeftModel."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("merge")
    base_dir, run = root / "base", root / "runs" / "force"
    model = transformers.Qwen3_5ForConditionalGeneration(tiny_config()).to(torch.bfloat16)
    model.save_pretrained(base_dir, safe_serialization=True)
    for name, text in PROCESSOR_FILES.items():
        (base_dir / name).write_text(text)
    (base_dir / "PROVENANCE.json").write_text("{}")
    lcfg = peft.LoraConfig(r=2, lora_alpha=4, target_modules=TARGETS)
    lora = peft.get_peft_model(model, lcfg)
    n_targets = 0
    for name, p in lora.named_parameters():
        if "lora_B" in name:  # peft zero-initialises B; a merge must then change the weights
            p.data = torch.randn_like(p) * 0.5
            n_targets += 1
    assert n_targets == 5  # q_proj of the one full-attention layer + gate_proj of 4 layers
    lora.save_pretrained(run / "adapter")
    assert (run / "adapter" / "adapter_config.json").is_file()
    return base_dir, run, lora


def _tensors(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    files = sorted(path.glob("*.safetensors"))
    assert files, f"no safetensors in {path}"
    out: dict[str, Any] = {}
    for f in files:
        out.update(load_file(str(f)))
    return out


def test_merge_writes_bf16_full_model_with_processor_files(
    base_and_run: tuple[Path, Path, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    base_dir, run, lora = base_and_run
    out = M.merge_adapter(run, str(base_dir), device_map="cpu")
    assert out == run / "merged" and I.merged_model_dir(str(run)) == out
    printed = capsys.readouterr().out
    assert "wrote" in printed and "GB in" in printed
    base_t, merged_t = _tensors(base_dir), _tensors(out)
    assert set(base_t) == set(merged_t)
    assert all(t.dtype == torch.bfloat16 for t in merged_t.values())
    scaling = 4 / 2
    changed = 0
    for key, b in base_t.items():
        m = merged_t[key]
        module = key.removesuffix(".weight")
        a_key = f"base_model.model.{module}.lora_A.default.weight"
        if ".q_proj." in key or ".gate_proj." in key:
            params = dict(lora.named_parameters())
            a, bb = params[a_key].float(), params[a_key.replace("lora_A", "lora_B")].float()
            expect = b.float() + scaling * (bb @ a)
            assert torch.allclose(m.float(), expect, rtol=2e-2, atol=2e-2), key
            assert not torch.equal(m, b), key
            changed += 1
        else:
            assert torch.equal(m, b), key
    assert changed == 1 + 4  # q_proj of the one full-attention layer, gate_proj of all four
    for name, text in PROCESSOR_FILES.items():
        assert (out / name).read_text() == text
    assert (out / "config.json").read_bytes() == (base_dir / "config.json").read_bytes()
    assert (out / "generation_config.json").is_file()
    assert (out / "PROVENANCE.json").is_file()  # every top-level json travels, as in batch_utils
    assert not (out / "adapter_config.json").exists()
    info = json.loads((out / "merge_info.json").read_text())
    assert info["adapter"] == str(run / "adapter")
    # bytes is measured before merge_info.json itself is written
    assert info["bytes"] == M.dir_size(out) - (out / "merge_info.json").stat().st_size
    assert info["device_map"] == "cpu" and "config.json" in info["copied"]

    reloaded = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        out, dtype=torch.bfloat16
    ).eval()
    ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        want = lora.eval()(input_ids=ids).logits.float()
        got = reloaded(input_ids=ids).logits.float()
    assert torch.allclose(got, want, rtol=5e-2, atol=5e-2)


def test_merge_refuses_existing_output_and_missing_adapter(
    base_and_run: tuple[Path, Path, Any], tmp_path: Path
) -> None:
    base_dir, run, _ = base_and_run
    out = run / "merged"
    assert out.is_dir()
    with pytest.raises(FileExistsError, match="--overwrite"):
        M.merge_adapter(run, str(base_dir), device_map="cpu")
    with pytest.raises(RuntimeError, match="adapter_config.json"):
        M.merge_adapter(tmp_path / "nope", str(base_dir), device_map="cpu")
    # an adapter directory passed directly also works, into an explicit --out
    other = M.merge_adapter(run / "adapter", str(base_dir), tmp_path / "m2", device_map="cpu")
    assert other == tmp_path / "m2" and I.is_model_dir(other)


def test_cli_merge_adapter(base_and_run: tuple[Path, Path, Any], tmp_path: Path) -> None:
    from tactile_flora import cli

    base_dir, run, _ = base_and_run
    res = CliRunner().invoke(
        cli.app,
        ["eval", "merge-adapter", "--run", str(run), "--base", str(base_dir), "--out",
         str(tmp_path / "cli"), "--device-map", "cpu"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "wrote" in res.output and I.is_model_dir(tmp_path / "cli")
    again = CliRunner().invoke(
        cli.app, ["eval", "merge-adapter", "--run", str(run), "--base", str(base_dir)]
    )
    assert again.exit_code == 1 and "FileExistsError" in again.output
