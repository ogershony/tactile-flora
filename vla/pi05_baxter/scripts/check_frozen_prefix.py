"""Check that the frozen-prefix forward gives the same loss and LoRA gradients as the joint one."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_prefix  # noqa: E402

frozen_prefix.apply()

from torch.utils.data import default_collate  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05 import modeling_pi05 as m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "assets/pi05_base_local"
RENAME = {"observation.images.head": "observation.images.base_0_rgb",
          "observation.images.wrist": "observation.images.left_wrist_0_rgb"}

cfg = PreTrainedConfig.from_pretrained(BASE)
cfg.pretrained_path = BASE
cfg.device = "cuda"
cfg.dtype = "bfloat16"
cfg.chunk_size = 10
cfg.n_action_steps = 10
ds = LeRobotDataset("OG7407/baxter-cube-pick-left-v0", root=ROOT / "data/baxter-cube-pick-left-v0-rel10",
                    episodes=[0], video_backend="pyav", delta_timestamps={"action": [k / 10 for k in range(10)]})
policy = make_policy(cfg=cfg, ds_meta=ds.meta, rename_map=RENAME)
pre, _ = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=str(BASE), dataset_stats=ds.meta.stats,
    preprocessor_overrides={"device_processor": {"device": "cuda"},
                            "rename_observations_processor": {"rename_map": RENAME},
                            "normalizer_processor": {"stats": ds.meta.stats,
                                                     "features": {**policy.config.input_features, **policy.config.output_features},
                                                     "norm_map": policy.config.normalization_mapping}})
peft_policy = policy.wrap_with_peft(peft_cli_overrides={
    "method_type": "LORA", "r": 32, "lora_alpha": 32, "init_type": None, "full_training_modules": None,
    "target_modules": r".*gemma_expert\.model\.layers\.[0-9]+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)"})
torch.manual_seed(0)
for n, p in peft_policy.named_parameters():
    if "lora_B" in n:
        torch.nn.init.normal_(p, std=0.02)  # nonzero B so every LoRA factor gets a gradient
batch = pre(default_collate([ds[i] for i in (3, 40)]))
pol = policy
images, img_masks = pol._preprocess_images(batch)
tokens, masks = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
actions = pol.prepare_action(batch)
noise = pol.model.sample_noise(actions.shape, actions.device)
time = pol.model.sample_time(actions.shape[0], actions.device)

out = {}
for name, fn in (("joint", m.PI05Pytorch._joint_forward), ("frozen", frozen_prefix.forward_frozen_prefix)):
    peft_policy.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = fn(pol.model, images, img_masks, tokens, masks, actions, noise, time)[:, :, :8].mean()
    loss.backward()
    g = {n: p.grad.detach().float().clone() for n, p in peft_policy.named_parameters() if p.requires_grad and p.grad is not None}
    out[name] = (loss.item(), g, torch.cuda.max_memory_allocated() / 2**30)
    print(name, "loss", round(loss.item(), 6), "grads", len(g), "peak GiB", round(out[name][2], 2), flush=True)

lj, gj, _ = out["joint"]
lf, gf, _ = out["frozen"]
print("loss rel diff", abs(lj - lf) / abs(lj))
cos = []
for n in gj:
    a, b = gj[n].flatten(), gf[n].flatten()
    if a.norm() > 0:
        cos.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
print("missing in frozen", set(gj) - set(gf))
print("grad cosine min/median", min(cos), sorted(cos)[len(cos) // 2], "n", len(cos))
