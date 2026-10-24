"""Frozen-prefix training forward for LeRobot's Pi0.5 (stop-gradient into the VLM).

LeRobot's training forward runs the image+text prefix and the noisy-action suffix through one joint
pass. The attention op mixes both, so even with every PaliGemma weight frozen, autograd records the
prefix path and backpropagates through the 2B language model: at batch 32 that does not fit in
24 GB, and it costs a VLM backward per step for gradients that reach no trainable parameter.

Pi0.5's prefix never attends to the suffix (block-causal mask), so the prefix keys and values do
not depend on the actions. This forward computes them once under no_grad, exactly as inference
does (`sample_actions`), and then runs the suffix through `denoise_step` with gradients on. The
loss is the same flow-matching MSE; only the trainable parts (action expert LoRA and projections)
receive gradients, which is also what the full joint pass gives them when the VLM is frozen.
It is the training-time analogue of pi0.5's knowledge-insulation stop-gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lerobot.policies.pi05 import modeling_pi05 as m


def forward_frozen_prefix(self, images, img_masks, tokens, masks, actions, noise, time):
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        lm = self.paligemma_with_expert.paligemma.model.language_model
        if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        prefix_att_2d_masks = m.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = m.prepare_attention_masks_4d(prefix_att_2d_masks)
        lm.config._attn_implementation = "eager"  # noqa: SLF001  (as sample_actions does)
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, time)
    return F.mse_loss(u_t, v_t, reduction="none")


def apply() -> None:
    m.PI05Pytorch._joint_forward = m.PI05Pytorch.forward
    m.PI05Pytorch.forward = forward_frozen_prefix
