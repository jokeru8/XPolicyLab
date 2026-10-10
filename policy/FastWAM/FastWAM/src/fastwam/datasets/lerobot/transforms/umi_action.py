"""UMI action transforms used before FastWAM normalization."""

from __future__ import annotations

from typing import Any, Dict

import torch

from XPolicyLab.utils.umi_protocol import adjacent_actions_to_chunk_targets


class UmiChunkActionTransform:
    """Convert stored adjacent-frame UMI deltas to fixed-reference chunks."""

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in batch:
            return batch
        action_is_pad = batch.get("action_is_pad")
        pad = (
            None
            if action_is_pad is None
            else torch.as_tensor(action_is_pad).cpu().numpy()
        )
        for key, action in batch["action"].items():
            tensor = torch.as_tensor(action)
            converted = adjacent_actions_to_chunk_targets(
                tensor.detach().cpu().numpy(), pad
            )
            batch["action"][key] = torch.as_tensor(
                converted, dtype=tensor.dtype, device=tensor.device
            )
        return batch

    def backward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Model outputs are intentionally kept in fixed-reference chunk form.
        return batch
