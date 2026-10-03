"""OpenPI transforms for the RoboTwin-UMI protocol."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import einops
import numpy as np
from XPolicyLab.utils.umi_protocol import UMI_BIMANUAL_ACTION_DIM
from XPolicyLab.utils.umi_protocol import adjacent_actions_to_chunk_targets

from openpi import transforms


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert a LeRobot/runtime RGB image to uint8 HWC without changing channels."""

    image = np.asarray(image)
    if image.ndim not in {3, 4}:
        raise ValueError(f"Expected a CHW/HWC image or batch, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 4:
        if image.shape[1] in {1, 3}:
            return einops.rearrange(image, "b c h w -> b h w c")
        if image.shape[-1] in {1, 3}:
            return image
    else:
        if image.shape[0] in {1, 3}:
            return einops.rearrange(image, "c h w -> h w c")
        if image.shape[-1] in {1, 3}:
            return image
    raise ValueError(f"Unsupported image layout: {image.shape}")


def _make_image_mask(image: np.ndarray, *, valid: bool) -> np.ndarray | np.bool_:
    if image.ndim == 4:
        return np.full((image.shape[0],), valid, dtype=bool)
    return np.bool_(valid)


@dataclasses.dataclass(frozen=True)
class RobotwinUmiInputs(transforms.DataTransformFn):
    """Map RoboTwin observations and adjacent UMI actions into π0.5 inputs."""

    use_head_camera: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected RoboTwin-UMI cameras: {tuple(sorted(unexpected))}")
        for required in ("cam_left_wrist", "cam_right_wrist"):
            if required not in in_images:
                raise ValueError(f"RoboTwin-UMI requires camera {required!r}")

        left_wrist = _parse_image(in_images["cam_left_wrist"])
        right_wrist = _parse_image(in_images["cam_right_wrist"])
        if left_wrist.shape != right_wrist.shape:
            raise ValueError(
                f"Wrist cameras must have matching shapes, got {left_wrist.shape} and {right_wrist.shape}"
            )

        if self.use_head_camera:
            if "cam_high" not in in_images:
                raise ValueError("use_head_camera=true requires camera 'cam_high'")
            base_image = _parse_image(in_images["cam_high"])
            base_valid = True
        else:
            # π0.5 has a fixed base-image slot.  A masked black placeholder keeps
            # the slot shape stable without leaking wrist content into it.
            base_image = np.zeros_like(left_wrist)
            base_valid = False

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": _make_image_mask(base_image, valid=base_valid),
                "left_wrist_0_rgb": _make_image_mask(left_wrist, valid=True),
                "right_wrist_0_rgb": _make_image_mask(right_wrist, valid=True),
            },
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            action_is_pad = data.get("action_is_pad")
            inputs["actions"] = adjacent_actions_to_chunk_targets(
                np.asarray(data["actions"]),
                None if action_is_pad is None else np.asarray(action_is_pad),
            ).astype(np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobotwinUmiOutputs(transforms.DataTransformFn):
    """Trim padded π0.5 output dimensions to the bimanual UMI protocol."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < UMI_BIMANUAL_ACTION_DIM:
            raise ValueError(
                f"π0.5 output has {actions.shape[-1]} dims, expected at least {UMI_BIMANUAL_ACTION_DIM}"
            )
        return {"actions": actions[..., :UMI_BIMANUAL_ACTION_DIM]}


@dataclasses.dataclass(frozen=True)
class RobotwinUmiNormInputs(transforms.DataTransformFn):
    """Prepare only numeric state/action fields for normalization statistics."""

    def __call__(self, data: dict) -> dict:
        action_is_pad = data.get("action_is_pad")
        return {
            "state": np.asarray(data["state"], dtype=np.float32),
            "actions": adjacent_actions_to_chunk_targets(
                np.asarray(data["actions"]),
                None if action_is_pad is None else np.asarray(action_is_pad),
            ).astype(np.float32),
        }
