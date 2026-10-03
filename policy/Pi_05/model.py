#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path
import dataclasses
import json
from typing import Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)
from XPolicyLab.utils.umi_protocol import UMI_BIMANUAL_ACTION_DIM
from XPolicyLab.utils.umi_protocol import UmiProtocolSpec
from XPolicyLab.utils.umi_protocol import chunk_targets_to_action_dicts


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self._is_umi = self.action_type in {"umi", "umi_relative", "relative_ee"}
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self.policy, self.train_config, self.model_root, self.umi_spec = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get(
            "train_config_name",
            "pi05_robotwin_umi" if self._is_umi else "pi05_aloha",
        )
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        umi_spec = None
        if self._is_umi:
            if not isinstance(config.data, _config.LeRobotRobotwinUmiDataConfig):
                raise ValueError(
                    f"action_type='umi' requires a RoboTwin-UMI train config, got {train_config_name!r}"
                )
            umi_spec = UmiProtocolSpec.from_mapping(
                model_cfg,
                action_horizon=config.model.action_horizon,
            )
            _validate_umi_checkpoint(model_root, umi_spec)
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(config.data, use_head_camera=umi_spec.use_head_camera),
            )
        elif isinstance(config.data, _config.LeRobotRobotwinUmiDataConfig):
            raise ValueError("The pi05_robotwin_umi train config requires action_type='umi'")

        data_config = config.data.create(config.assets_dirs, config.model)
        repo_id = model_cfg.get("repo_id", data_config.asset_id)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        policy = _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)
        return policy, config, model_root, umi_spec

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        state_action_type = "joint" if self._is_umi else self.action_type
        use_head_camera = self.umi_spec.use_head_camera if self.umi_spec is not None else True
        encoded_obs_list = [
            encode_obs(
                obs,
                state_action_type,
                self.robot_action_dim_info,
                use_head_camera=use_head_camera,
            )
            for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []

        for env_idx in env_idx_list:
            try:
                batch_index = self._latest_env_idx_list.index(env_idx)
            except ValueError as exc:
                raise ValueError(
                    f"Requested env_idx={env_idx!r} is not in the latest observation batch "
                    f"{self._latest_env_idx_list}"
                ) from exc
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]
            if self._is_umi:
                actions = np.asarray(actions, dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != UMI_BIMANUAL_ACTION_DIM:
                    raise ValueError(
                        f"pi05 UMI policy must return shape (H, {UMI_BIMANUAL_ACTION_DIM}), got {actions.shape}"
                    )
                if actions.shape[0] != self.umi_spec.action_horizon:
                    raise ValueError(
                        f"pi05 UMI policy returned horizon {actions.shape[0]}, "
                        f"expected {self.umi_spec.action_horizon}"
                    )
                action_list.append(chunk_targets_to_action_dicts(actions[: self.umi_spec.execute_steps]))
            elif self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def _validate_umi_checkpoint(model_root: Path, spec: UmiProtocolSpec) -> None:
    manifest_path = model_root / "assets" / "umi_spec.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"UMI checkpoint is missing protocol manifest: {manifest_path}. "
            "Use a checkpoint trained with pi05_robotwin_umi."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = spec.to_dict()
    for key in (
        "protocol",
        "model_action_representation",
        "state_representation",
        "use_head_camera",
        "action_dim",
        "action_horizon",
        "first_target_offset",
    ):
        if manifest.get(key) != expected[key]:
            raise ValueError(
                f"UMI checkpoint/config mismatch for {key}: checkpoint={manifest.get(key)!r}, "
                f"requested={expected[key]!r}"
            )


def encode_obs(observation, action_type, robot_action_dim_info, *, use_head_camera=True):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        if use_head_camera:
            images = {"cam_high": ensure_chw_uint8(observation["images"]["cam_high"]), **images}
        prompt = _get_prompt(observation)
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    if use_head_camera:
        images = {
            "cam_high": ensure_chw_uint8(
                extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
            ),
            **images,
        }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = _get_prompt(observation)
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not obs_list:
        raise ValueError("obs_list must not be empty")
    image_names = tuple(obs_list[0]["images"])
    for obs in obs_list[1:]:
        if tuple(obs["images"]) != image_names:
            raise ValueError("Every observation in a batch must contain the same ordered camera roles")
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            image_name: np.stack([obs["images"][image_name] for obs in obs_list], axis=0)
            for image_name in image_names
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            image_name: images[batch_index] for image_name, images in obs["images"].items()
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def _get_prompt(observation):
    prompt = observation.get("instruction")
    if prompt is None:
        instructions = observation.get("instructions")
        if isinstance(instructions, (list, tuple)) and instructions:
            prompt = instructions[0]
    if prompt is None:
        raise ValueError("Observation must contain an instruction prompt")
    return str(prompt)


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
