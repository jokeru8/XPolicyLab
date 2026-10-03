import json

import numpy as np
import pytest

from XPolicyLab.policy.Pi_05 import model as pi05_model
from XPolicyLab.utils.umi_protocol import UmiProtocolSpec


def test_encode_obs_can_exclude_head_camera():
    observation = {
        "images": {
            "cam_high": np.full((3, 4, 6), 255, dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 4, 6), dtype=np.uint8),
            "cam_right_wrist": np.ones((3, 4, 6), dtype=np.uint8),
        },
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "move",
    }

    encoded = pi05_model.encode_obs(observation, "joint", None, use_head_camera=False)
    assert tuple(encoded["images"]) == ("cam_left_wrist", "cam_right_wrist")


def test_encode_umi_obs_uses_relative_eef_state_not_joint_fields():
    relative_state = np.arange(14, dtype=np.float32)
    observation = {
        "vision": {
            "cam_left_wrist": {"rgb": np.zeros((4, 6, 3), dtype=np.uint8)},
            "cam_right_wrist": {"rgb": np.ones((4, 6, 3), dtype=np.uint8)},
        },
        "state": {
            "umi_relative_state": relative_state,
            # A wrong-size legacy qpos field must not be read by the UMI path.
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
        },
        "instruction": "move",
    }

    encoded = pi05_model.encode_obs(observation, "umi", None, use_head_camera=False)
    np.testing.assert_array_equal(encoded["state"], relative_state)
    assert tuple(encoded["images"]) == ("cam_left_wrist", "cam_right_wrist")


def test_checkpoint_manifest_must_match_camera_mode(tmp_path):
    spec = UmiProtocolSpec(use_head_camera=False)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "umi_spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    pi05_model._validate_umi_checkpoint(tmp_path, spec)

    mismatched = UmiProtocolSpec(use_head_camera=True)
    with pytest.raises(ValueError, match="use_head_camera"):
        pi05_model._validate_umi_checkpoint(tmp_path, mismatched)


class _FakePolicy:
    def infer(self, observation, **kwargs):
        del kwargs
        value = float(observation["state"][0])
        return {"actions": np.full((50, 14), value, dtype=np.float32)}


def test_batch_selection_uses_environment_ids():
    model = pi05_model.Model.__new__(pi05_model.Model)
    model.policy = _FakePolicy()
    model._is_umi = True
    model.umi_spec = UmiProtocolSpec(execute_steps=1)
    model._latest_env_idx_list = [3, 8]
    model.observation_window = {
        "state": np.asarray([[3.0], [8.0]], dtype=np.float32),
        "images": {},
        "prompt": ["first", "second"],
    }

    result = model.get_action_batch(env_idx_list=[8])
    np.testing.assert_array_equal(result[0][0]["left_umi_action"], 8.0)

    with pytest.raises(ValueError, match="env_idx=4"):
        model.get_action_batch(env_idx_list=[4])


def test_umi_output_horizon_must_match_protocol():
    model = pi05_model.Model.__new__(pi05_model.Model)
    model.policy = _FakePolicy()
    model._is_umi = True
    model.umi_spec = UmiProtocolSpec(action_horizon=16, execute_steps=1)
    model._latest_env_idx_list = [0]
    model.observation_window = {
        "state": np.asarray([[0.0]], dtype=np.float32),
        "images": {},
        "prompt": ["test"],
    }

    with pytest.raises(ValueError, match="returned horizon 50, expected 16"):
        model.get_action()
