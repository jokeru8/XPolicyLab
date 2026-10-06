import numpy as np
import pytest

from openpi.policies import robotwin_umi_policy


def _sample(*, use_head: bool = True) -> dict:
    images = {
        "cam_left_wrist": np.full((3, 8, 12), 31, dtype=np.uint8),
        "cam_right_wrist": np.full((3, 8, 12), 63, dtype=np.uint8),
    }
    if use_head:
        images["cam_high"] = np.full((3, 8, 12), 127, dtype=np.uint8)
    actions = np.zeros((3, 14), dtype=np.float32)
    actions[:, 0] = 0.1
    actions[:, 7] = 0.2
    return {
        "images": images,
        "state": np.stack(
            (np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32)),
        ),
        "actions": actions,
        "prompt": "move both arms",
    }


def test_wrist_only_uses_an_invalid_black_head_slot():
    result = robotwin_umi_policy.RobotwinUmiInputs(use_head_camera=False)(_sample(use_head=False))
    assert not bool(result["image_mask"]["base_0_rgb"])
    assert np.count_nonzero(result["image"]["base_0_rgb"]) == 0
    assert bool(result["image_mask"]["left_wrist_0_rgb"])
    assert bool(result["image_mask"]["right_wrist_0_rgb"])
    np.testing.assert_allclose(result["actions"][:, 0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, 7], [0.2, 0.4, 0.6], atol=1e-6)


def test_head_mode_requires_and_enables_the_real_head_image():
    transform = robotwin_umi_policy.RobotwinUmiInputs(use_head_camera=True)
    with pytest.raises(ValueError, match="cam_high"):
        transform(_sample(use_head=False))

    result = transform(_sample(use_head=True))
    assert bool(result["image_mask"]["base_0_rgb"])
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], 127)


def test_output_keeps_only_the_protocol_dimensions():
    actions = np.zeros((4, 32), dtype=np.float32)
    actions[:, :14] = 1.0
    result = robotwin_umi_policy.RobotwinUmiOutputs()({"actions": actions})
    assert result["actions"].shape == (4, 14)
    np.testing.assert_array_equal(result["actions"], 1.0)


def test_norm_inputs_do_not_require_images():
    sample = _sample(use_head=False)
    sample.pop("images")
    result = robotwin_umi_policy.RobotwinUmiNormInputs()(sample)
    assert set(result) == {"state", "actions"}
    np.testing.assert_allclose(result["actions"][:, 0], [0.1, 0.2, 0.3], atol=1e-6)


def test_proprioception_is_rebased_to_latest_pose_and_flattened():
    sample = _sample(use_head=False)
    sample["state"][0, 6] = 0.25
    sample["state"][0, 13] = 0.5
    sample["state"][1, 0] = 1.0
    sample["state"][1, 8] = 2.0
    sample["state"][1, 6] = 0.75
    sample["state"][1, 13] = 1.0

    result = robotwin_umi_policy.RobotwinUmiInputs(use_head_camera=False)(sample)
    state = result["state"].reshape(2, 14)
    np.testing.assert_allclose(state[0, :6], [-1.0, 0, 0, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(state[0, 7:13], [0, -2.0, 0, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(state[1, :6], 0.0, atol=1e-7)
    np.testing.assert_allclose(state[1, 7:13], 0.0, atol=1e-7)
    np.testing.assert_allclose(state[:, [6, 13]], [[0.25, 0.5], [0.75, 1.0]])


def test_no_proprioception_does_not_require_state_and_is_exactly_zero():
    sample = _sample(use_head=False)
    sample.pop("state")
    transform = robotwin_umi_policy.RobotwinUmiInputs(
        use_head_camera=False,
        use_proprioception=False,
    )
    result = transform(sample)
    np.testing.assert_array_equal(result["state"], np.zeros(28, dtype=np.float32))

    # This second transform runs after normalization in the configured pipeline.
    normalized = {"state": np.full(28, -1.0, dtype=np.float32)}
    robotwin_umi_policy.ZeroState()(normalized)
    np.testing.assert_array_equal(normalized["state"], np.zeros(28, dtype=np.float32))
