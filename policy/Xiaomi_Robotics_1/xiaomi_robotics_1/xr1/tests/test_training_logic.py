# Copyright (C) 2026 Xiaomi Corporation.
import numpy as np
import torch

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.models.VLA.xr1 import xr1


def test_flow_loss_with_empty_mask_is_finite():
    model = xr1.__new__(xr1)
    model.freq_excluded_dims = [17, 18, 19]
    pred = torch.randn(4, 24, 60, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.zeros_like(pred, dtype=torch.int32)

    loss_mse, loss_freq = model.compute_flow_loss(pred, target, mask, torch.ones_like(pred))

    assert loss_mse.item() == 0.0
    assert loss_freq.item() == 0.0
    assert torch.isfinite(loss_mse + loss_freq)
    (loss_mse + loss_freq).backward()
    assert pred.grad is not None


def test_flow_loss_without_full_horizon_has_zero_frequency_loss():
    model = xr1.__new__(xr1)
    model.freq_excluded_dims = [17, 18, 19]
    pred = torch.randn(4, 30, 60, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.zeros_like(pred, dtype=torch.int32)
    mask[:, :5] = 1

    loss_mse, loss_freq = model.compute_flow_loss(pred, target, mask, torch.ones_like(pred))

    assert torch.isfinite(loss_mse)
    assert loss_freq.item() == 0.0
    (loss_mse + loss_freq).backward()
    assert pred.grad is not None


def test_invalid_trajectory_masks_only_real_steps():
    dataset = JsonDataset.__new__(JsonDataset)
    dataset.action_length = 30

    mask = dataset._mask({"trajectory_type": "invalid"}, steps=3)

    np.testing.assert_array_equal(mask[:3], np.ones(3, dtype=np.int32))
    np.testing.assert_array_equal(mask[3:], np.zeros(27, dtype=np.int32))


def test_umi_arm_action_uses_fixed_reference_and_absolute_gripper():
    dataset = JsonDataset.__new__(JsonDataset)
    dataset.action_length = 2
    dataset.umi_mode = True
    eye = np.eye(3, dtype=np.float32).reshape(-1).tolist()
    traj = {
        "proprios": {
            "left_ee_rotm": [eye, eye],
            "left_ee_pos": [[0, 0, 0], [1, 0, 0]],
            "left_gripper_pos": [[0.2], [0.3]],
        },
        "actions": {
            "left_ee_rotm": [eye, eye],
            "left_ee_pos": [[1, 0, 0], [2, 0, 0]],
            "left_gripper_pos": [[0.4], [0.8]],
        },
    }
    pos, rot, grip = dataset._arm_action(traj, "left", 0, 2)
    np.testing.assert_allclose(pos[:, 0], [1, 2])
    np.testing.assert_allclose(rot, 0)
    np.testing.assert_allclose(grip[:, 0], [0.4, 0.8])


def test_umi_prompt_camera_switch():
    traj = {"instruction": {"text": "move."}}
    wrist = JsonDataset._umi_prompt(traj, use_head_camera=False)
    head = JsonDataset._umi_prompt(traj, use_head_camera=True)
    assert len(wrist["images"]) == 2
    assert len(head["images"]) == 3
