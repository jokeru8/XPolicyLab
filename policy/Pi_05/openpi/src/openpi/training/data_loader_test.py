import dataclasses
import types

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_full_action_horizon_indices_do_not_cross_episodes():
    dataset = types.SimpleNamespace(
        episode_data_index={
            "from": np.asarray([0, 3]),
            "to": np.asarray([3, 5]),
        }
    )
    assert _data_loader._full_action_horizon_indices(dataset, 2) == [0, 1, 3]  # noqa: SLF001


def test_full_action_horizon_requires_a_long_enough_episode():
    dataset = types.SimpleNamespace(
        episode_data_index={
            "from": np.asarray([0]),
            "to": np.asarray([1]),
        }
    )
    with pytest.raises(ValueError, match="No episode"):
        _data_loader._full_action_horizon_indices(dataset, 2)  # noqa: SLF001


def test_feature_name_validation_rejects_same_shape_with_wrong_semantics():
    expected = {"observation.state": ("left_rel_x", "left_gripper")}
    _data_loader._validate_feature_names(  # noqa: SLF001
        {"observation.state": {"names": [["left_rel_x", "left_gripper"]]}},
        expected,
    )
    with pytest.raises(ValueError, match="incompatible names"):
        _data_loader._validate_feature_names(  # noqa: SLF001
            {"observation.state": {"names": [["left_joint_0", "left_joint_1"]]}},
            expected,
        )


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
