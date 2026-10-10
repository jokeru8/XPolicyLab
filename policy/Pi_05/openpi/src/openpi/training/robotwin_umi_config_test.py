from pathlib import Path

import pytest

from openpi.training import robotwin_umi_config


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "deploy.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_umi_settings_are_read_from_deploy_yaml(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
use_head_camera: true
use_proprioception: false
observation_steps: 1
observation_stride: 5
""",
    )

    assert robotwin_umi_config.use_head_camera(deploy_path=path, environ={})
    assert not robotwin_umi_config.use_proprioception(deploy_path=path, environ={})
    assert robotwin_umi_config.observation_steps(deploy_path=path, environ={}) == 1
    assert robotwin_umi_config.observation_stride(deploy_path=path, environ={}) == 5


def test_environment_can_explicitly_override_yaml(tmp_path: Path):
    path = _write_config(tmp_path, "use_head_camera: false\n")

    assert robotwin_umi_config.use_head_camera(
        deploy_path=path, environ={"OPENPI_USE_HEAD_CAMERA": "true"}
    )


@pytest.mark.parametrize("value", ["sometimes", 2, None])
def test_invalid_head_camera_value_is_rejected(tmp_path: Path, value):
    path = _write_config(tmp_path, f"use_head_camera: {value!r}\n")

    with pytest.raises(ValueError, match="use_head_camera must be a boolean"):
        robotwin_umi_config.use_head_camera(deploy_path=path, environ={})
