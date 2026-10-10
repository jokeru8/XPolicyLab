"""Shared YAML-backed settings for RoboTwin-UMI training and statistics."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DEPLOY_CONFIG = Path(__file__).resolve().parents[4] / "deploy.yml"


def _load_deploy_config(deploy_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(deploy_path or os.environ.get("OPENPI_DEPLOY_CONFIG", DEFAULT_DEPLOY_CONFIG)).expanduser()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Pi_05 deploy config must contain a YAML mapping: {path}")
    return config


def _value(
    key: str,
    *,
    environment_name: str,
    default: Any,
    deploy_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    environment = os.environ if environ is None else environ
    if environment_name in environment:
        return environment[environment_name]
    return _load_deploy_config(deploy_path).get(key, default)


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _as_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def use_head_camera(
    *, deploy_path: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> bool:
    return _as_bool(
        _value(
            "use_head_camera",
            environment_name="OPENPI_USE_HEAD_CAMERA",
            default=False,
            deploy_path=deploy_path,
            environ=environ,
        ),
        name="use_head_camera",
    )


def use_proprioception(
    *, deploy_path: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> bool:
    return _as_bool(
        _value(
            "use_proprioception",
            environment_name="OPENPI_USE_PROPRIOCEPTION",
            default=True,
            deploy_path=deploy_path,
            environ=environ,
        ),
        name="use_proprioception",
    )


def observation_steps(
    *, deploy_path: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> int:
    return _as_positive_int(
        _value(
            "observation_steps",
            environment_name="OPENPI_UMI_OBSERVATION_STEPS",
            default=2,
            deploy_path=deploy_path,
            environ=environ,
        ),
        name="observation_steps",
    )


def observation_stride(
    *, deploy_path: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> int:
    return _as_positive_int(
        _value(
            "observation_stride",
            environment_name="OPENPI_UMI_OBSERVATION_STRIDE",
            default=3,
            deploy_path=deploy_path,
            environ=environ,
        ),
        name="observation_stride",
    )
