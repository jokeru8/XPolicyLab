"""Shared RoboTwin-UMI protocol and SE(3) action helpers.

The storage action is a bimanual 14-D vector.  Each 7-D arm block contains
``[translation_xyz, rotation_vector_xyz, absolute_gripper]`` and describes the
local transform from one frame to the next.  Policy targets use the same
numeric encoding, but every row in a chunk is relative to the observation that
triggered the prediction.

This module intentionally depends only on NumPy so it can be imported by data
workers, policy servers, and renderer-free tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


UMI_PROTOCOL = "umi_v1"
UMI_STORAGE_REPRESENTATION = "umi_relative_se3_gripper_v1"
UMI_CHUNK_REPRESENTATION = "umi_chunk_relative_se3_gripper_v1"
UMI_STATE_REPRESENTATION = "episode_initial_relative_se3_gripper_v1"

POSE_DIM = 7
UMI_ARM_ACTION_DIM = 7
UMI_BIMANUAL_ACTION_DIM = 14
ROBOTWIN_BIMANUAL_EE_ACTION_DIM = 16


@dataclass(frozen=True)
class UmiProtocolSpec:
    """Resolved configuration shared by UMI training and inference."""

    use_head_camera: bool = False
    observation_steps: int = 1
    action_horizon: int = 50
    execute_steps: int = 1
    protocol: str = UMI_PROTOCOL
    state_representation: str = UMI_STATE_REPRESENTATION
    storage_action_representation: str = UMI_STORAGE_REPRESENTATION
    model_action_representation: str = UMI_CHUNK_REPRESENTATION

    def __post_init__(self) -> None:
        if self.protocol != UMI_PROTOCOL:
            raise ValueError(f"Unsupported UMI protocol: {self.protocol!r}")
        if self.state_representation != UMI_STATE_REPRESENTATION:
            raise ValueError(f"Unsupported UMI state representation: {self.state_representation!r}")
        if self.storage_action_representation != UMI_STORAGE_REPRESENTATION:
            raise ValueError(
                f"Unsupported UMI storage action representation: {self.storage_action_representation!r}"
            )
        if self.model_action_representation != UMI_CHUNK_REPRESENTATION:
            raise ValueError(f"Unsupported UMI model action representation: {self.model_action_representation!r}")
        if self.observation_steps < 1:
            raise ValueError("observation_steps must be positive")
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if not 1 <= self.execute_steps <= self.action_horizon:
            raise ValueError("execute_steps must be in [1, action_horizon]")

    @property
    def camera_roles(self) -> tuple[str, ...]:
        if self.use_head_camera:
            return ("head", "left_wrist", "right_wrist")
        return ("left_wrist", "right_wrist")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["camera_roles"] = list(self.camera_roles)
        result["action_dim"] = UMI_BIMANUAL_ACTION_DIM
        result["first_target_offset"] = 1
        return result

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, action_horizon: int | None = None) -> "UmiProtocolSpec":
        observation = config.get("observation", {})
        temporal = config.get("temporal", {})
        execution = config.get("execution", {})
        if not isinstance(observation, Mapping):
            observation = {}
        if not isinstance(temporal, Mapping):
            temporal = {}
        if not isinstance(execution, Mapping):
            execution = {}

        resolved_horizon = int(
            action_horizon
            if action_horizon is not None
            else config.get("action_horizon", temporal.get("action_horizon", 50))
        )
        return cls(
            use_head_camera=_as_bool(
                config.get("use_head_camera", observation.get("use_head_camera", False)),
                name="use_head_camera",
            ),
            observation_steps=int(config.get("observation_steps", temporal.get("observation_steps", 1))),
            action_horizon=resolved_horizon,
            execute_steps=int(config.get("execute_steps", execution.get("execute_steps", 1))),
            protocol=str(config.get("action_protocol", config.get("umi_protocol", UMI_PROTOCOL))),
        )


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if rotvec.shape != (3,):
        raise ValueError(f"rotvec must have shape (3,), got {rotvec.shape}")
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(rotvec)
    axis_skew = _skew(rotvec / theta)
    return np.eye(3) + np.sin(theta) * axis_skew + (1.0 - np.cos(theta)) * (axis_skew @ axis_skew)


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta < 1e-8:
        return 0.5 * np.array(
            [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
        )
    if np.pi - theta < 1e-6:
        diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-8:
            if largest == 0:
                axis[1] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * axis[0])
                axis[2] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * axis[0])
            elif largest == 1:
                axis[0] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * axis[1])
                axis[2] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * axis[1])
            else:
                axis[0] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * axis[2])
                axis[1] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * axis[2])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            raise ValueError("cannot recover rotation axis from matrix")
        return theta * axis / norm
    vector = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    return theta * vector / (2.0 * np.sin(theta))


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        largest = int(np.argmax(np.diag(rotation)))
        if largest == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif largest == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (POSE_DIM,):
        raise ValueError(f"pose must have shape ({POSE_DIM},), got {pose.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_wxyz_to_matrix(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), got {matrix.shape}")
    return np.concatenate((matrix[:3, 3], matrix_to_quaternion_wxyz(matrix[:3, :3])))


def relative_action_to_matrix(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (6,):
        raise ValueError(f"relative pose action must have shape (6,), got {action.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = action[:3]
    result[:3, :3] = rotvec_to_matrix(action[3:])
    return result


def matrix_to_relative_action(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"relative transform must have shape (4, 4), got {matrix.shape}")
    return np.concatenate((matrix[:3, 3], matrix_to_rotvec(matrix[:3, :3])))


def pose_relative_to_reference(pose: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
    """Encode ``reference_pose^-1 * pose`` as ``xyz + rotvec``.

    RoboTwin poses are ``[x, y, z, qw, qx, qy, qz]``.  The returned translation
    is therefore expressed in the first-frame end-effector coordinate system,
    rather than in the world frame.
    """

    relative = np.linalg.inv(pose_to_matrix(reference_pose)) @ pose_to_matrix(pose)
    return matrix_to_relative_action(relative)


def bimanual_episode_relative_ee_state(
    left_poses: np.ndarray,
    left_grippers: np.ndarray,
    right_poses: np.ndarray,
    right_grippers: np.ndarray,
) -> np.ndarray:
    """Build the 14-D UMI proprioceptive state for one complete episode.

    The layout is ``[left_rel_xyz, left_rel_rotvec, left_gripper,
    right_rel_xyz, right_rel_rotvec, right_gripper]``.  Each EEF pose is
    relative to that arm's first recorded pose; grippers deliberately remain
    absolute.  Input poses use RoboTwin's scalar-first quaternion convention
    (``xyz + wxyz``).
    """

    def _poses(value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != POSE_DIM:
            raise ValueError(f"{name} must have shape (T, {POSE_DIM}), got {result.shape}")
        if result.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one frame")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result

    def _grippers(value: np.ndarray, name: str, horizon: int) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 2 and result.shape[1] == 1:
            result = result[:, 0]
        if result.ndim != 1 or result.shape[0] != horizon:
            raise ValueError(f"{name} must have shape (T,) or (T, 1), got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result

    left_poses = _poses(left_poses, "left_poses")
    right_poses = _poses(right_poses, "right_poses")
    if left_poses.shape[0] != right_poses.shape[0]:
        raise ValueError(
            "left_poses and right_poses must have the same horizon: "
            f"{left_poses.shape[0]} vs {right_poses.shape[0]}"
        )
    horizon = left_poses.shape[0]
    left_grippers = _grippers(left_grippers, "left_grippers", horizon)
    right_grippers = _grippers(right_grippers, "right_grippers", horizon)

    result = np.empty((horizon, UMI_BIMANUAL_ACTION_DIM), dtype=np.float32)
    left_reference = left_poses[0]
    right_reference = right_poses[0]
    for index in range(horizon):
        result[index, :6] = pose_relative_to_reference(left_poses[index], left_reference)
        result[index, 6] = left_grippers[index]
        result[index, 7:13] = pose_relative_to_reference(right_poses[index], right_reference)
        result[index, 13] = right_grippers[index]

    # Exact zero at the reference frame is a useful data invariant and avoids
    # numerical residue from matrix multiplication in serialized datasets.
    result[0, :6] = 0.0
    result[0, 7:13] = 0.0
    return result


def bimanual_relative_ee_state_from_reference(
    left_pose: np.ndarray,
    left_gripper: float,
    right_pose: np.ndarray,
    right_gripper: float,
    *,
    left_reference_pose: np.ndarray,
    right_reference_pose: np.ndarray,
) -> np.ndarray:
    """Build one UMI state row relative to explicit episode-start poses."""

    return bimanual_episode_relative_ee_state(
        np.stack((np.asarray(left_reference_pose), np.asarray(left_pose))),
        np.asarray((left_gripper, left_gripper)),
        np.stack((np.asarray(right_reference_pose), np.asarray(right_pose))),
        np.asarray((right_gripper, right_gripper)),
    )[1]


def adjacent_actions_to_chunk_targets(
    actions: np.ndarray,
    action_is_pad: np.ndarray | None = None,
) -> np.ndarray:
    """Compose adjacent-frame actions into fixed-reference UMI chunk targets.

    ``actions`` may be ``[H, 14]`` or have arbitrary leading batch dimensions.
    Padding must be trailing.  Padded rows repeat the latest valid target and are
    never composed as another physical motion.
    """

    actions = np.asarray(actions)
    if actions.ndim < 2 or actions.shape[-1] != UMI_BIMANUAL_ACTION_DIM:
        raise ValueError(
            f"actions must have shape (..., H, {UMI_BIMANUAL_ACTION_DIM}), got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must contain only finite values")

    horizon = actions.shape[-2]
    batch_shape = actions.shape[:-2]
    flat_actions = actions.reshape((-1, horizon, UMI_BIMANUAL_ACTION_DIM)).astype(np.float64, copy=False)
    if action_is_pad is None:
        flat_pad = np.zeros((flat_actions.shape[0], horizon), dtype=bool)
    else:
        pad = np.asarray(action_is_pad, dtype=bool)
        if pad.shape != (*batch_shape, horizon):
            raise ValueError(f"action_is_pad must have shape {(*batch_shape, horizon)}, got {pad.shape}")
        flat_pad = pad.reshape((-1, horizon))

    result = np.empty_like(flat_actions, dtype=np.float64)
    arm_offsets = (0, UMI_ARM_ACTION_DIM)
    for batch_index, (sample, sample_pad) in enumerate(zip(flat_actions, flat_pad, strict=True)):
        if np.any(sample_pad[:-1] & ~sample_pad[1:]):
            raise ValueError("action padding must be trailing within each chunk")
        cumulative = [np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)]
        latest_gripper = [float(sample[0, 6]), float(sample[0, 13])]
        for time_index in range(horizon):
            for arm_index, offset in enumerate(arm_offsets):
                if not sample_pad[time_index]:
                    cumulative[arm_index] = cumulative[arm_index] @ relative_action_to_matrix(
                        sample[time_index, offset : offset + 6]
                    )
                    latest_gripper[arm_index] = float(sample[time_index, offset + 6])
                result[batch_index, time_index, offset : offset + 6] = matrix_to_relative_action(
                    cumulative[arm_index]
                )
                result[batch_index, time_index, offset + 6] = latest_gripper[arm_index]

    return result.reshape(actions.shape).astype(actions.dtype if np.issubdtype(actions.dtype, np.floating) else np.float32)


def chunk_targets_to_ee_actions(
    left_reference_pose: np.ndarray,
    right_reference_pose: np.ndarray,
    chunk_actions: np.ndarray,
) -> np.ndarray:
    """Decode a fixed-reference UMI chunk into RoboTwin absolute EE targets."""

    chunk = np.asarray(chunk_actions, dtype=np.float64)
    squeeze = chunk.ndim == 1
    if squeeze:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[1] != UMI_BIMANUAL_ACTION_DIM:
        raise ValueError(f"chunk_actions must have shape (H, 14) or (14,), got {chunk_actions.shape}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("chunk_actions must contain only finite values")

    references = (pose_to_matrix(left_reference_pose), pose_to_matrix(right_reference_pose))
    output = np.empty((chunk.shape[0], ROBOTWIN_BIMANUAL_EE_ACTION_DIM), dtype=np.float64)
    for time_index, action in enumerate(chunk):
        output_offset = 0
        for arm_index, input_offset in enumerate((0, UMI_ARM_ACTION_DIM)):
            target = references[arm_index] @ relative_action_to_matrix(action[input_offset : input_offset + 6])
            output[time_index, output_offset : output_offset + POSE_DIM] = matrix_to_pose(target)
            output[time_index, output_offset + POSE_DIM] = np.clip(action[input_offset + 6], 0.0, 1.0)
            output_offset += POSE_DIM + 1
    output = output.astype(np.float32)
    return output[0] if squeeze else output


def chunk_targets_to_action_dicts(chunk_actions: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Convert a numeric chunk into XPolicyLab's bimanual UMI action dictionaries."""

    chunk = np.asarray(chunk_actions, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != UMI_BIMANUAL_ACTION_DIM:
        raise ValueError(f"chunk_actions must have shape (H, 14), got {chunk.shape}")
    return [
        {
            "left_umi_action": action[:UMI_ARM_ACTION_DIM].copy(),
            "right_umi_action": action[UMI_ARM_ACTION_DIM:].copy(),
        }
        for action in chunk
    ]
