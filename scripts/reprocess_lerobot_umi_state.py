#!/usr/bin/env python3
"""Upgrade a LeRobot v2.1 RoboTwin-UMI export to relative-EEF proprioception.

The source export stores valid adjacent UMI SE(3) actions, but its
``observation.state`` is the legacy bimanual qpos vector.  Videos and actions
do not need to change.  This tool reconstructs each frame's pose relative to
the episode's first frame by composing the existing actions, retains the old
absolute gripper values, and writes a new dataset root.

Videos are hard-linked by default when source and destination share a
filesystem.  Thus the new data root is immediately usable while avoiding a
second multi-gigabyte video copy.  Use ``--materialize-videos`` when the output
must remain valid after deleting the source dataset.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from XPolicyLab.utils.umi_protocol import (  # noqa: E402
    UMI_BIMANUAL_ACTION_DIM,
    UMI_STATE_REPRESENTATION,
    adjacent_actions_to_episode_relative_ee_state,
)


UMI_STATE_NAMES = (
    "left_rel_x",
    "left_rel_y",
    "left_rel_z",
    "left_rel_rotvec_x",
    "left_rel_rotvec_y",
    "left_rel_rotvec_z",
    "left_gripper",
    "right_rel_x",
    "right_rel_y",
    "right_rel_z",
    "right_rel_rotvec_x",
    "right_rel_rotvec_y",
    "right_rel_rotvec_z",
    "right_gripper",
)
_POSE_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


def _state_stats(state: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "min": state.min(axis=0).astype(float).tolist(),
        "max": state.max(axis=0).astype(float).tolist(),
        "mean": state.mean(axis=0).astype(float).tolist(),
        "std": state.std(axis=0).astype(float).tolist(),
        "count": [int(state.shape[0])],
    }


def _runs(values: np.ndarray):
    """Yield contiguous row ranges with the same episode index."""

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("episode_index must be a non-empty one-dimensional column")
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    start = 0
    for end in (*boundaries.tolist(), len(values)):
        yield int(values[start]), start, end
        start = end


def _replace_state_column(table: pa.Table, source_path: Path) -> tuple[pa.Table, dict[int, dict[str, Any]]]:
    required_columns = {"episode_index", "observation.state", "action"}
    missing = required_columns - set(table.column_names)
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")

    episode_indices = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
    legacy_state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if legacy_state.ndim != 2 or legacy_state.shape[1] != UMI_BIMANUAL_ACTION_DIM:
        raise ValueError(
            f"{source_path} legacy observation.state must have shape (N, {UMI_BIMANUAL_ACTION_DIM}), "
            f"got {legacy_state.shape}"
        )
    if actions.ndim != 2 or actions.shape != legacy_state.shape:
        raise ValueError(
            f"{source_path} action must have shape {legacy_state.shape}, got {actions.shape}"
        )

    new_state = np.empty_like(legacy_state)
    stats: dict[int, dict[str, Any]] = {}
    seen_episode_indices: set[int] = set()
    for episode_index, start, end in _runs(episode_indices):
        if episode_index in seen_episode_indices:
            raise ValueError(f"{source_path} contains non-contiguous rows for episode {episode_index}")
        seen_episode_indices.add(episode_index)
        episode_state = adjacent_actions_to_episode_relative_ee_state(
            actions[start:end],
            left_grippers=legacy_state[start:end, 6],
            right_grippers=legacy_state[start:end, 13],
        )
        if not np.array_equal(episode_state[0, _POSE_INDICES], np.zeros(len(_POSE_INDICES), dtype=np.float32)):
            raise AssertionError(f"{source_path}: episode {episode_index} first relative pose is not exactly zero")
        new_state[start:end] = episode_state
        stats[episode_index] = _state_stats(episode_state)

    column_index = table.schema.get_field_index("observation.state")
    state_array = pa.FixedSizeListArray.from_arrays(
        pa.array(new_state.reshape(-1), type=pa.float32()), UMI_BIMANUAL_ACTION_DIM
    )
    return table.set_column(column_index, "observation.state", state_array), stats


def _copy_dataset_tree(source: Path, destination: Path, *, materialize_videos: bool) -> None:
    def copy_file(source_file: str, destination_file: str) -> str:
        source_path = Path(source_file)
        relative = source_path.relative_to(source)
        if relative.parts and relative.parts[0] == "videos" and not materialize_videos:
            try:
                os.link(source_path, destination_file)
                return destination_file
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # A destination on another mount cannot hard-link.  Keep the
                # operation correct and make the fallback visible to the user.
                print(f"[warn] Cross-device video copy for {relative}")
        return shutil.copy2(source_path, destination_file)

    shutil.copytree(
        source,
        destination,
        copy_function=copy_file,
        ignore=shutil.ignore_patterns(".ms_upload_cache", "__pycache__", "*.pyc"),
    )


def _write_updated_episode_stats(path: Path, state_stats: dict[int, dict[str, Any]]) -> None:
    lines: list[str] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if episode_index not in state_stats:
                raise ValueError(f"No reconstructed state statistics for episode {episode_index}")
            row.setdefault("stats", {})["observation.state"] = state_stats[episode_index]
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(episode_index)
    missing = set(state_stats) - seen
    if missing:
        raise ValueError(f"episodes_stats.jsonl is missing episode indices: {sorted(missing)[:10]}")
    path.write_text("".join(lines), encoding="utf-8")


def _update_info(path: Path) -> None:
    info = json.loads(path.read_text(encoding="utf-8"))
    try:
        state_feature = info["features"]["observation.state"]
    except KeyError as exc:
        raise ValueError(f"{path} lacks features.observation.state") from exc
    if state_feature.get("shape") != [UMI_BIMANUAL_ACTION_DIM]:
        raise ValueError(
            f"Expected 14-D observation.state in {path}, got {state_feature.get('shape')!r}"
        )
    state_feature["names"] = [list(UMI_STATE_NAMES)]
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reprocess(source: Path, destination: Path, *, materialize_videos: bool) -> None:
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot v2.1 dataset root: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    _copy_dataset_tree(source, destination, materialize_videos=materialize_videos)
    state_stats: dict[int, dict[str, Any]] = {}
    parquet_files = sorted((destination / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {destination / 'data'}")

    for number, parquet_path in enumerate(parquet_files, start=1):
        table = pq.read_table(parquet_path)
        updated, per_episode_stats = _replace_state_column(table, parquet_path)
        duplicate = set(state_stats) & set(per_episode_stats)
        if duplicate:
            raise ValueError(f"Episodes span multiple Parquet files: {sorted(duplicate)[:10]}")
        state_stats.update(per_episode_stats)
        pq.write_table(updated, parquet_path, compression="snappy")
        if number % 100 == 0 or number == len(parquet_files):
            print(f"Reprocessed {number}/{len(parquet_files)} Parquet files")

    _write_updated_episode_stats(destination / "meta" / "episodes_stats.jsonl", state_stats)
    _update_info(destination / "meta" / "info.json")
    protocol = {
        "protocol": "umi_v1",
        "state_representation": UMI_STATE_REPRESENTATION,
        "state_layout": list(UMI_STATE_NAMES),
        "state_source": "reconstructed_from_adjacent_umi_actions_and_legacy_absolute_grippers",
        "storage_action_representation": "umi_relative_se3_gripper_v1",
    }
    (destination / "meta" / "umi_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {destination}")
    print(f"State representation: {UMI_STATE_REPRESENTATION}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing legacy-qpos RoboTwin-UMI LeRobot v2.1 root")
    parser.add_argument("destination", type=Path, help="New relative-EEF LeRobot v2.1 root; must not exist")
    parser.add_argument(
        "--materialize-videos",
        action="store_true",
        help="Copy videos instead of hard-linking them from the source dataset",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    reprocess(source, destination, materialize_videos=args.materialize_videos)


if __name__ == "__main__":
    main()
