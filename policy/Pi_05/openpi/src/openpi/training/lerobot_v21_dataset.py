"""Read-only LeRobot v2.1 compatibility layer for local RoboTwin-UMI data.

LeRobot 0.4.x intentionally only accepts v3 datasets.  RoboTwin-UMI remains a
published v2.1 dataset, so training uses this small reader instead of mutating
the source dataset or downgrading OpenPI's PyTorch stack.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import datasets
from lerobot.datasets.video_utils import decode_video_frames
import numpy as np
import torch


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def is_lerobot_v21_dataset(root: str | Path | None) -> bool:
    """Return whether ``root`` is an existing local LeRobot v2.1 dataset."""

    if root is None:
        return False
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.is_file():
        return False
    return str(_read_json(info_path).get("codebase_version")) == "v2.1"


class LeRobotV21Metadata:
    """Subset of LeRobot metadata used by OpenPI's data pipeline."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = _read_json(self.root / "meta" / "info.json")
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError(
                f"Expected a LeRobot v2.1 dataset at {self.root}, "
                f"got {self.info.get('codebase_version')!r}"
            )

        task_rows = _read_jsonl(self.root / "meta" / "tasks.jsonl")
        self.tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        episode_rows = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        self.episodes = {int(row["episode_index"]): row for row in episode_rows}

        expected_episodes = int(self.info["total_episodes"])
        if sorted(self.episodes) != list(range(expected_episodes)):
            raise ValueError("LeRobot v2.1 episodes must be contiguous and start at zero")

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.info["features"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] in {"image", "video"}]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "video"]

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        episode_chunk = episode_index // int(self.info["chunks_size"])
        return Path(
            self.info["video_path"].format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
        )


class LeRobotV21Dataset(torch.utils.data.Dataset):
    """Local, read-only LeRobot v2.1 dataset with temporal action queries."""

    def __init__(
        self,
        root: str | Path,
        *,
        delta_timestamps: dict[str, Sequence[float]] | None = None,
        camera_keys: Sequence[str] | None = None,
        video_backend: str = "pyav",
        tolerance_s: float = 1e-4,
    ):
        self.root = Path(root)
        self.meta = LeRobotV21Metadata(self.root)
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s

        selected_cameras = tuple(self.meta.camera_keys if camera_keys is None else camera_keys)
        missing_cameras = set(selected_cameras) - set(self.meta.camera_keys)
        if missing_cameras:
            raise ValueError(f"Dataset is missing configured cameras: {tuple(sorted(missing_cameras))}")
        non_video_cameras = set(selected_cameras) - set(self.meta.video_keys)
        if non_video_cameras:
            raise ValueError(
                "The RoboTwin-UMI v2.1 reader currently requires video cameras, got "
                f"{tuple(sorted(non_video_cameras))}"
            )
        self.camera_keys = selected_cameras

        self.delta_indices = self._make_delta_indices(delta_timestamps)
        starts = []
        ends = []
        cursor = 0
        for episode in self.meta.episodes.values():
            starts.append(cursor)
            cursor += int(episode["length"])
            ends.append(cursor)
        self.episode_data_index = {
            "from": torch.tensor(starts, dtype=torch.int64),
            "to": torch.tensor(ends, dtype=torch.int64),
        }

        # This mirrors LeRobot's v2.1 loader. Hugging Face stores one compact,
        # memory-mapped Arrow cache; workers do not repeatedly open 2,500 files.
        self.hf_dataset = datasets.load_dataset("parquet", data_dir=str(self.root / "data"), split="train")
        if len(self.hf_dataset) != cursor:
            raise ValueError(
                f"Frame count mismatch: episode metadata describes {cursor}, parquet data has {len(self.hf_dataset)}"
            )
        for episode_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            first = self.hf_dataset[start]
            last = self.hf_dataset[end - 1]
            if (
                int(first["episode_index"]) != episode_index
                or int(first["frame_index"]) != 0
                or int(last["episode_index"]) != episode_index
                or int(last["frame_index"]) != end - start - 1
            ):
                raise ValueError(
                    "Parquet rows are not ordered by contiguous episode/frame index; "
                    "temporal action windows would be ambiguous"
                )

    def _make_delta_indices(
        self, delta_timestamps: dict[str, Sequence[float]] | None
    ) -> dict[str, tuple[int, ...]]:
        if delta_timestamps is None:
            return {}
        result = {}
        for key, timestamps in delta_timestamps.items():
            if key not in self.meta.features:
                raise ValueError(f"Unknown temporal feature {key!r}")
            indices = []
            for timestamp in timestamps:
                frame_offset = round(float(timestamp) * self.meta.fps)
                if not np.isclose(timestamp, frame_offset / self.meta.fps, atol=self.tolerance_s):
                    raise ValueError(
                        f"Delta timestamp {timestamp} for {key!r} is not aligned to {self.meta.fps} Hz"
                    )
                indices.append(frame_offset)
            result[key] = tuple(indices)
        return result

    def __len__(self) -> int:
        return len(self.hf_dataset)

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        return torch.as_tensor(value)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        row = self.hf_dataset[index]
        episode_index = int(row["episode_index"])
        episode_start = int(self.episode_data_index["from"][episode_index])
        episode_end = int(self.episode_data_index["to"][episode_index])

        item = {key: self._to_tensor(value) for key, value in row.items()}
        for key, offsets in self.delta_indices.items():
            query_indices = [min(max(index + offset, episode_start), episode_end - 1) for offset in offsets]
            query_values = self.hf_dataset.select(query_indices)[key]
            item[key] = self._to_tensor(query_values)
            item[f"{key}_is_pad"] = torch.tensor(
                [index + offset < episode_start or index + offset >= episode_end for offset in offsets],
                dtype=torch.bool,
            )

        timestamp = float(row["timestamp"])
        for camera_key in self.camera_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, camera_key)
            item[camera_key] = decode_video_frames(
                video_path,
                [timestamp],
                self.tolerance_s,
                self.video_backend,
            ).squeeze(0)

        task_index = int(row["task_index"])
        item["task"] = self.meta.tasks[task_index]
        return item
