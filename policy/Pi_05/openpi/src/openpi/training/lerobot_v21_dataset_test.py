import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from openpi.training import lerobot_v21_dataset


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_dataset(root):
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        "action": {"dtype": "float32", "shape": [2], "names": None},
        "observation.images.cam_high": {"dtype": "video", "shape": [3, 4, 6], "names": None},
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [3, 4, 6],
            "names": None,
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [3, 4, 6],
            "names": None,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "total_episodes": 2,
            "total_frames": 6,
            "chunks_size": 1000,
            "fps": 10,
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
            "features": features,
        },
    )
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": "first task"}, {"task_index": 1, "task": "second task"}],
    )
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["first task"], "length": 3},
            {"episode_index": 1, "tasks": ["second task"], "length": 3},
        ],
    )

    for episode_index in range(2):
        rows = []
        for frame_index in range(3):
            global_index = episode_index * 3 + frame_index
            rows.append(
                {
                    "observation.state": [float(global_index), 0.0],
                    "action": [float(global_index), 1.0],
                    "timestamp": frame_index / 10,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": episode_index,
                }
            )
        data_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), data_path)


def test_v21_reader_queries_actions_without_crossing_episode(tmp_path, monkeypatch):
    _make_dataset(tmp_path)

    decoded_paths = []

    def fake_decode(path, timestamps, tolerance_s, backend):
        decoded_paths.append(str(path))
        assert timestamps == [0.2]
        assert tolerance_s == 1e-4
        assert backend == "pyav"
        return torch.ones((1, 3, 4, 6), dtype=torch.float32)

    monkeypatch.setattr(lerobot_v21_dataset, "decode_video_frames", fake_decode)
    dataset = lerobot_v21_dataset.LeRobotV21Dataset(
        tmp_path,
        delta_timestamps={"action": [0.0, 0.1]},
        camera_keys=(
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ),
    )

    item = dataset[2]
    np.testing.assert_array_equal(item["action"], [[2.0, 1.0], [2.0, 1.0]])
    np.testing.assert_array_equal(item["action_is_pad"], [False, True])
    assert item["task"] == "first task"
    assert len(decoded_paths) == 2
    assert all("cam_high" not in path for path in decoded_paths)
    assert lerobot_v21_dataset.is_lerobot_v21_dataset(tmp_path)
