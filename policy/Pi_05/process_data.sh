#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num_or_raw_task_dirs=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${OPENPI_DATA_MODE:-image}"

if [[ "${action_type}" == "umi" ]]; then
  : "${ROBOTWIN_UMI_DATASET:?Set ROBOTWIN_UMI_DATASET to the prepared LeRobot v2.1 dataset root}"
  info_path="${ROBOTWIN_UMI_DATASET}/meta/info.json"
  if [[ ! -f "${info_path}" ]]; then
    echo "[Pi_05][ERROR] Missing LeRobot metadata: ${info_path}" >&2
    exit 1
  fi
  python - "${info_path}" <<'PY'
import json
import pathlib
import sys

info_path = pathlib.Path(sys.argv[1])
info = json.loads(info_path.read_text(encoding="utf-8"))
if info.get("codebase_version") != "v2.1":
    raise SystemExit(f"Expected LeRobot v2.1, got {info.get('codebase_version')!r}")
required = {
    "observation.state",
    "action",
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}
missing = required - set(info.get("features", {}))
if missing:
    raise SystemExit(f"Dataset is missing required fields: {sorted(missing)}")
print(f"[Pi_05] Reusing prepared RoboTwin-UMI dataset: {info_path.parent.parent}")
PY
  exit 0
fi

py_args=(
  "${bench_name}"
  "${ckpt_name}"
  "${env_cfg_type}"
  "${action_type}"
  --mode "${mode}"
)
if [[ -n "${expert_data_num_or_raw_task_dirs}" ]]; then
  py_args+=("${expert_data_num_or_raw_task_dirs}")
fi
if [[ -n "${raw_task_dirs}" ]]; then
  py_args+=("${raw_task_dirs}")
fi

cd "${POLICY_DIR}/openpi"
python scripts/process_data.py "${py_args[@]}"
