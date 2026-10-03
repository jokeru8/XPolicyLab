# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. It supports the existing joint/EE path and the RoboTwin-UMI bimanual action protocol. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.
If uv cannot fetch the Python version in `.python-version`, select an already installed compatible interpreter, for example `OPENPI_PYTHON=/usr/bin/python3.12 bash install.sh`.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The dataset uses the official keys — `observation.state`, `action`, `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)); the bundled script exists because conversion must run inside openpi's own pinned LeRobot environment, which sets the dataset version. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

RoboTwin-UMI uses the prepared LeRobot v2.1 export directly. OpenPI remains pinned to LeRobot 0.4.4; a local, read-only compatibility reader handles the v2.1 parquet/video layout without rewriting the dataset. The keys match the official XPolicyLab v2.1 converter, `scripts/transform_lerobot_v21_format.py`, with `action_type=umi`. The `action` field is 14-D bimanual adjacent-frame local SE(3) motion plus absolute grippers. It is converted after temporal sampling into a fixed-reference UMI action chunk before normalization.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls

# UMI: validate and reuse the prepared v2.1 dataset (no conversion or overwrite)
ROBOTWIN_UMI_DATASET=/research_haidong_kpfs/zhoukr/datasets/robotwin_umi \
  bash process_data.sh RoboTwin robotwin_umi aloha_agilex umi
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# UMI wrist-only: first compute stats, then train
cd openpi
ROBOTWIN_UMI_DATASET=/research_haidong_kpfs/zhoukr/datasets/robotwin_umi \
OPENPI_USE_HEAD_CAMERA=false \
  .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_robotwin_umi
cd ..
ROBOTWIN_UMI_DATASET=/research_haidong_kpfs/zhoukr/datasets/robotwin_umi \
OPENPI_USE_HEAD_CAMERA=false \
  bash train.sh RoboTwin robotwin_umi aloha_agilex umi 0 0

# UMI head + dual wrist: use true for both stats and training
ROBOTWIN_UMI_DATASET=/research_haidong_kpfs/zhoukr/datasets/robotwin_umi \
OPENPI_USE_HEAD_CAMERA=true \
  bash train.sh RoboTwin robotwin_umi_head aloha_agilex umi 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

For UMI, the default config is `pi05_robotwin_umi`, with action horizon 50, one observation step, and `execute_steps=1`. Frames whose full action horizon would cross an episode boundary are excluded. Normalization reads only state/action parquet fields and does not decode video; both camera modes share the same `robotwin_umi` numeric statistics. Every saved UMI checkpoint contains `assets/umi_spec.json`; inference rejects mismatched camera mode, horizon, or action protocol, so wrist-only and head-enabled runs still require separate checkpoints.

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>

# UMI (deploy.yml use_head_camera must match the checkpoint manifest)
bash eval.sh RoboTwin <task_name> robotwin_umi aloha_agilex umi 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`. UMI additionally uses `action_protocol`, `use_head_camera`, `observation_steps`, and `execute_steps`. `execute_steps` defaults to 1; if increased, all targets in a predicted chunk remain relative to the observation that initiated that prediction request.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo/normalization asset id. UMI defaults to `robotwin_umi`; legacy modes use `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |
| `OPENPI_PYTHON` | Optional Python executable passed to `uv sync` during installation. |
| `ROBOTWIN_UMI_DATASET` | Required local root of the prepared LeRobot v2.1 dataset for UMI training/statistics. |
| `OPENPI_USE_HEAD_CAMERA` | UMI training camera mode: `false` for dual wrist (default), `true` for head plus dual wrist. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.

## Current validation scope

The UMI protocol math, camera masking, episode-boundary filtering, v2.1 reader, checkpoint manifest, and evaluation reference-pose semantics have unit coverage. Real v2.1 batches have been decoded and transformed in both camera modes, and a 64-frame normalization smoke run passed. Full-dataset statistics, π0.5 training, trained-checkpoint loading, and simulator closed-loop evaluation remain to be run.
