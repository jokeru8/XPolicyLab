import json

from etils import epath

from openpi.training import checkpoints
from openpi.training import config


def test_save_assets_writes_protocol_manifest(tmp_path):
    metadata = {"protocol": "umi_v1", "use_head_camera": False, "action_horizon": 50}
    data_config = config.DataConfig(protocol_metadata=metadata)

    checkpoints._save_assets(epath.Path(tmp_path), data_config)  # noqa: SLF001

    manifest = json.loads((tmp_path / "umi_spec.json").read_text(encoding="utf-8"))
    assert manifest == metadata
