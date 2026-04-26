from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_camera_json(path: str | Path) -> dict[str, Any]:
    camera_path = Path(path)
    with camera_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(camera_path.resolve())
    config["_config_dir"] = str(camera_path.resolve().parent)
    return config

