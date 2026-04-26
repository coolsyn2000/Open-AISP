from __future__ import annotations

import sys
from pathlib import Path


def ensure_raw_sim_importable() -> None:
    try:
        import raw_sim  # noqa: F401

        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        raw_sim_root = repo_root / "raw-sim"
        if raw_sim_root.exists():
            sys.path.insert(0, str(raw_sim_root))

