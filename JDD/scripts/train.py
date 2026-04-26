from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_SIM_ROOT = ROOT.parent / "raw-sim"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAW_SIM_ROOT))

from jdd.train import main  # noqa: E402


if __name__ == "__main__":
    main()
