"""lerobot-train with the frozen-prefix Pi0.5 forward (see frozen_prefix.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import frozen_prefix  # noqa: E402

frozen_prefix.apply()

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
