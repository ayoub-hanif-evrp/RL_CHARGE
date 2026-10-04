"""Capture runtime environment for reproducibility archives.

Does not invent versions. Records whatever is installed now.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def _ver(mod: str):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc.__class__.__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ENVIRONMENT_CAPTURE.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    payload = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": _ver("numpy"),
        "torch": _ver("torch"),
        "pyvrp": _ver("pyvrp"),
        "matplotlib": _ver("matplotlib"),
        "frvcpy": _ver("frvcpy"),
        "pip_freeze": [line for line in freeze.splitlines() if line.strip()],
    }
    try:
        import torch

        payload["cuda"] = bool(torch.cuda.is_available())
        payload["torch_version_detail"] = torch.__version__
    except Exception:  # noqa: BLE001
        payload["cuda"] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "n_freeze": len(payload["pip_freeze"])}, indent=2))


if __name__ == "__main__":
    main()
