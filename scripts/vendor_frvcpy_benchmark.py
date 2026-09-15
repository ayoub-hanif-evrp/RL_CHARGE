"""Download official Montoya / e-VRO FRVCP instances into data/external/frvcpy/benchmark/.

Tiny fixtures at data/external/frvcpy/{tiny-instance.json,routes.json} are left
untouched for unit tests. Benchmark IDs are never joined to EVRPTW-GR splits.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import EXTERNAL_DIR  # noqa: E402

DEST = EXTERNAL_DIR / "frvcpy" / "benchmark"
EVRO = "https://raw.githubusercontent.com/e-VRO/frvcpy/master/instances"
EVRPNL = "https://raw.githubusercontent.com/rafaelmartinelli/EVRPNLLib.jl/master/data"

C10_ZIPS = [
    "tc0c10s2cf1.zip",
    "tc0c10s2ct1.zip",
    "tc0c10s3cf1.zip",
    "tc0c10s3ct1.zip",
    "tc1c10s2cf2.zip",
    "tc1c10s2cf3.zip",
    "tc1c10s2cf4.zip",
    "tc1c10s2ct2.zip",
    "tc1c10s2ct3.zip",
    "tc1c10s2ct4.zip",
    "tc1c10s3cf2.zip",
    "tc1c10s3cf3.zip",
    "tc1c10s3cf4.zip",
    "tc1c10s3ct2.zip",
    "tc1c10s3ct3.zip",
    "tc1c10s3ct4.zip",
    "tc2c10s2cf0.zip",
    "tc2c10s2ct0.zip",
    "tc2c10s3cf0.zip",
    "tc2c10s3ct0.zip",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        dest.write_bytes(response.read())


def _n_customers_from_name(name: str) -> int:
    import re

    match = re.search(r"c(\d+)s", name)
    return int(match.group(1)) if match else 10


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    xml_dir = DEST / "xml"
    json_dir = DEST / "json"
    xml_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    src = EXTERNAL_DIR / "frvcpy"
    for name in ("frvcpy-instance.json", "vrprep-instance.xml", "frvcpy-instance.schema.json"):
        origin = src / name
        if origin.is_file():
            target_dir = json_dir if name.endswith(".json") else xml_dir
            shutil.copy2(origin, target_dir / name)
    for zip_name in C10_ZIPS:
        zpath = DEST / "zips" / zip_name
        if not zpath.is_file():
            _download(f"{EVRPNL}/{zip_name}", zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(xml_dir)
    xml_files = sorted(xml_dir.glob("*.xml"))
    try:
        from frvcpy.translator import translate
    except ImportError:
        translate = None
    if translate is not None:
        for xml in xml_files:
            out = json_dir / (xml.stem + ".json")
            if not out.is_file():
                try:
                    translate(str(xml), to_filename=str(out))
                except Exception:
                    continue
    json_files = sorted(p for p in json_dir.glob("*.json") if "schema" not in p.name)
    routes = []
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        n_nodes = len(payload.get("energy_matrix") or [])
        n_cs = len(payload.get("css") or [])
        n_cust = max(1, n_nodes - 1 - n_cs) if n_nodes else _n_customers_from_name(path.stem)
        if n_cust > 15:
            nodes = [0, 1, 2, 0]
        else:
            nodes = [0, *range(1, n_cust + 1), 0]
        max_q = float(payload.get("max_q") or 0.0)
        routes.append(
            {
                "route_id": f"official_{path.stem}_tour",
                "instance": f"json/{path.name}",
                "nodes": nodes,
                "q_init": max_q if max_q > 0 else None,
                "source": "Montoya_2017_VRPREP_2016-0020_or_e-VRO",
            }
        )
    for xml in xml_files:
        if any(xml.stem in r["route_id"] for r in routes):
            continue
        n_cust = _n_customers_from_name(xml.stem)
        routes.append(
            {
                "route_id": f"official_{xml.stem}_tour",
                "instance_xml": f"xml/{xml.name}",
                "nodes": [0, *range(1, n_cust + 1), 0],
                "q_init": None,
                "source": "Montoya_2017_VRPREP_2016-0020",
                "needs_translate": True,
            }
        )
    routes_payload = {
        "equivalent_to_evrptwgr": "native_frvcp",
        "never_join_to_evrptwgr_splits": True,
        "source": "e-VRO/frvcpy + rafaelmartinelli/EVRPNLLib.jl (Montoya 2017 / VRP-REP 2016-0020)",
        "piecewise_charging": "original_unchanged",
        "routes": routes,
    }
    (DEST / "routes.json").write_text(json.dumps(routes_payload, indent=2) + "\n", encoding="utf-8")
    hashes = {}
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".xml", ".json", ".md"}:
            hashes[path.relative_to(DEST).as_posix()] = _sha256(path)
    (DEST / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    readme = """# Official native FRVCP benchmark

Vendored from:

- [e-VRO/frvcpy](https://github.com/e-VRO/frvcpy) (`frvcpy-instance.json`, `vrprep-instance.xml`)
- [rafaelmartinelli/EVRPNLLib.jl](https://github.com/rafaelmartinelli/EVRPNLLib.jl) 10-customer
  Montoya et al. (2017) E-VRP-NL XML files (VRP-REP 2016-0020)

Original piecewise charging functions are copied unchanged. These instance IDs
are **never** joined to EVRPTW-GR train/validation/test splits.

`tiny-instance.json` and the two-route `../routes.json` remain unit-test/smoke
fixtures only.

`frvcpy.solver.Solver` is exact for these native FRVCP instances. The EVRPTW-GR
surrogate stays `not_equivalent`.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    hashes["README.md"] = _sha256(DEST / "README.md")
    hashes["routes.json"] = _sha256(DEST / "routes.json")
    (DEST / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(f"benchmark xml={len(xml_files)} json={len(json_files)} routes={len(routes)} -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
