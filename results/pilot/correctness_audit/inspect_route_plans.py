"""Map official EVRPTW-GR Routes Plans.xlsx onto parsed instances.

TEST is unused. Official tours are joint routing+charging multi-vehicle
sequences under MILP2 node order:

    0 depot-start
    1..nc customers in instance-file customer order
    nc+1 .. nc+ns stations in instance-file station order
    nc+ns+1 dummy depot-end

They are not PyVRP customer-only frozen routes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.parser import parse_instance  # noqa: E402
from data.paths import RAW_EVRPTW_GR_DIR, ROUTES_DIR, SPLITS_DIR  # noqa: E402
from domain.load_convention import LoadConvention  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from simulation.actions import ChargeAction, ContinueAction  # noqa: E402
from simulation.simulator import FixedRouteSimulator  # noqa: E402

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
FORBIDDEN = {"test", "testing"}


def _colrow(cell_ref: str) -> tuple[int, int]:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    row = int("".join(ch for ch in cell_ref if ch.isdigit()))
    col = 0
    for ch in letters:
        col = col * 26 + (ord(ch.upper()) - 64)
    return col, row


def _load_sheet(zf: ZipFile, strings: list[str], name: str) -> list[list]:
    root = ET.fromstring(zf.read(name))
    grid: dict[tuple[int, int], str] = {}
    max_c = 0
    max_r = 0
    for cell in root.findall(".//m:c", NS):
        ref = cell.attrib.get("r")
        if not ref:
            continue
        col, row = _colrow(ref)
        value_el = cell.find("m:v", NS)
        if value_el is None or value_el.text is None:
            continue
        raw = value_el.text
        if cell.attrib.get("t") == "s":
            raw = strings[int(raw)]
        grid[(col, row)] = raw
        max_c = max(max_c, col)
        max_r = max(max_r, row)
    table = []
    for r in range(1, max_r + 1):
        table.append([grid.get((c, r), "") for c in range(1, max_c + 1)])
    return table


def parse_routes_plans(path: Path) -> dict:
    with ZipFile(path) as zf:
        sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        strings = ["".join(t.text or "" for t in si.findall(".//m:t", NS)) for si in sst.findall("m:si", NS)]
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {}
        for rel in rels:
            rid_to_target[rel.attrib["Id"]] = rel.attrib["Target"]
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = {}
        for sheet in wb.findall("m:sheets/m:sheet", NS):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rid_to_target[rid].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheets[sheet.attrib["name"]] = _load_sheet(zf, strings, target)
    return sheets


def extract_tours(text: str) -> list[list[int]]:
    tours = []
    for match in re.finditer(r"\[([0-9, ]+)\]", str(text)):
        nums = [int(p.strip()) for p in match.group(1).split(",") if p.strip()]
        if nums:
            tours.append(nums)
    return tours


def _index_maps(instance):
    customers = list(instance.customers)
    stations = list(instance.stations)
    nc = len(customers)
    ns = len(stations)
    dummy = nc + ns + 1
    index_to_id = {0: instance.depot.string_id, dummy: instance.depot.string_id}
    for i, node in enumerate(customers, start=1):
        index_to_id[i] = node.string_id
    for k, node in enumerate(stations, start=1):
        index_to_id[nc + k] = node.string_id
    return index_to_id, nc, ns, dummy


def _customer_sequence(tour: list[int], nc: int) -> tuple[str, ...]:
    return tuple(str(i) for i in tour if 1 <= i <= nc)


def _named_customer_sequence(tour: list[int], index_to_id: dict, nc: int) -> tuple[str, ...]:
    return tuple(index_to_id[i] for i in tour if 1 <= i <= nc)


def replay_official_tour(instance, tour: list[int], index_to_id: dict, nc: int, dummy: int) -> dict:
    """Replay official node sequence: CONTINUE on customers/depot, charge-to-max on stations."""
    customers = _named_customer_sequence(tour, index_to_id, nc)
    if not customers:
        return {"replayed": False, "feasible": False, "reason": "no_customers"}
    profile = PhysicsProfile.from_instance(instance)
    sim = FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    for idx in tour:
        if idx in (0, dummy):
            continue
        node_id = index_to_id.get(idx)
        if node_id is None:
            return {"replayed": True, "feasible": False, "reason": "unknown_index", "index": idx}
        if sim.state.completed:
            break
        if 1 <= idx <= nc:
            result = sim.step(ContinueAction())
        else:
            result = sim.step(ChargeAction(node_id, 1.0))
        if not result.feasible:
            return {
                "replayed": True,
                "feasible": False,
                "reason": str(result.reason),
                "failed_node": node_id,
                "time": sim.state.time.value,
            }
    if not sim.state.completed:
        result = sim.step(ContinueAction())
        if not result.feasible:
            return {
                "replayed": True,
                "feasible": False,
                "reason": str(result.reason),
                "failed_node": "return_depot",
                "time": sim.state.time.value,
            }
    return {
        "replayed": True,
        "feasible": bool(sim.state.completed),
        "reason": "completed" if sim.state.completed else "not_completed",
        "time": sim.state.time.value,
        "n_station_visits": sim.state.metrics.number_of_station_visits,
    }


def _instance_files():
    files = {}
    for path in RAW_EVRPTW_GR_DIR.rglob("*.txt"):
        files.setdefault(path.stem, []).append(path)
    return files


def _load_split_ids(split: str) -> set[str]:
    if split.lower() in FORBIDDEN:
        raise SystemExit("TEST is forbidden")
    payload = json.loads((SPLITS_DIR / f"{split}.json").read_text(encoding="utf-8"))
    return set(payload["instance_ids"])


def _sheet_records(table: list[list]) -> list[dict]:
    if not table:
        return []
    header = [str(h).strip() for h in table[0]]
    rows = []
    for raw in table[1:]:
        if not any(str(c).strip() for c in raw):
            continue
        rec = {header[i]: raw[i] if i < len(raw) else "" for i in range(len(header))}
        name = str(
            rec.get("Instances") or rec.get("Instance") or rec.get(header[0]) or ""
        ).strip()
        if not name or name.lower().startswith("instance"):
            continue
        rec["_name"] = name
        rows.append(rec)
    return rows


def _official_rows(sheets: dict) -> list[dict]:
    rows = []
    for key in ("Small-Size Data", "Medium_Large-Size Data"):
        if key not in sheets:
            continue
        for rec in _sheet_records(sheets[key]):
            rec["_sheet"] = key
            rows.append(rec)
    return rows


def _stem_candidates(name: str, terrain: str | None = None) -> list[str]:
    name = name.strip()
    if name.endswith(".txt"):
        name = name[:-4]
    compact = re.match(
        r"^(?P<base>(rc|c|r)\d+)c(?P<nc>\d+)(?:-s(?P<ns>\d+))?$",
        name,
        re.I,
    )
    bases = []
    if compact:
        base = compact.group("base")
        nc = compact.group("nc")
        ns = compact.group("ns")
        bases.append(f"{base}C{nc}")
        if ns:
            bases.append(f"{base}_{nc}_{ns}")
    else:
        bases.append(name)
    suffixes = [terrain] if terrain else ["NL", "VG", "L"]
    stems = []
    for base in bases:
        if re.search(r"_(L|NL|VG)$", base, re.I):
            stems.append(base)
            continue
        for suf in suffixes:
            stems.append(f"{base}_{suf}")
        stems.append(base)
    return stems


def main() -> None:
    xlsx = ROOT / "results" / "pilot" / "correctness_audit" / "Routes_Plans.xlsx"
    out_path = ROOT / "results" / "pilot" / "correctness_audit" / "official_route_plan_mapping.json"
    sheets = parse_routes_plans(xlsx)
    print("sheets", list(sheets), flush=True)
    files = _instance_files()
    train_ids = _load_split_ids("train")
    val_ids = _load_split_ids("validation")
    corpus = read_jsonl(ROUTES_DIR / "corpus.jsonl")
    corpus_by_instance: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    corpus_by_parent: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for route in corpus:
        seq = tuple(route.customer_ids)
        corpus_by_instance[route.raw_instance_id].add(seq)
        corpus_by_parent[route.base_instance].add(seq)

    official_rows = _official_rows(sheets)
    mapped = []
    n_tours = 0
    n_replay_ok = 0
    n_replay_fail = 0
    n_seq_in_instance = 0
    n_seq_in_parent = 0
    n_seq_in_train = 0
    n_seq_in_val = 0
    n_unmapped_instance = 0

    for rec in official_rows:
        name = str(rec.get("_name") or "").strip()
        if not name:
            continue
        terrain_cols = []
        for key, val in rec.items():
            if key.startswith("_"):
                continue
            key_l = key.lower()
            if "nearly" in key_l or key_l.endswith("nl") or "very gentle" in key_l or key_l.endswith("vg") or "level" in key_l:
                tours = extract_tours(val)
                if tours:
                    terrain_cols.append((key, val, tours))
        if not terrain_cols:
            for key, val in rec.items():
                if key.startswith("_"):
                    continue
                tours = extract_tours(val)
                if tours:
                    terrain_cols.append((key, val, tours))
        for col, _raw, tours in terrain_cols:
            col_l = col.lower()
            terrain = None
            if "nearly" in col_l or col_l.endswith("nl"):
                terrain = "NL"
            elif "very" in col_l or "gentle" in col_l or col_l.endswith("vg"):
                terrain = "VG"
            elif col_l == "level" or col_l.endswith("_l"):
                terrain = "L"
            stems = _stem_candidates(name, terrain)
            path = None
            stem = None
            for cand in stems:
                hits = files.get(cand) or files.get(cand.replace("C", "C"))
                if hits:
                    path = hits[0]
                    stem = cand
                    break
            if path is None:
                n_unmapped_instance += 1
                mapped.append(
                    {
                        "official_name": name,
                        "column": col,
                        "mapped": False,
                        "n_tours": len(tours),
                    }
                )
                continue
            instance = parse_instance(path)
            index_to_id, nc, ns, dummy = _index_maps(instance)
            instance_id = instance.metadata.instance_id
            parent = instance.metadata.base_instance
            in_train = instance_id in train_ids
            in_val = instance_id in val_ids
            for tour in tours:
                n_tours += 1
                named = _named_customer_sequence(tour, index_to_id, nc)
                replay = replay_official_tour(instance, tour, index_to_id, nc, dummy)
                if replay.get("feasible"):
                    n_replay_ok += 1
                elif replay.get("replayed"):
                    n_replay_fail += 1
                hit_instance = named in corpus_by_instance.get(instance_id, set())
                hit_parent = named in corpus_by_parent.get(parent, set())
                if hit_instance:
                    n_seq_in_instance += 1
                if hit_parent:
                    n_seq_in_parent += 1
                if hit_instance and in_train:
                    n_seq_in_train += 1
                if hit_instance and in_val:
                    n_seq_in_val += 1
                mapped.append(
                    {
                        "official_name": name,
                        "column": col,
                        "stem": stem,
                        "instance_id": instance_id,
                        "base_instance": parent,
                        "in_train": in_train,
                        "in_validation": in_val,
                        "in_test": False,
                        "n_customers": nc,
                        "n_stations": ns,
                        "tour": tour,
                        "customer_ids": list(named),
                        "n_official_station_insertions": sum(1 for i in tour if i > nc and i != dummy),
                        "customer_seq_in_same_instance_corpus": hit_instance,
                        "customer_seq_in_same_parent_corpus": hit_parent,
                        "replay": replay,
                    }
                )

    summary = {
        "test_used": False,
        "xlsx": str(xlsx.relative_to(ROOT)).replace("\\", "/"),
        "xlsx_source_repository": "https://github.com/sinarastani/EVRPTW-GR",
        "xlsx_source_path": "Routes Plans.xlsx",
        "xlsx_sha256": hashlib.sha256(xlsx.read_bytes()).hexdigest(),
        "official_indexing": "MILP2: 0 depot, 1..nc customers (file order), nc+1..nc+ns stations (file order), dummy depot",
        "n_official_data_rows": len(official_rows),
        "n_mapped_tours": n_tours,
        "n_unmapped_instance_cells": n_unmapped_instance,
        "n_official_tours_replay_feasible_charge_to_max": n_replay_ok,
        "n_official_tours_replay_failed": n_replay_fail,
        "n_customer_seq_in_same_instance_corpus": n_seq_in_instance,
        "n_customer_seq_in_same_parent_corpus": n_seq_in_parent,
        "n_customer_seq_in_train_instance": n_seq_in_train,
        "n_customer_seq_in_validation_instance": n_seq_in_val,
        "note": (
            "Official tours are joint routing+charging under the original MILP. "
            "Successful charge-to-max replays show those tours are compatible "
            "with our simulator. Failed charge-to-max replays do not prove "
            "model disagreement: the spreadsheet does not reconstruct the "
            "official continuous charging quantities, and overcharging can "
            "create time-window violations. Customer sequence matches do not "
            "imply identical charging decisions. Official plans do not replace "
            "the PyVRP corpus."
        ),
    }
    out_path.write_text(json.dumps({"summary": summary, "tours": mapped}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
