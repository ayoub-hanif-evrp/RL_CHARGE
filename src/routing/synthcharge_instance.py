"""Build an in-memory instance from a SynthCharge v1.0 dict.

Altitude is identically zero. Energy is not computed here.
"""

from __future__ import annotations

from pathlib import Path

from data.models import (
    EVRPTWGRInstance,
    InstanceMetadata,
    NetworkGroup,
    Node,
    NodeType,
    TerrainVariant,
    VehicleParameter,
    VehicleParameters,
)

PROFILE_NAME = "synthcharge_linear"


def instance_from_synthcharge(
    raw: dict,
    *,
    instance_id: str,
    relative_path: str,
    layout: str,
    seed: int,
    path: Path | None = None,
) -> EVRPTWGRInstance:
    nodes = []
    depot = raw["depot"]
    horizon = float(raw["time_horizon"])
    nodes.append(
        Node("D0", NodeType.DEPOT, float(depot[0]), float(depot[1]), 0.0, 0.0, horizon, 0.0, 0.0, 2)
    )
    stations = list(raw["stations"])[1:]
    line = 3
    for index, station in enumerate(stations):
        nodes.append(
            Node(
                f"S{index}",
                NodeType.STATION,
                float(station[0]),
                float(station[1]),
                0.0,
                0.0,
                horizon,
                0.0,
                0.0,
                line,
            )
        )
        line += 1
    for index, (customer, demand, service, window) in enumerate(
        zip(raw["customers"], raw["demands"], raw["service_times"], raw["time_windows"]),
        start=1,
    ):
        nodes.append(
            Node(
                f"C{index}",
                NodeType.CUSTOMER,
                float(customer[0]),
                float(customer[1]),
                float(demand),
                float(window[0]),
                float(window[1]),
                float(service),
                0.0,
                line,
            )
        )
        line += 1
    vehicle = VehicleParameters(
        tank_capacity=float(raw["battery_capacity"]),
        load_capacity=float(raw["load_capacity"]),
        consumption_rate=float(raw["consumption_rate"]),
        inverse_refueling_rate=float(raw["refuel_rate"]),
        average_velocity=float(raw["velocity"]),
        curb_weight=None,
        raw=(
            VehicleParameter("Q", "battery capacity", float(raw["battery_capacity"])),
            VehicleParameter("C", "load capacity", float(raw["load_capacity"])),
            VehicleParameter("r", "consumption rate", float(raw["consumption_rate"])),
            VehicleParameter("g", "inverse refueling rate", float(raw["refuel_rate"])),
            VehicleParameter("v", "average velocity", float(raw["velocity"])),
        ),
    )
    distribution = {"R": "r", "C": "c", "RC": "rc"}[layout]
    metadata = InstanceMetadata(
        instance_id=instance_id,
        path=Path(path) if path is not None else Path(relative_path),
        relative_path=relative_path.replace("\\", "/"),
        network_group=NetworkGroup.SMALL,
        customer_folder="synthcharge",
        terrain_variant=TerrainVariant.LEVEL,
        base_instance=str(seed),
        customer_distribution=distribution,
        schedule_type=1,
        declared_customer_count=len(raw["customers"]),
        declared_station_count=len(stations),
        terrain_folder="none",
    )
    return EVRPTWGRInstance(metadata=metadata, nodes=tuple(nodes), vehicle=vehicle)


def write_synthcharge_txt(instance: EVRPTWGRInstance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "StringID\tType\tx\ty\tdemand\tReadyTime\tDueDate\tServiceTime\taltitude"
    ]
    for node in instance.nodes:
        lines.append(
            "\t".join(
                [
                    node.string_id,
                    node.node_type.value,
                    f"{node.x:.8f}",
                    f"{node.y:.8f}",
                    f"{node.demand:.8f}",
                    f"{node.ready_time:.8f}",
                    f"{node.due_date:.8f}",
                    f"{node.service_time:.8f}",
                    "0.00000000",
                ]
            )
        )
    vehicle = instance.vehicle
    lines.extend(
        [
            "",
            f"Q Vehicle fuel tank capacity /{vehicle.tank_capacity:.8f}/",
            f"C Vehicle load capacity /{vehicle.load_capacity:.8f}/",
            f"r fuel consumption rate /{vehicle.consumption_rate:.8f}/",
            f"g inverse refueling rate /{vehicle.inverse_refueling_rate:.8f}/",
            f"v average Velocity /{vehicle.average_velocity:.8f}/",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_synthcharge_instance(path: Path) -> EVRPTWGRInstance:
    text = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    footer = {}
    for line_number, line in enumerate(text, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("StringID"):
            continue
        if stripped[0] in {"Q", "C", "r", "g", "v"} and "/" in stripped:
            key = stripped.split()[0]
            footer[key] = float(stripped.split("/")[1])
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        altitude = float(parts[8]) if len(parts) > 8 else 0.0
        rows.append(
            Node(
                parts[0],
                NodeType.from_code(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
                altitude,
                line_number,
            )
        )
    customers = [node for node in rows if node.node_type is NodeType.CUSTOMER]
    stations = [node for node in rows if node.node_type is NodeType.STATION]
    stem = Path(path).stem
    layout = stem.split("_")[1] if stem.startswith("sc_") else "R"
    seed = stem.split("_s")[-1] if "_s" in stem else stem
    raw = {
        "depot": (rows[0].x, rows[0].y),
        "customers": [(node.x, node.y) for node in customers],
        "stations": [(rows[0].x, rows[0].y), *[(node.x, node.y) for node in stations]],
        "demands": [node.demand for node in customers],
        "service_times": [node.service_time for node in customers],
        "time_windows": [(node.ready_time, node.due_date) for node in customers],
        "battery_capacity": footer["Q"],
        "load_capacity": footer["C"],
        "consumption_rate": footer["r"],
        "refuel_rate": footer["g"],
        "velocity": footer["v"],
        "time_horizon": rows[0].due_date,
    }
    relative = path.as_posix()
    return instance_from_synthcharge(raw, instance_id=stem, relative_path=relative, layout=layout, seed=int(seed) if str(seed).isdigit() else 0)
