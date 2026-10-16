"""Descriptive audit of existing fixed-route construction for the FA-HPPO paper.

Read-only. Does not generate routes, retrain, or evaluate the consumed TEST.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

VAL_DIR = ROOT / "data" / "routes_v2" / "synthcharge_final" / "validation"
TRAIN_DIR = ROOT / "data" / "routes_v2" / "synthcharge_final" / "train"
TEST_DIR = ROOT / "data" / "routes_v2" / "synthcharge_v4_test"
V4_SPEC = ROOT / "results" / "v4" / "test" / "configs" / "synthcharge_v4_test.json"
V2_SPEC = ROOT / "configs" / "v2" / "synthcharge_final_benchmark.json"

LENGTH_BINS = (("short", 3, 5), ("medium", 6, 10), ("long", 11, 15))
SOURCE_SIZES = (15, 30, 50)

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#9A9A9A"
BLACK = "#222222"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def length_group(n_customers: int) -> str | None:
    for name, lo, hi in LENGTH_BINS:
        if lo <= int(n_customers) <= hi:
            return name
    return None


def load_nodes(instance_path: Path) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    for line in instance_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip() or line.startswith("Q ") or line.startswith("C ") or line.startswith("r ") or line.startswith("g ") or line.startswith("v "):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        kind = {"d": "depot", "f": "station", "c": "customer"}.get(parts[1])
        if kind is None:
            continue
        nodes[parts[0]] = {"id": parts[0], "kind": kind, "x": float(parts[2]), "y": float(parts[3])}
    return nodes


def witness_path(customer_ids: list[str], trace: list[dict]) -> list[dict]:
    """Replay CONTINUE/CHARGE decisions into an ordered node path."""
    path = [{"id": "D0", "role": "depot"}]
    next_customer = 0
    for step in trace:
        kind = step.get("kind")
        if kind == "CHARGE":
            station = str(step.get("station_id"))
            path.append(
                {
                    "id": station,
                    "role": "station",
                    "target_soc": step.get("target_soc"),
                    "arrival_soc": step.get("arrival_soc"),
                }
            )
        elif kind == "CONTINUE":
            if next_customer < len(customer_ids):
                path.append({"id": customer_ids[next_customer], "role": "customer", "order": next_customer + 1})
                next_customer += 1
            else:
                path.append({"id": "D0", "role": "depot"})
        else:
            raise ValueError(f"unrecognized certificate step kind {kind!r}")
    if next_customer != len(customer_ids):
        raise ValueError("certificate trace did not visit every frozen customer")
    if path[-1]["id"] != "D0" or path[-1]["role"] != "depot" or len(path) < 2:
        raise ValueError("certificate trace did not return to the depot")
    return path


def choose_validation_example(routes: list[dict], certificates: dict[str, dict]) -> dict:
    eligible = []
    for route in routes:
        n = int(route["n_customers"])
        if route.get("charge_class") != "charging_required":
            continue
        if not (6 <= n <= 10):
            continue
        instance = ROOT / str(route["relative_path"])
        cert = certificates.get(route["route_id"])
        if not instance.is_file() or not cert or not cert.get("trace"):
            continue
        eligible.append(route)
    if not eligible:
        raise SystemExit("no validation route matches the deterministic selection rule")
    eligible.sort(key=lambda row: (int(row["generator_seed"]), str(row["route_id"])))
    return eligible[0]


def _arrow(ax, start, end, color, style="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.15,
        color=color,
        linestyle=style,
        shrinkA=7,
        shrinkB=7,
        zorder=2,
    )
    ax.add_patch(patch)


# Visit-order offsets for the frozen example v2sc_sc_R_n15_s200000_00 only.
# Alternating sides keep labels 4–6 off the vertical customer stack.
_CUSTOMER_LABEL_OFFSETS = {
    "C14": (0, -10, "center", "top"),
    "C3": (0, -10, "center", "top"),
    "C9": (0, 9, "center", "bottom"),
    "C5": (9, -4, "left", "top"),
    "C10": (11, -7, "left", "top"),
    "C12": (9, -8, "left", "top"),
    "C4": (11, -9, "left", "top"),
    "C8": (-10, 1, "right", "center"),
}
_STATION_LABEL_OFFSETS = {
    "S1": (0, -13, "center", "top"),
    "S2": (11, 7, "left", "bottom"),
}


def _order_label(ax, node_id: str, order: int, nodes: dict[str, dict]) -> None:
    dx, dy, ha, va = _CUSTOMER_LABEL_OFFSETS[node_id]
    node = nodes[node_id]
    ax.annotate(
        str(order),
        (node["x"], node["y"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va=va,
        fontsize=7,
        color=BLUE,
        zorder=6,
        annotation_clip=False,
    )


def draw_fixed_route_figure(route: dict, nodes: dict[str, dict], trace: list[dict], out_dir: Path) -> None:
    customers = list(route["customer_ids"])
    if customers != list(_CUSTOMER_LABEL_OFFSETS):
        raise SystemExit("customer-label offsets are tied to the disclosed validation example")
    selected = set(customers)
    witness = witness_path(customers, trace)
    xs = [node["x"] for node in nodes.values()]
    ys = [node["y"] for node in nodes.values()]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    limits = (
        min(xs) - 0.11 * span_x,
        max(xs) + 0.08 * span_x,
        min(ys) - 0.09 * span_y,
        max(ys) + 0.13 * span_y,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.titlepad": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.16],
        hspace=0.16,
        wspace=0.08,
        left=0.025,
        right=0.985,
        top=0.90,
        bottom=0.02,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")

    def base(ax, label_artists: bool):
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        other = [node for node in nodes.values() if node["kind"] == "customer" and node["id"] not in selected]
        if other:
            ax.scatter(
                [n["x"] for n in other],
                [n["y"] for n in other],
                s=14,
                c="#D0D0D0",
                zorder=1,
                label="Unselected source customer" if label_artists else None,
            )
        stations = [node for node in nodes.values() if node["kind"] == "station"]
        ax.scatter(
            [n["x"] for n in stations],
            [n["y"] for n in stations],
            s=36,
            marker="^",
            facecolors="none",
            edgecolors=ORANGE,
            linewidths=0.9,
            zorder=3,
            label="Charging station" if label_artists else None,
        )
        depot = nodes["D0"]
        ax.scatter(
            [depot["x"]],
            [depot["y"]],
            s=42,
            marker="s",
            c=BLACK,
            zorder=4,
            label="Depot" if label_artists else None,
        )

    ax = axes[0]
    base(ax, True)
    seq = ["D0", *customers, "D0"]
    for a, b in zip(seq, seq[1:]):
        _arrow(ax, (nodes[a]["x"], nodes[a]["y"]), (nodes[b]["x"], nodes[b]["y"]), BLUE, "-")
    ax.scatter(
        [nodes[cid]["x"] for cid in customers],
        [nodes[cid]["y"] for cid in customers],
        s=28,
        c=BLUE,
        zorder=4,
        label="Selected customer",
    )
    for order, cid in enumerate(customers, start=1):
        _order_label(ax, cid, order, nodes)
    ax.set_title("(a) Customer-only PyVRP route", loc="left", fontsize=8)

    ax = axes[1]
    base(ax, False)
    for left, right in zip(witness, witness[1:]):
        charging = left["role"] == "station" or right["role"] == "station"
        _arrow(
            ax,
            (nodes[left["id"]]["x"], nodes[left["id"]]["y"]),
            (nodes[right["id"]]["x"], nodes[right["id"]]["y"]),
            ORANGE if charging else BLUE,
            "--" if charging else "-",
        )
    ax.scatter(
        [nodes[cid]["x"] for cid in customers],
        [nodes[cid]["y"] for cid in customers],
        s=28,
        c=BLUE,
        zorder=4,
    )
    for order, cid in enumerate(customers, start=1):
        _order_label(ax, cid, order, nodes)
    labeled = set()
    for point in witness:
        if point["role"] != "station" or point["id"] in labeled:
            continue
        labeled.add(point["id"])
        if point["id"] not in _STATION_LABEL_OFFSETS:
            raise SystemExit(f"no station-label offset for {point['id']}")
        node = nodes[point["id"]]
        dx, dy, ha, va = _STATION_LABEL_OFFSETS[point["id"]]
        target = point.get("target_soc")
        label = point["id"] if target is None else f"{point['id']}\nSOC {float(target):.2f}"
        ax.annotate(
            label,
            (node["x"], node["y"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.5,
            color=ORANGE,
            ha=ha,
            va=va,
            zorder=6,
            annotation_clip=False,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    detour, = ax.plot([], [], color=ORANGE, linestyle="--", linewidth=1.15, label="Charging detour")
    ax.set_title("(b) Charging-certificate witness", loc="left", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [
        "Depot",
        "Selected customer",
        "Unselected source customer",
        "Charging station",
        "Charging detour",
    ]
    by_label = dict(zip(labels, handles))
    by_label["Charging detour"] = detour
    legend_ax.legend(
        [by_label[name] for name in order],
        order,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=7,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in list(axes[0].texts) + list(axes[1].texts):
        boxes.append((artist.get_text().replace("\n", " "), artist.axes, artist.get_window_extent(renderer).expanded(1.5, 1.5)))
    overlaps = []
    for i, (left_text, left_ax, left_box) in enumerate(boxes):
        for right_text, right_ax, right_box in boxes[i + 1 :]:
            if left_ax is right_ax and left_box.overlaps(right_box):
                overlaps.append(f"{left_text} / {right_text}")
    if overlaps:
        raise SystemExit("figure labels overlap: " + "; ".join(overlaps))
    fig.savefig(out_dir / "figure_fixed_route_generation.pdf")
    fig.savefig(out_dir / "figure_fixed_route_generation.svg")
    fig.savefig(out_dir / "figure_fixed_route_generation.png", dpi=300)
    plt.close(fig)

    n_stations = sum(1 for step in trace if step.get("kind") == "CHARGE")
    caption = (
        "Fixed-route construction on one existing SynthCharge validation instance "
        f"({route['route_id']}; generator seed {route['generator_seed']}; "
        f"{route['n_customers']} customers on the frozen route). "
        "The legend applies to both panels and is placed below them. "
        "(a) PyVRP solves a customer-only capacitated pickup vehicle-routing problem with time windows. "
        "The solid arrows follow the selected customer sequence from the depot back to the depot. "
        "Charging stations are available facilities and are not inserted into this route. "
        "Faint markers are source-instance customers assigned to other vehicle routes and therefore not part of the frozen sequence. "
        "(b) The same customer order with the stored charging-certificate trajectory. "
        f"Dashed arrows are charging detours ({n_stations} station visits in this witness); "
        "station labels report the target state of charge recorded in that trace. "
        "This panel is an existing feasible charging witness from the certificate search. "
        "It is not a globally optimal charging plan and it was not produced by FA-HPPO. "
        "The customer visiting order is unchanged between panels."
    )
    (out_dir / "figure_fixed_route_generation_caption.txt").write_text(caption + "\n", encoding="utf-8")
    return caption


def audit_v4(audit_rows: list[dict], corpus: list[dict]) -> dict:
    n_examined = len(audit_rows)
    accepted = [row for row in audit_rows if row.get("accepted_route_id")]
    rejected_instances = [row for row in audit_rows if not row.get("accepted_route_id")]
    reason_events = Counter()
    events_on_accepted_instances = 0
    events_on_unaccepted_instances = 0
    for row in audit_rows:
        reasons = row.get("rejection_reasons") or []
        if row.get("accepted_route_id"):
            events_on_accepted_instances += len(reasons)
        else:
            events_on_unaccepted_instances += len(reasons)
        for item in reasons:
            reason_events[str(item.get("reason") or "missing_reason")] += 1
    by_layout = Counter(row["layout"] for row in accepted)
    by_source_n = Counter(int(row["n_customers"]) for row in accepted)
    corpus_by_id = {row["route_id"]: row for row in corpus}
    accepted_ids = [row["accepted_route_id"] for row in accepted]
    missing = sorted(set(accepted_ids) - set(corpus_by_id))
    extra = sorted(set(corpus_by_id) - set(accepted_ids))
    duplicate_ids = sorted(rid for rid, count in Counter(accepted_ids).items() if count > 1)
    length_counts = Counter()
    class_counts = Counter()
    bin_mismatches = []
    for row in accepted:
        route = corpus_by_id.get(row["accepted_route_id"])
        if route is None:
            continue
        group = length_group(int(route["n_customers"]))
        length_counts[group or "outside"] += 1
        class_counts[route.get("charge_class") or "missing"] += 1
        if route.get("length_bin") != group:
            bin_mismatches.append(route["route_id"])
    return {
        "n_candidate_instances": n_examined,
        "n_instances_one_accepted_route": len(accepted),
        "n_instances_no_accepted_route": len(rejected_instances),
        "n_rejection_events": int(sum(reason_events.values())),
        "rejection_events_by_reason": dict(reason_events),
        "rejection_events_on_eventually_accepted_instances": events_on_accepted_instances,
        "rejection_events_on_instances_with_no_accepted_route": events_on_unaccepted_instances,
        "accepted_by_layout": dict(by_layout),
        "accepted_by_source_customer_count": {str(k): int(by_source_n.get(k, 0)) for k in SOURCE_SIZES},
        "accepted_by_frozen_route_length_group": dict(length_counts),
        "accepted_by_charge_class": dict(class_counts),
        "corpus_route_count": len(corpus),
        "accepted_ids_missing_from_corpus": missing,
        "corpus_ids_missing_from_audit": extra,
        "duplicate_accepted_ids": duplicate_ids,
        "length_bin_mismatches": bin_mismatches,
        "counting_unit": {
            "audit_line": "one attempted source instance",
            "rejection_reasons": "candidate-route evaluations rejected before acceptance or before the instance was exhausted; not distinct rejected instances",
            "unverified_timeout_and_unverified_search_exhausted": "search did not certify a trajectory; not proof of infeasibility",
        },
    }


def write_audit_tables(summary: dict) -> None:
    rows = [
        ("instances", "candidate source instances examined", summary["n_candidate_instances"], "source instance", ""),
        ("instances", "instances yielding one accepted route", summary["n_instances_one_accepted_route"], "source instance", "at most one route is kept per instance"),
        ("instances", "instances yielding no accepted route", summary["n_instances_no_accepted_route"], "source instance", ""),
        ("rejections", "recorded route-evaluation rejections", summary["n_rejection_events"], "candidate-route evaluation", "not distinct source instances"),
    ]
    for reason, count in sorted(summary["rejection_events_by_reason"].items()):
        note = ""
        if reason in {"unverified_timeout", "unverified_search_exhausted"}:
            note = "unverified search outcome; not proof of infeasibility"
        rows.append(("rejection_reason", reason, count, "candidate-route evaluation", note))
    rows.append(
        (
            "rejections",
            "rejection events on instances that were later accepted",
            summary["rejection_events_on_eventually_accepted_instances"],
            "candidate-route evaluation",
            "earlier vehicle routes of an accepted instance",
        )
    )
    rows.append(
        (
            "rejections",
            "rejection events on instances with no accepted route",
            summary["rejection_events_on_instances_with_no_accepted_route"],
            "candidate-route evaluation",
            "",
        )
    )
    for layout, count in sorted(summary["accepted_by_layout"].items()):
        rows.append(("accepted_layout", layout, count, "accepted route", "spatial layout of the source instance"))
    for size, count in summary["accepted_by_source_customer_count"].items():
        rows.append(("accepted_source_size", f"source customers = {size}", count, "accepted route", "not the frozen-route length"))
    for group in ("short", "medium", "long"):
        rows.append(
            (
                "frozen_route_length",
                group,
                int(summary["accepted_by_frozen_route_length_group"].get(group, 0)),
                "accepted route",
                "short 3-5, medium 6-10, long 11-15 customers on the frozen route",
            )
        )
    for label in ("charging_required", "no_charge_required"):
        rows.append(
            (
                "charge_class",
                label,
                int(summary["accepted_by_charge_class"].get(label, 0)),
                "accepted route",
                "quota label from continue-only replay plus certificate; not an FA-HPPO outcome",
            )
        )
    rows.append(("cross_check", "frozen V4 TEST corpus routes", summary["corpus_route_count"], "frozen route", "expected 180"))

    fieldnames = ["section", "item", "count", "unit", "note"]
    with (OUT / "table_route_generation_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for section, item, count, unit, note in rows:
            writer.writerow({"section": section, "item": item, "count": count, "unit": unit, "note": note})

    md = [
        "# Table A. V4 TEST route-generation audit",
        "",
        "Each line of `candidate_audit.jsonl` is one attempted source instance. "
        "`rejection_reasons` counts candidate-route evaluations, not distinct rejected instances. "
        "`unverified_timeout` and `unverified_search_exhausted` mean the certificate search did not finish with a witness; "
        "they are not proofs of infeasibility. "
        "The retained benchmark is quota-balanced by spatial layout, actual frozen-route length, and charging requirement, "
        "not by the source-instance customer count (15, 30, or 50).",
        "",
        "| Section | Item | Count | Unit | Note |",
        "|---|---|---:|---|---|",
    ]
    for section, item, count, unit, note in rows:
        md.append(f"| {section} | {item} | {count} | {unit} | {note} |")
    md.append("")
    (OUT / "table_route_generation_audit.md").write_text("\n".join(md), encoding="utf-8")

    def tex(text: str) -> str:
        return (
            str(text)
            .replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    tex_lines = [
        "% Table A. V4 TEST route-generation audit",
        "\\begin{tabular}{llrll}",
        "\\hline",
        "Section & Item & Count & Unit & Note \\\\",
        "\\hline",
    ]
    for section, item, count, unit, note in rows:
        tex_lines.append(f"{tex(section)} & {tex(item)} & {count} & {tex(unit)} & {tex(note)} \\\\")
    tex_lines.extend(["\\hline", "\\end{tabular}", ""])
    (OUT / "table_route_generation_audit.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    (OUT / "route_generation_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_compact_audit_table(summary)


def write_compact_audit_table(summary: dict) -> None:
    """Journal subset of Table A. Counts are copied from the full audit summary."""
    reasons = summary["rejection_events_by_reason"]
    instance_total = summary["n_instances_one_accepted_route"] + summary["n_instances_no_accepted_route"]
    reason_total = (
        int(reasons["outside_length_bins"])
        + int(reasons["quota_full"])
        + int(reasons["unverified_timeout"])
        + int(reasons["unverified_search_exhausted"])
    )
    placement_total = (
        summary["rejection_events_on_eventually_accepted_instances"]
        + summary["rejection_events_on_instances_with_no_accepted_route"]
    )
    if instance_total != summary["n_candidate_instances"]:
        raise SystemExit("compact Table A instance rows do not sum to the examined-instance count")
    if reason_total != summary["n_rejection_events"] or placement_total != summary["n_rejection_events"]:
        raise SystemExit("compact Table A rejection rows do not sum to the rejection-event count")
    rows = [
        ("Source instance", "Candidate instances examined", summary["n_candidate_instances"], "Sum of the next two rows"),
        ("Source instance", "Instances with one accepted route", summary["n_instances_one_accepted_route"], "At most one route is kept per instance"),
        ("Source instance", "Instances with no accepted route", summary["n_instances_no_accepted_route"], ""),
        ("Route evaluation", "Rejection events", summary["n_rejection_events"], "Candidate vehicle routes, not source instances"),
        ("Route evaluation", "Outside length bins", int(reasons["outside_length_bins"]), "Frozen-route length outside 3–15 customers"),
        ("Route evaluation", "Quota full", int(reasons["quota_full"]), "Layout, length, and charge-class cell already filled"),
        ("Route evaluation", "Unverified timeout", int(reasons["unverified_timeout"]), "Search ended without a witness; not infeasibility"),
        ("Route evaluation", "Unverified search exhausted", int(reasons["unverified_search_exhausted"]), "Search ended without a witness; not infeasibility"),
        ("Route evaluation", "Events on instances later accepted", summary["rejection_events_on_eventually_accepted_instances"], "Earlier vehicle routes of an accepted instance"),
        ("Route evaluation", "Events on instances with no accepted route", summary["rejection_events_on_instances_with_no_accepted_route"], ""),
    ]
    note = (
        "Compact Table A. The first block counts source instances "
        f"({summary['n_candidate_instances']} = {summary['n_instances_one_accepted_route']} + {summary['n_instances_no_accepted_route']}). "
        "The second block counts candidate-route evaluations, not source instances. "
        f"The four reason rows sum to {summary['n_rejection_events']}, and the last two rows are a separate partition of the same events "
        f"({summary['rejection_events_on_eventually_accepted_instances']} + {summary['rejection_events_on_instances_with_no_accepted_route']}). "
        "Unverified timeout and unverified search exhausted mean the certificate search did not return a witness. "
        "They are not proofs of infeasibility. "
        "The full audit, including accepted layout, source size, length bin, and charge class, remains in `table_route_generation_audit`."
    )
    fieldnames = ["counting_unit", "item", "count", "note"]
    with (OUT / "table_route_generation_audit_compact.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for unit, item, count, row_note in rows:
            writer.writerow({"counting_unit": unit, "item": item, "count": count, "note": row_note})
    md = [
        "# Table A (compact). V4 TEST route generation",
        "",
        note,
        "",
        "| Counting unit | Item | Count | Note |",
        "|---|---|---:|---|",
    ]
    for unit, item, count, row_note in rows:
        md.append(f"| {unit} | {item} | {count} | {row_note} |")
    md.append("")
    (OUT / "table_route_generation_audit_compact.md").write_text("\n".join(md), encoding="utf-8")

    def tex(text: str) -> str:
        return (
            str(text)
            .replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    tex_lines = [
        "% Compact Table A. Source-instance counts are separate from route-evaluation events.",
        "\\begin{tabular}{llr}",
        "\\hline",
        "Counting unit & Item & Count \\\\",
        "\\hline",
    ]
    for unit, item, count, _row_note in rows:
        tex_lines.append(f"{tex(unit)} & {tex(item)} & {count} \\\\")
    tex_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "% Source instances: examined = accepted + not accepted.",
            "% Route-evaluation events: reason rows sum to the event total, and the last two rows partition that same total.",
            "% unverified timeout and unverified search exhausted are not proofs of infeasibility.",
            "",
        ]
    )
    (OUT / "table_route_generation_audit_compact.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    (OUT / "table_route_generation_audit_compact_caption.txt").write_text(note + "\n", encoding="utf-8")


def split_characteristics(name: str, routes: list[dict], certificates: dict[str, dict]) -> dict:
    customers = [int(row["n_customers"]) for row in routes]
    distances = [float(row["route_distance"]) for row in routes]
    layouts = Counter(row.get("layout") for row in routes)
    groups = Counter(length_group(int(row["n_customers"])) for row in routes)
    classes = Counter(row.get("charge_class") for row in routes)
    visits = []
    missing_cert = []
    class_vs_visits = {"charging_required_without_visit": 0, "no_charge_required_with_visit": 0}
    for row in routes:
        cert = certificates.get(row["route_id"])
        if cert is None or cert.get("trace") is None:
            missing_cert.append(row["route_id"])
            continue
        n_visits = sum(1 for step in cert["trace"] if step.get("kind") == "CHARGE")
        visits.append(n_visits)
        stored = row.get("certificate_station_visits")
        if stored is not None and int(stored) != n_visits:
            class_vs_visits.setdefault("stored_visit_mismatch", 0)
            class_vs_visits["stored_visit_mismatch"] = class_vs_visits.get("stored_visit_mismatch", 0) + 1
        if row.get("charge_class") == "charging_required" and n_visits < 1:
            class_vs_visits["charging_required_without_visit"] += 1
        if row.get("charge_class") == "no_charge_required" and n_visits >= 1:
            class_vs_visits["no_charge_required_with_visit"] += 1
    distance_field_mismatch = sum(
        1 for row in routes if abs(float(row["route_distance"]) - float(row["total_distance"])) > 1e-9
    )
    id_mismatch = sum(1 for row in routes if int(row["n_customers"]) != len(row["customer_ids"]))
    return {
        "split": name,
        "n_routes": len(routes),
        "customers_mean": float(statistics.fmean(customers)),
        "customers_min": min(customers),
        "customers_median": float(statistics.median(customers)),
        "customers_max": max(customers),
        "mean_customer_only_distance": float(statistics.fmean(distances)),
        "n_charging_required": int(classes.get("charging_required", 0)),
        "n_no_charge_required": int(classes.get("no_charge_required", 0)),
        "layout_C": int(layouts.get("C", 0)),
        "layout_R": int(layouts.get("R", 0)),
        "layout_RC": int(layouts.get("RC", 0)),
        "length_short": int(groups.get("short", 0)),
        "length_medium": int(groups.get("medium", 0)),
        "length_long": int(groups.get("long", 0)),
        "certificate_station_visits_mean": float(statistics.fmean(visits)) if visits else None,
        "certificate_station_visits_min": min(visits) if visits else None,
        "certificate_station_visits_median": float(statistics.median(visits)) if visits else None,
        "certificate_station_visits_max": max(visits) if visits else None,
        "n_missing_certificates": len(missing_cert),
        "distance_field_mismatches": distance_field_mismatch,
        "customer_count_mismatches": id_mismatch,
        "certificate_vs_charge_class": class_vs_visits,
    }


def write_characteristics(rows: list[dict]) -> None:
    fieldnames = [
        "split",
        "n_routes",
        "customers_mean",
        "customers_min",
        "customers_median",
        "customers_max",
        "mean_customer_only_distance_coordinate_units",
        "n_charging_required",
        "n_no_charge_required",
        "layout_C",
        "layout_R",
        "layout_RC",
        "length_short",
        "length_medium",
        "length_long",
        "certificate_station_visits_mean",
        "certificate_station_visits_min",
        "certificate_station_visits_median",
        "certificate_station_visits_max",
    ]
    with (OUT / "table_route_characteristics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row[key] for key in fieldnames if key != "mean_customer_only_distance_coordinate_units"}
            payload["mean_customer_only_distance_coordinate_units"] = row["mean_customer_only_distance"]
            writer.writerow(payload)
    distance_note = (
        "Mean customer-only distance is the mean length of the frozen depot–customers–depot sequence "
        "in normalized SynthCharge coordinate units. "
        "Instance coordinates lie in the generator's unit square. "
        "These distances are not kilometers."
    )
    md = [
        "# Table B. Frozen-route characteristics",
        "",
        "TRAIN and VALIDATION are the SynthCharge development corpora. "
        "TEST is the fresh V4 corpus only. "
        "Customer counts describe the frozen customer sequence. "
        f"{distance_note} "
        "Station-visit statistics describe the stored charging certificate, not FA-HPPO decisions. "
        "A no-charge-required label means a continue-only replay completed. "
        "In these corpora that label coincides with zero certificate station visits, but the certificate remains one feasible witness rather than an FA-HPPO trajectory.",
        "",
        "| Split | Routes | Customers mean [min, median, max] | Mean customer-only distance (coordinate units) | Charging required | No charge required | C / R / RC | Short / medium / long | Certificate station visits mean [min, median, max] |",
        "|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        md.append(
            "| {split} | {n_routes} | {customers_mean:.2f} [{customers_min}, {customers_median:.1f}, {customers_max}] | "
            "{mean_customer_only_distance:.3f} | {n_charging_required} | {n_no_charge_required} | "
            "{layout_C} / {layout_R} / {layout_RC} | {length_short} / {length_medium} / {length_long} | "
            "{certificate_station_visits_mean:.2f} [{certificate_station_visits_min}, {certificate_station_visits_median:.1f}, {certificate_station_visits_max}] |".format(
                **row
            )
        )
    md.append("")
    (OUT / "table_route_characteristics.md").write_text("\n".join(md), encoding="utf-8")
    write_compact_characteristics(rows, distance_note)


def write_compact_characteristics(rows: list[dict], distance_note: str) -> None:
    fieldnames = [
        "split",
        "n_routes",
        "customers_mean",
        "customers_min",
        "customers_median",
        "customers_max",
        "mean_distance_coordinate_units",
        "n_charging_required",
        "n_no_charge_required",
        "layout_C_R_RC",
        "length_short_medium_long",
        "certificate_station_visits_mean",
        "certificate_station_visits_min",
        "certificate_station_visits_median",
        "certificate_station_visits_max",
    ]
    packed = []
    for row in rows:
        packed.append(
            {
                "split": row["split"],
                "n_routes": row["n_routes"],
                "customers_mean": f"{row['customers_mean']:.2f}",
                "customers_min": row["customers_min"],
                "customers_median": f"{row['customers_median']:.1f}",
                "customers_max": row["customers_max"],
                "mean_distance_coordinate_units": f"{row['mean_customer_only_distance']:.3f}",
                "n_charging_required": row["n_charging_required"],
                "n_no_charge_required": row["n_no_charge_required"],
                "layout_C_R_RC": f"{row['layout_C']}/{row['layout_R']}/{row['layout_RC']}",
                "length_short_medium_long": f"{row['length_short']}/{row['length_medium']}/{row['length_long']}",
                "certificate_station_visits_mean": f"{row['certificate_station_visits_mean']:.2f}",
                "certificate_station_visits_min": row["certificate_station_visits_min"],
                "certificate_station_visits_median": f"{row['certificate_station_visits_median']:.1f}",
                "certificate_station_visits_max": row["certificate_station_visits_max"],
            }
        )
    with (OUT / "table_route_characteristics_compact.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(packed)
    caption = (
        "Frozen customer-route characteristics for TRAIN, VALIDATION, and the V4 TEST corpus. "
        f"{distance_note} "
        "Charging-required and no-charge-required counts are quota labels. "
        "Certificate station visits count CHARGE steps in the stored witness, not FA-HPPO decisions. "
        "Layouts are clustered (C), random (R), and mixed (RC). "
        "Length groups are short (3–5), medium (6–10), and long (11–15) customers on the frozen route."
    )
    md = [
        "# Table B (compact). Frozen-route characteristics",
        "",
        caption,
        "",
        "| Split | Routes | Customers mean [min, median, max] | Mean distance (coordinate units) | Charging / no charge | C / R / RC | Short / medium / long | Certificate visits mean [min, median, max] |",
        "|---|---:|---|---:|---|---|---|---|",
    ]
    for row in rows:
        md.append(
            "| {split} | {n_routes} | {customers_mean:.2f} [{customers_min}, {customers_median:.1f}, {customers_max}] | "
            "{mean_customer_only_distance:.3f} | {n_charging_required} / {n_no_charge_required} | "
            "{layout_C} / {layout_R} / {layout_RC} | {length_short} / {length_medium} / {length_long} | "
            "{certificate_station_visits_mean:.2f} [{certificate_station_visits_min}, {certificate_station_visits_median:.1f}, {certificate_station_visits_max}] |".format(
                **row
            )
        )
    md.append("")
    (OUT / "table_route_characteristics_compact.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "table_route_characteristics_compact_caption.txt").write_text(caption + "\n", encoding="utf-8")
    tex_lines = [
        "% Compact Table B. Distance is in normalized SynthCharge coordinate units, not kilometers.",
        "\\begin{tabular}{lrrrrlll}",
        "\\hline",
        "Split & Routes & Customers & Distance & Charging & No charge & C/R/RC & Short/medium/long \\\\",
        "\\hline",
    ]
    for row in rows:
        tex_lines.append(
            "{split} & {n_routes} & {customers_mean:.2f} [{customers_min}, {customers_median:.1f}, {customers_max}] & "
            "{mean_customer_only_distance:.3f} & {n_charging_required} & {n_no_charge_required} & "
            "{layout_C}/{layout_R}/{layout_RC} & {length_short}/{length_medium}/{length_long} \\\\".format(**row)
        )
    tex_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "% Distance: mean depot-customers-depot length in normalized SynthCharge coordinate units (unit square), not kilometers.",
            "\\begin{tabular}{lrrrr}",
            "\\hline",
            "Split & Certificate visits mean & Min & Median & Max \\\\",
            "\\hline",
        ]
    )
    for row in rows:
        tex_lines.append(
            "{split} & {certificate_station_visits_mean:.2f} & {certificate_station_visits_min} & {certificate_station_visits_median:.1f} & {certificate_station_visits_max} \\\\".format(
                **row
            )
        )
    tex_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "% Certificate station visits count CHARGE steps in the stored witness, not FA-HPPO decisions.",
            "",
        ]
    )
    (OUT / "table_route_characteristics_compact.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def draw_pipeline() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.4, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        (0.90, "#E8F1F8", "SynthCharge instance generation"),
        (0.78, "#E8F1F8", "Customer-only pickup-VRPTW\noptimization with PyVRP"),
        (0.66, "#E8F1F8", "Extraction of individual\nordered vehicle routes"),
        (0.54, "#F8EFE6", "Whole-route charging\ncertificate search"),
        (0.42, "#F8EFE6", "Classification by\ncharging requirement"),
        (0.30, "#F4F4F4", "Route-length, layout,\nand quota filtering"),
        (0.18, "#E7F6EF", "Frozen TRAIN, VALIDATION,\nand TEST route corpora"),
        (0.06, "#E8F1F8", "FA-HPPO charging decisions\non the frozen customer sequences"),
    ]
    boxes = []
    for y, color, text in steps:
        box = FancyBboxPatch(
            (0.18, y),
            0.64,
            0.09,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(0.50, y + 0.045, text, ha="center", va="center", fontsize=8.5)
        boxes.append(y)
    for y0, y1 in zip(boxes, boxes[1:]):
        ax.annotate(
            "",
            xy=(0.50, y1 + 0.09),
            xytext=(0.50, y0),
            arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 0.9},
        )
    ax.text(
        0.50,
        0.005,
        "PyVRP does not optimize battery charging.  "
        "The certificate search witnesses one feasible charging trajectory; it is not a globally exact charging optimizer.\n"
        "FA-HPPO does not create, reorder, or modify the customer sequence.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#333333",
    )
    fig.savefig(OUT / "figure_route_generation_pipeline.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure_route_generation_pipeline.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure_route_generation_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (OUT / "figure_route_generation_pipeline_caption.txt").write_text(
        "Route-generation pipeline used to build the frozen customer sequences. "
        "SynthCharge draws a source instance. PyVRP then solves a customer-only capacitated pickup vehicle-routing problem with time windows and does not decide battery charging. "
        "Each ordered vehicle route is submitted to a whole-route charging-certificate search, which looks for one feasible charging trajectory and is not a globally exact charging optimizer. "
        "Routes are classified by charging requirement and retained only while layout, frozen-route-length, and charging-requirement quotas remain open. "
        "The retained sequences are frozen. FA-HPPO later chooses charging insertions and charge amounts on those sequences; it does not create or reorder customers. "
        "A compact horizontal version with the unverified-search and quota-rejection branches is given in figure_route_generation_pipeline_horizontal.\n",
        encoding="utf-8",
    )
    draw_pipeline_horizontal()


def _flow_box(ax, x, y, w, h, text, face, fontsize=7.0, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face,
        edgecolor="#333333",
        linewidth=0.8,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#222222", linespacing=1.15)


def _flow_arrow(ax, start, end, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#333333",
            linestyle="--" if dashed else "-",
            shrinkA=1,
            shrinkB=1,
            zorder=3,
        )
    )


def draw_pipeline_horizontal() -> None:
    """Compact horizontal pipeline. The vertical figure is left unchanged."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.4, 2.72))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")
    main = [
        (2.0, "SynthCharge\ninstance", "#E8F1F8"),
        (18.2, "PyVRP\ncustomer-only\nVRPTW", "#E8F1F8"),
        (34.4, "Extract ordered\nvehicle routes", "#E8F1F8"),
        (50.6, "Charging-\ncertificate\nsearch", "#F8EFE6"),
        (66.8, "Classify and\napply quotas", "#F4F4F4"),
        (83.0, "Frozen routes;\nFA-HPPO", "#E7F6EF"),
    ]
    box_w, box_h, main_y = 14.6, 13.2, 20.2
    for x, text, face in main:
        _flow_box(ax, x, main_y, box_w, box_h, text, face, fontsize=6.7)
    for (x0, _, _), (x1, _, _) in zip(main, main[1:]):
        _flow_arrow(ax, (x0 + box_w, main_y + box_h / 2), (x1, main_y + box_h / 2))
    _flow_box(
        ax,
        42.4,
        1.3,
        22.6,
        12.4,
        "Unverified timeout or\nexhausted search\n(not infeasibility)",
        "#FBF6F1",
        fontsize=6.4,
        dashed=True,
    )
    _flow_box(
        ax,
        68.6,
        1.3,
        22.6,
        12.4,
        "Outside length bins\nor quota full\n(not retained)",
        "#F7F7F7",
        fontsize=6.4,
        dashed=True,
    )
    cert_x = main[3][0] + box_w / 2
    quota_x = main[4][0] + box_w / 2
    _flow_arrow(ax, (cert_x, main_y), (53.7, 13.7), dashed=True)
    _flow_arrow(ax, (quota_x, main_y), (79.9, 13.7), dashed=True)
    fig.savefig(OUT / "figure_route_generation_pipeline_horizontal.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "figure_route_generation_pipeline_horizontal.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "figure_route_generation_pipeline_horizontal.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    (OUT / "figure_route_generation_pipeline_horizontal_caption.txt").write_text(
        "Compact route-generation pipeline. "
        "The upper row is the path from a SynthCharge instance to a frozen customer sequence and the later FA-HPPO charging controller. "
        "PyVRP solves a customer-only capacitated pickup vehicle-routing problem with time windows and does not decide battery charging. "
        "The charging-certificate search looks for one feasible charging trajectory; it is not a globally exact charging optimizer. "
        "Dashed branches are routes that are not retained. "
        "An unverified timeout or an exhausted search means no witness was returned and is not a proof of infeasibility. "
        "A route outside the frozen-route length bins, or a route whose layout, length, and charging-requirement quota is already full, is not retained. "
        "FA-HPPO chooses charging insertions and charge amounts on the frozen sequences and does not create or reorder customers. "
        "The vertical pipeline figure remains available as figure_route_generation_pipeline.\n",
        encoding="utf-8",
    )


def expected_quotas() -> dict:
    v2 = json.loads(V2_SPEC.read_text(encoding="utf-8"))
    v4 = json.loads(V4_SPEC.read_text(encoding="utf-8"))
    def total(spec, split):
        quota = spec["quotas"][split]
        cells = len(spec["layouts"]) * len(spec["length_bins"])
        return cells * int(quota["routes_per_cell"])
    return {
        "train_expected": total(v2, "train"),
        "validation_expected": total(v2, "validation"),
        "v4_test_expected": total(v4, "test"),
        "v4_charging_required_expected": len(v4["layouts"]) * len(v4["length_bins"]) * int(v4["quotas"]["test"]["charging_required"]),
        "v4_no_charge_expected": len(v4["layouts"]) * len(v4["length_bins"]) * int(v4["quotas"]["test"]["no_charge_required"]),
        "v2_test_seed_start_not_used": int(v2["seed_starts"]["test"]),
        "v4_test_seed_start": int(v4["seed_starts"]["test"]),
        "train_seed_start": int(v2["seed_starts"]["train"]),
        "validation_seed_start": int(v2["seed_starts"]["validation"]),
    }


def collect_inconsistencies(summary: dict, characteristics: list[dict], quotas: dict, meta: dict) -> list[str]:
    issues = []
    if summary["corpus_route_count"] != 180 or summary["n_instances_one_accepted_route"] != 180:
        issues.append(
            f"V4 accepted/corpus counts are {summary['n_instances_one_accepted_route']}/{summary['corpus_route_count']}, expected 180/180."
        )
    if summary["accepted_ids_missing_from_corpus"] or summary["corpus_ids_missing_from_audit"] or summary["duplicate_accepted_ids"]:
        issues.append("Accepted audit route IDs do not match the frozen V4 corpus exactly.")
    if summary["length_bin_mismatches"]:
        issues.append(f"{len(summary['length_bin_mismatches'])} routes have a stored length bin that disagrees with the customer count.")
    test = next(row for row in characteristics if row["split"] == "V4 TEST")
    if test["n_routes"] != quotas["v4_test_expected"]:
        issues.append("V4 TEST route count disagrees with the V4 quota configuration.")
    if test["n_charging_required"] != quotas["v4_charging_required_expected"] or test["n_no_charge_required"] != quotas["v4_no_charge_expected"]:
        issues.append("V4 charge-class counts disagree with the configured quotas.")
    if meta["test"]["n_routes"] != test["n_routes"]:
        issues.append("V4 corpus_metadata n_routes disagrees with the corpus file.")
    train = next(row for row in characteristics if row["split"] == "TRAIN")
    val = next(row for row in characteristics if row["split"] == "VALIDATION")
    if train["n_routes"] != quotas["train_expected"] or meta["train"]["n_routes"] != train["n_routes"]:
        issues.append("TRAIN route count disagrees with its metadata or the V2 quota configuration.")
    if val["n_routes"] != quotas["validation_expected"] or meta["validation"]["n_routes"] != val["n_routes"]:
        issues.append("VALIDATION route count disagrees with its metadata or the V2 quota configuration.")
    for row in characteristics:
        if row["n_missing_certificates"] or row["distance_field_mismatches"] or row["customer_count_mismatches"]:
            issues.append(
                f"{row['split']} has missing certificates ({row['n_missing_certificates']}), "
                f"distance-field mismatches ({row['distance_field_mismatches']}), "
                f"or customer-count mismatches ({row['customer_count_mismatches']})."
            )
        if row["certificate_vs_charge_class"]["charging_required_without_visit"]:
            issues.append(f"{row['split']} has charging-required routes whose certificate contains no station visit.")
    if not issues:
        issues.append("No disagreement was found among corpus counts, metadata, audit IDs, length bins, and the declared quotas.")
    return issues


def write_report(example: dict, caption: str, summary: dict, characteristics: list[dict], issues: list[str], provenance: dict) -> None:
    for row in characteristics:
        note = row["certificate_vs_charge_class"]
        if note["no_charge_required_with_visit"] or note["charging_required_without_visit"]:
            raise SystemExit(f"{row['split']} no longer matches the reported charge-class and certificate-visit pattern")
    lines = [
        "# Fixed-route construction audit",
        "",
        "This note documents the existing route-generation process. No routes were regenerated, no policy was trained, and the consumed V4 TEST was not re-evaluated.",
        "",
        "## Authoritative inputs",
        "",
        "- TRAIN and VALIDATION: `data/routes_v2/synthcharge_final/{train,validation}/`, built from `configs/v2/synthcharge_final_benchmark.json` (seed starts 100000 and 200000).",
        "- V4 TEST: `data/routes_v2/synthcharge_v4_test/`, built from `results/v4/test/configs/synthcharge_v4_test.json` (seed start 500000) by `scripts/v4/test/generate_v4_test.py`.",
        "- The V2 configuration also declares a test seed start of 300000. That range is not the V4 TEST set and was not analyzed here. The V3 TEST corpus was not opened.",
        "- PyVRP builds customer-only pickup-VRPTW routes (`src/routing/pyvrp_generator.py`). The charging certificate is a best-first search for one simulator-valid trajectory (`src/exact/charging_certificate.py`); its `exact` flag is false.",
        "",
        "## Figure A",
        "",
        "Selection rule: among VALIDATION routes labeled charging-required, with 6 to 10 customers, and with both the source instance file and a certificate trace available, choose the lowest generator seed and then the route identifier.",
        "",
        f"Selected route: `{example['route_id']}` (seed {example['generator_seed']}, layout {example['layout']}, {example['n_customers']} customers, source instance `{example['raw_instance_id']}`).",
        "",
        caption,
        "",
        "## Table A counting units",
        "",
        "The full audit is `table_route_generation_audit`. "
        "The journal subset is `table_route_generation_audit_compact` (10 rows). "
        "Its first block counts source instances. Its second block counts candidate-route evaluations. Those units are not interchangeable.",
        "",
        f"- Candidate instances examined: {summary['n_candidate_instances']}.",
        f"- Instances with one accepted route: {summary['n_instances_one_accepted_route']}.",
        f"- Instances with no accepted route: {summary['n_instances_no_accepted_route']}.",
        f"- Rejection events: {summary['n_rejection_events']}. These are candidate vehicle routes rejected inside an instance, including routes examined before a later route from the same instance was accepted ({summary['rejection_events_on_eventually_accepted_instances']} such events).",
        "- `unverified_timeout` and `unverified_search_exhausted` are incomplete searches, not infeasibility certificates.",
        "- Acceptance is balanced on layout, frozen-route length (short 3-5, medium 6-10, long 11-15), and charging requirement. Source-instance sizes 15, 30, and 50 are generation settings, not the balancing factors.",
        "",
        "## Table B",
        "",
        "Customer-only distance is the stored `route_distance` of the depot-customer-depot sequence, expressed in normalized SynthCharge coordinate units. "
        "Instance coordinates lie in the generator's unit square. The values are not kilometers. "
        "`route_distance` matched `total_distance` on every analyzed route.",
        "The full table is `table_route_characteristics`. The journal subset is `table_route_characteristics_compact`.",
        "Certificate station visits count `CHARGE` steps in the stored trace. In these three corpora, every no-charge-required route has zero certificate station visits, and every charging-required route has at least one. "
        "The label is still not defined by that count: no-charge-required means a continue-only replay completed. "
        "The stored trace is one feasible witness, not the continue-only trajectory and not an FA-HPPO decision.",
        "",
        "## Figure B",
        "",
        "The vertical pipeline figure separates instance generation, customer-only PyVRP routing, certificate search, quota filtering, freezing, and the later FA-HPPO charging controller.",
        "The compact horizontal figure keeps that path and adds two dashed branches: unverified certificate searches, which are not proofs of infeasibility, and routes rejected because the frozen-route length bin is outside the accepted range or the quota cell is already full.",
        "",
        "## Checks against metadata and quotas",
        "",
    ]
    for issue in issues:
        lines.append(f"- {issue}")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Git commit: `{provenance['git_commit']}`",
            "- Input SHA-256 values are listed in `provenance.json`.",
            "",
            "## Limitations",
            "",
            "- The illustration is one validation route chosen by a disclosed rule. It is not a performance example and is not a TEST evaluation.",
            "- Non-selected customers are the source-instance customers omitted from the accepted vehicle route. Their assignment to other PyVRP routes is not redrawn.",
            "- Certificate target-SOC annotations are the values stored in the witness, not decisions of FA-HPPO.",
            "- Rejection counts must not be read as a failure rate over source instances.",
        ]
    )
    (OUT / "ROUTE_GENERATION_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(TRAIN_DIR / "corpus.jsonl")
    val = read_jsonl(VAL_DIR / "corpus.jsonl")
    test = read_jsonl(TEST_DIR / "corpus.jsonl")
    train_certs = {row["route_id"]: row for row in read_jsonl(TRAIN_DIR / "certificates.jsonl")}
    val_certs = {row["route_id"]: row for row in read_jsonl(VAL_DIR / "certificates.jsonl")}
    test_certs = {row["route_id"]: row for row in read_jsonl(TEST_DIR / "certificates.jsonl")}
    audit_rows = read_jsonl(TEST_DIR / "candidate_audit.jsonl")
    example = choose_validation_example(val, val_certs)
    nodes = load_nodes(ROOT / example["relative_path"])
    trace = val_certs[example["route_id"]]["trace"]
    witness_path(list(example["customer_ids"]), trace)
    caption = draw_fixed_route_figure(example, nodes, trace, OUT)
    summary = audit_v4(audit_rows, test)
    write_audit_tables(summary)
    characteristics = [
        split_characteristics("TRAIN", train, train_certs),
        split_characteristics("VALIDATION", val, val_certs),
        split_characteristics("V4 TEST", test, test_certs),
    ]
    write_characteristics(characteristics)
    draw_pipeline()
    quotas = expected_quotas()
    meta = {
        "train": json.loads((TRAIN_DIR / "corpus_metadata.json").read_text(encoding="utf-8")),
        "validation": json.loads((VAL_DIR / "corpus_metadata.json").read_text(encoding="utf-8")),
        "test": json.loads((TEST_DIR / "corpus_metadata.json").read_text(encoding="utf-8")),
    }
    issues = collect_inconsistencies(summary, characteristics, quotas, meta)
    inputs = [
        V2_SPEC,
        V4_SPEC,
        TRAIN_DIR / "corpus.jsonl",
        TRAIN_DIR / "certificates.jsonl",
        TRAIN_DIR / "corpus_metadata.json",
        VAL_DIR / "corpus.jsonl",
        VAL_DIR / "certificates.jsonl",
        VAL_DIR / "corpus_metadata.json",
        TEST_DIR / "corpus.jsonl",
        TEST_DIR / "certificates.jsonl",
        TEST_DIR / "candidate_audit.jsonl",
        TEST_DIR / "corpus_metadata.json",
        ROOT / example["relative_path"],
        ROOT / "src" / "routing" / "pyvrp_generator.py",
        ROOT / "src" / "routing" / "synthcharge_benchmark.py",
        ROOT / "src" / "routing" / "synthcharge_instance.py",
        ROOT / "src" / "exact" / "charging_certificate.py",
        ROOT / "scripts" / "v4" / "test" / "generate_v4_test.py",
    ]
    provenance = {
        "git_commit": git_head(),
        "selected_validation_route": {
            "route_id": example["route_id"],
            "generator_seed": example["generator_seed"],
            "raw_instance_id": example["raw_instance_id"],
            "n_customers": example["n_customers"],
            "layout": example["layout"],
            "length_bin": example["length_bin"],
            "charge_class": example["charge_class"],
            "selection_rule": "lowest generator_seed among charging-required VALIDATION routes with 6-10 customers and an available instance file plus certificate trace",
        },
        "quotas": quotas,
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in inputs},
        "not_opened": [
            "data/routes_v2/synthcharge_v3_test/",
            "V2 confirmatory test seed range starting at 300000",
        ],
        "inconsistencies": issues,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_report(example, caption, summary, characteristics, issues, provenance)
    print(json.dumps({
        "route": example["route_id"],
        "seed": example["generator_seed"],
        "audit_instances": summary["n_candidate_instances"],
        "accepted": summary["n_instances_one_accepted_route"],
        "issues": issues,
    }, indent=2))


if __name__ == "__main__":
    main()
