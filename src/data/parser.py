"""The canonical EVRPTW-GR instance parser.

This is the only place in the project that reads the raw instance format. It
opens files read-only and never writes to ``data/raw``.

Layout of every raw file:

    StringID<TAB>Type<TAB>x<TAB>y<TAB>demand<TAB>ReadyTime<TAB>DueDate<TAB>ServiceTime<TAB>altitude
    <one tab-separated row per node>
    <blank line>
    Q Vehicle fuel tank capacity /77.75/
    C Vehicle load capacity /200.0/
    r fuel consumption rate /1.0/
    g inverse refueling rate /3.47/
    v average Velocity /1.0/
    M Vehicle curb weight /6350.0/        (Medium_Network and Large_Network only)
"""

import re
from pathlib import Path

from .errors import MalformedInstanceError, MissingFieldError
from .models import (
    COLUMN_NAMES,
    EVRPTWGRInstance,
    InstanceMetadata,
    NetworkGroup,
    Node,
    NodeType,
    TerrainVariant,
    VehicleParameter,
    VehicleParameters,
)
from .paths import iter_instance_files, resolve_dataset_root

#: Footer keys that must be present in every file. ``M`` is deliberately absent
#: from this set: the Small_Network subset omits it by design.
REQUIRED_FOOTER_KEYS = ("Q", "C", "r", "g", "v")
OPTIONAL_FOOTER_KEYS = ("M",)

_FOOTER_RE = re.compile(r"^(?P<key>\S+)\s+(?P<label>.*?)\s*/(?P<value>[^/]+)/\s*$")

# "c101C25_NL.txt" and the like, used by Small_Network and Medium_Network.
_NAME_COMPACT_RE = re.compile(
    r"^(?P<base>(?P<dist>rc|c|r)(?P<schedule>[12])\d*)C(?P<customers>\d+)_(?P<variant>L|NL|VG)$",
    re.IGNORECASE,
)
# "c101_50_21_NL.txt", used by Large_Network.
_NAME_EXPANDED_RE = re.compile(
    r"^(?P<base>(?P<dist>rc|c|r)(?P<schedule>[12])\d*)_(?P<customers>\d+)_(?P<stations>\d+)_"
    r"(?P<variant>L|NL|VG)$",
    re.IGNORECASE,
)

_TERRAIN_BY_FOLDER = {
    "level": TerrainVariant.LEVEL,
    "nearly_level": TerrainVariant.NEARLY_LEVEL,
    "very_gentle": TerrainVariant.VERY_GENTLE,
}


def _to_float(raw, path, line_number, column):
    try:
        return float(raw)
    except ValueError:
        raise MalformedInstanceError(
            path, f"column {column!r} is not numeric: {raw!r}", line_number
        ) from None


def _parse_filename(path):
    """Recover instance identity from the filename.

    Handles both naming conventions in the dataset. Returns the base Schneider
    identifier, customer distribution, schedule type, declared customer count,
    declared station count (only the expanded form carries one), and variant.
    """
    stem = path.stem
    match = _NAME_COMPACT_RE.match(stem) or _NAME_EXPANDED_RE.match(stem)
    if match is None:
        raise MalformedInstanceError(path, f"filename does not follow any known convention: {stem!r}")

    groups = match.groupdict()
    stations = groups.get("stations")
    return {
        "base_instance": groups["base"].lower(),
        "customer_distribution": groups["dist"].lower(),
        "schedule_type": int(groups["schedule"]),
        "declared_customer_count": int(groups["customers"]),
        "declared_station_count": int(stations) if stations is not None else None,
        "terrain_variant": TerrainVariant(groups["variant"].upper()),
    }


def _parse_metadata(path, root):
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 2:
        raise MalformedInstanceError(
            path, f"expected <Network>/<Customers>/<Terrain>/<file>, got {relative.as_posix()!r}"
        )

    try:
        network_group = NetworkGroup(parts[0])
    except ValueError:
        raise MalformedInstanceError(path, f"unknown network group directory {parts[0]!r}") from None

    terrain_folder = parts[-2]
    customer_folder = parts[-3] if len(parts) >= 3 else ""

    # Folder casing is inconsistent in the raw dataset, so compare case-folded.
    folder_variant = _TERRAIN_BY_FOLDER.get(terrain_folder.casefold())
    if folder_variant is None:
        raise MalformedInstanceError(path, f"unknown terrain directory {terrain_folder!r}")

    name_fields = _parse_filename(path)
    if folder_variant is not name_fields["terrain_variant"]:
        raise MalformedInstanceError(
            path,
            f"terrain directory {terrain_folder!r} disagrees with filename variant "
            f"{name_fields['terrain_variant'].value!r}",
        )

    return InstanceMetadata(
        instance_id=path.stem,
        path=path,
        relative_path=relative.as_posix(),
        network_group=network_group,
        customer_folder=customer_folder,
        terrain_folder=terrain_folder,
        **name_fields,
    )


def _parse_header(line, path):
    columns = tuple(part for part in line.split("\t") if part.strip())
    normalised = tuple(part.strip() for part in columns)
    if normalised != COLUMN_NAMES:
        raise MalformedInstanceError(
            path, f"unexpected header columns {normalised!r}; expected {COLUMN_NAMES!r}", 1
        )
    return normalised


def _parse_node(line, path, line_number):
    fields = [part.strip() for part in line.split("\t") if part.strip()]
    if len(fields) != len(COLUMN_NAMES):
        raise MalformedInstanceError(
            path, f"expected {len(COLUMN_NAMES)} columns, found {len(fields)}: {line!r}", line_number
        )

    string_id, type_code = fields[0], fields[1]
    try:
        node_type = NodeType.from_code(type_code)
    except ValueError as exc:
        raise MalformedInstanceError(path, str(exc), line_number) from None

    numeric = [
        _to_float(fields[index], path, line_number, COLUMN_NAMES[index]) for index in range(2, 9)
    ]
    return Node(
        string_id=string_id,
        node_type=node_type,
        x=numeric[0],
        y=numeric[1],
        demand=numeric[2],
        ready_time=numeric[3],
        due_date=numeric[4],
        service_time=numeric[5],
        altitude=numeric[6],
        line_number=line_number,
    )


def _parse_footer(entries, path):
    by_key = {}
    for parameter in entries:
        if parameter.key in by_key:
            raise MalformedInstanceError(
                path, f"footer key {parameter.key!r} appears more than once"
            )
        by_key[parameter.key] = parameter

    for key in REQUIRED_FOOTER_KEYS:
        if key not in by_key:
            raise MissingFieldError(path, key, f"footer is missing required key {key!r}")

    unexpected = set(by_key) - set(REQUIRED_FOOTER_KEYS) - set(OPTIONAL_FOOTER_KEYS)
    if unexpected:
        raise MalformedInstanceError(path, f"unexpected footer keys {sorted(unexpected)!r}")

    curb_weight = by_key["M"].value if "M" in by_key else None
    return VehicleParameters(
        tank_capacity=by_key["Q"].value,
        load_capacity=by_key["C"].value,
        consumption_rate=by_key["r"].value,
        inverse_refueling_rate=by_key["g"].value,
        average_velocity=by_key["v"].value,
        curb_weight=curb_weight,
        raw=tuple(entries),
    )


def parse_instance_text(text, path, root=None):
    """Parse raw instance text. ``path`` is used for metadata and messages."""
    path = Path(path)
    root = resolve_dataset_root(root) if root is not None else None

    # splitlines() handles LF and CRLF alike, and files without a trailing
    # newline parse the same as those with one.
    lines = text.splitlines()
    if not lines:
        raise MalformedInstanceError(path, "file is empty")

    _parse_header(lines[0], path)

    nodes = []
    footer = []
    seen_footer = False
    for offset, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue

        footer_match = _FOOTER_RE.match(line.strip())
        if footer_match is not None:
            seen_footer = True
            value = _to_float(
                footer_match.group("value"), path, offset, footer_match.group("key")
            )
            footer.append(
                VehicleParameter(
                    key=footer_match.group("key"),
                    label=footer_match.group("label").strip(),
                    value=value,
                )
            )
            continue

        if seen_footer:
            raise MalformedInstanceError(
                path, f"node row appears after the footer block: {line!r}", offset
            )
        nodes.append(_parse_node(line, path, offset))

    if not nodes:
        raise MalformedInstanceError(path, "file contains no node rows")

    vehicle = _parse_footer(footer, path)
    # Without an explicit root, infer it from the required
    # <Network>/<Customers>/<Terrain>/<file> nesting.
    if root is None:
        if len(path.parents) < 4:
            raise MalformedInstanceError(
                path, "cannot infer dataset root; pass root= explicitly"
            )
        root = path.parents[3]
    metadata = _parse_metadata(path, root)
    return EVRPTWGRInstance(metadata=metadata, nodes=tuple(nodes), vehicle=vehicle)


def parse_instance(path, root=None):
    """Parse a single instance file. The file is only ever read."""
    path = Path(path).resolve()
    if not path.is_file():
        raise MalformedInstanceError(path, "instance file does not exist")
    text = path.read_text(encoding="utf-8")
    return parse_instance_text(text, path, root=root)


def load_dataset(root=None):
    """Parse every instance under ``root``, in stable order.

    Propagates the first failure. Use :func:`data.validation.scan_dataset` to
    collect problems across the whole dataset instead of stopping at the first.
    """
    resolved = resolve_dataset_root(root)
    return [parse_instance(path, root=resolved) for path in iter_instance_files(resolved)]
