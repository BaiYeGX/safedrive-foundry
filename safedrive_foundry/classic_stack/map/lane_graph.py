"""Lane graph construction, caching, queries and topology anomaly detection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .opendrive import OpenDriveDocument, OpenDriveParseError, parse_opendrive_file, parse_opendrive_xml


class MapError(ValueError):
    """Raised when map semantics or graph construction fails."""


@dataclass(frozen=True)
class TopologyAnomaly:
    code: str
    severity: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LaneNode:
    """A directed lane segment used as a graph node."""

    node_id: str
    road_id: str
    section_index: int
    lane_id: int
    lane_type: str
    length_m: float
    width_m: float
    speed_limit_mps: float
    junction_id: str | None
    s_start: float
    s_end: float
    driving: bool
    centerline: tuple[tuple[float, float], ...] = ()
    stop_line_s: float | None = None
    signal_ids: tuple[str, ...] = ()
    oracle_fields: tuple[str, ...] = ("centerline", "signal_ids", "speed_limit_mps", "stop_line_s")
    observable_fields: tuple[str, ...] = ("node_id", "road_id", "lane_id", "length_m", "speed_limit_mps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LaneEdge:
    """Directed connectivity between lane nodes."""

    edge_id: str
    source: str
    target: str
    kind: str  # successor | predecessor | lane_change_left | lane_change_right | junction
    cost: float
    road_id: str | None = None
    junction_id: str | None = None
    oracle_fields: tuple[str, ...] = ("cost", "kind", "junction_id")
    observable_fields: tuple[str, ...] = ("edge_id", "source", "target", "kind")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalFeature:
    signal_id: str
    road_id: str
    s: float
    name: str
    dynamic: bool
    type_code: str
    value: float | None
    orientation: str
    oracle_fields: tuple[str, ...] = ("signal_id", "road_id", "s", "dynamic", "type_code", "value")
    observable_fields: tuple[str, ...] = ("signal_id", "road_id", "name", "dynamic")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StopLineFeature:
    stop_line_id: str
    road_id: str
    s: float
    name: str
    oracle_fields: tuple[str, ...] = ("stop_line_id", "road_id", "s")
    observable_fields: tuple[str, ...] = ("stop_line_id", "road_id", "name")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MapSemantics:
    map_name: str
    map_hash: str
    roads: tuple[str, ...]
    junctions: tuple[str, ...]
    signals: tuple[SignalFeature, ...]
    stop_lines: tuple[StopLineFeature, ...]
    speed_limits_mps: Mapping[str, float]
    oracle_inputs: Mapping[str, Sequence[str]]
    observable_inputs: Mapping[str, Sequence[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "map_hash": self.map_hash,
            "roads": list(self.roads),
            "junctions": list(self.junctions),
            "signals": [item.to_dict() for item in self.signals],
            "stop_lines": [item.to_dict() for item in self.stop_lines],
            "speed_limits_mps": dict(self.speed_limits_mps),
            "oracle_inputs": {key: list(value) for key, value in self.oracle_inputs.items()},
            "observable_inputs": {key: list(value) for key, value in self.observable_inputs.items()},
        }


@dataclass
class LaneGraph:
    map_name: str
    map_hash: str
    nodes: dict[str, LaneNode]
    edges: dict[str, LaneEdge]
    adjacency: dict[str, list[str]]
    reverse_adjacency: dict[str, list[str]]
    semantics: MapSemantics
    anomalies: list[TopologyAnomaly] = field(default_factory=list)
    source_kind: str = "opendrive"
    cache_path: str | None = None

    def successors(self, node_id: str) -> list[LaneEdge]:
        return [self.edges[edge_id] for edge_id in self.adjacency.get(node_id, [])]

    def predecessors(self, node_id: str) -> list[LaneEdge]:
        return [self.edges[edge_id] for edge_id in self.reverse_adjacency.get(node_id, [])]

    def driving_nodes(self) -> list[LaneNode]:
        return [node for node in self.nodes.values() if node.driving]

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "map_hash": self.map_hash,
            "source_kind": self.source_kind,
            "nodes": {key: value.to_dict() for key, value in self.nodes.items()},
            "edges": {key: value.to_dict() for key, value in self.edges.items()},
            "adjacency": {key: list(value) for key, value in self.adjacency.items()},
            "reverse_adjacency": {key: list(value) for key, value in self.reverse_adjacency.items()},
            "semantics": self.semantics.to_dict(),
            "anomalies": [item.to_dict() for item in self.anomalies],
            "cache_path": self.cache_path,
        }

    def save_cache(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.cache_path = str(target)
        return target


def map_hash(payload: Mapping[str, Any] | str | bytes) -> str:
    if isinstance(payload, (str, bytes)):
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    else:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _node_id(road_id: str, section_index: int, lane_id: int) -> str:
    return f"R{road_id}:S{section_index}:L{lane_id}"


def _edge_id(source: str, target: str, kind: str) -> str:
    return f"{source}->{target}:{kind}"


def _synthetic_centerline(road_id: str, section_index: int, lane_id: int, length_m: float) -> tuple[tuple[float, float], ...]:
    # Deterministic synthetic geometry for offline fixtures; real CARLA maps can replace this later.
    base_x = float(int(road_id) if str(road_id).isdigit() else sum(ord(ch) for ch in str(road_id)) % 1000)
    base_y = float(section_index * 20 + lane_id * 4)
    steps = max(2, int(math.ceil(length_m / 5.0)) + 1)
    points: list[tuple[float, float]] = []
    for index in range(steps):
        s = length_m * index / (steps - 1)
        points.append((base_x + s, base_y))
    return tuple(points)


def _lane_speed(document: OpenDriveDocument, road_id: str, lane_speed: float | None) -> float:
    if lane_speed is not None:
        return float(lane_speed)
    road = document.road_by_id(road_id)
    if road.speed_limit_mps is not None:
        return float(road.speed_limit_mps)
    return 13.89  # ~50 km/h default


class LaneGraphBuilder:
    """Build a directed lane graph from OpenDRIVE or an equivalent dict fixture."""

    DEFAULT_LANE_CHANGE_COST = 4.0
    DEFAULT_JUNCTION_COST = 2.0

    def __init__(self, *, lane_change_enabled: bool = True) -> None:
        self.lane_change_enabled = lane_change_enabled

    def build_from_document(self, document: OpenDriveDocument) -> LaneGraph:
        nodes: dict[str, LaneNode] = {}
        edges: dict[str, LaneEdge] = {}
        adjacency: dict[str, list[str]] = {}
        reverse_adjacency: dict[str, list[str]] = {}
        anomalies: list[TopologyAnomaly] = []
        signals: list[SignalFeature] = []
        stop_lines: list[StopLineFeature] = []
        speed_limits: dict[str, float] = {}

        # First pass: create nodes.
        for road in document.roads:
            section_count = len(road.lane_sections)
            for section_index, section in enumerate(road.lane_sections):
                next_s = (
                    road.lane_sections[section_index + 1].s
                    if section_index + 1 < section_count
                    else road.length
                )
                length_m = max(0.1, next_s - section.s)
                junction_id = None if road.junction in {"", "-1"} else road.junction
                road_signals = [signal for signal in road.signals if section.s <= signal.s <= next_s]
                road_stop_objects = [
                    obj
                    for obj in road.objects
                    if obj.object_type.lower() in {"stopline", "stop_line", "stop"}
                    and section.s <= obj.s <= next_s
                ]
                for lane in section.all_lanes():
                    node_key = _node_id(road.road_id, section_index, lane.lane_id)
                    speed = _lane_speed(document, road.road_id, lane.speed_limit_mps)
                    speed_limits[node_key] = speed
                    driving = lane.lane_type.lower() in {"driving", "exit", "entry", "onramp", "offramp", "connectingramp"}
                    signal_ids = tuple(signal.signal_id for signal in road_signals)
                    stop_line_s = road_stop_objects[0].s if road_stop_objects else None
                    if any(signal.type_code in {"1000001", "traffic_light"} for signal in road_signals) and stop_line_s is None:
                        # OpenDRIVE often omits explicit stop-line objects; attach signal s as stop line.
                        stop_line_s = road_signals[0].s
                    nodes[node_key] = LaneNode(
                        node_id=node_key,
                        road_id=road.road_id,
                        section_index=section_index,
                        lane_id=lane.lane_id,
                        lane_type=lane.lane_type,
                        length_m=length_m,
                        width_m=lane.width,
                        speed_limit_mps=speed,
                        junction_id=junction_id,
                        s_start=section.s,
                        s_end=next_s,
                        driving=driving and lane.lane_id != 0,
                        centerline=_synthetic_centerline(road.road_id, section_index, lane.lane_id, length_m),
                        stop_line_s=stop_line_s,
                        signal_ids=signal_ids,
                    )

                for signal in road_signals:
                    signals.append(
                        SignalFeature(
                            signal_id=signal.signal_id,
                            road_id=road.road_id,
                            s=signal.s,
                            name=signal.name,
                            dynamic=signal.dynamic,
                            type_code=signal.type_code,
                            value=signal.value,
                            orientation=signal.orientation,
                        )
                    )
                for obj in road_stop_objects:
                    stop_lines.append(
                        StopLineFeature(
                            stop_line_id=obj.object_id,
                            road_id=road.road_id,
                            s=obj.s,
                            name=obj.name or obj.object_id,
                        )
                    )

        def add_edge(source: str, target: str, kind: str, cost: float, *, road_id: str | None = None, junction_id: str | None = None) -> None:
            if source not in nodes or target not in nodes:
                anomalies.append(
                    TopologyAnomaly(
                        code="MISSING_ENDPOINT",
                        severity="error",
                        message=f"edge {kind} references missing node {source}->{target}",
                        node_id=source if source not in nodes else target,
                    )
                )
                return
            edge_key = _edge_id(source, target, kind)
            if edge_key in edges:
                return
            edges[edge_key] = LaneEdge(
                edge_id=edge_key,
                source=source,
                target=target,
                kind=kind,
                cost=cost,
                road_id=road_id,
                junction_id=junction_id,
            )
            adjacency.setdefault(source, []).append(edge_key)
            reverse_adjacency.setdefault(target, []).append(edge_key)

        # Second pass: successor / predecessor / lane-change links.
        for road in document.roads:
            for section_index, section in enumerate(road.lane_sections):
                section_lanes = [lane for lane in section.all_lanes() if lane.lane_id != 0]
                lane_ids = sorted(lane.lane_id for lane in section_lanes)
                for lane in section_lanes:
                    source = _node_id(road.road_id, section_index, lane.lane_id)
                    source_node = nodes[source]
                    # same-road section succession via lane link ids
                    if lane.successor is not None and section_index + 1 < len(road.lane_sections):
                        target = _node_id(road.road_id, section_index + 1, lane.successor)
                        add_edge(source, target, "successor", source_node.length_m, road_id=road.road_id)
                    elif section_index + 1 < len(road.lane_sections):
                        # fallback same lane id
                        target = _node_id(road.road_id, section_index + 1, lane.lane_id)
                        if target in nodes:
                            add_edge(source, target, "successor", source_node.length_m, road_id=road.road_id)

                    if lane.predecessor is not None and section_index > 0:
                        target = _node_id(road.road_id, section_index - 1, lane.predecessor)
                        add_edge(target, source, "successor", nodes[target].length_m if target in nodes else source_node.length_m, road_id=road.road_id)

                    # road-level successor for last section
                    if section_index == len(road.lane_sections) - 1 and road.successor is not None:
                        if road.successor.element_type == "road":
                            try:
                                succ_road = document.road_by_id(road.successor.element_id)
                            except KeyError:
                                anomalies.append(
                                    TopologyAnomaly(
                                        code="MISSING_ENDPOINT",
                                        severity="error",
                                        message=(
                                            f"road {road.road_id} successor road "
                                            f"{road.successor.element_id} is missing"
                                        ),
                                        node_id=source,
                                    )
                                )
                            else:
                                succ_section = 0 if road.successor.contact_point == "start" else len(succ_road.lane_sections) - 1
                                # Prefer explicit lane successor id when present; else same lane id.
                                target_lane = lane.successor if lane.successor is not None else lane.lane_id
                                target = _node_id(succ_road.road_id, succ_section, target_lane)
                                add_edge(source, target, "successor", source_node.length_m, road_id=road.road_id)
                        elif road.successor.element_type == "junction":
                            # Junction expansion handled below.
                            pass

                    if self.lane_change_enabled and source_node.driving and source_node.junction_id is None:
                        # Adjacent lane change on same section: left is +1 for left-hand positive OpenDRIVE ids.
                        for neighbor in (lane.lane_id - 1, lane.lane_id + 1):
                            if neighbor == 0 or neighbor not in lane_ids:
                                continue
                            target = _node_id(road.road_id, section_index, neighbor)
                            if target not in nodes or not nodes[target].driving:
                                continue
                            kind = "lane_change_left" if neighbor > lane.lane_id else "lane_change_right"
                            # OpenDRIVE: positive lanes are left of center; negative are right.
                            if lane.lane_id > 0:
                                kind = "lane_change_left" if neighbor > lane.lane_id else "lane_change_right"
                            else:
                                kind = "lane_change_left" if neighbor < lane.lane_id else "lane_change_right"
                            add_edge(source, target, kind, self.DEFAULT_LANE_CHANGE_COST, road_id=road.road_id)

        # Junction connections.
        for junction in document.junctions:
            for connection in junction.connections:
                try:
                    incoming = document.road_by_id(connection.incoming_road)
                    connecting = document.road_by_id(connection.connecting_road)
                except KeyError as exc:
                    anomalies.append(
                        TopologyAnomaly(
                            code="JUNCTION_ROAD_MISSING",
                            severity="error",
                            message=f"junction {junction.junction_id} references missing road {exc}",
                            node_id=None,
                        )
                    )
                    continue
                in_section = len(incoming.lane_sections) - 1
                conn_section = 0 if connection.contact_point == "start" else len(connecting.lane_sections) - 1
                if connection.lane_links:
                    pairs = [(link.from_lane, link.to_lane) for link in connection.lane_links]
                else:
                    # fallback: connect matching lane ids that exist
                    in_lanes = {lane.lane_id for lane in incoming.lane_sections[in_section].all_lanes() if lane.lane_id != 0}
                    conn_lanes = {lane.lane_id for lane in connecting.lane_sections[conn_section].all_lanes() if lane.lane_id != 0}
                    pairs = [(lane_id, lane_id) for lane_id in sorted(in_lanes & conn_lanes)]
                for from_lane, to_lane in pairs:
                    source = _node_id(incoming.road_id, in_section, from_lane)
                    target = _node_id(connecting.road_id, conn_section, to_lane)
                    add_edge(
                        source,
                        target,
                        "junction",
                        self.DEFAULT_JUNCTION_COST + (nodes[source].length_m if source in nodes else 0.0),
                        road_id=connecting.road_id,
                        junction_id=junction.junction_id,
                    )

        # Topology anomaly scan.
        driving_nodes = [node for node in nodes.values() if node.driving]
        for node in driving_nodes:
            outs = adjacency.get(node.node_id, [])
            if not outs and node.junction_id is None:
                # dead end is allowed only if road has no successor; otherwise anomaly
                road = document.road_by_id(node.road_id)
                if road.successor is not None:
                    anomalies.append(
                        TopologyAnomaly(
                            code="DEAD_END_WITH_SUCCESSOR",
                            severity="warning",
                            message=f"driving node {node.node_id} has no outgoing edges but road successor exists",
                            node_id=node.node_id,
                        )
                    )
            if node.length_m <= 0:
                anomalies.append(
                    TopologyAnomaly(
                        code="NON_POSITIVE_LENGTH",
                        severity="error",
                        message=f"node {node.node_id} has non-positive length",
                        node_id=node.node_id,
                    )
                )
            if node.speed_limit_mps <= 0:
                anomalies.append(
                    TopologyAnomaly(
                        code="INVALID_SPEED_LIMIT",
                        severity="error",
                        message=f"node {node.node_id} has invalid speed limit",
                        node_id=node.node_id,
                    )
                )

        # orphan edge endpoints already captured; detect duplicate reverse-only islands
        reachable_roots = [node.node_id for node in driving_nodes]
        if reachable_roots:
            seen: set[str] = set()
            stack = [reachable_roots[0]]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                for edge_id in adjacency.get(current, []):
                    stack.append(edges[edge_id].target)
            unreachable = [node.node_id for node in driving_nodes if node.node_id not in seen and node.junction_id is None]
            for node_id in unreachable[:20]:
                # Limited report; multi-component maps are valid, so severity is info.
                anomalies.append(
                    TopologyAnomaly(
                        code="DISCONNECTED_COMPONENT",
                        severity="info",
                        message=f"node {node_id} not reachable from first driving node",
                        node_id=node_id,
                    )
                )

        payload = {
            "map_name": document.name,
            "roads": [road.road_id for road in document.roads],
            "junctions": [junction.junction_id for junction in document.junctions],
            "nodes": sorted(nodes),
            "edges": sorted(edges),
        }
        digest = map_hash(payload)
        semantics = MapSemantics(
            map_name=document.name,
            map_hash=digest,
            roads=tuple(road.road_id for road in document.roads),
            junctions=tuple(junction.junction_id for junction in document.junctions),
            signals=tuple(signals),
            stop_lines=tuple(stop_lines),
            speed_limits_mps=speed_limits,
            oracle_inputs={
                "topology": ["nodes", "edges", "junctions", "successors", "lane_changes"],
                "regulation": ["signals", "stop_lines", "speed_limits_mps"],
                "geometry": ["centerline", "width_m", "length_m"],
            },
            observable_inputs={
                "navigation": ["node_id", "road_id", "lane_id", "route_progress"],
                "regulation_visible": ["signal_ids", "stop_line_presence", "posted_speed"],
                "local_topology": ["successor_count", "lane_change_options"],
            },
        )
        # stable adjacency order
        for key in list(adjacency):
            adjacency[key] = sorted(adjacency[key])
        for key in list(reverse_adjacency):
            reverse_adjacency[key] = sorted(reverse_adjacency[key])
        return LaneGraph(
            map_name=document.name,
            map_hash=digest,
            nodes=nodes,
            edges=edges,
            adjacency=adjacency,
            reverse_adjacency=reverse_adjacency,
            semantics=semantics,
            anomalies=anomalies,
            source_kind="opendrive",
        )

    def build_from_xml(self, text: str, *, name: str | None = None, source_path: str | None = None) -> LaneGraph:
        document = parse_opendrive_xml(text, name=name, source_path=source_path)
        return self.build_from_document(document)

    def build_from_path(self, path: str | Path) -> LaneGraph:
        document = parse_opendrive_file(path)
        return self.build_from_document(document)


def build_lane_graph_from_dict(payload: Mapping[str, Any]) -> LaneGraph:
    """Build a graph from a compact JSON fixture (not full OpenDRIVE)."""

    map_name = str(payload.get("map_name") or payload.get("name") or "unnamed")
    nodes_payload = payload.get("nodes") or []
    edges_payload = payload.get("edges") or []
    nodes: dict[str, LaneNode] = {}
    edges: dict[str, LaneEdge] = {}
    adjacency: dict[str, list[str]] = {}
    reverse_adjacency: dict[str, list[str]] = {}
    anomalies: list[TopologyAnomaly] = []

    for item in nodes_payload:
        node = LaneNode(
            node_id=str(item["node_id"]),
            road_id=str(item["road_id"]),
            section_index=int(item.get("section_index", 0)),
            lane_id=int(item["lane_id"]),
            lane_type=str(item.get("lane_type", "driving")),
            length_m=float(item.get("length_m", 10.0)),
            width_m=float(item.get("width_m", 3.5)),
            speed_limit_mps=float(item.get("speed_limit_mps", 13.89)),
            junction_id=item.get("junction_id"),
            s_start=float(item.get("s_start", 0.0)),
            s_end=float(item.get("s_end", item.get("length_m", 10.0))),
            driving=bool(item.get("driving", True)),
            centerline=tuple(tuple(point) for point in item.get("centerline", [])),
            stop_line_s=item.get("stop_line_s"),
            signal_ids=tuple(item.get("signal_ids", ())),
        )
        nodes[node.node_id] = node

    for item in edges_payload:
        edge = LaneEdge(
            edge_id=str(item.get("edge_id") or _edge_id(item["source"], item["target"], item.get("kind", "successor"))),
            source=str(item["source"]),
            target=str(item["target"]),
            kind=str(item.get("kind", "successor")),
            cost=float(item.get("cost", 1.0)),
            road_id=item.get("road_id"),
            junction_id=item.get("junction_id"),
        )
        if edge.source not in nodes or edge.target not in nodes:
            anomalies.append(
                TopologyAnomaly(
                    code="MISSING_ENDPOINT",
                    severity="error",
                    message=f"edge {edge.edge_id} has missing endpoint",
                    edge_id=edge.edge_id,
                )
            )
            continue
        edges[edge.edge_id] = edge
        adjacency.setdefault(edge.source, []).append(edge.edge_id)
        reverse_adjacency.setdefault(edge.target, []).append(edge.edge_id)

    for key in list(adjacency):
        adjacency[key] = sorted(adjacency[key])
    for key in list(reverse_adjacency):
        reverse_adjacency[key] = sorted(reverse_adjacency[key])

    digest = map_hash(
        {
            "map_name": map_name,
            "nodes": sorted(nodes),
            "edges": sorted(edges),
            "payload_hash": payload.get("seed", map_name),
        }
    )
    signals = tuple(
        SignalFeature(
            signal_id=str(item["signal_id"]),
            road_id=str(item["road_id"]),
            s=float(item.get("s", 0.0)),
            name=str(item.get("name", "")),
            dynamic=bool(item.get("dynamic", True)),
            type_code=str(item.get("type_code", "1000001")),
            value=item.get("value"),
            orientation=str(item.get("orientation", "+")),
        )
        for item in payload.get("signals", [])
    )
    stop_lines = tuple(
        StopLineFeature(
            stop_line_id=str(item["stop_line_id"]),
            road_id=str(item["road_id"]),
            s=float(item.get("s", 0.0)),
            name=str(item.get("name", item["stop_line_id"])),
        )
        for item in payload.get("stop_lines", [])
    )
    semantics = MapSemantics(
        map_name=map_name,
        map_hash=digest,
        roads=tuple(sorted({node.road_id for node in nodes.values()})),
        junctions=tuple(sorted({node.junction_id for node in nodes.values() if node.junction_id})),
        signals=signals,
        stop_lines=stop_lines,
        speed_limits_mps={node_id: node.speed_limit_mps for node_id, node in nodes.items()},
        oracle_inputs={
            "topology": ["nodes", "edges", "junctions", "successors", "lane_changes"],
            "regulation": ["signals", "stop_lines", "speed_limits_mps"],
            "geometry": ["centerline", "width_m", "length_m"],
        },
        observable_inputs={
            "navigation": ["node_id", "road_id", "lane_id", "route_progress"],
            "regulation_visible": ["signal_ids", "stop_line_presence", "posted_speed"],
            "local_topology": ["successor_count", "lane_change_options"],
        },
    )
    return LaneGraph(
        map_name=map_name,
        map_hash=digest,
        nodes=nodes,
        edges=edges,
        adjacency=adjacency,
        reverse_adjacency=reverse_adjacency,
        semantics=semantics,
        anomalies=anomalies,
        source_kind=str(payload.get("source_kind", "fixture")),
    )


class LaneGraphQuery:
    def __init__(self, graph: LaneGraph) -> None:
        self.graph = graph

    def get_node(self, node_id: str) -> LaneNode:
        try:
            return self.graph.nodes[node_id]
        except KeyError as exc:
            raise MapError(f"unknown node_id {node_id}") from exc

    def lane_changes(self, node_id: str) -> list[LaneEdge]:
        return [edge for edge in self.graph.successors(node_id) if edge.kind.startswith("lane_change")]

    def successors(self, node_id: str, *, kinds: Iterable[str] | None = None) -> list[LaneEdge]:
        edges = self.graph.successors(node_id)
        if kinds is None:
            return edges
        allowed = set(kinds)
        return [edge for edge in edges if edge.kind in allowed]

    def nodes_for_road(self, road_id: str) -> list[LaneNode]:
        return [node for node in self.graph.nodes.values() if node.road_id == road_id]

    def anomalies(self, *, severity: str | None = None) -> list[TopologyAnomaly]:
        if severity is None:
            return list(self.graph.anomalies)
        return [item for item in self.graph.anomalies if item.severity == severity]

    def connected(self, source: str, target: str) -> bool:
        if source not in self.graph.nodes or target not in self.graph.nodes:
            return False
        seen: set[str] = set()
        stack = [source]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            for edge in self.graph.successors(current):
                stack.append(edge.target)
        return False


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_CARLA_FIXTURE_DIR = _FIXTURE_DIR / "carla"


def load_map_fixture(name: str, *, cache_dir: str | Path | None = None) -> LaneGraph:
    """Load a packaged synthetic fixture by map name (Town01/Town03/Town10HD style).

    Synthetic fixtures keep fixed node IDs for deterministic route/behavior unit
    tests. Real CARLA OpenDRIVE exports live under ``fixtures/carla/`` and are
    loaded via :func:`load_carla_map`.
    """

    xodr = _FIXTURE_DIR / f"{name}.xodr"
    json_path = _FIXTURE_DIR / f"{name}.json"
    if xodr.exists():
        graph = LaneGraphBuilder().build_from_path(xodr)
    elif json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        graph = build_lane_graph_from_dict(payload)
    else:
        raise FileNotFoundError(f"map fixture not found for {name}: expected {xodr.name} or {json_path.name}")

    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{graph.map_name}-{graph.map_hash[:12]}.json"
        graph.save_cache(cache_path)
    return graph


def load_carla_map(name: str, *, cache_dir: str | Path | None = None) -> LaneGraph:
    """Load a real CARLA 0.9.16 packaged OpenDRIVE map by town name.

    Files under ``fixtures/carla/`` are copied from
    ``CarlaUE4/Content/Carla/Maps/OpenDrive/<Town>.xodr`` (see manifest.json).
    This avoids ``client.load_world`` map switches that can fatal the server.
    """

    xodr = _CARLA_FIXTURE_DIR / f"{name}.xodr"
    if not xodr.exists():
        raise FileNotFoundError(
            f"CARLA OpenDRIVE fixture missing for {name}: expected {xodr}. "
            "Copy from CARLA install Content/Carla/Maps/OpenDrive/."
        )
    graph = LaneGraphBuilder().build_from_path(xodr)
    graph.source_kind = "carla_packaged_opendrive"
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{graph.map_name}-{graph.map_hash[:12]}.json"
        graph.save_cache(cache_path)
    return graph


def load_cached_graph(path: str | Path) -> LaneGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = build_lane_graph_from_dict(
        {
            "map_name": payload["map_name"],
            "source_kind": payload.get("source_kind", "cache"),
            "nodes": list(payload["nodes"].values()),
            "edges": list(payload["edges"].values()),
            "signals": payload.get("semantics", {}).get("signals", []),
            "stop_lines": payload.get("semantics", {}).get("stop_lines", []),
            "seed": payload.get("map_hash"),
        }
    )
    graph.map_hash = payload["map_hash"]
    graph.cache_path = str(path)
    graph.anomalies = [
        TopologyAnomaly(**item) for item in payload.get("anomalies", [])
    ]
    return graph

