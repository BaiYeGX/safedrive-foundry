"""Minimal OpenDRIVE subset parser used by the classic map stack.

This parser intentionally covers only the structural fields required by G1-03:
roads, lanes, successors/predecessors, lane links, signals, speed limits and
stop lines. It does not implement a full OpenDRIVE authoring toolchain.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Mapping


class OpenDriveParseError(ValueError):
    """Raised when an OpenDRIVE document is incomplete or inconsistent."""


@dataclass(frozen=True)
class RoadLink:
    element_type: str
    element_id: str
    contact_point: str = "end"


@dataclass(frozen=True)
class LaneLink:
    from_id: int
    to_id: int


@dataclass(frozen=True)
class LaneRecord:
    lane_id: int
    lane_type: str
    successor: int | None = None
    predecessor: int | None = None
    width: float = 3.5
    speed_limit_mps: float | None = None


@dataclass(frozen=True)
class LaneSection:
    s: float
    left: tuple[LaneRecord, ...] = ()
    center: tuple[LaneRecord, ...] = ()
    right: tuple[LaneRecord, ...] = ()

    def all_lanes(self) -> tuple[LaneRecord, ...]:
        return self.left + self.center + self.right


@dataclass(frozen=True)
class SignalRecord:
    signal_id: str
    s: float
    name: str
    dynamic: bool
    orientation: str
    country: str = "OpenDRIVE"
    type_code: str = "1000001"
    value: float | None = None
    unit: str = ""
    height: float = 0.0
    width: float = 0.0


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    s: float
    name: str
    object_type: str
    orientation: str = "+"
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass(frozen=True)
class RoadRecord:
    road_id: str
    name: str
    length: float
    junction: str
    predecessor: RoadLink | None
    successor: RoadLink | None
    lane_sections: tuple[LaneSection, ...]
    signals: tuple[SignalRecord, ...] = ()
    objects: tuple[ObjectRecord, ...] = ()
    speed_limit_mps: float | None = None


@dataclass(frozen=True)
class ConnectionLaneLink:
    from_lane: int
    to_lane: int


@dataclass(frozen=True)
class JunctionConnection:
    connection_id: str
    incoming_road: str
    connecting_road: str
    contact_point: str
    lane_links: tuple[ConnectionLaneLink, ...] = ()


@dataclass(frozen=True)
class JunctionRecord:
    junction_id: str
    name: str
    connections: tuple[JunctionConnection, ...] = ()


@dataclass(frozen=True)
class OpenDriveDocument:
    name: str
    rev_major: int
    rev_minor: int
    roads: tuple[RoadRecord, ...]
    junctions: tuple[JunctionRecord, ...]
    source_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def road_by_id(self, road_id: str) -> RoadRecord:
        for road in self.roads:
            if road.road_id == road_id:
                return road
        raise KeyError(road_id)

    def junction_by_id(self, junction_id: str) -> JunctionRecord:
        for junction in self.junctions:
            if junction.junction_id == junction_id:
                return junction
        raise KeyError(junction_id)


def _attr(node: ET.Element, name: str, default: str | None = None) -> str:
    value = node.attrib.get(name, default)
    if value is None:
        raise OpenDriveParseError(f"missing attribute '{name}' on <{node.tag}>")
    return value


def _float_attr(node: ET.Element, name: str, default: float | None = None) -> float:
    raw = node.attrib.get(name)
    if raw is None:
        if default is None:
            raise OpenDriveParseError(f"missing float attribute '{name}' on <{node.tag}>")
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise OpenDriveParseError(f"invalid float for '{name}' on <{node.tag}>: {raw}") from exc


def _int_attr(node: ET.Element, name: str, default: int | None = None) -> int:
    raw = node.attrib.get(name)
    if raw is None:
        if default is None:
            raise OpenDriveParseError(f"missing int attribute '{name}' on <{node.tag}>")
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise OpenDriveParseError(f"invalid int for '{name}' on <{node.tag}>: {raw}") from exc


def _bool_attr(node: ET.Element, name: str, default: bool = False) -> bool:
    raw = node.attrib.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _parse_road_link(parent: ET.Element | None, tag: str) -> RoadLink | None:
    if parent is None:
        return None
    node = parent.find(tag)
    if node is None:
        return None
    return RoadLink(
        element_type=_attr(node, "elementType"),
        element_id=_attr(node, "elementId"),
        contact_point=_attr(node, "contactPoint", "end"),
    )


def _parse_lane(node: ET.Element) -> LaneRecord:
    successor_node = node.find("./link/successor")
    predecessor_node = node.find("./link/predecessor")
    speed_node = node.find("speed")
    width_node = node.find("width")
    return LaneRecord(
        lane_id=_int_attr(node, "id"),
        lane_type=_attr(node, "type", "driving"),
        successor=_int_attr(successor_node, "id") if successor_node is not None else None,
        predecessor=_int_attr(predecessor_node, "id") if predecessor_node is not None else None,
        width=_float_attr(width_node, "a", 3.5) if width_node is not None else 3.5,
        speed_limit_mps=_float_attr(speed_node, "max") if speed_node is not None else None,
    )


def _parse_lane_section(node: ET.Element) -> LaneSection:
    left = tuple(_parse_lane(lane) for lane in node.findall("./left/lane"))
    center = tuple(_parse_lane(lane) for lane in node.findall("./center/lane"))
    right = tuple(_parse_lane(lane) for lane in node.findall("./right/lane"))
    return LaneSection(s=_float_attr(node, "s", 0.0), left=left, center=center, right=right)


def _parse_signal(node: ET.Element) -> SignalRecord:
    value_raw = node.attrib.get("value")
    value = float(value_raw) if value_raw not in (None, "") else None
    return SignalRecord(
        signal_id=_attr(node, "id"),
        s=_float_attr(node, "s", 0.0),
        name=_attr(node, "name", ""),
        dynamic=_bool_attr(node, "dynamic", False),
        orientation=_attr(node, "orientation", "+"),
        country=_attr(node, "country", "OpenDRIVE"),
        type_code=_attr(node, "type", "1000001"),
        value=value,
        unit=_attr(node, "unit", ""),
        height=_float_attr(node, "height", 0.0),
        width=_float_attr(node, "width", 0.0),
    )


def _parse_object(node: ET.Element) -> ObjectRecord:
    return ObjectRecord(
        object_id=_attr(node, "id"),
        s=_float_attr(node, "s", 0.0),
        name=_attr(node, "name", ""),
        object_type=_attr(node, "type", "obstacle"),
        orientation=_attr(node, "orientation", "+"),
        length=_float_attr(node, "length", 0.0),
        width=_float_attr(node, "width", 0.0),
        height=_float_attr(node, "height", 0.0),
    )


def _parse_road(node: ET.Element) -> RoadRecord:
    link = node.find("link")
    type_node = node.find("type")
    speed_node = type_node.find("speed") if type_node is not None else None
    speed_limit = None
    if speed_node is not None:
        max_speed = _float_attr(speed_node, "max")
        unit = _attr(speed_node, "unit", "m/s")
        if unit in {"km/h", "kph"}:
            speed_limit = max_speed / 3.6
        else:
            speed_limit = max_speed
    lane_sections = tuple(_parse_lane_section(section) for section in node.findall("./lanes/laneSection"))
    if not lane_sections:
        raise OpenDriveParseError(f"road {node.attrib.get('id')} has no laneSection")
    signals = tuple(_parse_signal(signal) for signal in node.findall("./signals/signal"))
    objects = tuple(_parse_object(obj) for obj in node.findall("./objects/object"))
    return RoadRecord(
        road_id=_attr(node, "id"),
        name=_attr(node, "name", ""),
        length=_float_attr(node, "length"),
        junction=_attr(node, "junction", "-1"),
        predecessor=_parse_road_link(link, "predecessor"),
        successor=_parse_road_link(link, "successor"),
        lane_sections=lane_sections,
        signals=signals,
        objects=objects,
        speed_limit_mps=speed_limit,
    )


def _parse_connection(node: ET.Element) -> JunctionConnection:
    lane_links = tuple(
        ConnectionLaneLink(from_lane=_int_attr(link, "from"), to_lane=_int_attr(link, "to"))
        for link in node.findall("laneLink")
    )
    return JunctionConnection(
        connection_id=_attr(node, "id"),
        incoming_road=_attr(node, "incomingRoad"),
        connecting_road=_attr(node, "connectingRoad"),
        contact_point=_attr(node, "contactPoint", "start"),
        lane_links=lane_links,
    )


def _parse_junction(node: ET.Element) -> JunctionRecord:
    connections = tuple(_parse_connection(conn) for conn in node.findall("connection"))
    return JunctionRecord(
        junction_id=_attr(node, "id"),
        name=_attr(node, "name", ""),
        connections=connections,
    )


def parse_opendrive_xml(text: str, *, source_path: str | None = None, name: str | None = None) -> OpenDriveDocument:
    """Parse an OpenDRIVE XML string into a typed document."""

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise OpenDriveParseError(f"invalid OpenDRIVE XML: {exc}") from exc
    if root.tag != "OpenDRIVE":
        raise OpenDriveParseError(f"root element must be OpenDRIVE, got {root.tag}")
    header = root.find("header")
    if header is None:
        raise OpenDriveParseError("OpenDRIVE header is required")
    roads = tuple(_parse_road(road) for road in root.findall("road"))
    junctions = tuple(_parse_junction(junction) for junction in root.findall("junction"))
    if not roads:
        raise OpenDriveParseError("OpenDRIVE document contains no roads")
    document_name = name or _attr(header, "name", source_path or "unnamed")
    return OpenDriveDocument(
        name=document_name,
        rev_major=_int_attr(header, "revMajor", 1),
        rev_minor=_int_attr(header, "revMinor", 4),
        roads=roads,
        junctions=junctions,
        source_path=source_path,
        metadata={
            "north": header.attrib.get("north"),
            "south": header.attrib.get("south"),
            "east": header.attrib.get("east"),
            "west": header.attrib.get("west"),
        },
    )


def parse_opendrive_file(path: str | bytes) -> OpenDriveDocument:
    from pathlib import Path

    file_path = Path(path)
    return parse_opendrive_xml(file_path.read_text(encoding="utf-8"), source_path=str(file_path), name=file_path.stem)
