"""Program-owned canonical evidence registry for the diagnostic formatter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from types import MappingProxyType
from typing import Any, Mapping

from langchain_core.messages import ToolMessage

from app.schemas.diagnostics import EvidenceItem


def _fail(message: str) -> None:
    raise RuntimeError(f"malformed successful tool payload: {message}")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{context} must be an object")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{context} must be a non-empty string")
    return value.strip()


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context} must be numeric")
    return float(value)


def _observed_at(value: Any, context: str) -> datetime:
    timestamp = _text(value, context)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"malformed successful tool payload: {context} must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{context} must include timezone information")
    return parsed


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    return value


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """Immutable binding between an evidence ID and program-verified facts."""

    evidence: EvidenceItem
    device_id: str
    tool_name: str
    facts: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvidenceRegistry:
    """Immutable canonical records that the formatter may only reference by ID."""

    entries: Mapping[str, RegistryEntry]
    unresolved_tool_errors: frozenset[str]

    def formatter_payload(self) -> dict[str, Any]:
        """Return deterministic, still-untrusted evidence context for the model."""

        return {
            "canonical_evidence": [
                {
                    "evidence_id": entry.evidence.evidence_id,
                    "evidence_type": entry.evidence.evidence_type,
                    "summary": entry.evidence.summary,
                    "observed_at": _json_safe(entry.evidence.observed_at),
                    "version": entry.evidence.version,
                    "facts": _json_safe(entry.facts),
                }
                for _, entry in sorted(self.entries.items())
            ]
        }

    def select(self, evidence_ids: list[str]) -> list[RegistryEntry]:
        unknown = set(evidence_ids) - set(self.entries)
        if unknown:
            raise RuntimeError(
                "structured diagnostic response referenced evidence IDs absent from the registry: "
                + ", ".join(sorted(unknown))
            )
        return [self.entries[evidence_id] for evidence_id in evidence_ids]


def _entry(
    *,
    evidence_id: str,
    evidence_type: str,
    source_id: str,
    summary: str,
    device_id: str,
    tool_name: str,
    facts: dict[str, Any],
    observed_at: datetime | None = None,
    version: str | None = None,
) -> RegistryEntry:
    return RegistryEntry(
        evidence=EvidenceItem(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            source_id=source_id,
            summary=summary,
            observed_at=observed_at,
            version=version,
        ),
        device_id=device_id,
        tool_name=tool_name,
        facts=MappingProxyType(dict(facts)),
    )


def _require_device_id(record: dict[str, Any], requested_device_id: str, context: str) -> str:
    device_id = _text(record.get("device_id"), f"{context}.device_id")
    if device_id != requested_device_id:
        raise RuntimeError(
            f"successful tool evidence device_id {device_id!r} does not match requested device {requested_device_id!r}"
        )
    return device_id


def _device_entries(payload: dict[str, Any], tool_name: str, requested_device_id: str) -> list[RegistryEntry]:
    source_id = _text(payload.get("source_id"), "get_device_info.source_id")
    device = _mapping(payload.get("device"), "get_device_info.device")
    device_id = _require_device_id(device, requested_device_id, "get_device_info.device")
    name = _text(device.get("name"), "get_device_info.device.name")
    status = _text(device.get("status"), "get_device_info.device.status")
    threshold = _number(
        device.get("vibration_alarm_threshold_mm_s"),
        "get_device_info.device.vibration_alarm_threshold_mm_s",
    )
    version = _text(payload.get("version"), "get_device_info.version")
    facts = {
        "device_id": device_id,
        "name": name,
        "status": status,
        "vibration_alarm_threshold_mm_s": threshold,
        "device_type": _text(device.get("device_type"), "get_device_info.device.device_type"),
        "location": _text(device.get("location"), "get_device_info.device.location"),
    }
    return [
        _entry(
            evidence_id=source_id,
            evidence_type="device",
            source_id=source_id,
            summary=(
                f"设备 {name}（{device_id}）状态为 {status}；振动报警阈值为 {threshold:g} mm/s。"
            ),
            device_id=device_id,
            tool_name=tool_name,
            facts=facts,
            version=version,
        )
    ]


def _sensor_entries(payload: dict[str, Any], tool_name: str, requested_device_id: str) -> list[RegistryEntry]:
    source_id = _text(payload.get("source_id"), "query_sensor_history.source_id")
    points = payload.get("points")
    if not isinstance(points, list):
        _fail("query_sensor_history.points must be a list")
    entries: list[RegistryEntry] = []
    for index, raw_point in enumerate(points):
        point = _mapping(raw_point, f"query_sensor_history.points[{index}]")
        device_id = _require_device_id(point, requested_device_id, f"query_sensor_history.points[{index}]")
        event_id = _text(point.get("event_id"), f"query_sensor_history.points[{index}].event_id")
        metric = _text(point.get("metric"), f"query_sensor_history.points[{index}].metric")
        value = _number(point.get("value"), f"query_sensor_history.points[{index}].value")
        unit = _text(point.get("unit"), f"query_sensor_history.points[{index}].unit")
        observed_at = _observed_at(point.get("observed_at"), f"query_sensor_history.points[{index}].observed_at")
        facts = {"device_id": device_id, "metric": metric, "value": value, "unit": unit, "observed_at": observed_at}
        entries.append(
            _entry(
                evidence_id=event_id,
                evidence_type="sensor",
                source_id=source_id,
                summary=f"测点 {metric}={value:g} {unit}，观测时间 {observed_at.isoformat()}。",
                device_id=device_id,
                tool_name=tool_name,
                facts=facts,
                observed_at=observed_at,
            )
        )
    return entries


def _alarm_entries(payload: dict[str, Any], tool_name: str, requested_device_id: str) -> list[RegistryEntry]:
    source_id = _text(payload.get("source_id"), "query_alarm_history.source_id")
    alarms = payload.get("alarms")
    if not isinstance(alarms, list):
        _fail("query_alarm_history.alarms must be a list")
    entries: list[RegistryEntry] = []
    for index, raw_alarm in enumerate(alarms):
        alarm = _mapping(raw_alarm, f"query_alarm_history.alarms[{index}]")
        device_id = _require_device_id(alarm, requested_device_id, f"query_alarm_history.alarms[{index}]")
        alarm_id = _text(alarm.get("alarm_id"), f"query_alarm_history.alarms[{index}].alarm_id")
        code = _text(alarm.get("code"), f"query_alarm_history.alarms[{index}].code")
        severity = _text(alarm.get("severity"), f"query_alarm_history.alarms[{index}].severity")
        message = _text(alarm.get("message"), f"query_alarm_history.alarms[{index}].message")
        observed_at = _observed_at(alarm.get("observed_at"), f"query_alarm_history.alarms[{index}].observed_at")
        facts = {"device_id": device_id, "code": code, "severity": severity, "message": message, "observed_at": observed_at}
        entries.append(
            _entry(
                evidence_id=alarm_id,
                evidence_type="alarm",
                source_id=source_id,
                summary=f"报警 {code}（{severity}）：{message} 观测时间 {observed_at.isoformat()}。",
                device_id=device_id,
                tool_name=tool_name,
                facts=facts,
                observed_at=observed_at,
            )
        )
    return entries


def _work_order_entries(payload: dict[str, Any], tool_name: str, requested_device_id: str) -> list[RegistryEntry]:
    source_id = _text(payload.get("source_id"), "query_work_order_history.source_id")
    work_orders = payload.get("work_orders")
    if not isinstance(work_orders, list):
        _fail("query_work_order_history.work_orders must be a list")
    entries: list[RegistryEntry] = []
    for index, raw_order in enumerate(work_orders):
        order = _mapping(raw_order, f"query_work_order_history.work_orders[{index}]")
        device_id = _require_device_id(order, requested_device_id, f"query_work_order_history.work_orders[{index}]")
        work_order_id = _text(order.get("work_order_id"), f"query_work_order_history.work_orders[{index}].work_order_id")
        status = _text(order.get("status"), f"query_work_order_history.work_orders[{index}].status")
        summary = _text(order.get("summary"), f"query_work_order_history.work_orders[{index}].summary")
        observed_at = _observed_at(order.get("observed_at"), f"query_work_order_history.work_orders[{index}].observed_at")
        facts = {"device_id": device_id, "status": status, "summary": summary, "observed_at": observed_at}
        entries.append(
            _entry(
                evidence_id=work_order_id,
                evidence_type="work_order",
                source_id=source_id,
                summary=f"工单 {work_order_id} 状态为 {status}：{summary} 时间 {observed_at.isoformat()}。",
                device_id=device_id,
                tool_name=tool_name,
                facts=facts,
                observed_at=observed_at,
            )
        )
    return entries


def _manual_entries(payload: dict[str, Any], tool_name: str, requested_device_id: str) -> list[RegistryEntry]:
    _text(payload.get("source_id"), "search_manual.source_id")
    default_version = _text(payload.get("version"), "search_manual.version")
    results = payload.get("results")
    if not isinstance(results, list):
        _fail("search_manual.results must be a list")
    entries: list[RegistryEntry] = []
    for index, raw_section in enumerate(results):
        section = _mapping(raw_section, f"search_manual.results[{index}]")
        device_id = _require_device_id(section, requested_device_id, f"search_manual.results[{index}]")
        source_id = _text(section.get("source_id"), f"search_manual.results[{index}].source_id")
        title = _text(section.get("title"), f"search_manual.results[{index}].title")
        content = _text(section.get("content"), f"search_manual.results[{index}].content")
        version_value = section.get("version", default_version)
        version = _text(version_value, f"search_manual.results[{index}].version")
        facts = {"device_id": device_id, "title": title, "content": content, "version": version}
        entries.append(
            _entry(
                evidence_id=source_id,
                evidence_type="manual",
                source_id=source_id,
                summary=f"手册《{title}》（版本 {version}）：{content}",
                device_id=device_id,
                tool_name=tool_name,
                facts=facts,
                version=version,
            )
        )
    return entries


_BUILDERS = {
    "get_device_info": _device_entries,
    "query_sensor_history": _sensor_entries,
    "query_alarm_history": _alarm_entries,
    "query_work_order_history": _work_order_entries,
    "search_manual": _manual_entries,
}


def entries_from_tool_payload(
    tool_name: str,
    payload: Mapping[str, Any],
    requested_device_id: str,
) -> list[RegistryEntry]:
    """Convert one successful tool payload into canonical registry entries.

    This is the shared conversion boundary for both the message-driven phase-one
    flow and the graph-driven phase-two flow. A ``not_found`` payload is a
    completed call that intentionally contributes no evidence.
    """

    payload = _mapping(payload, f"{tool_name}.payload")
    builder = _BUILDERS.get(tool_name)
    if builder is None:
        _fail(f"unsupported successful tool {tool_name!r}")
    status = payload.get("status")
    if status not in {"ok", "not_found"}:
        _fail(f"{tool_name}.payload.status must be 'ok' or 'not_found'")
    if status == "not_found":
        return []
    return builder(payload, tool_name, requested_device_id)


def serialize_entry(entry: RegistryEntry) -> dict[str, Any]:
    """Return a JSON-safe snapshot so graph state stays serializable."""

    return {
        "evidence": entry.evidence.model_dump(mode="json"),
        "device_id": entry.device_id,
        "tool_name": entry.tool_name,
        "facts": _json_safe(dict(entry.facts)),
    }


def deserialize_entry(data: Mapping[str, Any]) -> RegistryEntry:
    """Rebuild an immutable entry from its serialized snapshot."""

    return RegistryEntry(
        evidence=EvidenceItem.model_validate(data["evidence"]),
        device_id=_text(data.get("device_id"), "registry.device_id"),
        tool_name=_text(data.get("tool_name"), "registry.tool_name"),
        facts=MappingProxyType(dict(data["facts"])),
    )


def build_evidence_registry(messages: list[Any], requested_device_id: str) -> EvidenceRegistry:
    """Create canonical evidence and preserve tool failures until a valid retry clears them."""

    entries: dict[str, RegistryEntry] = {}
    unresolved_tool_errors: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        # A tool-argument failure is not evidence. It remains unresolved until
        # this same known tool returns a legal successful payload below.
        if message.status == "error":
            unresolved_tool_errors.add(message.name)
            continue
        if isinstance(message.content, str):
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"malformed successful tool payload: {message.name}.content is not JSON") from error
        elif isinstance(message.content, dict):
            payload = message.content
        else:
            _fail(f"{message.name}.content must be a JSON object")
        batch = entries_from_tool_payload(message.name, payload, requested_device_id)
        unresolved_tool_errors.discard(message.name)
        for entry in batch:
            previous = entries.get(entry.evidence.evidence_id)
            if previous is not None and previous != entry:
                _fail(f"conflicting stable evidence_id {entry.evidence.evidence_id!r}")
            entries[entry.evidence.evidence_id] = entry
    return EvidenceRegistry(
        entries=MappingProxyType(entries),
        unresolved_tool_errors=frozenset(unresolved_tool_errors),
    )
