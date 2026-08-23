"""Program-owned canonical evidence registry for the diagnostic formatter.

本模块维护一份由程序持有的"canonical 证据注册表"：把工具返回的成功 payload
解析成带稳定 evidence_id 的不可变条目（RegistryEntry），供诊断报告格式化器
按 ID 引用。

在整个项目中的位置：位于两段式诊断 Agent 的第一、二阶段之间——第一阶段产出
ToolMessage 列表后，由 build_evidence_registry 转换成注册表；第二阶段的模型
只能看到 formatter_payload 暴露的内容，最终通过 select 取回被引用的事实。

数据来源：仅接受"状态为成功"的 ToolMessage 内容（JSON 对象）；not_found 视为
一次合法完成但贡献零证据的调用。工具输出属于不可信输入，每个字段都要过类型
与取值校验后才准入册。

副作用边界：纯内存、只读转换，不访问数据库、网络或文件系统。

失败时的行为：payload 缺字段、类型不符、device_id 与请求不符、稳定 ID 冲突等
一律抛 RuntimeError；status=error 的 ToolMessage 会记入 unresolved_tool_errors，
直到同名工具后来返回合法成功结果才清除，否则上层必须阻断报告生成。
"""

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
    # bool 是 int 的子类，必须显式排除，否则 true/false 会被悄悄当成数值 1/0。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context} must be numeric")
    return float(value)


def _observed_at(value: Any, context: str) -> datetime:
    # 把结尾的 Z 替换成 +00:00，兼容不识别 Z 后缀的 fromisoformat 实现。
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
    """Immutable binding between an evidence ID and program-verified facts.

    不可变绑定：evidence_id 与程序逐字段校验过的 facts 捆绑在一起。facts 用
    MappingProxyType 包装，任何人都无法在入册后偷偷改写证据内容。
    """

    evidence: EvidenceItem
    device_id: str
    tool_name: str
    facts: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvidenceRegistry:
    """Immutable canonical records that the formatter may only reference by ID.

    格式化器（模型）只能通过 ID 引用这里的条目，不能直接书写证据事实；
    select 会拒绝注册表中不存在的 ID，从机制上杜绝"幻觉引用"。
    """

    entries: Mapping[str, RegistryEntry]
    unresolved_tool_errors: frozenset[str]

    def formatter_payload(self) -> dict[str, Any]:
        """Return deterministic, still-untrusted evidence context for the model.

        中文说明：为第二阶段的模型生成确定性上下文——按 evidence_id 排序输出，
        同样的注册表永远得到同样的字节序列。注意返回内容仍视为不可信数据，
        上层 Prompt 必须声明它不是指令。
        """

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
        """按模型给出的 ID 列表取回对应条目；出现注册表外的 ID 立即抛错。

        这是"幻觉引用"的最终防线：哪怕模型编造了一个看起来合理的 evidence_id，
        只要它不在注册表里，报告就无法生成。
        """
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
    flow and the graph-driven phase-two flow.     A ``not_found`` payload is a
    completed call that intentionally contributes no evidence.

    中文说明：单个成功 payload 转换为注册表条目的统一入口，消息驱动（第一阶段）
    与 Graph 驱动（第二阶段）两条链路共用这一转换边界。status 必须是 ok 或
    not_found；not_found 表示"查询合法执行但没有命中"，返回空列表而不是报错。
    不认识的工具名或其它 status 值都会抛 RuntimeError。
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
    """Return a JSON-safe snapshot so graph state stays serializable.

    中文说明：把条目转成可被 JSON 序列化（mode="json"）的快照，供 LangGraph
    state 与 checkpoint 保存——Graph State 必须可序列化才能支持持久化与恢复。
    """

    return {
        "evidence": entry.evidence.model_dump(mode="json"),
        "device_id": entry.device_id,
        "tool_name": entry.tool_name,
        "facts": _json_safe(dict(entry.facts)),
    }


def deserialize_entry(data: Mapping[str, Any]) -> RegistryEntry:
    """Rebuild an immutable entry from its serialized snapshot.

    中文说明：从快照重建不可变条目，与 serialize_entry 互为逆操作，
    用于从 checkpoint 恢复注册表内容。
    """

    return RegistryEntry(
        evidence=EvidenceItem.model_validate(data["evidence"]),
        device_id=_text(data.get("device_id"), "registry.device_id"),
        tool_name=_text(data.get("tool_name"), "registry.tool_name"),
        facts=MappingProxyType(dict(data["facts"])),
    )


def build_evidence_registry(messages: list[Any], requested_device_id: str) -> EvidenceRegistry:
    """Create canonical evidence and preserve tool failures until a valid retry clears them.

    中文说明：扫描消息列表中的全部 ToolMessage，把成功 payload 入册为
    canonical 证据；工具报错不会立即失败，而是记下工具名进入"未解决"集合，
    等同一工具后来返回合法成功结果时自动清除——这样模型的一次合法重试就能
    自我修复，而遗留的未解决错误由上层显式阻断报告。同一 evidence_id 出现
    内容不一致的两条记录视为数据冲突，直接抛错。
    """

    entries: dict[str, RegistryEntry] = {}
    unresolved_tool_errors: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        # 参数错误的工具调用本身不算证据：该工具保持"未解决"状态，
        # 直到下方同一工具返回一次合法的成功 payload 才清除。
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
