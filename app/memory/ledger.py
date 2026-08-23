"""Controlled append-only long-term memory.

本模块是第四阶段记忆能力的另一半：只追加（append-only）的长期台账，
记录"哪些维护动作经人批准并真正执行了"。与 session.py 的分工：短期
记忆服务于当前对话的上下文，长期台账服务于跨会话的历史追溯，并为
后续规划提供参考（例如"该设备上周刚处理过同类故障"）。

写入边界（安全核心）：
- 只有"已批准且已执行"的动作才允许入账，写入入口全系统仅此一处；
- 字段走白名单 _ALLOWED_FIELDS；自由格式的模型输出、用户原文提问、
  被拒绝的提案一律不予持久化；
- 文件采用 JSONL 格式，每行一条独立可审阅的 JSON 记录。

失败行为：写入前强制 request_id / device_id 非空，否则抛 ValueError；
读取时损坏的行被跳过而不是污染整份历史——台账宁缺毋滥，绝不带毒。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# 白名单：history_for_device 只回吐下列结构化程序字段，确保文件里
# 可能出现的其他内容不会被透传进 Prompt。
_ALLOWED_FIELDS = (
    "recorded_at",
    "action_type",
    "request_id",
    "device_id",
    "risk_level",
    "ticket_id",
    "decided_by",
)


class LongTermLedger:
    """只增不改的台账：记录已执行且经人工批准的维护动作。

    为什么 append-only：审计历史不容改写；配合每条记录的 UTC 时间戳
    与白名单字段，既能回答"这台设备过去发生过什么"，又不可能被下游
    逻辑偷偷修饰。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record_approved_action(
        self,
        *,
        request_id: str,
        device_id: str,
        risk_level: str,
        ticket_id: str,
        decided_by: str,
        action_type: str = "schedule_maintenance",
    ) -> dict[str, Any]:
        """追加一条已批准动作记录；这是台账唯一的写入入口。

        目录不存在会自动创建；记录带 UTC 时间戳落盘并原样返回。
        失败时：request_id / device_id 为空抛 ValueError；磁盘错误向上
        抛给调用方，绝不静默丢弃审计记录。
        """
        if not request_id.strip() or not device_id.strip():
            raise ValueError("request_id and device_id must not be empty")
        record = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "action_type": action_type,
            "request_id": request_id,
            "device_id": device_id,
            "risk_level": risk_level,
            "ticket_id": ticket_id,
            "decided_by": decided_by,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return dict(record)

    def history_for_device(self, device_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """返回某设备最近 limit 条记录（时间正序，仅白名单字段）。

        读取是防御式的：空行、JSON 损坏行、非 dict 行一律跳过，单行
        损坏不影响其余历史的可用性；limit 小于 1 或文件尚不存在时
        返回空列表。
        """

        if limit < 1:
            return []
        records: list[dict[str, Any]] = []
        if not self._path.exists():
            return records
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 单行损坏不得污染整份历史
                if not isinstance(parsed, dict):
                    continue
                if parsed.get("device_id") != device_id:
                    continue
                records.append({field: parsed.get(field) for field in _ALLOWED_FIELDS})
        return records[-limit:]

    def total_records(self) -> int:
        """统计可解析的记录总数；损坏的行不计入有效历史。"""

        if not self._path.exists():
            return 0
        count = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
        return count
