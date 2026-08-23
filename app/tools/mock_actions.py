"""Simulated, idempotent maintenance action; no external side effects exist.

这是本项目唯一的"业务动作"：排一次维护计划。它只写入进程内存里的一张
执行台账（ledger），绝不触碰真实工单系统、数据库或任何外部服务。

为什么用进程内存实现：在接入真实工单系统之前，先用零副作用的模拟实现，
把"审批 -> 执行 -> 幂等重试 -> 拒绝"整条链路端到端跑通并测试到位；将来
替换成真实实现时，函数签名与幂等契约保持不变。

失败行为：空 request_id / device_id 直接抛 ValueError；重复请求不是错误，
返回 ``already_executed`` 并保持原 ticket_id 不变。
"""

from __future__ import annotations

import threading
from typing import Any

_ACTION_TYPE = "schedule_maintenance"
# 执行台账：key 是 (action_type, request_id) 幂等键，value 是首次执行的完整记录。
_EXECUTION_LEDGER: dict[tuple[str, str], dict[str, Any]] = {}
# 即使是同步 invoke，LangGraph 也在后台线程上执行节点工作，
# 所以必须用锁保护这张进程级台账，避免并发下出现"检查-再写入"的竞态窗口。
_LEDGER_LOCK = threading.Lock()


def execute_maintenance_action(request_id: str, device_id: str) -> dict[str, Any]:
    """以 idempotency key 记录一次模拟的维护排程。

    幂等键是 (action_type, request_id)：同一个 request_id 无论重试多少次，
    都不可能生成第二张工单——这正是 Graph 在 Interrupt/Resume 之后重放节点
    时所依赖的安全保证。重复调用返回 ``already_executed``，表示"该请求此前
    已成功执行过"，本次只是幂等重复而非新动作，并回传原始 ticket 引用供
    调用方核对。

    失败行为：request_id 或 device_id 为空白字符串时抛 ValueError；
    台账写入由锁保护，不存在部分写入状态。
    """

    if not request_id.strip():
        raise ValueError("request_id must not be empty")
    if not device_id.strip():
        raise ValueError("device_id must not be empty")
    key = (_ACTION_TYPE, request_id)
    with _LEDGER_LOCK:
        previous = _EXECUTION_LEDGER.get(key)
        if previous is not None:
            return {
                "status": "already_executed",
                "ticket_id": previous["ticket_id"],
                "action_type": _ACTION_TYPE,
                "request_id": request_id,
                "device_id": previous["device_id"],
            }
        record = {
            "status": "executed",
            "ticket_id": f"MNT-{request_id}",
            "action_type": _ACTION_TYPE,
            "request_id": request_id,
            "device_id": device_id,
        }
        _EXECUTION_LEDGER[key] = record
    return dict(record)


def reset_execution_ledger() -> None:
    """仅供测试使用的清理入口，保证各单元测试之间不会看到彼此的工单。"""

    with _LEDGER_LOCK:
        _EXECUTION_LEDGER.clear()
