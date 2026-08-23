"""Short-term in-process session memory.

本模块提供进程内的短期会话记忆，是第四阶段"记忆"能力的一半：短期
（本文件）记"最近聊过什么"，长期（ledger.py）记"批准执行过哪些动作"。

在图编排中的位置：SessionMemory 不进 GraphState——它由外层（如 API 层）
在会话过程中写入，规划节点只通过注入引用读取最近几轮的摘要文本。

状态与边界：以显式 session_id 为键；每个会话最多保留 max_turns 轮
（deque(maxlen) 自动淘汰最旧轮次），每条文本截断到固定长度；纯内存态、
不落盘，进程重启即清空。

失败行为：空 session_id 或非法 max_turns 抛 ValueError；读取不存在的
会话返回空列表而不报错。存储的文本在下游一律视为不可信上下文：
只能提供背景，永远不能改变系统规则。
"""

from __future__ import annotations

from collections import deque


def _clip(value: str, limit: int) -> str:
    """压缩空白并截断到 limit 长度，限定单条记忆的最大体积。"""
    cleaned = " ".join(str(value).split())
    return cleaned[:limit]


class SessionMemory:
    """有界的按会话短期记忆：保存最近的提问与处置结果摘要。

    为什么限制轮数与长度：这些文本会作为参考拼进规划 Prompt，无界增长
    既浪费 token 又会放大 Prompt Injection 的影响面；轮数上限加字段
    截断把风险与成本都压在常数范围内。
    """

    def __init__(self, max_turns: int = 5) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least one")
        self._max_turns = max_turns
        self._sessions: dict[str, deque[dict[str, str]]] = {}

    def append_turn(
        self,
        session_id: str,
        *,
        question: str,
        device_id: str,
        risk_level: str,
        summary: str,
    ) -> None:
        """向指定会话追加一轮记录；超出 max_turns 时最旧一轮自动被挤出。"""
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        turns = self._sessions.setdefault(session_id, deque(maxlen=self._max_turns))
        turns.append(
            {
                "question": _clip(question, 200),
                "device_id": _clip(device_id, 128),
                "risk_level": _clip(risk_level, 32),
                "summary": _clip(summary, 300),
            }
        )

    def recent_turns(self, session_id: str) -> list[dict[str, str]]:
        """返回逐条复制的副本，调用方无法借道修改已存历史。"""

        return [dict(turn) for turn in self._sessions.get(session_id, ())]

    def forget(self, session_id: str | None = None) -> None:
        """删除单个会话或全部会话；各会话相互独立、可单独清理。"""

        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    def snapshot(self) -> dict[str, int]:
        """返回各会话的轮数统计，便于诊断观察与测试断言。"""

        return {session: len(turns) for session, turns in sorted(self._sessions.items())}
