"""Human-in-the-loop approval contracts for the phase-three gate.

本模块定义人工审批（Human-in-the-loop）环节的契约：ProposedAction 是程序从
定稿报告中派生的受控动作建议；ApprovalDecision 是人在 Interrupt 恢复后给出的
结构化决策（approved / modified / rejected 三选一）。

在整个项目中的位置：位于诊断报告之后、真实副作用之前——任何受控动作都必须
先经过这里的审批契约，未获批准绝不执行。

数据来源：动作建议只由 derive_proposed_action 从程序持有的报告字段派生，
模型文本永远不能定义业务动作；决策内容来自人工输入。

副作用边界：纯 schema 与纯函数派生，本模块自身不执行任何动作、不做任何写入。

失败时的行为：decided_by 含非法字符、modified 决策缺少 modified_actions、
非 modified 决策却携带 modified_actions 等都会触发 ValidationError 并向上抛出，
保证进入执行层的每条决策都是完整且可审计的。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.diagnostics import DiagnosisReport, StrictModel
from app.observability.tracing import _SAFE_VALUE

# 当前项目允许提议的受控动作类型白名单：只有"安排检修"一种。
CONTROLLED_ACTION_TYPES = ("schedule_maintenance",)


class ProposedAction(StrictModel):
    """The single controlled action type this project can propose.

    本项目唯一允许提议的动作类型。字段刻意精简：动作种类写死为
    schedule_maintenance，范围信息只来自程序的 request_id/device_id，
    模型无法借机扩大动作的能力。
    """

    action_type: Literal["schedule_maintenance"]
    request_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=400)


class ApprovalDecision(StrictModel):
    """Structured human decision consumed after an interrupt resumes.

    人在 Interrupt 恢复后提交的结构化决策。三种结果都要能表达：批准、
    修改（必须给出修改后的动作清单）、拒绝。
    """

    decision: Literal["approved", "modified", "rejected"]
    decided_by: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=400)
    # 字段约束与 DiagnosisReport.recommended_actions 对齐，保证"修改"后的
    # 决策回填报告时不会让最终校验失败。
    modified_actions: list[str] | None = Field(default=None, max_length=8)

    @field_validator("decided_by")
    @classmethod
    def decided_by_is_safe(cls, value: str) -> str:
        if not _SAFE_VALUE.fullmatch(value):
            raise ValueError("decided_by may contain only letters, digits, . _ : or -")
        return value

    @field_validator("modified_actions")
    @classmethod
    def modified_actions_are_valid(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if any(not action.strip() or len(action) > 400 for action in values):
            raise ValueError("modified actions must be non-empty and at most 400 characters")
        return values

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "ApprovalDecision":
        if self.decision == "modified":
            if not self.modified_actions:
                raise ValueError("a modified decision must provide modified_actions")
        elif self.modified_actions is not None:
            raise ValueError("modified_actions are only allowed for a modified decision")
        return self


def derive_proposed_action(report: dict[str, Any]) -> ProposedAction:
    """Derive the controlled action from a finalized report mapping.

    Only program-owned fields feed this derivation; model text never widens
    what the action can do.

    中文说明：从定稿报告派生受控动作建议。输入只取程序校验过的字段
    （request_id、device_id、risk_level 与截断后的 summary），模型自由文本
    无法影响动作的种类或范围；summary 截断到 300 字符是为了满足 reason
    字段的长度上限。传入非法 report 会触发 ValidationError。
    """

    validated = report if isinstance(report, DiagnosisReport) else DiagnosisReport.model_validate(report)
    return ProposedAction(
        action_type="schedule_maintenance",
        request_id=validated.request_id,
        device_id=validated.device_id,
        reason=f"risk_level={validated.risk_level}; {validated.summary[:300]}",
    )
