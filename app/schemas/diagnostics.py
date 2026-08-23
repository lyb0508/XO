"""Strict schema for a diagnostician's final, evidence-backed response.

本模块定义诊断结论的两份严格契约：DiagnosisDraft（模型可写的草稿——只能挑选
evidence_id，不能书写证据事实）与 DiagnosisReport（程序定稿的报告——证据条目
完全来自 EvidenceRegistry，模型无法掺入编造的测量值）。

在整个项目中的位置：DiagnosisDraft 是两段式诊断 Agent 第二阶段的结构化输出
目标；DiagnosisReport 是最终交付物，也是审批环节 derive_proposed_action 的
输入。

数据来源：Draft 由模型结构化输出产生（不可信，必须校验）；Report 由程序把
Draft 与注册表合并后生成。

副作用边界：纯 schema 定义，无任何 I/O 或写入。

失败时的行为：未知字段、超长文本、证据引用不一致、"证据充分却没给 ID"或
"高风险却不要求人工复核"等矛盾组合都会被 validator 拒绝，把坏输出挡在
报告环节之外。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Reject unknown output fields so model output cannot silently expand the API.

    公共基类：extra="forbid" 让模型多输出的未知字段直接导致校验失败，
    防止输出契约被悄悄扩宽；str_strip_whitespace 顺带清理首尾空白。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LikelyCause(StrictModel):
    """单条可能原因：必须挂靠至少一个证据 ID，置信度限制在 [0, 1]。"""

    cause: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must not contain duplicates")
        return values


class EvidenceItem(StrictModel):
    """一条 canonical 证据的对外形态：ID、类型、来源、摘要与观测时间/版本。"""

    evidence_id: str = Field(min_length=1, max_length=128)
    evidence_type: Literal["device", "sensor", "alarm", "work_order", "manual"]
    source_id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=800)
    observed_at: datetime | None = None
    version: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observed_at must include timezone information")
        return value


class DiagnosisDraft(StrictModel):
    """Model-writable diagnosis that can select evidence IDs but cannot write facts.

    模型可写的草稿：拥有结论字段（范围、风险、原因、建议），但只能引用
    evidence_id——真实证据事实由程序在 finalize 阶段从注册表回填。
    交叉校验保证草稿自洽：引用的 ID 必须在已选集合内，风险等级与
    evidence_sufficient、requires_human_review 之间不得互相矛盾。
    """

    request_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    scope_status: Literal["in_scope", "out_of_scope", "needs_clarification"]
    risk_level: Literal["low", "medium", "high", "critical", "unknown"]
    summary: str = Field(min_length=1, max_length=1200)
    evidence_sufficient: bool
    likely_causes: list[LikelyCause] = Field(max_length=5)
    evidence_ids: list[str] = Field(max_length=20)
    recommended_actions: list[str] = Field(max_length=8)
    requires_human_review: bool
    limitations: list[str] = Field(max_length=8)

    @field_validator("recommended_actions", "limitations")
    @classmethod
    def nonempty_text_items(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 400 for value in values):
            raise ValueError("text items must be non-empty and at most 400 characters")
        return values

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> "DiagnosisDraft":
        selected_ids = set(self.evidence_ids)
        referenced_ids = {
            evidence_id
            for cause in self.likely_causes
            for evidence_id in cause.evidence_ids
        }
        if not referenced_ids.issubset(selected_ids):
            raise ValueError("likely_causes may reference only selected evidence_ids")
        if self.evidence_sufficient and not self.evidence_ids:
            raise ValueError("at least one evidence_id is required when evidence is sufficient")
        if not self.evidence_sufficient:
            if self.risk_level != "unknown":
                raise ValueError("risk_level must be unknown when evidence is insufficient")
            if not self.limitations:
                raise ValueError("limitations are required when evidence is insufficient")
        if self.risk_level in {"high", "critical"} and not self.requires_human_review:
            raise ValueError("high or critical risk requires human review")
        return self


class DiagnosisReport(StrictModel):
    """Program-finalized report whose evidence facts originate only from the registry.

    程序定稿的报告：与 Draft 的关键差别是 evidence 字段装的是完整证据条目
    （EvidenceItem）而非 ID 列表，且这些条目只可能来自 EvidenceRegistry，
    因此报告中的每一条事实都可追溯到真实工具返回。
    """

    request_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    scope_status: Literal["in_scope", "out_of_scope", "needs_clarification"]
    risk_level: Literal["low", "medium", "high", "critical", "unknown"]
    summary: str = Field(min_length=1, max_length=1200)
    evidence_sufficient: bool
    likely_causes: list[LikelyCause] = Field(max_length=5)
    evidence: list[EvidenceItem] = Field(max_length=20)
    recommended_actions: list[str] = Field(max_length=8)
    requires_human_review: bool
    limitations: list[str] = Field(max_length=8)

    @field_validator("recommended_actions", "limitations")
    @classmethod
    def nonempty_text_items(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 400 for value in values):
            raise ValueError("text items must be non-empty and at most 400 characters")
        return values

    @field_validator("evidence")
    @classmethod
    def unique_evidence(cls, values: list[EvidenceItem]) -> list[EvidenceItem]:
        ids = [item.evidence_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique")
        return values

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> "DiagnosisReport":
        evidence_ids = {item.evidence_id for item in self.evidence}
        referenced_ids = {
            evidence_id
            for cause in self.likely_causes
            for evidence_id in cause.evidence_ids
        }
        if not referenced_ids.issubset(evidence_ids):
            raise ValueError("likely_causes may reference only included evidence_ids")
        if self.evidence_sufficient and not self.evidence:
            raise ValueError("at least one evidence item is required when evidence is sufficient")
        if not self.evidence_sufficient:
            if self.risk_level != "unknown":
                raise ValueError("risk_level must be unknown when evidence is insufficient")
            if not self.limitations:
                raise ValueError("limitations are required when evidence is insufficient")
        if self.risk_level in {"high", "critical"} and not self.requires_human_review:
            raise ValueError("high or critical risk requires human review")
        return self
