# 工业设备故障研判 Agent（LangChain 第一阶段）

这是一个用于学习 LangChain v1 的只读工业设备诊断示例。目标业务链路是“用户问题 → 证据采集 → 证据约束的结构化诊断报告”；当前只覆盖基础 Agent、工具契约、结构化输出、中间件限额和可选 LangSmith Trace。LangGraph 编排、RAG、真实设备/工单系统和生产写入不在本阶段。

## 快速开始（Windows PowerShell）

项目要求 Python 3.13（既有验证环境为 Python 3.13.12）。建议使用虚拟环境：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

从模板创建本地配置；`.env` 只保留在本机，不能提交：

```powershell
Copy-Item .env.example .env
```

默认配置是本地 Ollama：`PROVIDER=ollama`、模型 `qwen2.5:7b`、地址 `http://127.0.0.1:11434`。先在本机准备并启动该模型，然后运行：

```powershell
python -m app.main --question "研判 PUMP-003 的振动报警" --request-id request-001
```

也可以指定设备和 Trace 关联 ID：

```powershell
python -m app.main --question "查询设备信息" --device-id PUMP-003 --thread-id thread-001
```

Provider 必须二选一，不要同时配置为活动值：

- Ollama：保持 `.env.example` 的默认值。
- DeepSeek：设置 `INDUSTRIAL_AGENT_PROVIDER=deepseek`、`INDUSTRIAL_AGENT_MODEL=deepseek-v4-flash`、`INDUSTRIAL_AGENT_BASE_URL=https://api.deepseek.com/v1`、`INDUSTRIAL_AGENT_STRUCTURED_OUTPUT_METHOD=function_calling`，并在未跟踪的 `.env` 中填写 `INDUSTRIAL_AGENT_DEEPSEEK_API_KEY`。

缺失或空的 DeepSeek key 会 fail closed，不会回退到 Ollama。API 调用可能产生费用，只有在明确授权 live 调用时才应启用；不要把 key 发到聊天、日志、Trace 或 Git。

离线测试：

```powershell
python -m pytest
```

## 目录导航

- `app/config/`：集中式 `Settings`，负责 provider、模型、限额和 Trace 配置。
- `app/models/`：Ollama/DeepSeek 模型工厂，不在构造时发起网络调用。
- `app/tools/`：五个固定模拟数据的只读工具。
- `app/retrieval/`：手册 RAG：embedding 工厂、进程内向量库、带引用元数据的检索。
- `evaluations/`：LangSmith 评测：50 条固定数据集、确定性评测器、本地优先的实验运行器。
- `app/memory/`：有界短期会话记忆与仅记录已批准动作的长期台账。
- `app/graphs/`：LangGraph 编排：规划节点、并行 fan-out、Reducer 合并、条件路由、Checkpoint/Interrupt 人工审批与 fail-closed 分支。
- `app/agents/`：证据注册表转换边界、振动门控与报告格式化（两阶段 Agent 入口保留）。
- `app/schemas/`：工具输入、`QueryPlan` 与 `DiagnosisReport` 输出契约。
- `app/observability/`：显式开启才创建的 LangSmith client、allowlist 和脱敏。
- `app/main.py`：CLI 入口，输出一行 JSON 或脱敏错误。
- `tests/`：离线 fake model、契约、图流程、限额、来源闭环和脱敏测试。
- [`docs/phase1-langchain-agent.md`](docs/phase1-langchain-agent.md)：阶段学习材料、真实数据流、失败复盘和练习。
- [`docs/phase2-langgraph-orchestration.md`](docs/phase2-langgraph-orchestration.md)：LangGraph 编排设计、并行/Reducer 语义与 live 复盘。
- [`docs/phase3-persistence-approval.md`](docs/phase3-persistence-approval.md)：Checkpoint/Interrupt 审批流、幂等语义与线程隔离。
- [`docs/phase4-retrieval-memory.md`](docs/phase4-retrieval-memory.md)：手册 RAG、阈值校准纪律与受控记忆。
- [`docs/phase5-evaluation.md`](docs/phase5-evaluation.md)：50 条固定评测集、指标基线与根因分析。

2026-08-23 验证记录包括 Python 3.13.12、`pip check`、`compileall app tests`，以及离线 170 项测试（另有 2 项 live 测试默认跳过）通过；阶段 2/3/4 各有本地 Ollama live smoke 通过记录（含完整人工审批链路与台账落盘）；阶段 5 完成 50 条 live 全量评测基线：工具选择 0.81 / 拒答 0.727 / 轨迹 0.84，三项最终目标指标未达标且差距已量化。完整证据和未验证边界见阶段文档。`pylock.toml` 由 `pip lock` 生成，该命令仍属 experimental，不能把锁文件生成等同于部署验证。
