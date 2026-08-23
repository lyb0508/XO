# 第六阶段：工程化（HTTP API / SSE / 安全 / 交付）

## 1. 这一阶段解决什么问题

前五个阶段的能力都锁在 CLI 进程里：一次性运行、审批靠 stdin、无法被其他系统调用。第六阶段把诊断服务化——FastAPI 应用暴露同步与流式端点，人工审批走 HTTP 往返，加上认证、限流、容器化与 CI，使项目具备被真实系统集成的形态。

## 2. 端点与协议

| 端点 | 认证 | 说明 |
| --- | --- | --- |
| `GET /health` | 否 | 存活探针。 |
| `POST /diagnoses` | 是 | 同步执行完整诊断；审批中断时**服务端自动拒绝**（无人值守调用永不批准受控动作），返回 `{thread_id, report, approval, action_audit}`。 |
| `POST /diagnoses/stream` | 是 | `text/event-stream`。一条连接驱动一段图执行：每个节点完成推 `node` 事件；遇到审批中断推 `approval_required` 并结束流；正常结束推 `done`（含完整 outcome）；失败推 `error`（脱敏）。 |
| `POST /approvals/{thread_id}` | 是 | 提交结构化决策恢复检查点，同步返回最终 outcome。 |

SSE 客户端的完整交互循环：开流 → 收到 `approval_required` → `POST /approvals/{thread_id}` 提交决策（或重开流观察后续）→ 检查点续跑。

## 3. 安全设计

- **fail-closed 启动**：`INDUSTRIAL_AGENT_API_KEY` 未配置或空白时 `create_app()` 直接抛错拒绝启动。
- **常量时间比较**：`secrets.compare_digest` 防时序侧信道。
- **限流**：进程内滑动窗口（默认 30 次/分钟/key），超限 429 + `Retry-After`；键为客户端地址+key 组合。
- **错误脱敏**：所有 HTTP 错误详情经 `redact_payload`（与 CLI/Trace 同一层）；标识符复用 `_SAFE_VALUE` 白名单校验，非法字符直接 400。
- **已知边界（如实声明）**：同步端点的自动拒绝意味着纯 HTTP 调用方永远无法批准动作——批准必须走 `/approvals` 显式决策；InMemorySaver 与限流窗口均为进程内状态，多副本部署需要外部存储（未实现）。

## 4. 文件责任

| 文件 | 责任 |
| --- | --- |
| `app/api/__init__.py` | `create_app()` 工厂（模型可注入供测试）、三个诊断端点、认证、限流器、SSE 编码。 |
| `app/asgi.py` | 无参 ASGI 入口（容器平台用）。 |
| `Dockerfile` + `.dockerignore` | python:3.13-slim、非 root 用户、`--factory` 启动；容器内需将 Ollama 地址指向 `host.docker.internal:11434`。 |
| `.github/workflows/tests.yml` | CI：3.13 环境、editable 安装、pip check、compileall、离线 pytest 全量。 |
| `.env.example` | 新增 `INDUSTRIAL_AGENT_API_KEY` / `INDUSTRIAL_AGENT_API_RATE_LIMIT` 占位。 |

## 5. 验证证据

以下为 2026-08-24 实际执行的验证：

- 默认离线套件 182 项收集：180 通过，2 项 live 默认跳过；连续三次稳定（约 3.0 秒）。API 新增 8 项：无 key/错 key 401、outcome 契约、非法标识符 400、限流 429+Retry-After 且按 key 隔离、SSE 事件序列（node…done）、审批决策校验 422、服务无 key 拒绝启动、错误脱敏。
- Live Ollama HTTP 全链路：uvicorn 真实启动 → `/health` 200 → 无 key 请求 401 → 同步诊断返回 4 条证据 → SSE 流收到 6 个 `node` + 1 个 `done`（报告含 3 个振动测点与设备阈值证据）。
- 未验证：Docker 镜像构建与容器内运行（本机无 Docker 守护进程验证记录）、CI workflow 的真实 GitHub 运行、多副本部署、TLS 终结。

## 6. 过程复盘

- `Settings.api_key` 使用 `SecretStr`，`str()` 会得到掩码导致全部请求 401；修复为 `get_secret_value()` 解析并保留两种注入方式的兼容。
- SSE 处理器里残留了一段未实现的"优雅停机"属性访问，使每条流首事件变成 error；删除死代码后事件序列符合契约。
- 测试夹具的 fake draft 固定了 request_id，而 API 未传 request_id 时自动生成 uuid，身份校验正确拦截——测试数据必须与生产校验规则对齐。

## 7. 项目全景与剩余工作

六个阶段全部完成：LangChain Agent → LangGraph 编排 → 持久化与审批 → RAG 与记忆 → LangSmith 评测 → 工程化交付。当前明确的遗留项：

1. 三项评测最终目标指标未达标（0.81/0.727/0.84），改进方向见 phase5 文档 §4。
2. 多副本部署所需的持久 Checkpointer、分布式限流与台账存储。
3. Docker 构建/CI 真实运行的验证（需相应环境授权）。
4. TLS 终结、请求日志审计与在线监控（AGENTS.md 归入上线后能力）。
