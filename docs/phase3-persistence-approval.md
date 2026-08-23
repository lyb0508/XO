# 第三阶段：持久化与人工审批

## 1. 这一阶段解决什么问题

第二阶段的图每次运行都是无状态的：进程结束一切归零，`thread_id` 只是 Trace 标签，报告里的"需要人工复核"只是一个字段，没有任何机制真正停下来等人决策。第三阶段引入 LangGraph Checkpoint 与 Interrupt，使高风险结论在执行任何受控动作前强制暂停，等待结构化的人工批准/修改/拒绝，并用幂等键保证恢复重跑不会重复产生副作用。

核心规则来自官方文档并在本项目验证：`interrupt()` 需要 checkpointer 和 `configurable.thread_id`；恢复用 `Command(resume=...)`；**节点从开头重跑**，因此 interrupt 之前的代码必须无副作用；interrupt payload 必须 JSON 可序列化；不要 try/except 包裹 interrupt。

## 2. 文件责任与真实数据流

| 文件 | 当前责任 |
| --- | --- |
| `app/schemas/approval.py` | `ProposedAction`（程序派生的唯一受控动作）、`ApprovalDecision`（三种决策 + 审批人 + 理由 + 修订动作列表）；`derive_proposed_action` 只从报告的程序字段派生动作，模型无权定义业务动作。 |
| `app/tools/mock_actions.py` | 唯一模拟副作用：以 `(action_type, request_id)` 为幂等键记录"安排检修"；重复调用返回 `already_executed` 且票据号稳定；`reset_execution_ledger()` 仅供测试清理；无任何外部 I/O。 |
| `app/graphs/nodes.py` 新增节点 | `approval_gate`：纯计算派生拟议动作 → `interrupt(payload)` → 校验人工输入为 `ApprovalDecision` → 按 decision 用 `Command(goto=...)` 路由；modified 时同步改写报告的 `recommended_actions`。`execute_approved_action`：仅在批准后运行，调用幂等模拟工具并写审计。`record_rejection`：仅写拒绝审计，无副作用。 |
| `app/graphs/routing.py` | 新增 `route_after_finalize`：按 `report.requires_human_review` 分流到审批门或直接完成。 |
| `app/graphs/builder.py` | `checkpointer` 参数：提供时才接线审批分支；不提供则 finalize 直接连 complete——没有持久化就没有恢复 interrupt 的能力，图拒绝提供会产生中断却无法恢复的路径。 |
| `app/main.py` | CLI 以 `InMemorySaver` 编译并把 `thread_id` 放进 `configurable`（它现在是真正的状态指针）；`_run_with_approval` 循环处理 `__interrupt__`；stdin 收集决策（approve/modify/reject、decided_by、理由、修改动作逐行输入）；EOF 或多次非法输入时报错退出，**绝不伪造人工决策**；最终输出 `{report, approval, action_audit}` 对象。 |

```text
finalize_report -> route_after_finalize
    requires_human_review=false -> complete -> END
    true -> approval_gate:
        纯计算派生 ProposedAction（无副作用，重跑安全）
        interrupt({proposed_action, report_summary})
        resume 值校验为 ApprovalDecision
        approved            -> execute_approved_action -> complete
        modified(附新动作)   -> 改写 recommended_actions -> execute_approved_action
        rejected            -> record_rejection -> complete
```

## 3. 幂等与恢复语义

- interrupt 前只有纯计算；唯一的"写"发生在恢复后的独立节点。
- 模拟工具的幂等键是 `(schedule_maintenance, request_id)`：同一请求无论恢复重放多少次只会有一张票；重复调用返回首次票据号。
- `test_threads_are_isolated_under_a_shared_checkpointer` 证明两个线程在同一 InMemorySaver 下各自暂停、各自决策、互不污染。
- 局限如实声明：InMemorySaver 进程退出即失，跨进程持久化（如 SqliteSaver）与生产部署属于后续工程化阶段；当前"可恢复"指同一进程生命周期内。

## 4. 失败复盘：live 暴露的过度 Schema 约束

live Ollama 运行中 qwen2.5:7b 在未请求 manual 证据的计划里多填了 `manual_query`，QueryPlan 的反向校验（"metrics/manual_query 仅允许出现在对应证据类型的请求中"）直接把整个诊断变成解析失败。该约束本意是防越权，实际毫无防御价值：fan-out 只由 `requested_evidence_types` 驱动，多余字段天然不被程序消费。修复为"正向必填严格、反向多余宽容"：请求某类证据时对应参数仍强制提供，但对未请求类型的多余字段不再拒绝，同时在 PLAN_PROMPT 中说明程序会忽略它们。教训：Schema 应约束程序要消费的东西，而不是惩罚模型的冗余输出；否则小模型的每一次啰嗦都变成一次故障。

另一次 live 运行中模型给出 `requires_human_review=false`，审批分支未被触发——这是合法行为而非缺陷：审批只在报告要求复核时发生。

## 5. 验证证据与边界

以下为 2026-08-23 实际执行的验证：

- 默认离线套件 142 项收集：140 通过，2 项 live 测试默认跳过；连续三次稳定（约 2.3–2.5 秒）。新增覆盖：三种决策的 Schema 形状、幂等工具语义、approve/modified/reject 三条图级路径、无复核直通路径、共享 checkpointer 下的线程隔离、CLI 交互审批与 EOF 不伪造决策。
- live Ollama 全链路通过一次：high 风险报告触发 interrupt，管道注入 approve 后 `action_audit.status=executed`、票据 `MNT-live-p3-smoke-005` 生成；另有一次 out-of-scope 判定未触发审批（合法）与一次计划解析失败（已修复并复盘）。
- 未验证：跨进程持久化恢复、DeepSeek provider、远端 LangSmith 上传、并发多线程同时 resume 的竞争行为。

## 6. 学习练习与验收答案

1. **为什么 approval_gate 在 interrupt 前只能做纯计算？** 答：resume 时节点从头重跑，interrupt 前的副作用会重复执行；把副作用放在恢复后的独立节点才能保证恰好一次。
2. **幂等键为什么含 action_type 和 request_id？** 答：同一请求的同一动作只应产生一张票；不同动作类型或不同请求互不干扰。
3. **checkpointer=None 时为什么直接砍掉审批分支？** 答：interrupt 依赖 checkpoint 保存现场；没有持久化就无法恢复，保留中断路径等于制造必然失败的死胡同，编译期移除是 fail-safe。
4. **EOF 时为什么不自动拒绝？** 答：自动拒绝也是替人做决定；CLI 选择显式失败，线程在 checkpointer 生命周期内保持可恢复状态。
5. **modified 决策改了什么？** 答：仅替换报告中 `recommended_actions` 文本列表后照常执行拟议动作；动作类型和范围始终由程序派生，人不能扩大它。

## 7. 明确未实现与后续阶段

RAG 与记忆（第四阶段）、LangSmith Dataset/Evaluator 固定评测（第五阶段）、FastAPI/SSE/权限/限流与真实持久化部署（第六阶段）均未实现。当前审计记录只存在于图状态与进程内存中。
