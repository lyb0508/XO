# 第二阶段：LangGraph 状态编排

## 1. 这一阶段解决什么问题

第一阶段的证据采集由 `create_agent` 在单轮循环里自主决定调用哪个工具。它证明了工具契约和限额可行，但整个流程是一个黑盒 Runnable：没有显式状态、没有可测试的路由、也没有图级并行能力。第二阶段把诊断流程改写成 LangGraph 图，使"查什么"（模型规划）与"怎么查"（程序执行）分离，并引入 State、条件路由、并行 fan-out、Reducer 合并与显式错误分支。

核心设计决策：**规划式编排**。模型只在两个 Schema 绑定节点中出现——`plan_queries` 输出 `QueryPlan`，`format_report` 输出 `DiagnosisDraft`；五个只读工具全部由程序以确定性参数调用。放弃"模型多轮自主补查"换来完全可预测的执行面，这符合工业场景对外层流程确定性的要求，也是 AGENTS.md"外层尽量确定性、动态推理限制在明确 Agent 节点内"的落实。

## 2. 文件责任与真实数据流

| 文件 | 当前责任 |
| --- | --- |
| `app/graphs/state.py` | `GraphState` TypedDict；三个并行写入键的确定性 Reducer（payload 排序合并、错误去重排序、注册表快照排序合并）。状态只存 JSON 可序列化原始数据，为第三阶段 Checkpoint 做准备。 |
| `app/graphs/nodes.py` | 规划节点（一次结构化调用）、哑分发节点、五个查询节点工厂、join 合并冲突检测、格式化节点、定稿节点、fail_closed 终端节点。 |
| `app/graphs/routing.py` | 三个纯函数路由：范围分流、动态 fan-out 节点列表、错误分支。不依赖图或模型即可单测。 |
| `app/graphs/builder.py` | 组装与编译；`GRAPH_RECURSION_LIMIT = 25` 作为步数上限兜底。 |
| `app/schemas/query_plan.py` | `QueryPlan` 契约：in_scope 必须有 device 和证据类型；时间类证据必须有完整时区窗口；sensor 必须恰好一个 metric；manual 必须有查询词；非 in_scope 禁止请求证据。 |
| `app/agents/evidence.py` | 新增 `entries_from_tool_payload`（payload→规范条目的共享转换边界）、`serialize_entry`/`deserialize_entry`（图状态序列化往返）；`build_evidence_registry` 重构为复用同一转换函数，行为等价。 |
| `app/agents/diagnostic.py` | 私有定稿逻辑公开为 `finalize_report`/`validate_vibration_gate`；新增 `repair_vibration_selection`（见 §5）。 |
| `app/main.py` | CLI 切换到图入口；新增空问题显式拒绝；`AGENT_VERSION = "phase2-graph"`；fail_closed 分支走 stderr 错误路径退出码 1。 |

真实链路：

```text
CLI
  -> Settings / model factory（同第一阶段）
  -> build_diagnosis_graph(model)
       plan_queries   （1 次结构化调用 -> QueryPlan）
       -> route_after_plan：
            out_of_scope / needs_clarification -> format_report（0 次工具调用）
            in_scope -> dispatch
                 -> 条件边按 requested_evidence_types 返回节点列表，同一超步并行：
                    fetch_device_info / query_sensor_history / query_alarm_history /
                    query_work_order_history / search_manual_docs
                 -> 各节点把 payload 与序列化注册表快照经 Reducer 合并回 State
            -> join_registry：按 evidence_id 去重并检出冲突
            -> route_after_join：
                 有未解决错误或冲突 -> fail_closed（report=None）-> END
                 干净 -> format_report（1 次结构化调用 -> DiagnosisDraft）
            -> finalize_report：repair + 振动门控 + 注册表事实替换 -> END
  -> CLI 校验 report 存在且合法 -> 脱敏 -> 一行 JSON
```

## 3. 并行、Reducer 与执行顺序

- fan-out 通过 `add_conditional_edges("dispatch", route_to_queries)` 实现：路由函数返回节点名列表，LangGraph 在同一超步内并行执行。
- 三个被并行写入的键都有 `Annotated[..., reducer]`。开发中真实触发过 `InvalidUpdateError: Can receive only one value per step`——这正是缺少 Reducer 的症状，修复即补上 `merge_registry_snapshots`。该教训已固化为测试。
- 所有 Reducer 都做规范化排序，合并结果与完成顺序无关；`test_graph_routing.py` 用不同合并方向断言相等性。
- 同步 `invoke` 下同超步节点物理上顺序执行；真正的物理并发需要异步运行时（本阶段未使用）。图的并行语义是结构性的：fan-out/fan-in 拓扑 + 顺序无关合并。

## 4. 安全边界

- 模型无权选择工具参数：查询参数全部由 `_payload_args` 从已校验的 QueryPlan 和权威的 `state["device_id"]` 构造；即使 planner 输出其他 device_id，也会在 `plan_queries` 中被程序覆写。
- 动态 fan-out 的路由依据是 Pydantic 校验后的 `requested_evidence_types` 字面量映射，不是自由文本。
- 请求 sensor 证据时代码强制追加 device 阈值查询，振动门控的前置证据不依赖模型意愿。
- 工具异常写入 `tool_errors` 进入 fail_closed 分支，格式化器在该分支零调用（测试断言 `calls == 0`）；错误消息经脱敏后才离开进程。
- `QueryPlan.metrics` 收紧为至多一个元素：计划承诺必须等于执行能力（每类型一次调用），避免"多指标计划、单指标执行"的静默降权。

## 5. 失败复盘：live 验证暴露的引用不全问题

首次 live Ollama（qwen2.5:7b）运行中，格式化模型输出的 evidence_ids 只含振动测点、漏选了设备阈值条目，振动门控按设计拒绝："vibration diagnosis requires selected device threshold evidence"。门控本身正确（fail closed 生效），但正常路径被阻断。

处置不是放宽门控，而是新增 `repair_vibration_selection`：当 draft 已引用至少一个振动点时，由程序把注册表中全部振动点与唯一一致的阈值条目并入引用集合。修复只追加程序采集的事实、绝不编造；矛盾阈值不会被自动注入；repair 后仍完整执行门控；若 repair 产物违反 Schema（如超过 20 个 ID 上限）则回退原 draft 交由门控显式拒绝。对应回归测试覆盖：缺阈值、漏测点、低风险冲突仍拒、无振动引用不动、阈值矛盾不注入。

第二次 live 运行走通全图：4 条证据（3 sensor + 1 device）、repair 生效、输出合法报告。模型给出 `evidence_sufficient=true` 且 `risk_level=unknown` 属于其保守判断并在 limitations 中说明了理由，Schema 允许该组合，不属于程序缺陷。

## 6. 验证证据与边界

以下为 2026-08-23 实际执行的验证：

- 默认离线套件 128 项收集：126 通过，2 项 live 测试按设计跳过；连续三次运行稳定（约 2.3 秒）。
- live Ollama smoke 三次（两次失败暴露问题并修复、一次通过），命令与结果见 §5；本地免费模型，无付费调用，无外部副作用。
- 第一轮独立只读审查（PARTIAL）发现的 4 项 P2 全部关闭：路由测试改为冻结字面量期望并整表比对；join 冲突检测补直接单测；repair 增加 ValidationError 回退；metrics 收紧为单值。3 项 P3 一并修复（防御默认值统一、CLI 空问题校验、附带回归断言）。延期 3 项 P3：异常分类细化（阶段3 错误路径统一处理）、gate 对缺失 facts 的裸 KeyError（上游构造保证存在）、deserialize 快照一致性校验（状态仅程序写入，阶段3 引入持久化恢复时再加固）。
- 未验证：远端 LangSmith Trace 上传、DeepSeek provider 的图模式 live 调用、alarm/work_order/manual 三类的完整图级 live 流程。

## 7. 学习练习与验收答案

1. **为什么并行键必须有 Reducer？** 答：多个节点在同一超步写同一键时，默认 last_value 只接受一个值，会抛 `InvalidUpdateError`；用 `Annotated[list, merge_fn]` 声明合并策略后框架才会归并更新。
2. **fan-out 如何实现？** 答：条件边路由函数返回节点名列表；列表成员在同一超步并行，之后各自连到 join 节点汇合。
3. **为什么 out_of_scope 不调用任何工具？** 答：`route_after_plan` 直接把非 in_scope 计划送到 format_report；工具只能从 dispatch 可达。
4. **repair 会引入模型编造的事实吗？** 答：不会。它只追加注册表已有条目 ID；事实本体在 finalize 时始终由注册表注入。
5. **recursion_limit 为什么还要设？** 答：当前图最多 6 步，25 是防未来拓扑改动失控的兜底上限，属于代码级而非提示词级的循环保护。

## 8. 明确未实现与后续阶段

Checkpoint、`thread_id` 持久化、Interrupt/人工审批与恢复幂等属于第三阶段；RAG 与记忆属于第四阶段；LangSmith Dataset/Evaluator 固定评测属于第五阶段。当前图无检查点，`thread_id` 仍仅用于 Trace 关联。
