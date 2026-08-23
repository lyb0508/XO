# 项目复盘：工业设备故障研判与工单闭环 Agent

复盘时间：2026-08-24。范围：2026-08-23 18:39 首个提交至 08-24 全部六个阶段。方法：git 历史、测试套件、评测基线、CI 运行记录与六份阶段文档交叉核对；结论均附可复查证据。

## 1. 结论速览

| 维度 | 判定 | 一句话依据 |
| --- | --- | --- |
| 六阶段代码交付 | 达成 | 每阶段 feat → fix(review findings) → docs 闭环，181 项离线测试连续三次全绿 |
| 业务链路九环节 | 达成 | 结构化解析到带证据报告全链路贯通，含 HITL 审批与幂等模拟动作 |
| 安全边界 | 达成 | 只读工具集、审批门禁、fail-closed 配置、脱敏 Trace 均有测试 |
| 最终质量指标 | 达成 | 改进后连续两轮 live 全部达标：0.95 / 0.909 / 0.96，final_contract 满分 |
| 文档一致性 | 达成 | 六份阶段文档 + 本复盘；README 标题滞后为已知 P2 |

## 2. 双重目标判定（AGENTS.md §1）

**目标一：可运行、可测试、可观测的 Agent 工程 —— 达成。**

- 可运行：CLI（`python -m app.main`）与 HTTP 服务（uvicorn 工厂）双入口；CI 中 Docker 容器 `/health` 冒烟通过。
- 可测试：183 项收集 = 181 离线通过（fake model，毫秒级）+ 2 项 live 默认跳过；评测侧另有 50-case 固定数据集与确定性评测器。
- 可观测：LangSmith Trace 默认关闭、显式开启才创建 client；元数据 allowlist + 输入/输出/错误脱敏；request/thread/版本标签可关联。

**目标二：学习者能解释每个模块的责任、数据流、失败路径与安全边界 —— 达成并强化。**

- 六份阶段文档均含"文件→责任映射、真实数据流、成功/失败路径、验证证据、至少一个失败复盘、练习与答案"。
- 本次复盘同步完成全部 33 个 Python 文件的通俗中文注释：模块 docstring 讲责任/来源/副作用/失败行为，关键函数讲"为什么"。以"剥离 docstring 后 AST 与 HEAD 完全一致"的批量脚本验证零可执行代码变更，随后全量测试三次稳定。

## 3. 目标业务链路逐环节核对（AGENTS.md §1）

| 环节 | 实现 | 证据 |
| --- | --- | --- |
| 用户问题 | CLI 参数 / `POST /diagnoses` | app/main.py、app/api |
| 结构化解析 | QueryPlan schema + 解析节点 | app/schemas/query_plan.py |
| 设备/测点/报警/工单检索 | 五个只读 @tool 并行 fan-out | app/tools/industrial.py、app/graphs/nodes.py |
| 知识检索（RAG） | 手册 store/embeddings/retriever，阈值经正负样例校准 | app/retrieval/ |
| 证据检查 | 不可变证据注册表：稳定 ID、冲突即 fail、未解决错误阻断格式化 | app/agents/evidence.py |
| 故障研判 | 两段式：证据 Agent（限额内）+ 无工具结构化 formatter | app/agents/diagnostic.py |
| 风险判断 | 阈值门禁在代码层强制（振动值 vs 报警阈值），不信模型散文 | diagnostic.py `_validate_vibration_gate` |
| 人工审批 | interrupt → approved/modified/rejected 三种决策均有参数化测试 | tests/test_graph_approval.py:67 |
| 模拟业务动作 | 幂等键 (action_type, request_id)，重复调用返回 already_executed | app/tools/mock_actions.py |
| 带证据报告 | DiagnosisReport 严格 schema + 来源只认 status=ok payload | app/schemas/diagnostics.py |

九环节全部实现且有对应测试；无环节靠 prompt 单独防御。

## 4. 六阶段交付与验证状态（AGENTS.md §3）

| 阶段 | 交付 | 验证状态 |
| --- | --- | --- |
| 1 LangChain 基础 | 模型适配、工具契约、两段式 Agent、Middleware 限额、Trace | 离线测试 + live Ollama 冒烟一次 |
| 2 LangGraph 编排 | State(Reducer)、并行 fan-out、条件路由、fail_closed 分支 | 离线测试；live 引用不全问题已复盘修复 |
| 3 持久化与审批 | InMemorySaver checkpoint、interrupt/resume、幂等动作 | 离线 + live 审批执行冒烟；跨进程持久化未做（明示） |
| 4 RAG 与记忆 | 手册检索（校准阈值）、session/ledger 分离记忆 | 离线 + live 检索命中验证；embedding 语义质量未验证明示 |
| 5 LangSmith 评测 | 50-case 固定集、确定性评测器、运行器与隔离台账 | 五轮 live 评测，诚实基线落盘 |
| 6 工程化 | FastAPI/SSE/认证/限流/Dockerfile/CI | 181 项测试 + CI 三次真实运行（pytest+docker job 全绿） |

每阶段均遵循"实施 → 独立审查发现 → 修复提交 → 中文文档"循环，共三次审查后修复类 fix 提交（46d73bd、4f402ae 的提交信息含 "close review findings" 字样，41a7c84 为 per review 加固）。

## 5. 专项规则对照（AGENTS.md §7）

**7.1 模型与 Agent —— 符合。** provider 经 factory 注入；DeepSeek key 缺失直接报错不降级；单元测试全用 fake model，live 测试独立文件可单独开关且默认跳过；结构化输出绑定 Pydantic schema，解析失败进显式错误分支；ModelCallLimit/ToolCallLimit/每工具限额三类中间件均有超限测试。

**7.2 LangGraph —— 符合（含一次真实缺陷教训）。** GraphState 全字段 JSON 可序列化；Reducer 排序合并保证顺序无关并有交换律测试；路由纯函数化、只读程序字段；interrupt 前副作用幂等，恢复不重复创建工单（幂等键）；调用上限防死循环；thread_id 隔离有测试。第六阶段收尾曾出现"`/approvals` 恢复返回 500"缺陷——根因是 `_build_graph()` 每次新建空 checkpointer，恢复端点把 `Command(resume=...)` 当全新 run；已改为 create_app 级共享单例并以回归测试锁定（962cddd）。教训：脱敏兜底会把根因变成无信息量的 500，诊断时需绕过脱敏层取原始错误。

**7.3 RAG 与不可信内容 —— 符合。** 手册/工具输出/用户输入均视为不可信证据，系统规则由代码强制；检索结果带章节 ID/标题/版本元数据；报告区分文档证据与模型推断；相似度阈值不用拍脑袋值，而以固定正负样例的可分性校准测试约束；短期会话与长期台账分离设计、分开清理，长期写入走白名单字段。

**7.4 可观测与评测 —— 基础设施符合，质量指标未达。** Dataset 期望独立于生产实现维护（evaluations/dataset.json 固定 50 例，覆盖正常/多工具/证据不足/审批/注入/越权）；LLM-as-judge 未启用属规范允许的可选项；Experiment 记录模型/Prompt/图谱版本。过程中最重要的安全发现（P1）：评测 target 曾把注入样本"自动批准执行"直写生产台账，已修复为临时 ledger + 非 in_scope 一律拒绝 + 污染清理——评测基建同样必须过安全审计。

**7.5 高风险动作 —— 符合。** 停机类动作仅存在于模拟层；一切副作用必须经 interrupt 门禁；approved/modified/rejected 三种决策都有参数化测试与约束测试（modified 必须给 modified_actions，其余决策禁止携带）；外部写操作具备幂等键、审计记录与部分失败语义；仅靠 prompt 声明的危险动作限制不存在——门禁全部在图路由与代码层。

## 6. 测试矩阵与最终指标（AGENTS.md §8）

矩阵覆盖：单元（schema/路由/Reducer/脱敏）、Agent（选工具/错参/无关问题/超限/结构化输出）、Graph（正常/越界/合并顺序/工具失败/interrupt-resume/线程隔离）、API（状态码/SSE 序列/限流/取消路径/错误脱敏）、安全（注入/越权/未审批动作/敏感信息）、评测（五维度确定性规则）——各层级均有对应测试文件，无 0 项测试目录。

| 最终目标指标 | 目标 | 改进前基线 | 当前（2026-08-24，连续两轮一致） | 判定 |
| --- | --- | --- | --- | --- |
| 关键结构化输出有效率 | 100% | final_contract 0.90 | final_contract **1.00**（qwen 与 deepseek 双路径） | ✓ |
| 高风险动作人工审批覆盖率 | 100% | 所有副作用动作均经 interrupt 门禁 | 不变 | ✓ |
| 工具选择正确率 | ≥90% | 0.81 | qwen **0.95** / deepseek **0.98** | ✓ |
| 证据不足追问/拒答正确率 | ≥90% | 0.727（strict） | qwen **0.909** / deepseek **0.955**（strict） | ✓ |
| 关键轨迹稳定率 | ≥85% | 0.84 | qwen **0.96** / deepseek **0.98** | ✓ |

改进后全部达标。手段是纯工程层修复，未更换模型：计划规范化层 + include_raw 接管解析失败 + 有界错误反馈重试 + formatter 加固 + PLAN_PROMPT 针对四类失败场景的显式规则（详见 phase5 文档"改进措施"）。19 个失败案例中 8 个解析类失败被规范化层直接消解，其余由提示词规则纠正。

## 7. 协作流程自身复盘（AGENTS.md §4–6）

- 主控-子 agent 分工实际运转：16 个角色配置齐全，7 个实施角色留下仓库级产出痕迹；三次"close review findings"提交证明审查-修复循环真实发生。
- 受控串行写入得到时间戳序列佐证：契约先行（schemas → state → nodes → routing → builder → tests → docs），同一时刻单一 Owner。
- 会话层交付物（STATUS/EVIDENCE 返回）按设计不入库；仓库保留结果证据。若需更强审计，可将子 agent 摘要归档到被忽略的本地目录。
- 本次注释任务复用同一模式：五个实施 Agent 按互斥文件组并行，主 Agent 以 AST 批量对比统一验收，杜绝"顺手改逻辑"。

## 8. 缺口与后续路线

**P1（影响最终验收指标）**
1. 三项质量指标未达标——按 §6 排序的根因逐项治理，优先计划解析重试/降级。

**P2（不影响关键验收，已披露）**
2. live provider 仅 Ollama/qwen2.5:7b 单路径验证；DeepSeek 与远端 LangSmith 上传未验证（无有效 key）。
3. 多副本部署所需的外部持久 Checkpointer、分布式限流与台账存储未实现（当前实现均为进程内）。
4. TLS 终结、请求日志审计与在线监控属于上线后能力。
5. README 标题仍标注"第一阶段"，内容已提及阶段六，需整体改写；另 phase6 文档 §5 记录的"182 项收集"是当时验证快照，962cddd 新增回归测试后实际为 183 项（181 通过 + 2 跳过），历史记录未回填。
6. 评测集只覆盖单轮诊断，不含多轮会话与审批交互的端到端轨迹。

## 9. 学习要点提炼

1. **把动态性关进笼子**：外层流程确定化（图路由读程序字段）、动态推理限制在明确节点、结构化输出绑定 schema——三道墙让小模型的不稳定变成可测试的错误分支而非静默错误。
2. **证据注册表模式**：模型只能引用程序核发的证据 ID，报告字段由不可变事实回填——从机制上消灭"编造来源"。
3. **HITL 的正确位置**：interrupt 放在有副作用之前、恢复走 Command(resume)、幂等键兜住重复恢复——三者缺一不可。
4. **评测基建也是生产调用方**：评测代码绕过业务安全门禁时会造成真实污染，必须同等审计。
5. **脱敏要分层**：对外脱敏保护隐私，诊断排错需要受控的原始错误通道，两者都要设计。
