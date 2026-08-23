# 第五阶段：LangSmith 评测

## 1. 这一阶段解决什么问题

前四个阶段的能力都只有"单次验证"：一次 live smoke 通过不代表五十种问法都通过。第五阶段建立固定评测集、确定性评测器与可重复的实验运行器，把系统质量从"感觉可用"变成可量化、可对比、可追踪的数字，并如实暴露当前本地小模型的真实水平。

## 2. 文件责任

| 文件 | 责任 |
| --- | --- |
| `evaluations/dataset.json` | **手工维护**的 50 条固定样例与期望值，8 类场景；期望从不从生产代码派生（`tests/test_evaluation_pipeline.py` 校验其结构与规模）。 |
| `evaluations/target.py` | 评测目标函数：跑完整图并输出标准化中间产物（范围/计划/轨迹/报告契约）；单例崩溃转为显式 error 输出而非缺口。 |
| `evaluations/evaluators.py` | 六个确定性评测器：工具选择（全对1/子集0.5/错0）、范围分类（拒答类两值等价）、最终契约、拒答行为（仅对拒答类计分）、轨迹（must_include/exclude）、安全。 |
| `evaluations/run_evaluation.py` | 运行器：默认 `upload_results=False` 纯本地运行；`--upload` 才需要 LangSmith key；`--report-file` 落盘含逐案明细的完整报告；本地模式清理 LangSmith 环境变量并静音其日志。 |
| `tests/test_evaluation_pipeline.py` | 数据集结构/规模守卫、ID 稳定性、各评测器打分边界、汇总口径（not-applicable 不稀释拒答指标）。 |

## 3. 运行方式与指标口径

```powershell
python -m evaluations.run_evaluation                       # 全量 50 条，live Ollama
python -m evaluations.run_evaluation --limit 3             # 冒烟
python -m evaluations.run_evaluation --upload              # 需要有效 LANGSMITH_API_KEY
```

指标定义（对照 AGENTS.md §8 最终目标）：

| 指标 | 含义 | 目标 | 基线（干净运行） | 改进后（2026-08-24） |
| --- | --- | --- | --- | --- |
| tool_selection | 计划证据类型集合 vs 期望（全对1/子集0.5/越界0） | ≥0.90 | **0.81** ✗ | **0.95** ✓ |
| refusal_behavior (strict) | 拒答类场景正确追问或拒答（排除不适用样例） | ≥0.90 | **0.727** ✗ | **0.909** ✓ |
| trajectory | 实际触达的数据源满足 must_include/exclude | ≥0.85 | **0.84** ✗ | **0.96** ✓ |
| security | 注入/越权样本的计划面受控 | — | 0.88 | 0.96 |
| scope_classification / final_contract | 范围分类 / 报告契约 | — | 0.68 / 0.90 | 0.92 / 1.00 |

改进措施（2026-08-24 实施，连续两轮全量 live 运行结果完全一致、零 error case）：

1. **计划规范化层**（`_normalize_plan_payload`）：在校验前由程序执行提示词已声明的约定——非 in_scope 计划清空全部证据字段而非拒绝；sensor 请求缺 metrics 时补默认 `vibration_mm_s`；in_scope 历史请求缺显式时间窗时降级为 needs_clarification。根因是"语义正确但违反反向约束"的输出被 schema 打成解析失败。
2. **include_raw=True 接管解析失败**：`with_structured_output` 默认在内部校验并抛 OutputParserException，规范化层没有执行机会；改为 include_raw 契约后，解析失败返回原始输出，程序提取 JSON → 规范化 → 本地校验。
3. **有界错误反馈重试**：首次失败把真实错误回传模型再试一次（官方推荐模式），仅一层、不嵌套。
4. **formatter 加固**：evidence_ids 保序去重；high/critical 风险强制 requires_human_review=True（朝更严格方向）。
5. PLAN_PROMPT 增补四类高频失败场景的显式范围规则（设备档案/手册检索属 in_scope、模糊时间窗必须追问、多工具请求必须完整列举）。

## 4. 基线结果与根因分析（50 条 live，qwen2.5:7b）

干净运行（隔离台账 + 非 in_scope 自动拒绝后）共 19 个失败案例，归因：

1. **计划解析失败 ×5**：`OutputParserException`——小模型生成的 QueryPlan JSON 不合规（跨轮次波动于 5–7 条）。单一最大失败源。
2. **device_info 场景 4/4 失败**：模型把纯设备信息查询判为 out_of_scope 或 needs_clarification——对"不需要澄清的简单查询"过度保守。
3. **insufficient 场景 4/6 失败**："昨天""上周"这类模糊时间窗未被澄清而是被自行假设后直接查询。
4. **manual_only 3/4 失败**：手册检索请求被误判需澄清。
5. **multi_tool 子集偏差 2 条**：多请求 work_order 或漏请求 device/manual。

结论：**评测基础设施达标（本阶段交付），三项最终目标指标未达标且差距已被精确量化**。AGENTS.md §8 明确这些是"项目最终阶段"的目标值；本阶段的门禁是建成可信的测量体系并给出诚实基线——0.81 的工具选择正确率就是后续迭代的起点。改进方向按预期收益排序：a) 计划输出重试/降级策略（可回收约 10% 失败）；b) PLAN_PROMPT 针对 device_info/manual 场景的显式引导；c) 更强的规划模型或 function_calling 方法迁移。

## 5. 过程复盘

- `_expected()` 曾在已扁平的期望对象上再找一层 `"expected"` 键，导致首轮全量所有期望读空；单元测试当时编码了同样的错误嵌套形状而未拦截。修复同时更正了两处——教训：测试夹具的形状必须从真实数据流推导，而不是从实现反推。
- `upload_results=False` 模式下 `run.outputs` 不被填充，逐案明细最初全是 None；改为在评测器层捕获目标输出。
- **最重要的一次审查发现（P1）**：评测 target 的自动批准曾把越权/注入/无关样本也"批准执行"并直写生产长期记忆台账——实证积累了 33 条 eval-runner 记录，且会经规划节点的记忆注入泄入后续实验。修复为三重防线：评测 ledger 指向一次性临时文件、非 in_scope 计划一律自动拒绝而非批准、清理全部污染记录（保留 1 条真实人工审批）。教训：评测基建本身也是生产路径的调用方，必须接受与业务代码相同的安全审计。
- inj-002 期望修正：混合注入（合法查询+恶意指令）的整体拒绝在安全上完全正确且更强，原"只认可执行合法部分"的定义过窄，已在数据集中修正。

## 6. 验证证据与边界

以下为 2026-08-23 实际执行的验证：

- 默认离线套件 172 项收集：170 通过，2 项 live 测试默认跳过。
- 全量 live 评测共执行五轮（两轮冒烟 + 三轮全量）；最终基线以上表为准，报告含逐案明细落盘 `tmp/eval_report.json`（gitignore 内生成物）。
- P1 修复后的干净运行验证：生产台账前后行数不变（1 条真实记录），评测写入与生产完全隔离。
- LLM-as-judge 未启用：本版以确定性规则覆盖全部维度，同模型自评的偏见风险不值得在基线阶段引入（AGENTS.md 允许 judge 为可选）。
- 未验证：--upload 真实上传（无有效 key）、DeepSeek provider 下的评测表现、跨轮次稳定性统计（需多次全量运行对比）。

## 7. 明确未实现与后续阶段

在线评测与生产监控属于第六阶段之后；FastAPI/SSE/权限/限流/部署尚未实现。评测集目前只覆盖单轮诊断，不含多轮会话与审批交互路径的端到端轨迹评测。
