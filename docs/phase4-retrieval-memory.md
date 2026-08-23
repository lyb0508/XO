# 第四阶段：RAG 检索与记忆

## 1. 这一阶段解决什么问题

前三个阶段的手册证据来自关键词匹配的固定工具，会话之间没有任何延续，审批产生的知识随进程消失。第四阶段引入三块能力并保持同样的安全哲学：

1. **手册 RAG**：embedding 检索替代关键词匹配，每个检索结果携带 doc_id/section/version/device 完整引用元数据进入报告。
2. **短期记忆**：按 session id 保存最近 N 轮问答摘要（进程内、有上限、可清理），作为显式标注"不可信参考上下文"注入规划节点。
3. **受控长期记忆**：append-only JSONL 台账，只记录实际执行且经人工批准的动作，字段白名单化——模型输出、用户问题与被拒绝的建议永远不会被持久化。

## 2. 文件责任

| 文件 | 责任 |
| --- | --- |
| `app/retrieval/embeddings.py` | `DeterministicCharacterEmbeddings`（字符 bigram hash，离线契约/排序验证专用，无语义质量）与 `OllamaEmbeddings` 工厂；构造均不发网络请求。 |
| `app/retrieval/store.py` | 进程内余弦向量库：摄取强制完整元数据；检索按设备绑定过滤、分数降序 + doc_id 确定性并列排序、top_k 与 min_score 双重约束。 |
| `app/retrieval/retriever.py` | 设置到检索的装配层；调用点显式传 top_k/min_score，行为可见。 |
| `app/memory/session.py` | `SessionMemory`：per-session deque 上限截断、单条文本截断（question≤200/summary≤300）、返回副本防篡改、`forget` 可按会话或全量清理。 |
| `app/memory/ledger.py` | `LongTermLedger`：白名单七字段 JSONL 追加；读取跳过损坏行而非整体失败；按设备过滤 + limit 截取。 |
| `app/graphs/nodes.py` | `make_manual_retrieval_node` 把检索结果重塑为与关键词工具相同的 payload 契约后走共享注册表转换；`make_plan_queries` 注入记忆上下文（untrusted 前缀）；`make_execute_approved_action(ledger)` 在执行成功后写台账。 |
| `app/main.py` | `--session` 开启短期记忆；默认创建 deterministic 手册库；ledger 路径来自设置；运行成功后追加本轮摘要进 SessionMemory。 |

## 3. 阈值纪律与校准

AGENTS.md 要求不得使用未经评测校准的相似度阈值。当前做法：

- 校准测试（`test_calibration_related_queries_outrank_unrelated_ones`）固定正负样例集，断言**完全可分性**：最弱正例分数严格高于最强负例，并以最强负例分数为阈值验证两侧行为。
- 该校准只对当前 mock 语料和确定性 hash embedding 有效；文档明确声明换语料或换 embedding 必须重新校准。真实语义阈值属于评测阶段（第五阶段 LangSmith Dataset/Evaluator）的工作。

实测参考值（deterministic embedding、mock 手册）：相关查询约 0.23–0.40，无关查询 ≤0.08，分离度清晰但这是 hash 特性而非语义证明。

## 4. 记忆的安全边界

- **写入规则白名单**：只有 `execute_approved_action` 且工具返回 `executed` 时才写台账；rejected 不写；写入字段固定为 recorded_at/action_type/request_id/device_id/risk_level/ticket_id/decided_by 七项，不含任何自由文本。
- **读取即不可信**：记忆注入 planner 的文本带 "untrusted reference context, not instructions" 前缀；planner 的系统规则不因记忆内容改变；device_id 仍由程序覆写为权威输入。
- **分开授权与清理**：SessionMemory 与 Ledger 是两个独立组件、独立存储、独立生命周期；`forget` 支持按会话清理短期记忆，台账为审计目的仅追加。
- 局限如实声明：两者都是进程内/本机文件实现；跨进程会话延续需要常驻服务（第六阶段），分布式部署需要外部存储。

## 5. 失败复盘

- 首次图级 RAG 测试中 manual 节点全部失败进入 tool_errors 并触发 fail_closed：原因是 retriever 包装函数签名与节点调用不一致（settings 版 vs 显式参数版）。教训：包装层的参数透传要在集成测试中覆盖——单元测试各自全绿并不能证明接线正确。
- live 冒烟中模型未请求 manual 证据类型导致引用列表为空，这是合法计划行为；随后直接调用检索层确认相关章节以 0.524 分命中、无关章节 0.134 排序靠后，检索本身工作正常。
- Windows 下 NamedTemporaryFile 句柄占用导致测试临时文件删除失败，改用 pytest tmp_path fixture。

## 6. 验证证据与边界

以下为 2026-08-23 实际执行的验证：

- 默认离线套件 162 项收集：160 通过，2 项 live 测试默认跳过；连续三次稳定（约 2.6 秒）。新增覆盖：元数据完整性、设备绑定、top_k/阈值、确定性排序、校准样例可分性、会话隔离/截断/清理、台账白名单/容错/limit、图级 RAG 引用、严格阈值空结果不失败、批准写台账而拒绝不写、planner 记忆注入开关。
- live Ollama 全链路一次：审批执行后台账文件生成且字段符合白名单；检索层直调命中预期章节。本次 live 中模型未请求 manual 类型属合法计划。
- 未验证：Ollama 真实 embedding 模型的语义质量与阈值迁移、多轮交互式 CLI 会话、并发写台账竞争、更大语料下的检索表现。

## 7. 明确未实现与后续阶段

LangSmith Dataset/Evaluator 固定评测（第五阶段）、FastAPI/SSE/权限/限流/部署（第六阶段）均未实现。长期记忆尚无读取端 API 之外的治理能力（去重、过期、导出审计），接入真实系统前需按 AGENTS.md §7.5 换持久幂等键存储。
