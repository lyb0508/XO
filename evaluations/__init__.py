"""Phase-five evaluation suite: datasets, evaluators, and the runner.

本包是第五阶段评测套件的入口。整体按四个环节组织：

1. 固定数据集（``dataset.json``）：手工维护的输入与期望值，期望值完全独立于
   生产代码，避免“用实现生成答案再考实现”的循环论证。
2. 目标函数（``target.py``）：唯一知道如何驱动诊断 Graph 的模块，把一次完整
   运行投影成可比较的结构化结果。
3. 确定性 Evaluator（``evaluators.py``）：只依赖目标函数输出与数据集期望值
   的纯函数，不访问网络、不调用模型，可离线单测。
4. 运行器（``run_evaluation.py``）：本地优先执行整套评测并汇总报告；
   仅当显式传入 ``--upload`` 且环境中配置了 LANGSMITH_API_KEY 时，
   才会把结果作为真实 Experiment 上传到 LangSmith。

学习入口建议按 1 → 2 → 3 → 4 的顺序阅读，对应“考什么、怎么跑、怎么判、怎么汇”。
"""
