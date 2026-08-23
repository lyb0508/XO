"""Read-only deterministic industrial data tools.

工具包的对外出口：上层模块只需 ``from app.tools import INDUSTRIAL_TOOLS``
即可获得全部五个只读工业数据工具；写动作（mock_actions）刻意不在此处
导出，避免被当成普通查询工具随手挂载。
"""

from app.tools.industrial import INDUSTRIAL_TOOLS

__all__ = ["INDUSTRIAL_TOOLS"]

