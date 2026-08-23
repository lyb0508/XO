"""Uvicorn entrypoint for container platforms that ignore --factory.

有些容器平台或托管环境不支持 ``uvicorn app.api:create_app --factory`` 的工厂
写法，只接受一个现成的模块级 ``app`` 对象。本文件在导入时调用工厂生成应用
实例，供 ``uvicorn app.asgi:app`` 直接引用。

注意导入即执行 create_app()：由于未配置 INDUSTRIAL_AGENT_API_KEY 时工厂会
直接抛错（fail-closed），"导入失败"本身就是这道安全闸的一部分。
"""

from app.api import create_app

app = create_app()
