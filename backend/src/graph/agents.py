"""ReAct 子图工厂：同时兼容新旧两套 Agent 构造 API。

背景
----
LangGraph 1.x 起，``langgraph.prebuilt.create_react_agent`` 被标记为
**Deprecated since V1.0, removed in V2.0**，官方迁移目标是把 Agent 循环收归到
``langchain.agents.create_agent``（``langchain>=1.0`` 提供）。

两个 API 的差别很小但会直接影响调用：

====================== ========================== ==========================
项目                    langgraph.prebuilt         langchain.agents
====================== ========================== ==========================
函数名                  ``create_react_agent``     ``create_agent``
系统提示词参数          ``prompt=``                ``system_prompt=``
状态结构                ``{"messages": [...]}``    ``{"messages": [...]}``
====================== ========================== ==========================

本模块把它们收敛成一个 ``build_react_agent``：装了 ``langchain`` 就用新 API
（无弃用告警、面向未来），只用 ``langgraph`` 就退回旧 API（保持最小依赖可运行）。
上层节点不需要知道当前跑在哪一种上。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

#: 当前生效的 API 来源：langchain | langgraph | 空（尚未构建过）
_API_MODE: str = ""


def build_react_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """构造一个 ReAct 工具循环子图（返回已编译图）。

    Args:
        llm: 聊天模型。
        tools: 该 Agent 可用的工具（**按需裁剪**，用不到的工具只会增加误调用概率）。
        system_prompt: 系统提示词。
        name: 子图名称，出现在 trace 与图结构里。
    """

    global _API_MODE

    if _API_MODE in ("", "langchain"):
        try:
            from langchain.agents import create_agent  # type: ignore[import-not-found]

            kwargs: dict[str, Any] = {}
            if system_prompt:
                kwargs["system_prompt"] = system_prompt
            if name:
                kwargs["name"] = name
            _API_MODE = "langchain"
            return create_agent(llm, list(tools), **kwargs)
        except ImportError:
            if _API_MODE == "langchain":  # 之前成功过，现在失败属于真异常
                raise
            logger.info(
                "未安装 langchain 主包，使用 langgraph.prebuilt.create_react_agent"
                "（该 API 在 LangGraph V2 中会移除；`pip install langchain` 可切到 create_agent）"
            )

    from langgraph.prebuilt import create_react_agent

    kwargs = {}
    if system_prompt:
        kwargs["prompt"] = system_prompt
    if name:
        kwargs["name"] = name
    _API_MODE = "langgraph"
    return create_react_agent(llm, list(tools), **kwargs)


def agent_api_mode() -> str:
    """返回当前生效的 Agent 构造 API（供启动日志与体检脚本展示）。"""

    return _API_MODE or "unknown"
