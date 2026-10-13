"""LangGraph 图：状态、节点、工具、组装与调用封装。

分层顺序（自下而上）::

    schemas.py   结构化输出模型（Pydantic）
    state.py     图状态（TypedDict + reducer）
    events.py    SSE 事件构造（前端契约的唯一出口）
    tools.py     web_search / note 系列工具
    llm.py       ChatModel 工厂 + 结构化调用降级
    nodes.py     节点实现
    builder.py   StateGraph 组装
    runner.py    面向 FastAPI 的调用封装（把图事件翻译成前端事件）
"""

from graph.builder import build_research_graph, get_research_graph, graph_mermaid
from graph.nodes import ResearchContext
from graph.state import ResearchState, TaskResult, TodoItem, initial_state

__all__ = [
    "ResearchContext",
    "ResearchState",
    "TaskResult",
    "TodoItem",
    "build_research_graph",
    "get_research_graph",
    "graph_mermaid",
    "initial_state",
]
