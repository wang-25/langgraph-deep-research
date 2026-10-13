"""StateGraph 组装。

图结构
------
::

                              ┌──────────────┐
    START ───────────────────►│  plan_todos  │   结构化输出任务清单
                              └──────┬───────┘
                                     │  fan_out_tasks 返回 N 个 Send
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ research_task 1 │     │ research_task 2 │ ... │ research_task N │   同一 superstep 并发
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             └───────────────────────┼───────────────────────┘
                                     ▼  fan-in（全部完成后才执行）
                              ┌──────────────┐
                              │ write_report │   整合结构化结果，流式产出报告
                              └──────┬───────┘
                                     ▼
                              ┌────────────────┐
                              │ persist_report │   结论笔记落盘（幂等）
                              └──────┬─────────┘
                                     ▼
                                    END

为什么用图而不是「顺序调用 + 线程池」
------------------------------------
旧版 ``run_stream`` 是这样写的：手工起 N 个 ``Thread``，各跑一遍
「搜索 → 总结」，用 ``Queue`` 收集事件，再用 ``Lock`` 保护共享的 state
（``backend/src/agent.py:201-265``）。它能跑，但有三处隐性成本：

1. **并发上限写在代码里**，调节并发要改逻辑；
2. **归并语义靠人肉保证**：``web_research_results`` 这些列表的 append 顺序、
   计数器的自增都得自己加锁，否则就是数据竞争；
3. **失败处理是特例代码**：某个任务抛异常要用 try/except 包住整段 worker。

换成图之后，这三件事分别变成：``Send`` 扇出（并发由运行时调度）、
字段上的 ``operator.add`` reducer（归并语义声明式）、
节点内的本地 try/except（失败不影响其他分支）。
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import (
    ResearchContext,
    fan_out_tasks,
    persist_report,
    plan_todos,
    research_task,
    write_report,
)
from graph.state import ResearchState

logger = logging.getLogger(__name__)

#: 节点名常量（条件边的 path_map 里也用它们，避免手写字符串拼错）
NODE_PLAN = "plan_todos"
NODE_RESEARCH = "research_task"
NODE_REPORT = "write_report"
NODE_PERSIST = "persist_report"

_graph: Optional[CompiledStateGraph] = None


def build_research_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """组装并编译深度研究图。

    Args:
        checkpointer: 断点存储器。本版本默认不用（研究流程是一次性任务），
            但接口保留——接上 ``InMemorySaver`` / ``AsyncSqliteSaver`` 之后，
            「进程重启继续跑」和「interrupt() 人工介入」都能直接启用，
            图结构一行都不用改。
    """

    builder = StateGraph(ResearchState, context_schema=ResearchContext)

    builder.add_node(NODE_PLAN, plan_todos)
    builder.add_node(NODE_RESEARCH, research_task)
    builder.add_node(NODE_REPORT, write_report)
    builder.add_node(NODE_PERSIST, persist_report)

    builder.add_edge(START, NODE_PLAN)

    # ---- 动态扇出：条件边直接返回 Send 列表，N 个任务进同一个 superstep ----
    builder.add_conditional_edges(NODE_PLAN, fan_out_tasks, [NODE_RESEARCH])

    # ---- fan-in：所有 research_task 分支都指向 write_report，它会等它们全部结束 ----
    builder.add_edge(NODE_RESEARCH, NODE_REPORT)
    builder.add_edge(NODE_REPORT, NODE_PERSIST)
    builder.add_edge(NODE_PERSIST, END)

    return builder.compile(checkpointer=checkpointer)


def get_research_graph() -> CompiledStateGraph:
    """进程内共享的已编译图（编译有成本，且图是无状态的）。"""

    global _graph
    if _graph is None:
        _graph = build_research_graph()
        logger.info(
            "LangGraph 深度研究图编译完成，节点: %s", sorted(_graph.get_graph().nodes)
        )
    return _graph


def reset_research_graph() -> None:
    """丢弃已编译的图（测试或切换 checkpointer 时用）。"""

    global _graph
    _graph = None


def graph_mermaid() -> str:
    """导出 Mermaid 源码。

    直接来自**真实图结构**而不是手画：``README.md`` 里的架构图就是这个函数
    的输出，改了边就会同步变化，不会文档漂移。
    """

    return get_research_graph().get_graph().draw_mermaid()
