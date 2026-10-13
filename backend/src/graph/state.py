"""图状态定义。

与旧版 ``backend/src/models.py`` 的对比
---------------------------------------
旧版是 dataclass，且并行靠 ``threading.Thread`` + ``Queue`` 手工汇总，
所以 ``SummaryState`` 既当「数据容器」又当「线程间共享内存」，还要自己加
``Lock``（``agent.py:354`` 的 ``with self._state_lock``）。

这里换成 ``TypedDict`` + **reducer**，把「怎么合并」声明在字段上：

* ``task_results`` / ``sources_gathered`` / ``model_calls`` 标了
  ``Annotated[..., operator.add]``：N 个子任务节点在同一 superstep 并发写，
  LangGraph 在 fan-in 时自动归并，**不需要锁，也不需要顺序假设**；
* ``todos`` / ``report`` 这类单值字段不加 reducer（last-write-wins），
  因为它们只会被一个节点写一次。

额外好处：TypedDict 天然可 JSON 序列化，将来接 checkpointer 存断点不用改结构。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class TodoItem(TypedDict, total=False):
    """规划阶段产出的任务（图状态里的可序列化形态）。"""

    id: int
    title: str
    intent: str
    query: str
    status: str
    note_id: Optional[str]
    note_path: Optional[str]


class TaskResult(TypedDict, total=False):
    """一个子任务的执行结果，由 ``research_task`` 节点产出。"""

    task_id: int
    title: str
    status: str  # completed | skipped | failed
    summary: str
    key_findings: list[str]
    not_verified: list[str]
    confidence: str
    sources_summary: str
    raw_context: str
    backend: str
    note_id: Optional[str]
    note_path: Optional[str]
    error: Optional[str]


class ResearchState(TypedDict, total=False):
    """整张图的状态。

    ``total=False`` 让每个字段都是可选的，节点函数因此可以只返回「增量」，
    而不是构造完整状态——这也是 LangGraph 节点的推荐写法。
    """

    # ---------------- 入参 ----------------
    research_topic: str
    """研究主题。"""

    # ---------------- 规划结果 ----------------
    todos: list[TodoItem]
    """规划出的任务清单（单值字段，无 reducer）。"""

    # ---------------- Send 扇出的载荷 ----------------
    # 这两个字段只在 ``research_task`` 节点被 Send 调用时被填充：
    # Send 的第二个参数会**整体作为该节点的输入状态**，所以并行子任务
    # 读不到父状态里的 todos，必须把需要的东西一并塞进载荷。
    current_task: TodoItem
    task_index: int

    # ---------------- fan-in 归并（带 reducer） ----------------
    task_results: Annotated[list[TaskResult], operator.add]
    """N 个子任务的并发结果，自动 append。"""

    sources_gathered: Annotated[list[str], operator.add]
    """各任务的来源清单，供最终报告引用。"""

    raw_contexts: Annotated[list[str], operator.add]
    """各任务拿到的检索原文（便于溯源与调试）。"""

    model_calls: Annotated[int, operator.add]
    """LLM 调用次数累加——reducer 让「计数」不需要任何共享变量。"""

    errors: Annotated[list[str], operator.add]
    """非致命错误（某个子任务失败），最终报告会显式声明。"""

    # ---------------- 最终产物 ----------------
    report: str
    report_note_id: Optional[str]
    report_note_path: Optional[str]


def initial_state(research_topic: str) -> ResearchState:
    """构造干净的初始状态。

    带 reducer 的 channel 若从未被写入，读取时会 KeyError，所以显式给空列表。
    """

    return ResearchState(
        research_topic=research_topic,
        todos=[],
        task_results=[],
        sources_gathered=[],
        raw_contexts=[],
        model_calls=0,
        errors=[],
        report="",
        report_note_id=None,
        report_note_path=None,
    )


def result_by_task(results: list[TaskResult]) -> dict[int, TaskResult]:
    """把 fan-in 的结果按 task_id 建索引（供报告节点使用）。"""

    indexed: dict[int, TaskResult] = {}
    for item in results:
        task_id = item.get("task_id")
        if task_id is not None:
            indexed[int(task_id)] = item
    return indexed


def serialize_todo(todo: TodoItem, result: Optional[TaskResult] = None) -> dict[str, Any]:
    """转成前端 ``todo_list`` / ``task_status`` 事件需要的字段。

    字段名与旧版 ``DeepResearchAgent._serialize_task`` 完全一致——
    前端契约不变是这次重写的硬性约束。
    """

    payload: dict[str, Any] = {
        "id": todo.get("id"),
        "title": todo.get("title"),
        "intent": todo.get("intent"),
        "query": todo.get("query"),
        "status": todo.get("status", "pending"),
        "summary": None,
        "sources_summary": None,
        "note_id": todo.get("note_id"),
        "note_path": todo.get("note_path"),
        "stream_token": f"task_{todo.get('id')}",
    }
    if result:
        payload["status"] = result.get("status", payload["status"])
        payload["summary"] = result.get("summary")
        payload["sources_summary"] = result.get("sources_summary")
        payload["note_id"] = result.get("note_id") or payload["note_id"]
        payload["note_path"] = result.get("note_path") or payload["note_path"]
    return payload
