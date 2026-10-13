"""SSE 事件构造与工具调用抽取。

这一层是「前端契约」的唯一出口
------------------------------
旧版把这些 payload 零散地写在 ``agent.py`` 的各个 yield 里，字段名一旦改动
就会悄悄破坏前端。这里集中成**构造函数**，好处有两个：

1. 前端需要哪些字段一目了然（``frontend/src/App.vue`` 的事件分支与之一一对应）；
2. 新增节点时不必重读 agent.py 去猜格式。

事件类型保持不变：``status`` / ``todo_list`` / ``task_status`` / ``sources`` /
``task_summary_chunk`` / ``tool_call`` / ``report_note`` / ``final_report`` / ``done`` /
``error``；本次新增 ``usage``（前端遇到未知类型会忽略，属安全扩展）。

关于「工具调用事件从哪来」
--------------------------
旧版是 HelloAgents 的 ``ToolCallTracker`` 由框架回调写入。新版用标准 LangChain
消息对象：ReAct 循环结束后，``AIMessage.tool_calls`` 里有**真实调用参数**，
配对的 ``ToolMessage`` 里有**真实返回原文**。``tool_call_events`` 就是把这两者
按 ``tool_call_id`` 配对成前端要的事件——参数和结果都来自事实，不靠模型转述。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from graph.state import TaskResult, TodoItem, serialize_todo
from utils import truncate

logger = logging.getLogger(__name__)

#: 前端工具调用面板里单条结果的展示上限
TOOL_RESULT_DISPLAY_LIMIT = 800


class EventIds:
    """进程内自增的事件 ID（前端用它做列表 key 与去重）。"""

    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        self._value += 1
        return self._value


def emit(payload: dict[str, Any]) -> None:
    """把一条事件推给 ``stream_mode="custom"``。

    非流式调用（``ainvoke``）时没有 writer，属正常情况，静默跳过即可。
    """

    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        writer(payload)
    except Exception:  # noqa: BLE001 - 没有 writer / 不在图执行上下文中
        logger.debug("事件未能推送（非流式调用？）: %s", payload.get("type"))


# ---------------------------------------------------------------------------
# 事件构造
# ---------------------------------------------------------------------------


def status_event(
    message: str,
    *,
    task_id: Optional[int] = None,
    step: Optional[int] = None,
    level: str = "info",
) -> dict[str, Any]:
    """流程状态提示（前端写入进度日志）。"""

    payload: dict[str, Any] = {"type": "status", "message": message, "level": level}
    if task_id is not None:
        payload["task_id"] = task_id
    if step is not None:
        payload["step"] = step
    return payload


def todo_list_event(todos: Sequence[TodoItem], *, step: int = 0) -> dict[str, Any]:
    """任务清单（前端据此渲染任务卡片）。"""

    return {
        "type": "todo_list",
        "tasks": [serialize_todo(todo) for todo in todos],
        "step": step,
    }


def task_status_event(
    todo: TodoItem,
    status: str,
    *,
    step: Optional[int] = None,
    summary: Optional[str] = None,
    sources_summary: Optional[str] = None,
    detail: Optional[str] = None,
    note_id: Optional[str] = None,
    note_path: Optional[str] = None,
) -> dict[str, Any]:
    """单个任务的状态变化（in_progress / completed / skipped / failed）。"""

    payload: dict[str, Any] = {
        "type": "task_status",
        "task_id": todo.get("id"),
        "status": status,
        "title": todo.get("title"),
        "intent": todo.get("intent"),
        "note_id": note_id if note_id is not None else todo.get("note_id"),
        "note_path": note_path if note_path is not None else todo.get("note_path"),
    }
    if summary is not None:
        payload["summary"] = summary
    if sources_summary is not None:
        payload["sources_summary"] = sources_summary
    if detail:
        payload["detail"] = detail
    if step is not None:
        payload["step"] = step
    return payload


def sources_event(
    task_id: int,
    *,
    latest_sources: str,
    raw_context: str = "",
    backend: str = "",
    step: Optional[int] = None,
    note_id: Optional[str] = None,
    note_path: Optional[str] = None,
) -> dict[str, Any]:
    """检索来源（前端渲染来源列表）。"""

    payload: dict[str, Any] = {
        "type": "sources",
        "task_id": task_id,
        "latest_sources": latest_sources,
        "raw_context": raw_context,
        "backend": backend,
        "note_id": note_id,
        "note_path": note_path,
    }
    if step is not None:
        payload["step"] = step
    return payload


def summary_chunk_event(
    task_id: int,
    content: str,
    *,
    step: Optional[int] = None,
    note_id: Optional[str] = None,
) -> dict[str, Any]:
    """任务总结的增量片段。

    说明：结构化输出（``with_structured_output``）不支持 token 级增量，
    因此这里一次下发**完整的结构化总结**——前端「逐块追加」的渲染路径保持不变，
    不需要为本次重构改动任何前端代码。
    """

    payload: dict[str, Any] = {
        "type": "task_summary_chunk",
        "task_id": task_id,
        "content": content,
        "note_id": note_id,
    }
    if step is not None:
        payload["step"] = step
    return payload


def report_note_event(
    note_id: str, title: str, content: str, note_path: Optional[str] = None
) -> dict[str, Any]:
    """结论笔记已落盘。"""

    payload: dict[str, Any] = {
        "type": "report_note",
        "note_id": note_id,
        "title": title,
        "content": content,
    }
    if note_path:
        payload["note_path"] = note_path
    return payload


def final_report_event(
    report: str, *, note_id: Optional[str] = None, note_path: Optional[str] = None
) -> dict[str, Any]:
    """最终报告。"""

    return {
        "type": "final_report",
        "report": report,
        "note_id": note_id,
        "note_path": note_path,
    }


def usage_event(usage: dict[str, Any]) -> dict[str, Any]:
    """本次运行的 token/成本统计（新增事件类型，前端可选择性消费）。"""

    return {"type": "usage", **usage}


def error_event(detail: str) -> dict[str, Any]:
    return {"type": "error", "detail": detail}


def done_event() -> dict[str, Any]:
    return {"type": "done"}


# ---------------------------------------------------------------------------
# 从消息历史抽取「真实发生过的事」
# ---------------------------------------------------------------------------


def tool_call_events(
    messages: Sequence[BaseMessage],
    *,
    agent: str,
    task_id: Optional[int],
    event_ids: EventIds,
    step: Optional[int] = None,
    notes_workspace: Optional[str] = None,
) -> list[dict[str, Any]]:
    """把消息历史里的工具调用与返回配对成前端事件。

    Args:
        messages: ReAct 循环的完整消息历史。
        agent: 展示用的 Agent 名称（如「研究执行专家」）。
        task_id: 该调用所属的任务 ID（并行时用于前端分组）。
    """

    pending: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                pending[call.get("id") or ""] = {
                    "name": call.get("name") or "unknown",
                    "args": call.get("args") or {},
                }
        elif isinstance(message, ToolMessage):
            call_id = message.tool_call_id or ""
            meta = pending.get(call_id, {})
            tool_name = meta.get("name") or getattr(message, "name", None) or "unknown"
            parameters = meta.get("args") or {}
            result_text = _stringify(message.content)
            note_id = _extract_note_id(parameters, result_text)

            payload: dict[str, Any] = {
                "type": "tool_call",
                "event_id": event_ids.next(),
                "agent": agent,
                "tool": tool_name,
                "parameters": parameters,
                "result": truncate(result_text, TOOL_RESULT_DISPLAY_LIMIT),
                "task_id": task_id,
                "note_id": note_id,
            }
            if note_id and notes_workspace:
                payload["note_path"] = str(_note_path(notes_workspace, note_id))
            if step is not None:
                payload["step"] = step
            events.append(payload)

    return events


def parse_search_payloads(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """从 ``web_search`` 的 ToolMessage 里还原结构化搜索结果。

    工具返回的是 JSON 文本，这里原样解析——**报告里的来源永远来自工具真实返回**，
    而不是模型复述的链接（避免幻觉 URL）。
    """

    payloads: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if (getattr(message, "name", None) or "") != "web_search":
            continue
        try:
            data = json.loads(_stringify(message.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            payloads.append(data)
    return payloads


def merge_search_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """把同一任务内多次搜索的结果合并（按 URL 去重）。"""

    merged: dict[str, Any] = {"backend": "", "results": [], "answer": None, "notices": []}
    seen: set[str] = set()

    for payload in payloads:
        merged["backend"] = payload.get("backend") or merged["backend"]
        if payload.get("answer"):
            merged["answer"] = payload["answer"]
        for notice in payload.get("notices") or []:
            if notice not in merged["notices"]:
                merged["notices"].append(notice)
        for item in payload.get("results") or []:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            merged["results"].append(item)

    return merged


def result_event_payloads(
    result: TaskResult,
    todo: TodoItem,
    *,
    step: Optional[int],
) -> list[dict[str, Any]]:
    """把一个已完成任务的结果转成事件序列（总结片段 + 状态收尾）。"""

    task_id = int(result.get("task_id") or todo.get("id") or 0)
    summary = result.get("summary") or ""
    events: list[dict[str, Any]] = []

    if summary:
        events.append(
            summary_chunk_event(
                task_id,
                summary,
                step=step,
                note_id=result.get("note_id") or todo.get("note_id"),
            )
        )

    events.append(
        task_status_event(
            todo,
            result.get("status") or "completed",
            step=step,
            summary=summary,
            sources_summary=result.get("sources_summary"),
            detail=result.get("error"),
            note_id=result.get("note_id"),
            note_path=result.get("note_path"),
        )
    )
    return events


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _extract_note_id(parameters: dict[str, Any], result_text: str) -> Optional[str]:
    """笔记 ID 可能在入参里（update/read），也可能在返回值里（create）。"""

    note_id = parameters.get("note_id")
    if isinstance(note_id, str) and note_id.strip():
        return note_id.strip()

    import re

    match = re.search(r"笔记ID[:：]\s*([A-Za-z0-9._-]+)", result_text or "")
    return match.group(1) if match else None


def _note_path(notes_workspace: str, note_id: str) -> str:
    from pathlib import Path

    return str(Path(notes_workspace) / f"{note_id}.md")
