"""节点实现。

节点清单
--------
===================== ==========================================================
节点                  职责
===================== ==========================================================
``plan_todos``        结构化输出任务清单（with_structured_output）
``research_task``     单个子任务：保底检索 → ReAct 工具循环 → 结构化总结（×N 并行）
``write_report``      把所有结构化结果拼成材料，流式产出最终报告
``persist_report``    结论笔记落盘
===================== ==========================================================

几个刻意的设计选择
------------------
1. **并行靠图，不靠线程。** 旧版在 ``run_stream`` 里手写 ``Thread`` + ``Queue`` +
   ``Lock`` 汇总 N 个任务（``agent.py:201-265``）。这里由 ``builder.py`` 用
   ``Send`` 扇出，LangGraph 负责调度与 fan-in，节点函数是**纯异步函数**，
   不用关心并发。
2. **代码保底，模型增量。** 每个子任务先由代码做一次确定性检索（保证一定有来源），
   再让 ReAct 循环用 ``web_search`` 补充。这样即便模型不支持 tool calling、
   或者干脆不调工具，任务也不会「零来源」地失败。
3. **降级要诚实。** 结构化输出失败 → 退化为纯结构化调用；连模型调用都失败 →
   用检索原文拼一份降级总结，并把原因写进 ``errors``，最终报告会显式声明
   哪些内容没能核实（同 ch13 旅游规划图的做法）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from langgraph.types import Send

from config import Settings
from graph import events
from graph.agents import build_react_agent
from graph.events import EventIds
from graph.llm import ainvoke_structured
from graph.schemas import TaskSummary, TodoList
from graph.state import ResearchState, TaskResult, TodoItem, result_by_task
from prompts import (
    REPORT_WRITER_SYSTEM_PROMPT,
    TASK_SUMMARY_SYSTEM_PROMPT,
    TASK_SUMMARIZER_SYSTEM_PROMPT,
    TODO_PLANNER_SYSTEM_PROMPT,
    build_conclusion_note_content,
    build_planner_prompt,
    build_report_prompt,
    build_summary_prompt,
    build_task_prompt,
)
from services import search as search_service
from services.notes import NoteStore
from services.text_processing import strip_tool_calls
from utils import deduplicate_and_format_sources, format_sources, strip_thinking_tokens, truncate

logger = logging.getLogger(__name__)

#: 塞进总结 prompt 的每个来源 token 上限（与旧版一致）
MAX_TOKENS_PER_SOURCE = 2000
#: 报告节点引用笔记摘录的长度上限
NOTE_EXCERPT_LIMIT = 1200


@dataclass
class ResearchContext:
    """图的运行时依赖（**不进 checkpoint**，每次请求现构造）。

    放进 context 而不是 state 的三条理由：

    1. ``ChatModel`` / 工具句柄是「活对象」，本来就不该被序列化进断点文件；
    2. 它们对所有节点是只读共享的，塞进 state 会让状态图变得嘈杂；
    3. 测试时换一份 context 就能注入假模型／假工具，图结构完全不用动。
    """

    settings: Settings
    llm: BaseChatModel
    task_tools: list[BaseTool] = field(default_factory=list)
    note_store: Optional[NoteStore] = None
    event_ids: EventIds = field(default_factory=EventIds)
    _task_agent: Any = field(default=None, repr=False)

    @property
    def notes_workspace(self) -> Optional[str]:
        return str(self.settings.notes_path()) if self.note_store else None

    def task_agent(self) -> Any:
        """惰性构建 ReAct 子图（一次运行内复用，避免每个任务重复编译）。

        **刻意不加 ``response_format``**：结构化输出在部分端点（DeepSeek thinking
        模式、部分兼容网关）不可用，而工具调用是可用的。把两者绑进同一个 agent，
        结构化那步一旦 400 就会让整个 agent 抛错，连带丢掉已经拿到的工具结果。
        拆开之后：工具调用失败 → 退化为纯结构化调用；结构化不可用 → 退回自由文本。
        """

        if self._task_agent is None:
            self._task_agent = build_react_agent(
                self.llm,
                self.task_tools,
                system_prompt=TASK_SUMMARIZER_SYSTEM_PROMPT,
                name="research_agent",
            )
        return self._task_agent


# ---------------------------------------------------------------------------
# 规划节点
# ---------------------------------------------------------------------------


async def plan_todos(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict[str, Any]:
    """把研究主题拆成任务清单。

    与旧版最大的差别：不再要求模型「输出 JSON 文本」再由 ``planner.py`` 抠出来，
    而是 ``with_structured_output(TodoList)`` 直接得到 Pydantic 对象。
    """

    context = runtime.context
    settings = context.settings
    topic = state["research_topic"]

    events.emit(events.status_event("初始化研究流程"))
    events.emit(events.status_event(f"正在为「{topic}」规划研究任务", step=0))

    messages: list[BaseMessage] = [
        SystemMessage(content=TODO_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=build_planner_prompt(topic, settings.max_tasks)),
    ]

    planned: list[Any] = []
    try:
        result = await ainvoke_structured(
            context.llm, TodoList, messages, settings=settings
        )
        planned = list(result.tasks)[: settings.max_tasks]
    except Exception as exc:  # noqa: BLE001 - 规划失败要能退化，不能 500
        logger.exception("任务规划失败，退化为单任务")
        events.emit(
            events.status_event(
                f"任务规划失败，退化为单任务：{type(exc).__name__}", level="warn"
            )
        )

    todos: list[TodoItem] = [
        TodoItem(
            id=index,
            title=(item.title or f"任务{index}").strip(),
            intent=(item.intent or "聚焦主题的关键问题").strip(),
            query=(item.query or topic).strip() or topic,
            status="pending",
            note_id=None,
            note_path=None,
        )
        for index, item in enumerate(planned, start=1)
    ]

    if not todos:
        logger.info("规划结果为空，使用兜底任务")
        todos = [
            TodoItem(
                id=1,
                title="基础背景梳理",
                intent="收集主题的核心背景与最新动态",
                query=f"{topic} 最新进展",
                status="pending",
                note_id=None,
                note_path=None,
            )
        ]

    events.emit(events.todo_list_event(todos, step=0))
    events.emit(events.status_event(f"已生成 {len(todos)} 个研究任务", step=0))
    return {"todos": todos, "model_calls": 1}


def fan_out_tasks(state: ResearchState) -> list[Send]:
    """条件边：为每个任务发一条 ``Send``，让它们落在同一个 superstep 并发执行。

    注意 ``Send`` 的第二个参数会**整体成为该节点的输入状态**——并行分支读不到
    父状态里的 ``todos``，所以主题和任务本体必须随载荷一起传下去。
    """

    todos = state.get("todos") or []
    topic = state.get("research_topic") or ""
    return [
        Send(
            "research_task",
            {
                "research_topic": topic,
                "current_task": todo,
                "task_index": index,
            },
        )
        for index, todo in enumerate(todos, start=1)
    ]


# ---------------------------------------------------------------------------
# 子任务节点
# ---------------------------------------------------------------------------


async def research_task(
    state: ResearchState, runtime: Runtime[ResearchContext]
) -> dict[str, Any]:
    """执行单个研究任务：保底检索 → 工具循环 → 结构化总结。"""

    context = runtime.context
    settings = context.settings
    todo: TodoItem = state["current_task"]
    index = int(state.get("task_index") or todo.get("id") or 1)
    task_id = int(todo.get("id") or index)

    started = time.monotonic()
    errors: list[str] = []
    model_calls = 0

    # ---- 1. 代码负责创建任务笔记，保证「每个任务一定有笔记」不依赖模型自觉 ----
    # 目录已经由 runner 按「本次主题」建好；这里给文件名加上序号与任务名，便于人翻看。
    note = None
    if context.note_store is not None:
        try:
            note = context.note_store.create(
                title=f"任务 {task_id}: {todo.get('title', '')}",
                content=(
                    f"## 任务概览\n"
                    f"- 任务目标：{todo.get('intent', '')}\n"
                    f"- 检索查询：{todo.get('query', '')}\n\n"
                    f"## 来源概览\n（详见任务来源事件与最终报告）"
                ),
                note_type="task_state",
                tags=["deep_research", f"task_{task_id}"],
                task_id=task_id,
                note_id=f"task-{task_id:02d}",
                filename=f"{task_id:02d}-{todo.get('title', '')}.md",
            )
        except Exception as exc:  # noqa: BLE001 - 笔记失败不该拖垮研究
            logger.warning("创建任务笔记失败: %s", exc)
            errors.append(f"任务 {task_id} 笔记创建失败：{exc}")

    if note is not None:
        todo["note_id"] = note.id
        todo["note_path"] = str(note.path) if note.path else None

    events.emit(events.task_status_event(todo, "in_progress", step=index))
    events.emit(
        events.status_event(f"开始检索：{todo.get('query', '')}", task_id=task_id, step=index)
    )

    # ---- 2. 确定性保底检索（不依赖模型是否愿意调工具） ----
    baseline = await search_service.dispatch_search(
        todo.get("query") or state.get("research_topic") or "",
        api=None,
        settings=settings,
    )
    for notice in baseline.notices:
        events.emit(events.status_event(notice, task_id=task_id, step=index, level="warn"))

    # ---- 3. 交给 ReAct 工具循环：模型自行决定补充检索/写笔记 ----
    context_text = deduplicate_and_format_sources(
        baseline.as_payload(),
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=settings.fetch_full_page,
    )
    if baseline.answer:
        context_text = f"AI直接答案：\n{baseline.answer}\n\n{context_text}"

    note_excerpt = None
    if note is not None:
        note_excerpt = truncate(note.content, NOTE_EXCERPT_LIMIT)

    prompt = build_task_prompt(
        state.get("research_topic") or "",
        todo,
        context=context_text or "（本轮没有拿到检索结果）",
        note_id=note.id if note is not None else None,
        note_excerpt=note_excerpt,
    )

    messages: list[BaseMessage] = []
    structured: Optional[TaskSummary] = None
    agent_error: Optional[str] = None
    agent_text = ""

    # ---- 4. 工具循环：只负责「调工具 + 自由文本」，不掺结构化输出 ----
    # 刻意不传 response_format：结构化输出在部分端点（例如 DeepSeek thinking 模式）
    # 不可用，一旦把两者绑在一起，结构化那步 400 会把整个 agent 拖垮，
    # 连已经搜到的工具结果都一起丢掉。拆成两步后，能用的部分（工具调用）先拿到结果。
    if context.task_tools:
        try:
            agent = context.task_agent()
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]}),
                timeout=settings.task_timeout,
            )
            messages = list(result.get("messages") or [])
            agent_text = _last_ai_text(messages)
            model_calls += sum(1 for message in messages if isinstance(message, AIMessage)) or 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("任务 %s 的工具循环失败：%s: %s", task_id, type(exc).__name__, exc)
            agent_error = f"{type(exc).__name__}: {exc}"
            events.emit(
                events.status_event(
                    f"工具循环失败，退化为直接总结：{type(exc).__name__}",
                    task_id=task_id,
                    step=index,
                    level="warn",
                )
            )
            errors.append(f"任务 {task_id} 工具循环失败：{agent_error}")

    # ---- 5. 汇总来源：模型补充的检索结果与保底结果合并 ----
    agent_payloads = events.parse_search_payloads(messages)
    merged = events.merge_search_payloads([baseline.as_payload(), *agent_payloads])
    sources_summary = format_sources(merged)
    if merged.get("results"):
        full_context = deduplicate_and_format_sources(
            merged, max_tokens_per_source=MAX_TOKENS_PER_SOURCE, fetch_full_page=False
        )
        if merged.get("answer"):
            full_context = f"AI直接答案：\n{merged['answer']}\n\n{full_context}"
    else:
        full_context = context_text

    # ---- 6. 单独一步做结构化总结 ----
    # 放在「汇总来源」之后：总结要基于**合并后的全部材料**（含模型自己补搜的部分），
    # 而不是只用保底检索那一段。提示词用 tool-free 的 TASK_SUMMARY_SYSTEM_PROMPT，
    # 否则模型会照着工具说明去"调用"不存在的工具、输出 DSML 标记而不是 JSON。
    try:
        structured = await ainvoke_structured(
            context.llm,
            TaskSummary,
            [
                SystemMessage(content=TASK_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_summary_prompt(
                        state.get("research_topic") or "",
                        todo,
                        context=full_context or "（本轮没有检索到任何材料）",
                        draft=agent_text,
                        sources_summary=sources_summary,
                    )
                ),
            ],
            settings=settings,
        )
        model_calls += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("任务 %s 结构化总结失败：%s", task_id, exc)
        errors.append(f"任务 {task_id} 结构化总结失败：{type(exc).__name__}: {exc}")
        events.emit(
            events.status_event(
                f"结构化总结不可用，退回自由文本：{type(exc).__name__}",
                task_id=task_id,
                step=index,
                level="warn",
            )
        )

    events.emit(
        events.sources_event(
            task_id,
            latest_sources=sources_summary,
            raw_context=truncate(full_context, 4000),
            backend=str(merged.get("backend") or ""),
            step=index,
            note_id=note.id if note is not None else None,
            note_path=str(note.path) if note and note.path else None,
        )
    )

    # 工具调用事件：参数与结果都来自真实消息，不靠模型转述
    for payload in events.tool_call_events(
        messages,
        agent="研究执行专家",
        task_id=task_id,
        event_ids=context.event_ids,
        step=index,
        notes_workspace=context.notes_workspace,
    ):
        events.emit(payload)

    # ---- 7. 定状态 ----
    # 总结来源按可信度排序：结构化输出 → 工具循环的自由文本 → 原始来源线索（代码拼）
    summary = ""
    if structured is not None:
        summary = strip_tool_calls(structured.summary_markdown or "").strip()
    if not summary:
        summary = strip_tool_calls(agent_text).strip()
    if not summary:
        summary = _fallback_summary(todo, merged, agent_error)

    # 「一个来源都没有，且没拿到结构化结果」才叫 skipped——此时结论无从核实。
    # 注意不要把 errors 当成 summary，否则用户看到的「总结」就只剩一行警告。
    status = "skipped" if (not merged.get("results") and structured is None) else "completed"

    if errors:
        summary = f"{summary}\n\n> ⚠️ 执行提示：{'; '.join(errors)}"

    # ---- 8. 代码负责把总结写进笔记（模型可能忘了调 note_update） ----
    note_id = note.id if note is not None else None
    note_path = str(note.path) if note is not None and note.path else None
    if note is not None and summary:
        try:
            context.note_store.update(  # type: ignore[union-attr]
                note.id,
                content=f"## 任务总结\n\n{summary}",
                append=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入任务笔记失败: %s", exc)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    result_dict: TaskResult = TaskResult(
        task_id=task_id,
        title=todo.get("title", ""),
        status=status,
        summary=summary,
        key_findings=list(structured.key_findings) if structured else [],
        not_verified=list(structured.not_verified) if structured else [],
        confidence=structured.confidence if structured else "low",
        sources_summary=sources_summary,
        raw_context=truncate(full_context, 4000),
        backend=str(merged.get("backend") or ""),
        note_id=note_id,
        note_path=note_path,
        error="; ".join(errors) or None,
    )

    for payload in events.result_event_payloads(result_dict, todo, step=index):
        events.emit(payload)
    events.emit(
        events.status_event(
            f"任务 {task_id} 完成（{elapsed_ms} ms，来源 {len(merged.get('results') or [])} 条）",
            task_id=task_id,
            step=index,
        )
    )

    payload: dict[str, Any] = {
        "task_results": [result_dict],
        "model_calls": model_calls,
    }
    if sources_summary:
        payload["sources_gathered"] = [f"### 任务 {task_id}：{todo.get('title', '')}\n{sources_summary}"]
    if full_context:
        payload["raw_contexts"] = [full_context]
    if errors:
        payload["errors"] = errors
    return payload


# ---------------------------------------------------------------------------
# 报告节点
# ---------------------------------------------------------------------------


async def write_report(
    state: ResearchState, runtime: Runtime[ResearchContext]
) -> dict[str, Any]:
    """整合所有结构化结果，流式产出最终报告。

    这里用 ``astream`` 而不是 ``ainvoke``：报告是唯一「长文本」产物，
    流式能在服务端逐块累计、边生成边写日志（也方便将来把增量直接推给前端）。
    """

    context = runtime.context
    settings = context.settings
    todos = state.get("todos") or []
    results = result_by_task(state.get("task_results") or [])
    errors = list(state.get("errors") or [])
    final_step = len(todos) + 1

    events.emit(events.status_event("正在整合任务总结，撰写报告", step=final_step))

    notes: dict[int, str] = {}
    if context.note_store is not None:
        for todo in todos:
            task_id = int(todo.get("id") or 0)
            note_id = (results.get(task_id) or {}).get("note_id") or todo.get("note_id")
            if not note_id:
                continue
            note = context.note_store.read(str(note_id))
            if note is not None:
                notes[task_id] = truncate(note.content, NOTE_EXCERPT_LIMIT)

    messages: list[BaseMessage] = [
        SystemMessage(content=REPORT_WRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=build_report_prompt(
                state.get("research_topic") or "",
                todos,
                results,
                notes=notes,
                errors=errors,
            )
        ),
    ]

    chunks: list[str] = []
    try:
        async for chunk in context.llm.astream(messages):
            text = _message_text(chunk)
            if text:
                chunks.append(text)
    except Exception as exc:  # noqa: BLE001 - 报告失败要有兜底产物
        logger.exception("报告生成失败")
        errors.append(f"报告生成失败：{type(exc).__name__}: {exc}")
        events.emit(
            events.status_event(
                f"报告生成失败，改用任务总结拼接：{type(exc).__name__}",
                step=final_step,
                level="warn",
            )
        )

    report = strip_tool_calls(strip_thinking_tokens("".join(chunks))).strip()
    if not report:
        report = _fallback_report(state.get("research_topic") or "", todos, results)

    payload: dict[str, Any] = {"report": report, "model_calls": 1}
    if errors:
        payload["errors"] = errors
    return payload


async def persist_report(
    state: ResearchState, runtime: Runtime[ResearchContext]
) -> dict[str, Any]:
    """把报告落成 ``conclusion`` 笔记，并写一份本次研究的目录索引。

    幂等：同一个 ID（``report``）已存在时走更新而不是新建文件。
    """

    context = runtime.context
    report = (state.get("report") or "").strip()
    topic = state.get("research_topic") or ""

    if context.note_store is None or not report:
        return {}

    title = f"研究报告：{topic}".strip() or "研究报告"
    content = build_conclusion_note_content(topic, report)

    try:
        existing = context.note_store.read("report")
        if existing is not None:
            note = context.note_store.update("report", content=content, title=title)
        else:
            note = context.note_store.create(
                title=title,
                content=content,
                note_type="conclusion",
                tags=["deep_research", "report"],
                note_id="report",
                filename="研究报告.md",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("保存结论笔记失败: %s", exc)
        return {}

    if note is None:
        return {}

    _write_run_index(context, state, report_note=note)

    note_path = str(note.path) if note.path else None
    events.emit(events.report_note_event(note.id, title, content, note_path))
    return {"report_note_id": note.id, "report_note_path": note_path}


def _write_run_index(
    context: ResearchContext,
    state: ResearchState,
    *,
    report_note: Any,
) -> None:
    """写 ``00-研究概要.md``：本次研究的任务清单、状态与文件索引。

    这样一次查询产生的目录是**自解释**的：打开文件夹先看概要，
    再按序号点进各任务笔记或报告。
    """

    store = context.note_store
    if store is None:
        return

    todos = state.get("todos") or []
    results = result_by_task(list(state.get("task_results") or []))
    topic = state.get("research_topic") or ""

    lines = [
        f"# 研究概要：{topic}",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 任务数：{len(todos)}",
        f"- 模型调用：{int(state.get('model_calls') or 0)} 次",
        f"- 检索后端：{context.settings.search_api.value}",
        "",
        "## 任务清单",
        "",
        "| # | 任务 | 状态 | 可信度 | 来源 | 笔记 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for todo in todos:
        task_id = int(todo.get("id") or 0)
        result = results.get(task_id) or {}
        note_file = _task_note_filename(store, task_id)
        sources = _count_sources(result.get("sources_summary") or "")
        lines.append(
            f"| {task_id} | {todo.get('title', '')} | {result.get('status', 'unknown')} | "
            f"{result.get('confidence', '-')} | {sources} | "
            f"{f'[{note_file}](<{note_file}>)' if note_file else '-'} |"
        )

    lines.extend(["", "## 产物", ""])
    if report_note is not None and report_note.path:
        lines.append(f"- 最终报告：[{report_note.filename}](<{report_note.filename}>)")
    else:
        lines.append("- 最终报告：（未生成）")

    errors = list(state.get("errors") or [])
    if errors:
        lines.extend(["", "## 执行提示", ""])
        lines.extend(f"- {item}" for item in errors)

    try:
        existing = store.read("index")
        if existing is not None:
            store.update("index", content="\n".join(lines))
        else:
            store.create(
                title=f"研究概要：{topic}",
                content="\n".join(lines),
                note_type="index",
                tags=["deep_research", "index"],
                note_id="index",
                filename="00-研究概要.md",
            )
    except Exception as exc:  # noqa: BLE001 - 索引失败不该影响报告本身
        logger.warning("写入研究概要失败: %s", exc)


def _task_note_filename(store: Any, task_id: int) -> Optional[str]:
    """按 id 找到任务笔记的真实文件名（供索引里做相对链接）。"""

    note = store.read(f"task-{task_id:02d}")
    return note.filename if note is not None else None


def _count_sources(sources_summary: str) -> int:
    return sum(1 for line in sources_summary.splitlines() if line.strip().startswith("*"))


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _message_text(message: Any) -> str:
    """从 AIMessage / AIMessageChunk 里取纯文本（内容可能是分块列表）。"""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "text_delta"}:
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return ""


def _last_ai_text(messages: Sequence[BaseMessage]) -> str:
    """取工具循环里最后一条**有内容的** AI 文本，作为结构化总结的草稿。"""

    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            text = _message_text(message).strip()
            # 只发起工具调用、没有正文的 AIMessage 不算草稿
            if text and not getattr(message, "tool_calls", None):
                return text
    return ""


def _fallback_summary(todo: TodoItem, merged: dict[str, Any], agent_error: Optional[str]) -> str:
    """模型完全不可用时的降级总结：只陈述事实（检索到了什么），不编内容。"""

    lines = [
        "## 任务总结",
        "",
        f"- 本任务的自动总结未能生成（{agent_error or '模型未返回结构化结果'}）。",
        f"- 检索查询：{todo.get('query', '')}",
        f"- 检索到 {len(merged.get('results') or [])} 条来源：",
    ]
    for item in (merged.get("results") or [])[:5]:
        lines.append(f"  - [{item.get('title') or item.get('url')}]({item.get('url')})")
    if merged.get("answer"):
        lines.extend(["", "### 检索后端直接答案", "", str(merged["answer"])])
    lines.extend(["", "以上内容未经模型归纳，仅作原始来源线索。"])
    return "\n".join(lines)


def _fallback_report(
    topic: str, todos: list[TodoItem], results: dict[int, TaskResult]
) -> str:
    """报告生成失败时的兜底产物：直接把各任务总结拼起来，保证有交付物。"""

    parts = [
        f"# 研究报告：{topic}",
        "",
        "> ⚠️ 报告撰写节点未能正常产出，以下内容由各任务的结构化总结直接拼接而成。",
        "",
    ]
    for todo in todos:
        task_id = int(todo.get("id") or 0)
        result = results.get(task_id) or {}
        parts.extend(
            [
                f"## 任务 {task_id}：{todo.get('title', '')}",
                "",
                f"- 状态：{result.get('status', 'unknown')}（可信度：{result.get('confidence', 'unknown')}）",
                f"- 检索查询：{todo.get('query', '')}",
                "",
                result.get("summary") or "暂无可用信息",
                "",
                "### 来源",
                "",
                result.get("sources_summary") or "暂无来源",
                "",
            ]
        )
    if not todos:
        parts.append("暂无相关信息。")
    return "\n".join(parts)
