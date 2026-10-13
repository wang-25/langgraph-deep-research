"""面向 FastAPI 的调用封装。

这一层负责三件事，把「图」和「HTTP/SSE」解耦：

1. **依赖装配**：按请求构造 ``ResearchContext``（模型、工具、笔记库、事件计数器），
   并把请求级的 ``search_api`` 覆盖合并进配置；
2. **事件翻译**：节点用 ``stream_mode="custom"`` 推的事件原样转发给前端，
   同时用 ``stream_mode="values"`` 拿最终状态，补上 ``final_report`` / ``usage`` / ``done``；
3. **指标收集**：每个请求一个 ``UsageTracker``，通过 invoke config 挂上——
   LangChain 的 callback 会传播到所有子调用（含并行 superstep 里的每个任务）。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator, Optional

from config import SearchAPI, Settings, get_settings
from graph import events
from graph.builder import get_research_graph
from graph.events import EventIds
from graph.llm import get_chat_model
from graph.nodes import ResearchContext
from graph.state import ResearchState, initial_state, result_by_task, serialize_todo
from graph.tools import build_task_tools
from observability import UsageTracker
from services.notes import NoteStore

logger = logging.getLogger(__name__)


def new_run_id() -> str:
    """生成一次运行的 ID（日志串联与 configurable.thread_id 都用它）。"""

    return uuid.uuid4().hex


def resolve_search_override(value: Any) -> Optional[SearchAPI]:
    """把请求体里的 ``search_api`` 转成枚举；非法值抛 ``ValueError``。"""

    if value is None or value == "":
        return None
    try:
        return SearchAPI(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SearchAPI)
        raise ValueError(f"不支持的 search_api: {value!r}（可选：{allowed}）") from exc


class DeepResearchRunner:
    """LangGraph 深度研究流程的调用入口。"""

    # ------------------------------------------------------------------
    # 依赖装配
    # ------------------------------------------------------------------
    def build_settings(self, search_api: Any = None) -> Settings:
        """合并请求级覆盖（目前只有搜索后端）。"""

        settings = get_settings()
        override = resolve_search_override(search_api)
        if override is None:
            return settings
        return settings.model_copy(update={"search_api": override})

    def build_context(self, settings: Settings, *, topic: str = "") -> ResearchContext:
        """构造本次运行的运行时依赖。

        ``topic`` 会用来给本次研究的笔记开一个独立子目录
        （``notes/<主题>-<时间戳>/``），这样一次查询的产物天然聚在一起。
        """

        note_store = None
        if settings.enable_notes:
            note_store = NoteStore(settings.notes_path()).scope_to_run(topic or "research")
        context = ResearchContext(
            settings=settings,
            llm=get_chat_model(),
            task_tools=build_task_tools(settings, note_store),
            note_store=note_store,
            event_ids=EventIds(),
        )
        logger.info(
            "研究上下文就绪: model=%s search_api=%s structured_method=%s notes=%s tools=%s",
            settings.resolved_llm_model(),
            settings.search_api.value,
            settings.llm_structured_method,
            note_store.workspace if note_store else "off",
            [tool.name for tool in context.task_tools],
        )
        return context

    def _config(self, run_id: str, settings: Settings, tracker: UsageTracker) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": settings.graph_recursion_limit,
            # 回调挂在 invoke config 上，随图执行传播到每一次 LLM/工具调用
            "callbacks": [tracker],
        }

    # ------------------------------------------------------------------
    # 结果组装
    # ------------------------------------------------------------------
    @staticmethod
    def to_todo_items(state: ResearchState) -> list[dict[str, Any]]:
        """把最终状态里的任务清单与执行结果合并成前端需要的列表。"""

        results = result_by_task(list(state.get("task_results") or []))
        items: list[dict[str, Any]] = []
        for todo in state.get("todos") or []:
            task_id = int(todo.get("id") or 0)
            items.append(serialize_todo(todo, results.get(task_id)))
        return items

    # ------------------------------------------------------------------
    # 非流式：一次跑完
    # ------------------------------------------------------------------
    async def run(
        self,
        topic: str,
        *,
        search_api: Any = None,
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """跑完整张图并返回报告（``POST /research`` 的实现）。

        对外字段与旧版一致（``report_markdown`` / ``todo_items``），
        额外附加 ``usage`` 等统计信息；前端不需要任何改动。
        """

        run_id = run_id or new_run_id()
        settings = self.build_settings(search_api)
        context = self.build_context(settings, topic=topic)
        tracker = UsageTracker(default_model=settings.resolved_llm_model())

        logger.info("开始研究(run=%s): topic=%r", run_id[:8], topic)
        try:
            state = await get_research_graph().ainvoke(
                initial_state(topic),
                config=self._config(run_id, settings, tracker),
                context=context,
            )
        finally:
            logger.info("研究结束(run=%s): %s", run_id[:8], tracker.summary_line())

        return {
            "run_id": run_id,
            "report_markdown": (state.get("report") or "").strip(),
            "todo_items": self.to_todo_items(state),
            "model_calls": int(state.get("model_calls") or 0),
            "errors": list(state.get("errors") or []),
            "usage": tracker.snapshot(),
        }

    # ------------------------------------------------------------------
    # 流式：SSE
    # ------------------------------------------------------------------
    async def stream(
        self,
        topic: str,
        *,
        search_api: Any = None,
        run_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行图并逐条产出前端事件（``POST /research/stream`` 的实现）。

        事件类型与旧版完全一致：``status`` / ``todo_list`` / ``task_status`` /
        ``sources`` / ``task_summary_chunk`` / ``tool_call`` / ``report_note`` /
        ``final_report`` / ``done`` / ``error``，另加 ``usage``。
        """

        run_id = run_id or new_run_id()
        settings = self.build_settings(search_api)
        context = self.build_context(settings, topic=topic)
        tracker = UsageTracker(default_model=settings.resolved_llm_model())
        graph = get_research_graph()

        logger.info("开始流式研究(run=%s): topic=%r", run_id[:8], topic)

        last_values: dict[str, Any] = {}
        try:
            async for mode, chunk in graph.astream(
                initial_state(topic),
                config=self._config(run_id, settings, tracker),
                context=context,
                stream_mode=["custom", "updates", "values"],
            ):
                if mode == "custom":
                    # 节点用 get_stream_writer() 推的事件，原样转发
                    if isinstance(chunk, dict):
                        yield chunk
                elif mode == "values":
                    # 每个 superstep 结束后的完整状态；最后一个就是最终状态
                    if isinstance(chunk, dict):
                        last_values = chunk
        except Exception as exc:  # noqa: BLE001 - 兜底成一条 error 事件，避免 SSE 半路断开
            logger.exception("流式研究失败")
            yield events.error_event(f"{type(exc).__name__}: {exc}")
            yield events.done_event()
            return
        finally:
            logger.info("流式研究结束(run=%s): %s", run_id[:8], tracker.summary_line())

        report = (last_values.get("report") or "").strip()
        yield events.final_report_event(
            report,
            note_id=last_values.get("report_note_id"),
            note_path=last_values.get("report_note_path"),
        )
        yield events.usage_event(tracker.snapshot())
        yield events.done_event()


_runner: Optional[DeepResearchRunner] = None


def get_runner() -> DeepResearchRunner:
    """进程级单例。"""

    global _runner
    if _runner is None:
        _runner = DeepResearchRunner()
    return _runner
