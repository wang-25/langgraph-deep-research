"""工具定义。

这就是「真工具调用」落地的地方
------------------------------
旧版工具是提示词协议：模型输出 ``[TOOL_CALL:note:{...}]``，上层用正则解析
（``backend/src/services/planner.py`` 的 ``TOOL_CALL_PATTERN``）。
参数写错、少个引号、多个换行都会让解析失败，而失败方式是「静默地没调用工具」。

新版是 ``@tool`` + ``bind_tools``：

* **调用决策由模型给出结构化参数**（``AIMessage.tool_calls``），框架负责校验与执行；
* 参数非法会变成一条可读的 ``ToolMessage`` 错误回到模型，模型可以自我修正重试；
* 每个工具的能力边界写在 docstring 里，而 docstring 就是送给模型的 schema 描述。

工具本身只是**普通异步函数**，不依赖 LangGraph——所以 ``scripts/`` 里的离线脚本
可以单独调用它们做测试。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool

from config import Settings
from services import search as search_service
from services.notes import NoteStore
from utils import truncate

logger = logging.getLogger(__name__)

#: 单条来源正文塞进模型上下文的上限
SOURCE_CONTENT_LIMIT = 1500


def build_web_search_tool(settings: Settings) -> BaseTool:
    """构造 ``web_search`` 工具。"""

    @tool("web_search")
    async def web_search(query: str, max_results: int = 5) -> str:
        """联网检索网页，返回标题、链接、摘要与正文片段。

        Args:
            query: 检索语句，尽量具体（可包含年份、厂商名等限定词）。
            max_results: 期望返回的结果条数，默认 5。

        Returns:
            JSON 字符串：``{"backend":..., "answer":..., "notices":[...], "results":[...]}``。
            调用失败时 ``results`` 为空、``notices`` 里带原因——这种情况下请如实说明
            「检索失败」，不要用常识替代检索结果。
        """

        outcome = await search_service.dispatch_search(query, api=None, settings=settings)
        payload: dict[str, Any] = {
            "backend": outcome.backend,
            "answer": outcome.answer,
            "notices": outcome.notices,
            "results": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": truncate(item.get("content", "") or "", SOURCE_CONTENT_LIMIT),
                    "raw_content": truncate(
                        item.get("raw_content", "") or "", SOURCE_CONTENT_LIMIT
                    ),
                }
                for item in outcome.results[: max(1, min(int(max_results or 5), 20))]
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    return web_search


def build_note_tools(store: NoteStore) -> list[BaseTool]:
    """构造笔记相关工具（read / update / create）。"""

    @tool("note_read")
    def note_read(note_id: str) -> str:
        """读取指定 ID 的任务笔记全文。

        Args:
            note_id: 笔记 ID，形如 ``note-20250101010101-ab12``。
        """

        note = store.read(note_id)
        if note is None:
            return f"笔记不存在：{note_id}"
        return note.to_tool_text()

    @tool("note_update")
    def note_update(
        note_id: str,
        content: str,
        append: bool = True,
        title: Optional[str] = None,
    ) -> str:
        """把本轮的新增发现写回任务笔记，保持既有段落结构。

        Args:
            note_id: 要更新的笔记 ID。
            content: 需要写入的内容（Markdown）。
            append: 为 True 时追加到笔记末尾，为 False 时整体替换。
            title: 可选，同时更新笔记标题。
        """

        note = store.update(note_id, content=content, append=append, title=title)
        if note is None:
            return f"更新失败，笔记不存在：{note_id}"
        return note.to_tool_text()

    @tool("note_create")
    def note_create(
        title: str,
        content: str = "",
        note_type: str = "task_state",
        tags: Optional[list[str]] = None,
        task_id: Optional[int] = None,
    ) -> str:
        """新建一条笔记并返回它的 ID。

        Args:
            title: 笔记标题。
            content: 初始内容（Markdown）。
            note_type: 笔记类型，任务笔记用 ``task_state``，结论用 ``conclusion``。
            tags: 标签列表，便于检索。
            task_id: 关联的任务编号。
        """

        note = store.create(
            title=title,
            content=content,
            note_type=note_type,
            tags=tags or [],
            task_id=task_id,
        )
        return note.to_tool_text()

    return [note_read, note_update, note_create]


def build_task_tools(settings: Settings, store: Optional[NoteStore]) -> list[BaseTool]:
    """子任务执行节点可用的工具集合。

    笔记相关的工具只在开启笔记时注入。**裁剪工具是有意为之**：给模型的每个工具
    都会占用 schema 描述和决策成本，用不到的工具只会增加误调用概率
    （这也是旧版把 note 工具塞给所有 Agent 的副作用）。
    """

    tools: list[BaseTool] = [build_web_search_tool(settings)]
    if store is not None:
        # 子任务只读 + 更新：创建由代码负责（保证每个任务一定有笔记，不依赖模型自觉）
        read_tool, update_tool, _ = build_note_tools(store)
        tools.extend([read_tool, update_tool])
    return tools
