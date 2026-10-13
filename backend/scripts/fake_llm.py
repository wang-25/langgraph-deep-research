"""离线脚本化假模型：无需 API Key 也能跑通整张图。

它实现的正是本项目真正依赖的那几个接口：

* ``bind_tools``              → 返回带 ``tool_calls`` 的 ``AIMessage``（真工具调用协议）
* ``with_structured_output`` → 返回符合 schema 的 Pydantic 实例（结构化输出协议）
* ``_astream``                → 分块吐字（报告节点的流式累计）

于是「图能不能编译、Send 扇出对不对、事件契约对不对、笔记有没有落盘」
这些问题都可以在**没有网络、没有 key** 的环境下验证。
这是把 Agent 工程化的必要一步：**模型是外部依赖，测试不能依赖它的可用性。**
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Iterable, Literal, Optional, Sequence, Union, get_args, get_origin

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

_NOTE_ID_RE = re.compile(r"任务笔记 ID[:：]\s*([A-Za-z0-9._-]+)")

#: 固定的假用量，用于验证成本统计链路
_FAKE_USAGE = {"input_tokens": 1200, "output_tokens": 400, "total_tokens": 1600}


class ScriptedChatModel(BaseChatModel):
    """按「消息历史状态」决定下一步动作的假模型。"""

    # `gpt-4o-mini` 在价目表里，这样 usage 事件能给出真实量级的成本估算
    offline_model_name: str = "gpt-4o-mini"
    bound_tool_names: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # LangChain 接口
    # ------------------------------------------------------------------
    @property
    def _llm_type(self) -> str:
        return "scripted-offline"

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> "ScriptedChatModel":
        """记录可用工具名并返回自身副本（等价于真实模型的工具绑定）。"""

        names = [getattr(tool, "name", None) or str(tool) for tool in tools]
        return self.model_copy(update={"bound_tool_names": names})

    def with_structured_output(
        self, schema: Any, *, method: Optional[str] = None, **kwargs: Any
    ) -> RunnableLambda:
        """直接返回一个「按 schema 造对象」的 Runnable。"""

        del method, kwargs
        return RunnableLambda(lambda _input: build_canned_instance(schema))

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        message = self.respond(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """分块吐字：让报告节点真的走一遍流式累计逻辑。"""

        del stop, run_manager, kwargs
        message = self.respond(messages)

        if getattr(message, "tool_calls", None):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"], ensure_ascii=False),
                            "id": call["id"],
                            "index": index,
                        }
                        for index, call in enumerate(message.tool_calls)
                    ],
                )
            )
            return

        text = message.content if isinstance(message.content, str) else ""
        step = max(1, len(text) // 4)
        chunks = [text[start : start + step] for start in range(0, len(text), step)]

        for index, chunk in enumerate(chunks):
            is_last = index == len(chunks) - 1
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=chunk,
                    # 真实 OpenAI 流式响应会在最后一个 chunk 带上 usage；
                    # 这里模拟同样的行为，让成本统计在流式路径上也能闭环
                    usage_metadata=dict(_FAKE_USAGE) if is_last else None,
                )
            )

    # ------------------------------------------------------------------
    # 剧本
    # ------------------------------------------------------------------
    def respond(self, messages: list[BaseMessage]) -> AIMessage:
        """根据消息历史决定这一步返回什么。

        剧本刻意做得「像真模型」：先检索 → 再写笔记 → 最后收尾，
        这样并行任务的工具调用事件、笔记联动都能被真实地验证一遍。
        """

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        if self.bound_tool_names:
            if not tool_messages and "web_search" in self.bound_tool_names:
                return self._ai_message(
                    "",
                    [
                        {
                            "name": "web_search",
                            "args": {"query": "LangGraph deep research 架构", "max_results": 3},
                            "id": "call_search_1",
                        }
                    ],
                )

            note_id = self._find_note_id(messages)
            if len(tool_messages) == 1 and note_id and "note_update" in self.bound_tool_names:
                return self._ai_message(
                    "",
                    [
                        {
                            "name": "note_update",
                            "args": {
                                "note_id": note_id,
                                "content": "## 关键发现\n\n- （离线冒烟）模型通过真实工具调用写回了笔记。",
                                "append": True,
                            },
                            "id": "call_note_1",
                        }
                    ],
                )

            return self._ai_message("（离线冒烟）检索与笔记同步已完成，下面输出结构化总结。")

        # 没有绑定工具：规划 / 报告走纯文本
        if any("研究报告" in _text(m) for m in messages):
            return self._ai_message(
                "# 研究报告：离线冒烟\n\n"
                "## 背景概览\n本报告由离线假模型生成，用于验证报告节点的流式累计与落盘。\n\n"
                "## 核心洞见\n1. Send 扇出让子任务真正并行。\n"
                "2. 结构化输出消除了正则解析。\n\n"
                "## 参考来源\n- 见各任务来源概览。\n"
            )

        return self._ai_message("（离线冒烟）模型回复。")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _ai_message(self, content: str, tool_calls: Optional[list[dict[str, Any]]] = None) -> AIMessage:
        return AIMessage(
            content=content,
            tool_calls=tool_calls or [],
            usage_metadata=dict(_FAKE_USAGE),
            response_metadata={"model_name": self.offline_model_name},
        )

    @staticmethod
    def _find_note_id(messages: Iterable[BaseMessage]) -> Optional[str]:
        for message in messages:
            match = _NOTE_ID_RE.search(_text(message))
            if match:
                return match.group(1)
        return None


# ---------------------------------------------------------------------------
# 按 schema 造数据
# ---------------------------------------------------------------------------


def build_canned_instance(schema: Any, depth: int = 0) -> Any:
    """按 Pydantic schema 递归造一个合法实例。"""

    if isinstance(schema, dict):  # 少数调用方传 JSON schema dict
        return {"result": "（离线冒烟）"}

    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return "（离线冒烟）"

    values: dict[str, Any] = {}
    for name, field_info in schema.model_fields.items():
        values[name] = _value_for(name, field_info.annotation, depth)
    return schema(**values)


def _value_for(name: str, annotation: Any, depth: int) -> Any:
    origin = get_origin(annotation)

    if origin is list:
        args = get_args(annotation)
        inner = args[0] if args else str
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            if inner.__name__ == "TodoItemPlan":
                return [
                    inner(
                        title=f"离线演示任务 {index + 1}",
                        intent="验证 LangGraph 流水线在无 API Key 环境下的行为",
                        query=f"LangGraph deep research 演示 {index + 1}",
                    )
                    for index in range(2)
                ]
            return [build_canned_instance(inner, depth + 1) for _ in range(2)]
        if inner is str:
            return ["（离线冒烟）要点一", "（离线冒烟）要点二"]
        return []

    if origin is Union:  # Optional[...]
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _value_for(name, args[0], depth) if args else None

    if origin is Literal:
        args = get_args(annotation)
        return args[0] if args else None

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return build_canned_instance(annotation, depth + 1)

    if annotation is str:
        return _string_for(name)
    if annotation is int:
        return 1
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    return None


def _string_for(name: str) -> str:
    return {
        "summary_markdown": (
            "## 任务总结\n\n"
            "- （离线冒烟）本段文字由假模型生成，用于验证事件与笔记链路。\n"
            "- 真实运行时这里会是模型基于检索结果写的多维度总结。\n"
            "- 结构化输出保证字段一定存在，不需要正则兜底。"
        ),
        "title": "离线演示任务",
        "intent": "验证 LangGraph 流水线在无 API Key 环境下也能跑通",
        "query": "LangGraph deep research 离线演示",
    }.get(name, "（离线演示内容）")


def _text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)
