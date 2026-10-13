"""模型输出清洗。

旧版这里叫 ``strip_tool_calls``：因为工具调用是靠**文本协议** ``[TOOL_CALL:note:{...}]``
表达的，模型偶尔会把协议原文写进给用户看的正文里，只能在渲染前用正则擦掉。

新版用真正的函数调用（``bind_tools`` / ``ToolMessage``），协议文本根本不会出现在
正文里——这个函数因此从「必需环节」降级为「防御性兜底」：提示词里残留的旧协议样例、
或者换了不支持 tool calling 的模型时,依然不会把裸协议泄给用户。
"""

from __future__ import annotations

import re

_TOOL_CALL_RE = re.compile(r"\[TOOL_CALL:[^\]]*\]")
#: 少数模型会照抄 OpenAI/Anthropic 风格的伪 XML 工具调用
_PSEUDO_XML_RE = re.compile(r"<(tool_call|function_call)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def strip_tool_calls(text: str) -> str:
    """移除文本里残留的工具调用标记（兼容旧版文本协议）。"""

    if not text:
        return text

    cleaned = _PSEUDO_XML_RE.sub("", text)
    cleaned = _TOOL_CALL_RE.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
