"""纯文本工具函数。

这一层**从旧版 ``backend/src/utils.py`` 原样保留**（只做了类型注解现代化）：
它没有任何框架依赖，是「升级框架时不该被重写的部分」。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Union

CHARS_PER_TOKEN = 4

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def strip_thinking_tokens(text: str) -> str:
    """移除 ``<think>...</think>`` 段落（推理型模型的思考过程）。"""

    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def html_to_text(html: str) -> str:
    """极简 HTML → 文本（不引入 bs4，够用即可）。"""

    text = _TAG_RE.sub(" ", html)
    text = _HTML_RE.sub(" ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return _WS_RE.sub("\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def deduplicate_and_format_sources(
    search_response: Dict[str, Any] | List[Dict[str, Any]],
    max_tokens_per_source: int,
    *,
    fetch_full_page: bool = False,
) -> str:
    """按 URL 去重，并格式化成可直接塞进 prompt 的文本。"""

    if isinstance(search_response, dict):
        sources_list = search_response.get("results", []) or []
    else:
        sources_list = search_response

    unique_sources: dict[str, Dict[str, Any]] = {}
    for source in sources_list:
        url = source.get("url")
        if url and url not in unique_sources:
            unique_sources[url] = source

    formatted_parts: List[str] = []
    for source in unique_sources.values():
        title = source.get("title") or source.get("url", "")
        content = source.get("content", "") or ""
        formatted_parts.append(f"信息来源: {title}\n\n")
        formatted_parts.append(f"URL: {source.get('url', '')}\n\n")
        formatted_parts.append(f"信息内容: {content}\n\n")

        if fetch_full_page:
            raw_content = source.get("raw_content") or ""
            if not raw_content:
                logger.debug("raw_content missing for %s", source.get("url", ""))
            char_limit = max_tokens_per_source * CHARS_PER_TOKEN
            if len(raw_content) > char_limit:
                raw_content = f"{raw_content[:char_limit]}... [truncated]"
            formatted_parts.append(
                f"详细信息内容限制为 {max_tokens_per_source} 个 token: {raw_content}\n\n"
            )

    return "".join(formatted_parts).strip()


def format_sources(search_results: Dict[str, Any] | None) -> str:
    """把结果渲染成前端 ``sources`` 面板用的短列表。"""

    if not search_results:
        return ""

    results = search_results.get("results", []) or []
    return "\n".join(
        f"* {item.get('title', item.get('url', ''))} : {item.get('url', '')}"
        for item in results
        if isinstance(item, dict) and item.get("url")
    )


def truncate(text: str, limit: int) -> str:
    """截断长文本，附上被截断的长度说明（便于排查而非静默丢数据）。"""

    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…（已截断，原文 {len(text)} 字）"
