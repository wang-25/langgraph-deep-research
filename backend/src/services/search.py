"""搜索后端 dispatch。

设计要点
--------
1. **不引入搜索 SDK**：Tavily / Perplexity / SearXNG 都是「一个 POST」的 REST 接口，
   直接用 ``httpx``（已经是 langchain-openai 的传递依赖）省掉三个包。
   只有 DuckDuckGo 没有官方 REST，需要 ``ddgs`` 这个可选依赖，未安装时给出明确提示。
2. **失败不抛异常**：返回 ``SearchOutcome(results=[], notices=[...])``。
   旧版是把后端返回的提示文本塞进 notices 让上层展示；这里保留同样的语义——
   某个后端挂了，应该表现为「这个任务没有来源」，而不是整条研究流水线 500。
3. **全异步**：``astream`` 里并行跑 N 个子任务时，同步阻塞的搜索会退化成串行。
   所有阻塞调用（ddgs、页面抓取）都走 ``asyncio.to_thread``。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from config import SearchAPI, Settings
from utils import html_to_text, truncate

logger = logging.getLogger(__name__)

#: 每个来源正文最多保留的字符数（fetch_full_page 打开时）
MAX_RAW_CHARS = 8000
#: 最多抓取多少个页面的正文，避免一个任务打十几个请求
MAX_FETCH_PAGES = 3

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_PERPLEXITY_DEFAULT_BASE = "https://api.perplexity.ai"


@dataclass
class SearchOutcome:
    """归一化后的搜索结果。"""

    backend: str
    results: list[dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None
    notices: list[str] = field(default_factory=list)

    def as_payload(self) -> dict[str, Any]:
        """转成旧版使用的 dict 形状，方便复用 ``utils`` 里的格式化函数。"""

        return {
            "backend": self.backend,
            "results": self.results,
            "answer": self.answer,
            "notices": self.notices,
        }


async def dispatch_search(
    query: str,
    *,
    api: str | SearchAPI | None = None,
    settings: Settings,
) -> SearchOutcome:
    """按配置执行一次搜索。

    Args:
        query: 检索语句。
        api: 覆盖配置的后端名（来自 HTTP 请求体，可能为 ``None``）。
        settings: 全局配置。

    Returns:
        归一化的 ``SearchOutcome``；任何后端异常都会被收敛成 notice。
    """

    backend = _resolve_backend(api, settings)
    logger.info("搜索: backend=%s query=%r", backend, query)

    try:
        if backend == SearchAPI.TAVILY.value:
            outcome = await _search_tavily(query, settings)
        elif backend == SearchAPI.PERPLEXITY.value:
            outcome = await _search_perplexity(query, settings)
        elif backend == SearchAPI.SEARXNG.value:
            outcome = await _search_searxng(query, settings)
        else:
            outcome = await _search_duckduckgo(query, settings)
    except Exception as exc:  # noqa: BLE001 - 网络/配额问题都不该炸掉图
        logger.warning("搜索后端 %s 失败: %s: %s", backend, type(exc).__name__, exc)
        return SearchOutcome(
            backend=backend,
            results=[],
            notices=[f"搜索后端 {backend} 调用失败：{type(exc).__name__}: {exc}"],
        )

    if settings.fetch_full_page:
        await _enrich_with_full_page(outcome, settings)

    logger.info(
        "搜索完成: backend=%s results=%d answer=%s notices=%d",
        outcome.backend,
        len(outcome.results),
        bool(outcome.answer),
        len(outcome.notices),
    )
    return outcome


def _resolve_backend(api: str | SearchAPI | None, settings: Settings) -> str:
    """决定实际使用的后端。

    ``advanced`` 是个「尽量用好后端」的别名：有 Tavily key 就用 Tavily，
    否则退回免 key 的 DuckDuckGo。
    """

    raw = api.value if isinstance(api, SearchAPI) else api
    raw = (raw or settings.search_api.value).strip().lower()
    if raw == SearchAPI.ADVANCED.value:
        return SearchAPI.TAVILY.value if settings.tavily_api_key else SearchAPI.DUCKDUCKGO.value
    if raw not in {item.value for item in SearchAPI}:
        logger.warning("未知搜索后端 %r，回退到 duckduckgo", raw)
        return SearchAPI.DUCKDUCKGO.value
    return raw


# ---------------------------------------------------------------------------
# 各后端实现
# ---------------------------------------------------------------------------


async def _search_tavily(query: str, settings: Settings) -> SearchOutcome:
    if not settings.tavily_api_key:
        return SearchOutcome(
            backend="tavily",
            notices=["未配置 TAVILY_API_KEY，无法使用 Tavily；请改 SEARCH_API=duckduckgo 或补 key"],
        )

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": settings.max_results_per_query,
        "include_raw_content": settings.fetch_full_page,
        "include_answer": True,
    }

    async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
        response = await client.post(_TAVILY_ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()

    results = [
        {
            "title": item.get("title") or item.get("url", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "") or "",
            "raw_content": item.get("raw_content") or "",
        }
        for item in (data.get("results") or [])
        if item.get("url")
    ]

    return SearchOutcome(
        backend="tavily",
        results=results,
        answer=data.get("answer") or None,
    )


async def _search_perplexity(query: str, settings: Settings) -> SearchOutcome:
    if not settings.perplexity_api_key:
        return SearchOutcome(
            backend="perplexity",
            notices=["未配置 PERPLEXITY_API_KEY，无法使用 Perplexity"],
        )

    base = (settings.llm_base_url or _PERPLEXITY_DEFAULT_BASE).rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": settings.llm_model_id or "sonar",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {settings.perplexity_api_key}"}

    async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    message = (data.get("choices") or [{}])[0].get("message") or {}
    answer = message.get("content") or None
    citations = data.get("citations") or message.get("citations") or []

    results = [
        {"title": _domain(url_item), "url": url_item, "content": "", "raw_content": ""}
        for url_item in citations
        if isinstance(url_item, str)
    ]

    return SearchOutcome(backend="perplexity", results=results, answer=answer)


async def _search_searxng(query: str, settings: Settings) -> SearchOutcome:
    url = f"{settings.searxng_url.rstrip('/')}/search"
    params = {"q": query, "format": "json", "safesearch": 0}

    async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    results = [
        {
            "title": item.get("title") or item.get("url", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "") or "",
            "raw_content": "",
        }
        for item in (data.get("results") or [])[: settings.max_results_per_query]
        if item.get("url")
    ]

    return SearchOutcome(backend="searxng", results=results)


async def _search_duckduckgo(query: str, settings: Settings) -> SearchOutcome:
    """DuckDuckGo 需要 ``ddgs`` 包（可选依赖），没装就给出安装提示。"""

    def _run() -> list[dict[str, Any]]:
        try:
            from ddgs import DDGS  # type: ignore[import-not-found]
        except ImportError:  # 兼容旧包名
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]

        with DDGS() as client:
            return list(client.text(query, max_results=settings.max_results_per_query))

    try:
        raw_items = await asyncio.to_thread(_run)
    except ImportError:
        return SearchOutcome(
            backend="duckduckgo",
            notices=["未安装 ddgs，无法使用 DuckDuckGo：pip install ddgs（或改用 tavily/searxng）"],
        )

    results = [
        {
            "title": item.get("title") or item.get("href", ""),
            "url": item.get("href") or item.get("url", ""),
            "content": item.get("body") or item.get("content", "") or "",
            "raw_content": "",
        }
        for item in raw_items
        if item.get("href") or item.get("url")
    ]

    return SearchOutcome(backend="duckduckgo", results=results)


# ---------------------------------------------------------------------------
# 正文补全
# ---------------------------------------------------------------------------


async def _enrich_with_full_page(outcome: SearchOutcome, settings: Settings) -> None:
    """给前 ``MAX_FETCH_PAGES`` 条结果补上正文。

    Tavily 在 ``include_raw_content=True`` 时已经返回正文，这里只处理缺口；
    抓取失败只写 notice，不影响已有摘要。
    """

    targets = [item for item in outcome.results if not item.get("raw_content")][:MAX_FETCH_PAGES]
    if not targets:
        return

    async with httpx.AsyncClient(
        timeout=settings.search_timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/0.1)"},
    ) as client:

        async def fetch(item: dict[str, Any]) -> None:
            try:
                response = await client.get(item["url"])
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type and "text" not in content_type:
                    return
                item["raw_content"] = truncate(html_to_text(response.text), MAX_RAW_CHARS)
            except Exception as exc:  # noqa: BLE001
                logger.debug("抓取正文失败 %s: %s", item.get("url"), exc)

        await asyncio.gather(*(fetch(item) for item in targets))


def _domain(url: str) -> str:
    try:
        return httpx.URL(url).host or url
    except Exception:  # noqa: BLE001
        return url
