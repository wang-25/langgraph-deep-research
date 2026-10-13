"""FastAPI 入口（LangGraph 版）。

对外契约与旧版 ``backend/src/main.py`` **完全一致**：

* ``GET  /healthz``           健康检查
* ``POST /research``          同步返回 ``{report_markdown, todo_items}``
* ``POST /research/stream``   SSE 事件流

因此 ``frontend/`` 一行都不用改就能切到这一版后端。额外新增两个只读端点：
``GET /research/graph``（Mermaid 源码）与 ``GET /research/usage`` 不在服务端留存统计、
只是说明用——用量随每次 SSE 的 ``usage`` 事件返回。
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

# 让 `python src/main.py` / `uvicorn main:app --app-dir src` 两种方式都能跑起来
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import PlainTextResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import Settings, get_settings  # noqa: E402
from graph.builder import graph_mermaid  # noqa: E402
from graph.runner import get_runner  # noqa: E402
from observability import setup_langsmith  # noqa: E402

logger = logging.getLogger("deep_research")


def _configure_logging(level: str) -> None:
    """统一用标准库 logging。

    旧版是 loguru + stdlib logging 混用（``main.py`` 用 loguru，服务层用 logging），
    结果是同一次请求的日志分散在两种格式里、级别过滤也要配两遍。
    这里统一到标准库：少一个依赖，uvicorn 的 ``--log-level`` 也能一并管住。
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-4s | %(name)s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


_configure_logging(get_settings().log_level)


class ResearchRequest(BaseModel):
    """触发一次研究。"""

    topic: str = Field(..., description="研究主题")
    search_api: Optional[str] = Field(
        default=None,
        description="覆盖默认搜索后端：duckduckgo / tavily / perplexity / searxng / advanced",
    )


class ResearchResponse(BaseModel):
    """同步研究的响应（字段与旧版兼容，新增项都有默认值）。"""

    report_markdown: str = Field(..., description="Markdown 格式研究报告")
    todo_items: list[dict[str, Any]] = Field(default_factory=list, description="任务清单与结果")
    run_id: Optional[str] = Field(default=None, description="本次运行 ID（便于对日志/LangSmith）")
    model_calls: int = Field(default=0, description="图上累计的模型调用次数")
    errors: list[str] = Field(default_factory=list, description="非致命错误")
    usage: Optional[dict[str, Any]] = Field(default=None, description="token 与成本统计")


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """打日志前遮盖密钥。"""

    if not value:
        return "unset"
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}...{value[-visible:]}"


def _log_startup(settings: Settings) -> None:
    logger.info(
        "Deep Research (LangGraph) 启动: provider=%s model=%s base_url=%s search_api=%s "
        "max_tasks=%s fetch_full_page=%s notes=%s api_key=%s",
        settings.llm_provider,
        settings.resolved_llm_model(),
        settings.resolved_llm_base_url() or "unset",
        settings.search_api.value,
        settings.max_tasks,
        settings.fetch_full_page,
        settings.notes_path() if settings.enable_notes else "off",
        _mask_secret(settings.resolved_llm_api_key()),
    )
    logger.info("CORS 放行来源: %s", ", ".join(settings.cors_origin_list()))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_langsmith(settings)
    _log_startup(settings)
    # 提前编译图，把「图有问题」暴露在启动阶段而不是第一个请求
    logger.info("图结构:\n%s", graph_mermaid())
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Deep Research (LangGraph)", lifespan=lifespan)

    origins = settings.cors_origin_list()
    allow_all = origins == ["*"]
    if allow_all:
        logger.warning("CORS_ORIGINS=* ：仅建议本地开发使用，生产环境请写死具体域名")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else origins,
        # 通配符与 credentials 不能同时开（浏览器会拒绝带凭证的通配符响应）
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index() -> Dict[str, Any]:
        """服务索引。

        后端是纯 API，没有首页；直接访问 ``/`` 会看到这个索引而不是 404。
        想手工调接口用 ``/docs``（FastAPI 自带的 Swagger UI）。
        真正的界面在 ``frontend/``：``npm install && npm run dev``。
        """

        return {
            "service": "Deep Research (LangGraph)",
            "docs": "/docs",
            "endpoints": {
                "GET /healthz": "健康检查",
                "GET /config": "当前生效配置（密钥已遮盖）",
                "GET /research/graph": "Mermaid 图结构",
                "POST /research": "同步研究，返回 {report_markdown, todo_items}",
                "POST /research/stream": "SSE 事件流（前端使用）",
            },
            "cors_origins": origins,
            "ui": "前端在 frontend/ 目录：npm install && npm run dev（默认 http://localhost:5174）",
        }

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def show_config() -> Dict[str, Any]:
        """当前**生效**的配置（密钥已遮盖）。

        存在的理由很实际：``get_settings()`` 是进程级缓存，改完 ``.env`` 必须重启才生效。
        「前端查不到结果、日志里配置却没错」这种问题，用它一眼就能确认后端到底在用哪套配置
        （搜索后端、结构化方式、任务数、笔记目录都在这里）。
        """

        current = get_settings()
        return {
            "llm": {
                "provider": current.llm_provider,
                "model": current.resolved_llm_model(),
                "base_url": current.resolved_llm_base_url(),
                "structured_method": current.llm_structured_method,
                "max_tokens": current.llm_max_tokens,
                "temperature": current.llm_temperature,
                "api_key": _mask_secret(current.resolved_llm_api_key()),
            },
            "search": {
                "api": current.search_api.value,
                "max_results_per_query": current.max_results_per_query,
                "fetch_full_page": current.fetch_full_page,
                "tavily_key": "set" if current.tavily_api_key else "unset",
                "perplexity_key": "set" if current.perplexity_api_key else "unset",
                "searxng_url": current.searxng_url,
            },
            "workflow": {
                "max_tasks": current.max_tasks,
                "task_timeout": current.task_timeout,
                "recursion_limit": current.graph_recursion_limit,
            },
            "notes": {
                "enabled": current.enable_notes,
                "root": str(current.notes_path()),
            },
            "observability": {
                "langsmith_enabled": current.langsmith_enabled(),
                "langsmith_project": current.langsmith_project,
            },
            "server": {
                "cors_origins": current.cors_origin_list(),
                "log_level": current.log_level,
            },
        }

    @app.get("/research/graph", response_class=PlainTextResponse)
    def research_graph() -> PlainTextResponse:
        """返回 Mermaid 源码（可以直接粘进 Markdown / 画图工具）。"""

        return PlainTextResponse(graph_mermaid())

    @app.post("/research", response_model=ResearchResponse)
    async def run_research(payload: ResearchRequest) -> ResearchResponse:
        runner = get_runner()
        try:
            result = await runner.run(payload.topic, search_api=payload.search_api)
        except ValueError as exc:  # 配置类问题（例如未知搜索后端）
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - 兜底
            logger.exception("研究失败")
            raise HTTPException(status_code=500, detail=f"Research failed: {exc}") from exc

        return ResearchResponse(
            report_markdown=result["report_markdown"],
            todo_items=result["todo_items"],
            run_id=result.get("run_id"),
            model_calls=result.get("model_calls", 0),
            errors=result.get("errors", []),
            usage=result.get("usage"),
        )

    @app.post("/research/stream")
    async def stream_research(payload: ResearchRequest) -> StreamingResponse:
        runner = get_runner()

        try:
            # 提前校验一次参数，避免把 400 藏在 SSE 流里
            runner.build_settings(payload.search_api)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def event_iterator() -> AsyncIterator[str]:
            try:
                async for event in runner.stream(payload.topic, search_api=payload.search_api):
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            except Exception as exc:  # pragma: no cover - 兜底
                logger.exception("流式研究失败")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "main:app",
        host=_settings.host,
        port=_settings.port,
        reload=False,
        log_level=_settings.log_level.lower(),
        app_dir=str(Path(__file__).resolve().parent),
    )
