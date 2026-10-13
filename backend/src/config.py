"""集中式配置。

与旧版 ``backend/src/config.py`` 的区别
---------------------------------------
1. 旧版是「手写 ``from_env`` + 两段 env 别名映射」，新增一个配置项要改三处；
   这里换成 ``pydantic-settings``：字段名 → 环境变量名是**约定**（大小写不敏感），
   新增配置只写一行字段声明。
2. 旧版把「LLM 客户端怎么建」散落在 ``DeepResearchAgent._init_llm`` 里；
   这里只保留**配置解析**（``resolved_llm_base_url`` 之类的纯函数），
   客户端构造交给 ``graph/llm.py``，职责单一。

环境变量名与旧版 ``backend/.env.example`` 完全兼容，原来的 key 可以直接复用。
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 项目根目录（本文件在 <root>/src/config.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SearchAPI(str, Enum):
    """支持的搜索后端。

    用 ``str, Enum`` 而不是普通 Enum：这样它能直接被 FastAPI 序列化，
    也能当字符串用（``settings.search_api == "tavily"``）。
    """

    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Settings(BaseSettings):
    """全部运行时配置。

    ``extra="ignore"`` 让 .env 里多余的变量（例如旧版遗留项）不会导致启动失败。
    """

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- LLM ----------------
    llm_provider: str = Field(default="custom", description="custom / ollama / lmstudio")
    llm_model_id: Optional[str] = Field(default=None, description="模型名，优先级高于 local_llm")
    local_llm: str = Field(default="llama3.2", description="本地模型名（Ollama/LMStudio）")
    llm_api_key: Optional[str] = Field(default=None)
    llm_base_url: Optional[str] = Field(default=None, description="OpenAI 兼容端点")
    ollama_base_url: str = Field(default="http://localhost:11434")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1")
    llm_temperature: float = Field(default=0.0)
    llm_timeout: float = Field(default=90.0)
    llm_max_retries: int = Field(default=2)
    llm_max_tokens: Optional[int] = Field(
        default=None,
        description=(
            "单次响应的最大输出 token 数。**thinking 模型的推理 token 也算在里面**，"
            "不设上限时（用端点默认值）容易出现「推理吃掉全部预算、正文为空」——"
            "表现为结构化总结输出空 JSON。建议 thinking 模型显式设为 8192。"
        ),
    )
    llm_structured_method: str = Field(
        default="json_schema",
        description=(
            "with_structured_output 的实现方式：json_schema / json_mode / "
            "function_calling / prompted_json（后者为自实现：纯提示词 + JSON 抽取）"
        ),
    )

    # ---------------- 搜索 ----------------
    search_api: SearchAPI = Field(default=SearchAPI.DUCKDUCKGO)
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    search_timeout: float = Field(default=30.0)
    fetch_full_page: bool = Field(default=True, description="是否抓取正文并塞进上下文")
    tavily_api_key: Optional[str] = Field(default=None)
    perplexity_api_key: Optional[str] = Field(default=None)
    searxng_url: str = Field(default="http://localhost:8888")

    # ---------------- 工作流 ----------------
    max_tasks: int = Field(default=5, ge=1, le=10, description="规划阶段最多保留多少个子任务")
    task_timeout: float = Field(default=300.0, description="单个子任务（含工具循环）的超时秒数")
    graph_recursion_limit: int = Field(default=50)

    # ---------------- 笔记 ----------------
    enable_notes: bool = Field(default=True)
    notes_workspace: str = Field(default="./notes")

    # ---------------- 可观测性 ----------------
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: Optional[str] = Field(default=None)
    langsmith_project: str = Field(default="deep-research-langgraph")
    langsmith_endpoint: Optional[str] = Field(default=None)

    # ---------------- 服务 ----------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    # 默认放行本仓库前端的开发端口：`frontend/vite.config.ts` 里 server.port = 5174。
    # localhost / 127.0.0.1 两种写法都放行——浏览器把这两个当不同 Origin，
    # 只放行其中一个，换种打开方式就会被 CORS 拦掉。填 "*" 表示放行全部（仅本地开发用）。
    cors_origins: str = Field(
        default=(
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:5174,http://127.0.0.1:5174,"
            "http://localhost:3000,http://127.0.0.1:3000"
        )
    )
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @field_validator("search_api", mode="before")
    @classmethod
    def _normalise_search_api(cls, value: Any) -> Any:
        """允许大小写/空格的宽松写法：``"Tavily "`` → ``tavily``。"""

        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("llm_structured_method", mode="before")
    @classmethod
    def _normalise_method(cls, value: Any) -> Any:
        allowed = {"json_schema", "json_mode", "function_calling", "prompted_json"}
        if isinstance(value, str) and value.strip().lower() in allowed:
            return value.strip().lower()
        return "json_schema"

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    def resolved_llm_model(self) -> str:
        """模型名：``LLM_MODEL_ID`` 优先，其次 ``LOCAL_LLM``。"""

        return self.llm_model_id or self.local_llm

    def resolved_llm_base_url(self) -> Optional[str]:
        """按 provider 解析出 OpenAI 兼容 base_url。"""

        provider = (self.llm_provider or "custom").strip().lower()
        if provider == "ollama":
            base = self.ollama_base_url.rstrip("/")
            return base if base.endswith("/v1") else f"{base}/v1"
        if provider == "lmstudio":
            return self.lmstudio_base_url
        return self.llm_base_url

    def resolved_llm_api_key(self) -> Optional[str]:
        """API key；本地推理服务通常不校验，给个占位符即可。"""

        if self.llm_api_key:
            return self.llm_api_key
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        provider = (self.llm_provider or "custom").strip().lower()
        if provider in {"ollama", "lmstudio"}:
            return "not-needed"
        return None

    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def notes_path(self) -> Path:
        """笔记目录的绝对路径（相对路径按项目根解析，避免受 CWD 影响）。"""

        path = Path(self.notes_workspace).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def langsmith_enabled(self) -> bool:
        """是否真的会启用 LangSmith：开关打开 **且** 有 key。"""

        return bool(self.langsmith_tracing and self.langsmith_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内缓存的配置单例。"""

    return Settings()


def reset_settings_cache() -> None:
    """清空缓存（测试或运行时改环境变量后调用）。"""

    get_settings.cache_clear()
