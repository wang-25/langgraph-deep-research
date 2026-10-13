"""可观测性：LangSmith 追踪 + token/成本统计。

两个独立的东西，正好对应可观测性的两个层面：

1. **``setup_langsmith``（外部追踪）**：把配置翻译成 LangSmith 需要的环境变量。
   启用后，每一次 LLM 调用、工具调用、图节点执行都会自动上报，
   能在 Web UI 上看到完整的 trace 树——面试演示时这是最有说服力的一屏。
   注意：**不需要改任何业务代码**，LangChain 的 callback 机制天然贯穿全链路。

2. **``UsageTracker``（内部计量）**：一个 ``BaseCallbackHandler``，
   从每次 LLM 结束事件里读 ``usage_metadata`` 累加 token，并按价目表估算成本。
   它被挂进图的 invoke config，所以并行子任务的用量也会被同一个实例收齐
   （这正是「回调随 config 传播」带来的便利，不需要自己写线程安全的计数器）。

指标属于**运行时依赖**而不是业务状态：所以它走 ``context`` / config，
不进图状态、不进 checkpoint（模型句柄和计数器本来也没法序列化）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from config import Settings

logger = logging.getLogger(__name__)

#: 每 100 万 token 的价格（USD），按模型名前缀匹配；未命中则只报 token 不报钱。
PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
}


def setup_langsmith(settings: Settings) -> bool:
    """按配置打开/关闭 LangSmith 追踪。

    Returns:
        ``True`` 表示本次进程会向 LangSmith 上报。

    同时写 ``LANGSMITH_*``（新名）与 ``LANGCHAIN_TRACING_V2`` /
    ``LANGCHAIN_API_KEY``（旧名），兼容不同版本的 SDK 与文档习惯。
    """

    if not settings.langsmith_enabled():
        # 明确关掉，避免宿主环境里残留的变量导致「以为没开其实在传数据」
        for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            os.environ.pop(key, None)
        logger.info("LangSmith 追踪未启用（LANGSMITH_TRACING=False 或缺少 API key）")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    logger.info(
        "LangSmith 追踪已启用: project=%s endpoint=%s",
        settings.langsmith_project,
        settings.langsmith_endpoint or "(默认)",
    )
    return True


@dataclass
class ModelUsage:
    """单个模型的累计用量。"""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self, model: str, cost: Optional[float]) -> dict[str, Any]:
        return {
            "model": model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "estimated_cost_usd": cost,
        }


class UsageTracker(BaseCallbackHandler):
    """统计 LLM 调用次数、token 用量与预估成本。

    挂载方式（见 ``graph/runner.py``）::

        config = {"callbacks": [tracker], "recursion_limit": ...}
        async for ... in graph.astream(state, config=config, context=context):
            ...

    LangChain 的 callback 会随 config 传播到所有子 Runnable（含 ReAct agent 内部的
    每一次模型调用、以及并行 superstep 里的各个任务），所以一个实例就能收齐全局用量。
    """

    def __init__(self, default_model: Optional[str] = None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._models: dict[str, ModelUsage] = {}
        self._started_at = time.monotonic()
        self._run_started: dict[str, float] = {}
        #: 少数流式响应不回传模型名，这种情况下把它归到本次配置的模型上
        #: （宁可归因到配置值并保留在明细里，也不要静默丢成「unknown」）
        self._default_model = default_model
        self.llm_calls = 0

    # ------------------------------------------------------------------
    # LangChain 回调
    # ------------------------------------------------------------------
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            self._run_started[run_id] = time.monotonic()

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, **kwargs: Any
    ) -> None:
        self.on_llm_start(serialized, [], **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        elapsed = 0.0
        if run_id:
            started = self._run_started.pop(run_id, None)
            if started is not None:
                elapsed = time.monotonic() - started

        model, input_tokens, output_tokens, total_tokens = _extract_usage(response)
        if model == "unknown" and self._default_model:
            model = self._default_model

        with self._lock:
            self.llm_calls += 1
            usage = self._models.setdefault(model, ModelUsage())
            usage.calls += 1
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.total_tokens += total_tokens or (input_tokens + output_tokens)
            usage.elapsed_seconds += elapsed

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        logger.warning("LLM 调用失败(run=%s): %s: %s", run_id[:8], type(error).__name__, error)
        self._run_started.pop(run_id, None)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """汇总用量，供 SSE ``usage`` 事件与日志使用。"""

        with self._lock:
            models = list(self._models.items())

        per_model: list[dict[str, Any]] = []
        total_in = total_out = total_tokens = 0
        total_cost = 0.0
        pricing_known = True

        for name, usage in sorted(models, key=lambda item: -item[1].total_tokens):
            cost = estimate_cost(name, usage.input_tokens, usage.output_tokens)
            if cost is None and usage.total_tokens > 0:
                # 有 token 记录但价目表里没有这个模型 → 整体成本只能标为未知
                pricing_known = False
            elif cost is not None:
                total_cost += cost
            per_model.append(usage.as_dict(name, cost))
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            total_tokens += usage.total_tokens

        return {
            "llm_calls": self.llm_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(total_cost, 6) if pricing_known else None,
            "pricing_known": pricing_known,
            "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
            "models": per_model,
        }

    def summary_line(self) -> str:
        """一行人类可读的日志摘要。"""

        data = self.snapshot()
        cost = data["estimated_cost_usd"]
        cost_text = f"${cost:.4f}" if cost is not None else "价格未知"
        return (
            f"LLM 调用 {data['llm_calls']} 次 | "
            f"tokens in/out {data['input_tokens']}/{data['output_tokens']} | "
            f"预估成本 {cost_text} | 耗时 {data['elapsed_seconds']}s"
        )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """按价目表估算成本；未收录的模型返回 ``None``（宁可不知道，不要编一个数）。"""

    if not model:
        return None

    name = model.lower()
    for prefix, (input_price, output_price) in PRICING_USD_PER_1M.items():
        if prefix in name:
            return (input_tokens / 1_000_000) * input_price + (
                output_tokens / 1_000_000
            ) * output_price
    return None


def _extract_usage(response: LLMResult) -> tuple[str, int, int, int]:
    """从 LLMResult 里尽力抽取 (模型名, 输入 token, 输出 token, 总 token)。

    LangChain 不同版本/不同端点的字段位置不一样，这里按可靠性依次尝试：
    ``AIMessage.usage_metadata``（标准） → ``llm_output`` 里的 token_usage（OpenAI 旧格式）。
    """

    model = ""
    llm_output = response.llm_output or {}
    if isinstance(llm_output, dict):
        model = str(llm_output.get("model_name") or llm_output.get("model") or "")

    input_tokens = output_tokens = total_tokens = 0

    for generation_list in response.generations or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if usage:
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
                if not model:
                    metadata = getattr(message, "response_metadata", None) or {}
                    model = str(metadata.get("model_name") or metadata.get("model") or "")
                break  # 一次响应只统计一次，避免 n>1 的候选重复计数

    if not input_tokens and not output_tokens and isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(token_usage, dict):
            input_tokens = int(token_usage.get("prompt_tokens") or 0)
            output_tokens = int(token_usage.get("completion_tokens") or 0)
            total_tokens = int(token_usage.get("total_tokens") or 0)

    return model or "unknown", input_tokens, output_tokens, total_tokens
