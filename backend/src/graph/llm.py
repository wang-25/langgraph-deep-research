"""ChatModel 工厂与结构化输出调用。

旧版 ``DeepResearchAgent._init_llm`` 返回的是 HelloAgents 自己包装的 LLM 对象，
它只暴露 ``run`` / ``stream_run`` 两个方法，因此：

* 模型无法参与工具调用决策（工具是「提示词协议 + 上层正则解析」实现的）；
* 结构化输出只能靠抠 JSON；
* 换 provider 要改上层代码。

这里返回标准的 ``BaseChatModel``：``bind_tools`` / ``with_structured_output`` /
``astream`` 这些能力由 LangChain 统一提供，上层只和接口打交道。

⚠️ 关于三种结构化输出方式（真机踩坑记录）
------------------------------------------
``with_structured_output`` 有三种实现方式，**端点支持情况差异很大**：

====================== ============================================== ====================================
method                 底层做法                                        典型失败场景
====================== ============================================== ====================================
``json_schema``        ``response_format={"type":"json_schema"}``       OpenAI 支持；**DeepSeek 直接 400**
                                                                        （"This response_format type is
                                                                        unavailable now"）
``json_mode``          ``response_format={"type":"json_object"}``       需要 prompt 里出现 "JSON" 字样
                       + 把 JSON Schema 写进提示词                       （本模块会自动补上）
``function_calling``   强制 ``tool_choice=<某个函数>``                    DeepSeek **thinking 模式**会 400：
                                                                        "Thinking mode does not support
                                                                        this tool_choice"
``prompted_json``      本模块自实现：不打 response_format，             无（代价是约束最弱，
（自实现）             纯提示词要求 JSON + 自己解析 + Pydantic 校验        所以排在最后）
====================== ============================================== ====================================

还有一个**跨方式**的坑：thinking 模型的推理 token 与正文共享输出预算。
不设 ``LLM_MAX_TOKENS`` 时，长上下文任务（例如一个任务抓回 40 条来源）会出现
「推理写满预算、JSON 正文为空」——LangChain 报 ``Invalid json output:``（后面什么都没有），
三种原生方式会**一起**失败。所以：thinking 模型建议显式设 ``LLM_MAX_TOKENS=8192``，
并把 ``prompted_json`` 作为最后一道保险。

所以这里按「配置优先 + 其余兜底」的顺序逐个尝试，把端点差异收敛在一处。
这也解释了为什么 ``research_task`` 节点要把**工具调用**和**结构化总结**拆成两步：
工具调用（``bind_tools``，不指定 tool_choice）在 thinking 模型上是可用的，
而结构化输出可能不可用——两者绑在一起会让能用的部分一起失败。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Sequence, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import Settings, get_settings

logger = logging.getLogger(__name__)

#: 结构化输出的候选实现方式，按「配置优先、其余兜底」排序。
#: 前三者是 LangChain 原生 method，``prompted_json`` 是本模块自实现的兜底。
STRUCTURED_METHODS: Tuple[str, ...] = (
    "json_schema",
    "json_mode",
    "function_calling",
    "prompted_json",
)

_FALLBACK_METHODS: Tuple[str, ...] = STRUCTURED_METHODS

_model_cache: Optional[BaseChatModel] = None


def create_chat_model(
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    settings: Optional[Settings] = None,
    **overrides: Any,
) -> ChatOpenAI:
    """按配置创建一个 OpenAI 兼容的 ChatModel。

    DeepSeek、通义、vLLM、Ollama、LMStudio 等只要提供 OpenAI 兼容端点即可复用，
    差别只在 ``LLM_BASE_URL`` / ``LLM_MODEL_ID``。
    """

    settings = settings or get_settings()
    api_key = settings.resolved_llm_api_key()
    if not api_key:
        raise ValueError(
            "未配置 LLM 密钥：请在 backend-langgraph/.env 中设置 "
            "LLM_API_KEY（或 OPENAI_API_KEY）"
        )

    params: dict[str, Any] = {
        "model": model or settings.resolved_llm_model(),
        "api_key": api_key,
        "base_url": settings.resolved_llm_base_url(),
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }
    if settings.llm_max_tokens:
        # 显式给输出预算：thinking 模型的推理 token 也占这个额度，
        # 不设时容易「推理写满、正文为空」（实测 deepseek-flash + 长上下文）
        params["max_tokens"] = settings.llm_max_tokens
    params.update(overrides)
    return ChatOpenAI(**params)


def get_chat_model() -> BaseChatModel:
    """进程内共享的 ChatModel 单例。

    ChatModel 本身是无状态的（对话历史由消息列表携带），共享实例可以复用
    httpx 连接池，省掉每条请求的 TLS 握手。
    """

    global _model_cache
    if _model_cache is None:
        _model_cache = create_chat_model()
        logger.info(
            "LLM 初始化完成: model=%s base_url=%s",
            getattr(_model_cache, "model_name", "?"),
            getattr(_model_cache, "openai_api_base", "?"),
        )
    return _model_cache


def set_chat_model(model: Optional[BaseChatModel]) -> None:
    """注入自定义 ChatModel（离线冒烟测试 / 运行时切换到用）。"""

    global _model_cache
    _model_cache = model


async def ainvoke_structured(
    llm: BaseChatModel,
    schema: type[BaseModel],
    messages: Sequence[BaseMessage],
    *,
    preferred_method: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BaseModel:
    """调用模型并把结果约束到 ``schema``，失败时自动换一种实现方式重试。

    Raises:
        RuntimeError: 所有候选方式都失败（错误信息带上已尝试的方式，便于定位端点问题）。
    """

    settings = settings or get_settings()
    first = preferred_method or settings.llm_structured_method
    if first not in _FALLBACK_METHODS:
        first = _FALLBACK_METHODS[0]
    methods = (first,) + tuple(method for method in _FALLBACK_METHODS if method != first)

    last_error: Optional[Exception] = None

    for method in methods:
        try:
            if method == "prompted_json":
                result: Any = await _invoke_prompted_json(llm, schema, messages)
            else:
                runnable = llm.with_structured_output(schema, method=method)
                payload = list(messages)
                if method == "json_mode":
                    # json_object 模式要求提示词里明确出现 "JSON"，并最好附上字段约束
                    payload = payload + [HumanMessage(content=_json_instruction(schema))]
                result = await runnable.ainvoke(payload)
        except Exception as exc:  # noqa: BLE001 - 需要兜住任意端点异常以尝试下一种方式
            last_error = exc
            logger.warning(
                "结构化输出 method=%s 失败: %s: %s%s",
                method,
                type(exc).__name__,
                exc,
                _hint_for(exc),
            )
            continue

        if isinstance(result, schema):
            return result
        # 少数实现返回 dict / 带 .parsed 的包装对象，交给 Pydantic 校验
        return schema.model_validate(getattr(result, "parsed", result))

    raise RuntimeError(f"结构化输出失败（已尝试 {methods}）") from last_error


async def _invoke_prompted_json(
    llm: BaseChatModel,
    schema: type[BaseModel],
    messages: Sequence[BaseMessage],
) -> BaseModel:
    """自实现的兜底：**不设置 response_format**，纯提示词要求 JSON，再解析并校验。

    为什么还要这一层：``json_mode`` 会给请求打上 ``response_format={"type":"json_object"}`，
    而部分端点（实测 DeepSeek thinking 模式 + 长上下文）在这个组合下会把输出预算
    全花在 reasoning 上，``content`` 直接返回空 —— 于是 LangChain 的 JsonOutputParser
    报 ``Invalid json output:``（后面什么都没有）。

    去掉 ``response_format`` 之后就是一次普通对话，模型照提示词输出 JSON 文本；
    解析 + Pydantic 校验由我们自己做，失败也依然是**结构化**的失败，而不是静默降级。
    """

    payload = list(messages) + [HumanMessage(content=_json_instruction(schema))]
    response = await llm.ainvoke(payload)
    text = _message_text(response).strip()

    if not text:
        raise ValueError(
            "模型返回空内容（thinking 模型的推理 token 可能吃掉了输出预算，"
            "试试调大 LLM_MAX_TOKENS；或该端点对当前请求做了截断）"
        )

    return schema.model_validate(_extract_json_object(text))


def _extract_json_object(text: str) -> Any:
    """从模型回复里抽出 JSON 对象（容忍 ```json 围栏与前后解释文字）。"""

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"回复里找不到 JSON 对象：{text[:200]!r}")

    return json.loads(candidate[start : end + 1])


def _message_text(message: Any) -> str:
    """取消息纯文本（内容可能是分块列表）。"""

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


def _hint_for(exc: BaseException) -> str:
    """把常见失败翻译成一句可操作的提示，避免只看到异常类名。"""

    message = str(exc)
    if "Invalid json output" in message or "OUTPUT_PARSING_FAILURE" in message:
        return (
            "  ← 对方返回的 JSON 为空或不可解析。thinking 模型的推理 token 会占用输出预算，"
            "建议设置 LLM_MAX_TOKENS=8192，或改用 LLM_STRUCTURED_METHOD=prompted_json"
        )
    if "response_format" in message:
        return "  ← 该端点不支持这种 response_format，试试 LLM_STRUCTURED_METHOD=prompted_json"
    if "tool_choice" in message:
        return "  ← 该端点不允许强制 tool_choice（thinking 模式常见），试试 prompted_json"
    return ""


def _json_instruction(schema: type[BaseModel]) -> str:
    """要求模型只输出 JSON，并把字段约束摆给它看。

    两个作用：满足 OpenAI/DeepSeek 对「prompt 中必须出现 JSON」的硬性要求；
    把 schema 直接写进提示词，弥补 json_object 模式不做强约束的短板。
    """

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return (
        "请只输出一个 JSON 对象，不要 Markdown 代码块、不要任何解释文字。\n"
        "JSON 必须满足以下 JSON Schema：\n"
        f"{schema_json}"
    )
