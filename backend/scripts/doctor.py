"""配置体检（doctor）：把「跑不出结果」拆成可定位的几项。

真实联调最容易遇到的不是代码 bug，而是**配置/网络/端点能力**问题：

1. 模型名或 base_url 写错 → 连不上或 404；
2. 端点不支持 ``json_schema`` 严格模式 → 结构化输出失败（代码会自动降级到 function_calling）；
3. 端点不支持 tool calling → ReAct 循环失败（代码会降级为纯结构化调用）；
4. 搜索后端在当前网络不可达（例如 Google/DDG 被墙）→ 检索永远 0 结果。

本脚本逐项真打一次，把**异常原文**打出来，避免只看到「降级了」却不知道降级原因。

用法::

    python scripts/doctor.py                  # 全项检查
    python scripts/doctor.py --skip-search    # 只查模型
    python scripts/doctor.py --skip-llm       # 只查搜索
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import get_settings  # noqa: E402


@tool("echo")
def echo(text: str) -> str:
    """把输入原样返回。"""

    return text


class Probe(BaseModel):
    """结构化输出探针 schema。"""

    summary: str = Field(description="一句话总结")
    keywords: list[str] = Field(description="2-3 个关键词")


def _line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _exc_text(exc: BaseException, limit: int = 900) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + "…"


async def check_llm() -> bool:
    """三步探针：普通对话 → 工具调用 → 结构化输出（两种 method）。"""

    from graph.llm import create_chat_model

    settings = get_settings()
    _line("1. 模型端点体检")
    print(f"provider={settings.llm_provider}")
    print(f"model={settings.resolved_llm_model()}")
    print(f"base_url={settings.resolved_llm_base_url()}")
    print(f"structured_method={settings.llm_structured_method}")

    try:
        llm = create_chat_model(settings=settings)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 构造 ChatModel 失败: {_exc_text(exc)}")
        return False

    ok = True

    # ---- 1a. 普通对话 ----
    try:
        reply = await llm.ainvoke([HumanMessage(content="只回复两个字：收到")])
        print(f"✅ 普通对话可用: {str(reply.content)[:60]!r}")
        usage = getattr(reply, "usage_metadata", None)
        print(f"   usage_metadata={usage}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 普通对话失败（模型名/base_url/key 有问题）: {_exc_text(exc)}")
        return False

    # ---- 1b. 工具调用 ----
    try:
        bound = llm.bind_tools([echo])
        reply = await bound.ainvoke([HumanMessage(content="用 echo 工具回显 hello")])
        calls = getattr(reply, "tool_calls", None) or []
        if calls:
            print(f"✅ 工具调用可用: {[c.get('name') for c in calls]}")
        else:
            print("⚠️ 端点接受 bind_tools，但模型没有返回 tool_calls（可能不支持工具调用）")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 工具调用失败（ReAct 循环会降级）: {_exc_text(exc)}")
        ok = False

    # ---- 1c. 结构化输出：四种方式逐个「作为首选」试，并区分「本身可用」与「靠降级成功」 ----
    from graph.llm import STRUCTURED_METHODS, ainvoke_structured

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    capture = _Capture()
    llm_logger = logging.getLogger("graph.llm")
    llm_logger.addHandler(capture)

    try:
        for method in STRUCTURED_METHODS:
            captured.clear()
            try:
                result = await ainvoke_structured(
                    llm,
                    Probe,
                    [
                        HumanMessage(
                            content="用一句话总结：LangGraph 用图组织 Agent 流程，并支持并行扇出。"
                        )
                    ],
                    preferred_method=method,
                    settings=settings.model_copy(update={"llm_structured_method": method}),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 结构化输出 method={method}: 不可用（且降级也失败）: {_exc_text(exc)}")
                continue

            degraded = any(f"method={method} 失败" in item for item in captured)
            if degraded:
                print(f"⚠️ 结构化输出 method={method}: 本身不可用，靠降级拿到结果 {result!r}")
            else:
                print(f"✅ 结构化输出 method={method}: 可用 {result!r}")

        # ---- 1c-bis. 长上下文压力测试：thinking 模型最容易在这里露馅 ----
        # 推理 token 与正文共享输出预算，上下文一大就可能「推理写满、正文为空」。
        filler = "LangGraph 的 Send API 用于动态扇出并行分支。" * 400
        captured.clear()
        try:
            result = await ainvoke_structured(
                llm,
                Probe,
                [
                    HumanMessage(
                        content=f"以下是一段很长的检索材料：\n{filler}\n\n"
                        "请据此用一句话总结，并给出 2-3 个关键词。"
                    )
                ],
            )
            degraded = any("失败" in item for item in captured)
            tag = "（走的是降级路径）" if degraded else ""
            print(f"✅ 长上下文（约 {len(filler)} 字）结构化输出可用{tag}: {result!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 长上下文结构化输出失败: {_exc_text(exc)}")
            print("   → 多半是输出预算被推理吃掉了：试试 LLM_MAX_TOKENS=8192")
            ok = False
    finally:
        llm_logger.removeHandler(capture)

    # ---- 1d. 迷你 ReAct 子图（验证工具循环本身能不能跑完） ----
    try:
        from graph.agents import agent_api_mode, build_react_agent

        agent = build_react_agent(
            llm, [echo], system_prompt="你需要用 echo 工具回显用户给的词。", name="echo_agent"
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="回显 hello")]})
        tool_messages = [
            m for m in (result.get("messages") or []) if type(m).__name__ == "ToolMessage"
        ]
        print(
            f"✅ ReAct 工具循环跑通，工具返回 {len(tool_messages)} 条"
            f"（构造 API: {agent_api_mode()}）"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"❌ ReAct 工具循环失败: {_exc_text(exc)}")
        ok = False

    return ok


async def check_search() -> bool:
    """逐个搜索后端真打一次，并逐个 ddgs backend 试。"""

    from services import search as search_service

    settings = get_settings()
    _line("2. 搜索后端体检")
    print(f"search_api={settings.search_api.value} fetch_full_page={settings.fetch_full_page}")
    print(f"tavily_key={'yes' if settings.tavily_api_key else 'no'} "
          f"perplexity_key={'yes' if settings.perplexity_api_key else 'no'}")

    query = "LangGraph tutorial"
    ok = False

    # ---- 配置的后端 ----
    outcome = await search_service.dispatch_search(query, settings=settings)
    print(
        f"\n[{settings.search_api.value}] results={len(outcome.results)} "
        f"notices={outcome.notices}"
    )
    for item in outcome.results[:3]:
        print(f"   - {item.get('title')} <{item.get('url')}>")
    ok = bool(outcome.results)

    # ---- 直连 ddgs，逐个 backend 试（定位「是不是某个上游被墙」）----
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
        import inspect

        print(f"\nddgs 已安装，DDGS.text 签名: {inspect.signature(DDGS.text)}")
        for backend in ("auto", "duckduckgo", "brave", "google", "mojeek", "yahoo"):
            try:
                items = await asyncio.to_thread(
                    lambda b=backend: list(DDGS().text(query, max_results=3, backend=b))
                )
                print(f"   backend={backend:<12} -> {len(items)} 条")
                if items:
                    ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"   backend={backend:<12} -> ❌ {_exc_text(exc, 160)}")
    except ImportError:
        print("\n⚠️ 未安装 ddgs：pip install -e \".[duckduckgo]\"")
    except Exception as exc:  # noqa: BLE001
        print(f"\n⚠️ ddgs 探测异常: {_exc_text(exc)}")

    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description="Deep Research 配置体检")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()

    llm_ok = True
    search_ok = True

    if not args.skip_llm:
        try:
            llm_ok = await check_llm()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            llm_ok = False

    if not args.skip_search:
        try:
            search_ok = await check_search()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            search_ok = False

    _line("体检结论")
    print(f"模型端点: {'✅ 可用' if llm_ok else '❌ 有项目不可用（详见上方异常）'}")
    print(f"搜索后端: {'✅ 可用' if search_ok else '❌ 全部不可用（检索会返回 0 结果）'}")
    if not (llm_ok and search_ok):
        print(
            "\n排查建议：\n"
            "  * 模型 404/401 → 核对 LLM_MODEL_ID / LLM_BASE_URL / LLM_API_KEY；\n"
            "  * 工具调用或 json_schema 报错 → 换 LLM_STRUCTURED_METHOD，或换支持 function calling 的模型；\n"
            "  * 搜索全超时 → 当前网络到不了 Google/DDG，改用 Tavily(key) / Perplexity(key) / 自建 SearXNG。"
        )
    return 0 if (llm_ok and search_ok) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
