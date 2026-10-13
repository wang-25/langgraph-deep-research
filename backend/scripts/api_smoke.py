"""HTTP / SSE 层冒烟：验证 FastAPI 端点与前端契约。

与 ``dry_run.py`` 的区别：那个直接调 runner，这个**走完整的 HTTP 栈**
（FastAPI 路由 → SSE 编码 → 客户端解析），确保 ``main.py`` 与
``frontend/src/services/api.ts`` 的解析逻辑真的对得上。

跑法::

    cd backend-langgraph
    python scripts/api_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

NOTES_DIR = ROOT / "notes-offline-smoke"

os.environ["LLM_PROVIDER"] = "custom"
os.environ["LLM_MODEL_ID"] = "gpt-4o-mini"
os.environ["LLM_API_KEY"] = "offline-smoke"
os.environ["LLM_BASE_URL"] = "http://localhost/unused"
os.environ["ENABLE_NOTES"] = "True"
os.environ["NOTES_WORKSPACE"] = str(NOTES_DIR)
os.environ["SEARCH_API"] = "duckduckgo"
os.environ["LANGSMITH_TRACING"] = "False"

from fastapi.testclient import TestClient  # noqa: E402
from fake_llm import ScriptedChatModel  # noqa: E402

import graph.llm as llm_module  # noqa: E402
from services import search as search_service  # noqa: E402

from dry_run import fake_dispatch_search  # noqa: E402  (复用同一个假搜索后端)

TOPIC = "LangGraph 与 Deep Research 的工程化"


def parse_sse(body: str) -> list[dict]:
    """按前端 ``api.ts`` 的方式解析 SSE：以空行分段，取 ``data:`` 之后的 JSON。"""

    events: list[dict] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        payload = block[5:].strip()
        if payload:
            events.append(json.loads(payload))
    return events


def main() -> int:
    if NOTES_DIR.exists():
        shutil.rmtree(NOTES_DIR)

    llm_module.set_chat_model(ScriptedChatModel())
    search_service.dispatch_search = fake_dispatch_search  # type: ignore[assignment]

    # 在导入 main 之前完成桩注入：create_app 里会编译图、读配置
    from main import app

    failures: list[str] = []

    with TestClient(app) as client:
        # ---- 0. 服务索引（不是 404） ----
        response = client.get("/")
        print(f"GET / -> {response.status_code}")
        if response.status_code != 200 or "endpoints" not in response.json():
            failures.append("/ 未返回服务索引")

        # ---- 0b. CORS 预检：前端 dev server 的 Origin 必须被放行 ----
        # （真实踩过的坑：frontend/vite.config.ts 的端口是 5174，不在默认放行列表里）
        for origin in ("http://localhost:5174", "http://127.0.0.1:5174"):
            response = client.options(
                "/research/stream",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            allowed = response.headers.get("access-control-allow-origin")
            print(f"OPTIONS (Origin={origin}) -> {response.status_code}, allow-origin={allowed}")
            if response.status_code != 200 or allowed != origin:
                failures.append(f"CORS 未放行前端 Origin {origin}")

        # ---- 1. 健康检查 ----
        response = client.get("/healthz")
        print(f"GET /healthz -> {response.status_code} {response.json()}")
        if response.status_code != 200 or response.json().get("status") != "ok":
            failures.append("/healthz 异常")

        # ---- 2. 图结构端点 ----
        response = client.get("/research/graph")
        graph_text = response.text
        print(f"GET /research/graph -> {response.status_code}, {len(graph_text)} 字符")
        for node in ("plan_todos", "research_task", "write_report", "persist_report"):
            if node not in graph_text:
                failures.append(f"图里缺少节点 {node}")

        # ---- 3. 同步研究 ----
        response = client.post("/research", json={"topic": TOPIC})
        print(f"POST /research -> {response.status_code}")
        if response.status_code != 200:
            failures.append(f"/research 返回 {response.status_code}: {response.text[:200]}")
        else:
            data = response.json()
            print(
                "   report={} 字符, todo_items={}, model_calls={}, cost={}".format(
                    len(data.get("report_markdown") or ""),
                    len(data.get("todo_items") or []),
                    data.get("model_calls"),
                    (data.get("usage") or {}).get("estimated_cost_usd"),
                )
            )
            if not data.get("report_markdown"):
                failures.append("/research 未返回报告")
            if len(data.get("todo_items") or []) < 1:
                failures.append("/research 未返回任务列表")
            first = (data.get("todo_items") or [{}])[0]
            for key in ("id", "title", "intent", "query", "status", "summary", "sources_summary"):
                if key not in first:
                    failures.append(f"/research 的任务项缺少字段 {key}")

        # ---- 4. 流式研究（前端真正使用的路径） ----
        response = client.post("/research/stream", json={"topic": TOPIC})
        print(f"POST /research/stream -> {response.status_code}, {len(response.text)} 字节")
        if response.status_code != 200:
            failures.append(f"/research/stream 返回 {response.status_code}")
        else:
            events = parse_sse(response.text)
            types = [event.get("type") for event in events]
            print(f"   解析出 {len(events)} 条事件: {types}")
            for required in ("status", "todo_list", "task_status", "sources", "tool_call", "final_report", "usage", "done"):
                if required not in types:
                    failures.append(f"SSE 缺少事件类型 {required}")
            if types and types[-1] != "done":
                failures.append("SSE 最后一条不是 done")
            report_events = [event for event in events if event.get("type") == "final_report"]
            if not (report_events and report_events[0].get("report")):
                failures.append("SSE 的 final_report 为空")

        # ---- 5. 参数校验：非法 search_api 应返回 400 ----
        response = client.post("/research", json={"topic": TOPIC, "search_api": "not-a-backend"})
        print(f"POST /research (非法 search_api) -> {response.status_code}")
        if response.status_code != 400:
            failures.append(f"非法 search_api 未返回 400（实际 {response.status_code}）")

        # ---- 6. 覆盖 search_api 应生效 ----
        response = client.post("/research/stream", json={"topic": TOPIC, "search_api": "tavily"})
        if response.status_code != 200:
            failures.append("search_api 覆盖请求失败")
        else:
            print("POST /research/stream (search_api=tavily 覆盖) -> 200")

    print("-" * 78)
    if failures:
        print("❌ HTTP 冒烟失败：")
        for item in failures:
            print(f"   - {item}")
        return 1

    print("✅ HTTP 冒烟通过：路由、SSE 编码、事件契约、参数校验全部正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
