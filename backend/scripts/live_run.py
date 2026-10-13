"""真机联调脚本：对**正在运行**的后端发一次真实研究请求，实时打印事件流。

与两个冒烟脚本的分工：

* ``dry_run.py`` / ``api_smoke.py``：进程内、假模型、零成本，验证契约与回归；
* ``live_run.py``（本脚本）：走真实 HTTP + 真实 LLM + 真实搜索，验证「端到端真的能用」。

用法::

    python scripts/live_run.py --topic "LangGraph 的核心概念"
    python scripts/live_run.py --topic "向量数据库选型" --search-api tavily
    python scripts/live_run.py --topic "..." --url http://127.0.0.1:8000

退出码非 0 表示没有拿到有效报告。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Iterator

import httpx

#: 每个事件类型在控制台里的显示宽度
_LABEL_WIDTH = 20


def iter_sse(response: httpx.Response) -> Iterator[dict[str, Any]]:
    """逐行解析 SSE（与前端 ``api.ts`` 的解析逻辑一致：空行分段 + ``data:`` 前缀）。"""

    for line in response.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def describe(event: dict[str, Any]) -> str:
    """把一条事件压成一行人类可读的摘要。"""

    kind = event.get("type")
    if kind == "status":
        return str(event.get("message") or "")
    if kind == "todo_list":
        tasks = event.get("tasks") or []
        return f"{len(tasks)} 个任务: " + " / ".join(str(t.get("title")) for t in tasks)
    if kind == "task_status":
        detail = event.get("status")
        if event.get("summary"):
            detail = f"{detail}, 总结 {len(str(event['summary']))} 字"
        return f"任务 {event.get('task_id')} -> {detail}"
    if kind == "sources":
        raw = str(event.get("latest_sources") or "")
        return f"任务 {event.get('task_id')} backend={event.get('backend')} 来源 {raw.count('*')} 条"
    if kind == "tool_call":
        return f"[任务 {event.get('task_id')}] {event.get('tool')}({_short(event.get('parameters'))})"
    if kind == "task_summary_chunk":
        return f"任务 {event.get('task_id')} 总结 {len(str(event.get('content') or ''))} 字"
    if kind == "report_note":
        return f"结论笔记 {event.get('note_id')}"
    if kind == "final_report":
        return f"报告 {len(str(event.get('report') or ''))} 字"
    if kind == "usage":
        return (
            f"{event.get('llm_calls')} 次调用, {event.get('total_tokens')} tokens, "
            f"${event.get('estimated_cost_usd')}"
        )
    if kind == "error":
        return str(event.get("detail") or "")
    return ""


def _short(value: Any, limit: int = 70) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description="对运行中的后端发一次真实研究请求")
    parser.add_argument("--topic", required=True, help="研究主题")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="后端地址")
    parser.add_argument("--search-api", default=None, help="覆盖搜索后端")
    parser.add_argument("--timeout", type=float, default=900.0, help="整体超时（秒）")
    args = parser.parse_args()

    payload: dict[str, Any] = {"topic": args.topic}
    if args.search_api:
        payload["search_api"] = args.search_api

    print("=" * 84)
    print(f"真实研究请求 -> {args.url}/research/stream")
    print(f"主题: {args.topic}")
    print("=" * 84)

    started = time.monotonic()
    events: list[dict[str, Any]] = []
    final_report = ""
    usage: dict[str, Any] = {}
    error_detail = ""

    try:
        with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=10.0)) as client:
            with client.stream("POST", f"{args.url}/research/stream", json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    print(f"❌ HTTP {response.status_code}: {response.text[:500]}")
                    return 1
                for event in iter_sse(response):
                    events.append(event)
                    elapsed = time.monotonic() - started
                    print(
                        f"[{elapsed:7.2f}s] {str(event.get('type')):<{_LABEL_WIDTH}} "
                        f"{describe(event)}"
                    )
                    if event.get("type") == "final_report":
                        final_report = str(event.get("report") or "")
                    elif event.get("type") == "usage":
                        usage = event
                    elif event.get("type") == "error":
                        error_detail = str(event.get("detail") or "")
    except httpx.HTTPError as exc:
        print(f"❌ 请求失败: {type(exc).__name__}: {exc}")
        print("   后端起了吗？试试 GET /healthz")
        return 1

    elapsed = time.monotonic() - started
    kinds = [event.get("type") for event in events]
    tasks = next((e.get("tasks") or [] for e in events if e.get("type") == "todo_list"), [])
    completed = [e for e in events if e.get("type") == "task_status" and e.get("status") == "completed"]
    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    tools_used = sorted({str(e.get("tool")) for e in tool_calls})

    print("-" * 84)
    print(f"耗时: {elapsed:.1f}s | 事件: {len(events)} 条 {sorted(set(kinds))}")
    print(f"任务: {len(tasks)} 个，完成 {len(completed)} 个")
    print(f"工具调用: {len(tool_calls)} 次 {tools_used}")
    print(
        "用量: {} 次调用 | tokens in/out {}/{} | 预估成本 ${}".format(
            usage.get("llm_calls"),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("estimated_cost_usd"),
        )
    )
    for item in usage.get("models") or []:
        print(f"   - {item.get('model')}: calls={item.get('calls')} tokens={item.get('total_tokens')}")
    print(f"报告长度: {len(final_report)} 字")

    if final_report.strip():
        preview = final_report.strip()[:600]
        print("-" * 84)
        print("报告开头：")
        print(preview + ("…" if len(final_report) > 600 else ""))

    print("-" * 84)
    if error_detail:
        print(f"❌ 流程报错: {error_detail}")
        return 1
    if not final_report.strip():
        print("❌ 没有拿到有效报告")
        return 1
    print("✅ 真实链路跑通")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
