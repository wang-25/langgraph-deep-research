"""离线端到端冒烟：不联网、不需要 API Key。

做三件事：

1. 用 ``ScriptedChatModel`` 顶替真实模型（真工具调用 + 真结构化输出协议）；
2. 用假搜索后端顶替 ``dispatch_search``（真 ToolMessage + 真来源格式化）；
3. 跑完整张图，把 SSE 事件逐条打印并**断言前端契约的每一个事件类型都出现过**，
   同时校验任务笔记、结论笔记确实落盘。

跑法::

    cd backend-langgraph
    python scripts/dry_run.py                # 纯离线：假模型 + 假搜索（CI 回归用）
    python scripts/dry_run.py --live-llm     # 真模型（读 .env）+ 假搜索

``--live-llm`` 的用途是**隔离变量**：真实联调时「搜索不通」和「模型端点不给力」
会同时出现，很容易互相掩盖。固定住来源、只换模型，就能单独判断模型这一侧的问题。

退出码非 0 表示契约被破坏。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

NOTES_DIR = ROOT / "notes-offline-smoke"

_args = argparse.ArgumentParser(description="离线端到端冒烟")
_args.add_argument(
    "--live-llm",
    action="store_true",
    help="使用 .env 里的真实模型（搜索仍用假后端，便于隔离变量）",
)
ARGS = _args.parse_args()

# 必须在导入 config 之前设置：Settings 是带缓存的单例
if not ARGS.live_llm:
    os.environ["LLM_PROVIDER"] = "custom"
    os.environ["LLM_MODEL_ID"] = "gpt-4o-mini"
    os.environ["LLM_API_KEY"] = "offline-smoke"
    os.environ["LLM_BASE_URL"] = "http://localhost/unused"
os.environ["ENABLE_NOTES"] = "True"
os.environ["NOTES_WORKSPACE"] = str(NOTES_DIR)
os.environ["SEARCH_API"] = "duckduckgo"
os.environ["LANGSMITH_TRACING"] = "False"

from fake_llm import ScriptedChatModel  # noqa: E402

import graph.llm as llm_module  # noqa: E402
from graph.runner import get_runner  # noqa: E402
from services import search as search_service  # noqa: E402

TOPIC = "LangGraph 在深度研究 Agent 中的工程实践"

_FAKE_RESULTS = [
    {
        "title": "LangGraph 官方文档：Map-Reduce 与 Send API",
        "url": "https://langchain-ai.github.io/langgraph/",
        "content": "Send 用于动态扇出并行分支，配合 reducer 做 fan-in 归并。",
        "raw_content": "Send(node, state) 会把 state 作为该子节点的输入状态。" * 20,
    },
    {
        "title": "Deep Agents 与 open_deep_research",
        "url": "https://github.com/langchain-ai/open_deep_research",
        "content": "官方深度研究实现：规划、并行检索、总结、成稿。",
        "raw_content": "open_deep_research 使用 LangGraph 组织 planner / researcher / writer。" * 20,
    },
    {
        "title": "Structured output 实践",
        "url": "https://python.langchain.com/docs/concepts/structured_outputs/",
        "content": "with_structured_output 支持 json_schema 与 function_calling 两种实现。",
        "raw_content": "结构化输出把解析环节交给框架，避免正则抠 JSON。" * 20,
    },
]


async def fake_dispatch_search(query, *, api=None, settings=None):  # noqa: ANN001, ARG001
    """假搜索后端：返回固定来源，形状与真实后端一致。"""

    await asyncio.sleep(0.05)  # 模拟网络延迟，顺便验证并行度
    return search_service.SearchOutcome(
        backend="offline-fake",
        results=[dict(item) for item in _FAKE_RESULTS],
        answer="（离线冒烟）检索后端直接答案：LangGraph 用 Send 做并行扇出。",
        notices=[],
    )


EXPECTED_TYPES = {
    "status",
    "todo_list",
    "task_status",
    "sources",
    "task_summary_chunk",
    "tool_call",
    "report_note",
    "final_report",
    "usage",
    "done",
}


async def main() -> int:
    if NOTES_DIR.exists():
        shutil.rmtree(NOTES_DIR)

    if not ARGS.live_llm:
        llm_module.set_chat_model(ScriptedChatModel())
    search_service.dispatch_search = fake_dispatch_search  # type: ignore[assignment]

    mode = "真实模型 + 固定假来源" if ARGS.live_llm else "假模型 + 假来源"
    print("=" * 78)
    print(f"端到端冒烟：{mode}")
    print("=" * 78)

    events: list[dict] = []
    async for event in get_runner().stream(TOPIC):
        events.append(event)
        preview = json.dumps(event, ensure_ascii=False)
        print(f"  [{event.get('type'):<20}] {preview[:120]}")

    counts = Counter(event.get("type") for event in events)
    failures: list[str] = []

    print("\n" + "-" * 78)
    print("事件统计:", dict(counts))

    # ---- 1. 事件类型契约 ----
    # 真实模型下 note_update 是「模型自觉」行为，不保证发生，因此不算硬性契约；
    # 其余事件都由代码路径决定，两种模式都必须出现。
    expected = set(EXPECTED_TYPES)
    if ARGS.live_llm:
        expected.discard("tool_call")
    missing = expected - set(counts)
    if missing:
        failures.append(f"缺少事件类型: {sorted(missing)}")

    # ---- 2. 任务数量与状态 ----
    todo_event = next((e for e in events if e["type"] == "todo_list"), None)
    tasks = (todo_event or {}).get("tasks") or []
    completed = [e for e in events if e["type"] == "task_status" and e.get("status") == "completed"]
    in_progress = [e for e in events if e["type"] == "task_status" and e.get("status") == "in_progress"]

    print(f"任务数: {len(tasks)} | in_progress 事件: {len(in_progress)} | completed 事件: {len(completed)}")
    if not tasks:
        failures.append("未生成任何任务")
    if len(in_progress) != len(tasks):
        failures.append(f"in_progress 事件数({len(in_progress)}) != 任务数({len(tasks)})")
    if len(completed) != len(tasks):
        failures.append(f"completed 事件数({len(completed)}) != 任务数({len(tasks)})")

    # ---- 3. 工具调用事件必须带真实参数与结果 ----
    tool_events = [e for e in events if e["type"] == "tool_call"]
    tool_names = sorted({e["tool"] for e in tool_events})
    print(f"工具调用: {len(tool_events)} 次，工具={tool_names}")
    if "web_search" not in tool_names:
        failures.append("没有观察到 web_search 的真实工具调用")
    if "note_update" not in tool_names and not ARGS.live_llm:
        # 假模型的剧本一定会调 note_update；真实模型只要求「不报错」，不强制它照做
        failures.append("没有观察到 note_update 的真实工具调用")
    for event in tool_events:
        if not event.get("parameters"):
            failures.append(f"工具调用 {event['tool']} 缺少参数")
        if not event.get("result"):
            failures.append(f"工具调用 {event['tool']} 缺少返回内容")
        if event.get("task_id") is None:
            failures.append(f"工具调用 {event['tool']} 缺少 task_id（前端无法归组）")

    # ---- 4. 结构化输出字段 ----
    summary_chunks = [e for e in events if e["type"] == "task_summary_chunk"]
    if not all(e.get("content") for e in summary_chunks):
        failures.append("存在空的 task_summary_chunk")
    print(f"结构化总结片段: {len(summary_chunks)} 条")

    # ---- 5. 报告与用量 ----
    final = next((e for e in events if e["type"] == "final_report"), None)
    report = (final or {}).get("report") or ""
    print(f"最终报告长度: {len(report)} 字符")
    if len(report) < 50:
        failures.append("最终报告过短，流式累计可能失效")

    usage = next((e for e in events if e["type"] == "usage"), None) or {}
    print(
        "用量: llm_calls={} tokens={} cost={} pricing_known={}".format(
            usage.get("llm_calls"),
            usage.get("total_tokens"),
            usage.get("estimated_cost_usd"),
            usage.get("pricing_known"),
        )
    )
    print("分模型明细:")
    for item in usage.get("models") or []:
        print(
            "   - {model}: calls={calls} in={input_tokens} out={output_tokens} cost={estimated_cost_usd}".format(
                **item
            )
        )
    if not usage.get("llm_calls"):
        failures.append("usage 事件没有统计到 LLM 调用")
    if not usage.get("total_tokens"):
        failures.append("usage 事件没有统计到 token")
    if not ARGS.live_llm and usage.get("estimated_cost_usd") in (None, 0):
        # 真实模型可能不在价目表里（例如 deepseek-flash），此时 pricing_known=False 是**正确**行为
        failures.append("成本估算未生效（gpt-4o-mini 应在价目表内）")

    # ---- 6. 笔记落盘：一次研究 = 一个目录 + 有意义的文件名 + 合法 YAML 头 ----
    run_dirs = sorted(p for p in NOTES_DIR.iterdir() if p.is_dir()) if NOTES_DIR.exists() else []
    print(f"研究目录: {[p.name for p in run_dirs]}")
    if len(run_dirs) != 1:
        failures.append(f"应生成 1 个研究目录，实际 {len(run_dirs)} 个")
        note_files: list[Path] = []
        run_dir = None
    else:
        run_dir = run_dirs[0]
        note_files = sorted(run_dir.glob("*.md"))

    print(f"笔记文件: {len(note_files)} 个 -> {[p.name for p in note_files]}")
    # 任务笔记 + 00-研究概要 + 研究报告
    if len(note_files) != len(tasks) + 2:
        failures.append(f"笔记文件数({len(note_files)}) != 任务数+2({len(tasks) + 2})")

    # 文件名必须是「序号-任务名」这种可读形式，不能再是 note-<时间戳>-<随机>
    for path in note_files:
        if re.match(r"^note-\d{14}-[0-9a-f]+\.md$", path.name):
            failures.append(f"文件名仍是旧的随机形式：{path.name}")

    task_notes = [
        p
        for p in note_files
        if re.match(r"^\d{2}-", p.name) and p.name != "00-研究概要.md"
    ]
    if len(task_notes) != len(tasks):
        failures.append(f"带序号的任务笔记数({len(task_notes)}) != 任务数({len(tasks)})")

    report_path = run_dir / "研究报告.md" if run_dir else None
    if report_path is None or not report_path.exists():
        failures.append("没有生成「研究报告.md」")
    elif report[:40] not in report_path.read_text(encoding="utf-8"):
        failures.append("报告笔记内容与最终报告不一致")

    index_path = run_dir / "00-研究概要.md" if run_dir else None
    if index_path is None or not index_path.exists():
        failures.append("没有生成「00-研究概要.md」目录索引")
    else:
        index_text = index_path.read_text(encoding="utf-8")
        if "研究报告.md" not in index_text or "任务清单" not in index_text:
            failures.append("研究概要缺少任务清单或报告链接")

    # front matter 必须是合法 YAML——用真解析器验，避免又出现「编辑器加载失败」
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        print("（未安装 PyYAML，跳过 front matter 校验）")
    else:
        for path in note_files:
            raw = path.read_text(encoding="utf-8")
            match = re.match(r"^---\n(.*?)\n---\n", raw, re.DOTALL)
            if not match:
                failures.append(f"{path.name} 缺少 front matter")
                continue
            try:
                yaml.safe_load(match.group(1))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path.name} 的 YAML 头解析失败：{str(exc).splitlines()[0]}")
        print(f"front matter YAML 校验: {len(note_files)} 个文件")

    print("-" * 78)
    if failures:
        print("❌ 冒烟失败：")
        for item in failures:
            print(f"   - {item}")
        return 1

    if ARGS.live_llm:
        print("✅ 冒烟通过：真实模型 + 固定来源下，规划/工具循环/结构化总结/报告/笔记全链路正常")
    else:
        print("✅ 冒烟通过：图、并行扇出、真工具调用、结构化输出、SSE 契约、笔记链路全部正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
