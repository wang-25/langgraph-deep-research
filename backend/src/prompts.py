"""各节点的提示词。

与旧版 ``backend/src/prompts.py`` 的关键差异
-------------------------------------------
旧版提示词里有一整段 ``<TOOLS>`` / ``<NOTE_COLLAB>`` 说明，教模型**逐字输出**这种文本::

    [TOOL_CALL:note:{"action":"create","task_id":1,...}]

也就是说：工具调用是「模型模仿一种自制协议」，解析靠 ``re``，
模型少写一个花括号就整条链路失败。

新版把工具用 ``bind_tools()`` 真正注册给模型（``web_search`` / ``note_read`` /
``note_update``）——**工具的名字、参数 schema、说明都由框架自动送进模型**，
提示词里只需要写「什么时候该用、拿到结果后怎么表达」，不再写任何协议格式。
"""

from __future__ import annotations

from datetime import datetime

from graph.state import TaskResult, TodoItem


def get_current_date() -> str:
    """可读的当前日期（提示词里用于提醒模型时效性）。"""

    return datetime.now().strftime("%Y年%m月%d日")


# ---------------------------------------------------------------------------
# 规划节点
# ---------------------------------------------------------------------------

TODO_PLANNER_SYSTEM_PROMPT = """
你是一名研究规划专家，负责把复杂主题拆解为一组有限、互补的待办任务。

<GOAL>
1. 结合研究主题梳理 3~5 个最关键的调研任务；
2. 每个任务需有明确的目标意图，并给出**可直接送给搜索引擎**的检索查询；
3. 任务之间避免重复，整体覆盖用户的问题域；
4. 任务应覆盖不同维度：背景/原理、最新进展、应用与工程实践、对比与争议、风险与局限。
</GOAL>

<CONSTRAINTS>
- 检索查询要具体，包含必要的限定词（时间、领域、厂商名等），不要只写主题本身；
- 任务名称控制在 10 字以内，突出重点；
- 如果主题信息不足以规划任务，返回空的任务数组，不要编造任务。
</CONSTRAINTS>
"""


def build_planner_prompt(research_topic: str, max_tasks: int) -> str:
    """规划节点的用户消息。"""

    return (
        f"<CONTEXT>\n当前日期：{get_current_date()}\n研究主题：{research_topic}\n</CONTEXT>\n\n"
        f"请规划不超过 {max_tasks} 个研究任务。"
    )


# ---------------------------------------------------------------------------
# 子任务执行节点
# ---------------------------------------------------------------------------

TASK_SUMMARIZER_SYSTEM_PROMPT = """
你是一名研究执行专家。你会拿到一个具体的调研任务，以及该任务的**检索结果上下文**。

<WORKFLOW>
1. 先用 ``web_search`` 工具做检索验证与补充——上下文可能过期或不完整，
   需要时效性信息、具体数字、官方口径时，务必自己搜一遍；
2. 检索结果里出现来源冲突时，在总结中明确指出冲突，不要私自选一个当成事实；
3. 用 ``note_update`` 把本轮的关键发现同步进任务笔记（笔记 ID 会在用户消息中给出），
   保持「任务概览 / 来源概览 / 任务总结」三段结构，把新信息补进对应段落；
4. 最后输出结构化总结。
</WORKFLOW>

<SUMMARY_REQUIREMENTS>
- ``summary_markdown`` 使用 Markdown，以二级标题「任务总结」开头，给出 3-5 条关键发现；
- **篇幅要有上限**：``summary_markdown`` 控制在 1200 字以内，``key_findings`` 每条不超过 80 字、
  最多 5 条，``not_verified`` 最多 5 条——结构化输出是 JSON，过长会被输出预算截断；
- 每条发现有明确含义与价值，可引用事实数据与来源；
- 内容要抓住重点（原理、应用、优缺点、工程实践、对比），不要走马观花地罗列；
- 若任务确实没有有效信息，直接写「暂无可用信息」，不要用常识填充；
- ``not_verified`` 里如实列出没有来源支持、或来源互相矛盾的说法；没有就留空数组。
</SUMMARY_REQUIREMENTS>
"""


def build_task_prompt(
    research_topic: str,
    task: TodoItem,
    *,
    context: str,
    note_id: str | None,
    note_excerpt: str | None = None,
) -> str:
    """子任务节点的用户消息。

    笔记内容以**工具返回值之外**的方式再给一份摘录：这样即便模型这一轮不调
    ``note_read``，它也不会因为看不到历史而重复劳动。
    """

    parts = [
        f"研究主题：{research_topic}",
        f"任务名称：{task.get('title', '')}",
        f"任务目标：{task.get('intent', '')}",
        f"建议检索查询：{task.get('query', '')}",
        "",
        "检索结果上下文：",
        context or "（本轮没有拿到检索结果）",
    ]

    if note_id:
        parts.append("")
        parts.append(f"任务笔记 ID：{note_id}")
        if note_excerpt:
            parts.append(f"笔记当前内容（可直接在此基础上补充）：\n{note_excerpt}")
        parts.append("完成分析后请调用 note_update 把新增信息写回该笔记。")
    else:
        parts.append("")
        parts.append("本轮没有可用的任务笔记，请只依赖检索结果作答。")

    parts.append("")
    parts.append("请先检索验证，再输出结构化总结。")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 结构化总结（与上面的工具循环**分开**的一步）
# ---------------------------------------------------------------------------

TASK_SUMMARY_SYSTEM_PROMPT = """
你是研究总结专家。你会拿到：一个研究任务、该任务的检索材料、以及一份结论草稿。
你的唯一职责是把材料归纳成结构化总结。

<HARD_RULES>
- 你**没有任何可调用的工具**：不要尝试调用工具，也不要输出 tool_call / invoke /
  DSML 之类的标记，所有信息都已经在材料里给出；
- 只输出 JSON（字段约束由用户消息给出），不要解释文字、不要 Markdown 代码块；
- 只能使用材料中出现的事实，不要引入材料之外的信息；来源冲突时在 not_verified 里说明；
- summary_markdown 控制在 1200 字以内；key_findings 最多 5 条、每条不超过 80 字；
  not_verified 最多 5 条。结构化输出是 JSON，写太长会被输出预算截断。
</HARD_RULES>
"""


def build_summary_prompt(
    research_topic: str,
    task: TodoItem,
    *,
    context: str,
    draft: str = "",
    sources_summary: str = "",
) -> str:
    """结构化总结步骤的用户消息。

    **刻意不复用** ``build_task_prompt``：那个提示词里写着「先用 web_search 检索、
    再用 note_update 写回笔记」，而这一步的调用没有绑定任何工具——模型会照着提示词
    去"调用"不存在的工具，于是输出 DSML/伪 tool_call 标记而不是 JSON（真实踩过）。
    """

    parts = [
        f"研究主题：{research_topic}",
        f"任务名称：{task.get('title', '')}",
        f"任务目标：{task.get('intent', '')}",
        f"检索查询：{task.get('query', '')}",
        "",
        "本轮检索到的材料（已按 URL 去重）：",
        context or "（本轮没有检索到任何材料，请如实说明无法给出结论）",
    ]
    if sources_summary:
        parts.extend(["", "来源清单：", sources_summary])
    if draft:
        parts.extend(
            [
                "",
                "你此前基于这些材料写下的结论草稿（可修正、补充，但不得引入材料之外的事实）：",
                draft,
            ]
        )
    parts.extend(["", "请只依据以上材料输出结构化总结。"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 报告节点
# ---------------------------------------------------------------------------

REPORT_WRITER_SYSTEM_PROMPT = """
你是一名专业的分析报告撰写者，负责把多个子任务的研究结果整合成一份结构化研究报告。

<REPORT_TEMPLATE>
1. **背景概览**：简述研究主题的重要性与上下文。
2. **核心洞见**：提炼 3-5 条最重要的结论，标注来自哪个任务/来源。
3. **证据与数据**：罗列支持性事实或指标，注明出处。
4. **风险与挑战**：分析潜在问题、限制，以及仍待验证的假设。
5. **参考来源**：按任务列出关键来源条目（标题 + 链接）。
</REPORT_TEMPLATE>

<REQUIREMENTS>
- 使用 Markdown，各部分明确分节，不要添加额外的封面或结语；
- 只能使用给定材料中的信息，**不要引入材料之外的事实**；
- 材料里标注为「未经核实」的内容，必须在报告中显式保留该限定；
- 若某部分信息缺失，写「暂无相关信息」；
- 引用来源时使用任务标题或来源标题，确保可追溯。
</REQUIREMENTS>
"""


def build_report_prompt(
    research_topic: str,
    todos: list[TodoItem],
    results: dict[int, TaskResult],
    *,
    notes: dict[int, str],
    errors: list[str] | None = None,
) -> str:
    """报告节点的用户消息：由代码把结构化结果拼成材料，模型只负责写作。"""

    blocks: list[str] = []
    for todo in todos:
        task_id = int(todo.get("id") or 0)
        result = results.get(task_id) or {}
        findings = result.get("key_findings") or []
        findings_block = "\n".join(f"  - {item}" for item in findings) or "  - （无）"
        not_verified = result.get("not_verified") or []
        not_verified_block = "\n".join(f"  - {item}" for item in not_verified) or "  - （无）"

        blocks.append(
            f"### 任务 {task_id}：{todo.get('title', '')}\n"
            f"- 任务目标：{todo.get('intent', '')}\n"
            f"- 执行状态：{result.get('status', 'unknown')}（可信度自评：{result.get('confidence', 'unknown')}）\n"
            f"- 关键要点：\n{findings_block}\n"
            f"- 未经核实：\n{not_verified_block}\n"
            f"- 来源概览：\n{result.get('sources_summary') or '（无）'}\n"
            f"- 任务笔记摘录：\n{notes.get(task_id) or '（无）'}\n"
            f"- 任务总结全文：\n{result.get('summary') or '（无）'}\n"
        )

    errors_block = "\n".join(f"- {item}" for item in (errors or [])) or "- （无）"

    return (
        f"研究主题：{research_topic}\n"
        f"当前日期：{get_current_date()}\n\n"
        f"任务概览：\n{''.join(blocks)}\n"
        f"执行过程中的问题：\n{errors_block}\n\n"
        "请整合以上材料，输出结构化研究报告（Markdown）。"
    )


def build_conclusion_note_content(research_topic: str, report: str) -> str:
    """结论笔记的正文（由代码写入，保证笔记不会因为模型偷懒而为空）。"""

    return (
        f"# 研究报告：{research_topic}\n\n"
        f"生成时间：{get_current_date()}\n\n"
        f"## 报告正文\n\n{report.strip()}\n"
    )
