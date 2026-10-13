"""结构化输出模型（Pydantic）。

旧版靠**提示词里写 JSON 模板 + 用正则从回复里抠 JSON** 来拿到任务清单：

    text.find("{"); json.loads(...)   # backend/src/services/planner.py

这种方式有三个必然的坑：模型前面加一句解释就解析失败、字段名写错无法察觉、
缺字段只能 ``item.get("title") or f"任务{idx}"`` 兜底。

这里全部换成 ``with_structured_output``：字段约束由 schema 交给模型（json_schema
或 function_calling），返回值一定是校验过的 Pydantic 实例，解析代码整段消失。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TodoItemPlan(BaseModel):
    """规划阶段产出的单个研究任务。"""

    title: str = Field(description="任务名称，10 字以内，突出重点")
    intent: str = Field(description="任务要解决的核心问题，1-2 句")
    query: str = Field(description="建议使用的检索关键词，可直接送给搜索引擎")


class TodoList(BaseModel):
    """规划节点的结构化输出。"""

    tasks: list[TodoItemPlan] = Field(
        default_factory=list,
        description="3~5 个互补的研究任务；信息不足时返回空数组",
    )


class TaskSummary(BaseModel):
    """子任务的结构化总结。

    为什么要拆成字段而不是一段自由文本：

    * ``summary_markdown`` 直接渲染给用户；
    * ``key_findings`` 是**可程序化消费**的要点，写报告时按任务聚合，
      不必再让模型从 Markdown 里重新读一遍；
    * ``not_verified`` 强制模型显式声明「哪些内容没能用检索结果证实」——
      这是抑制幻觉最便宜的一招：把「不确定」变成必填字段，
      模型就没有「装作全都知道」的余地。
    """

    summary_markdown: str = Field(
        description="面向用户的 Markdown 总结，以二级标题“任务总结”开头，3-5 条关键发现",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="可独立成立的结论要点，每条一句话，便于汇总进最终报告",
    )
    not_verified: list[str] = Field(
        default_factory=list,
        description="本任务中未能被检索结果证实、或来源相互矛盾的内容；没有则留空数组",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="对本次总结可信度的自评：来源充分为 high，来源稀少或有冲突为 low",
    )
