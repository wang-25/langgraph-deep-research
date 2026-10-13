"""Deep research 后端（LangChain / LangGraph 版）。

包结构::

    src/
      config.py           配置（pydantic-settings）
      prompts.py          各节点的系统提示词
      utils.py            纯文本工具函数
      observability.py    LangSmith 开关 + token/成本统计
      graph/
        state.py          图状态（TypedDict + reducer）
        schemas.py        结构化输出模型（Pydantic）
        llm.py            ChatModel 工厂 + 结构化调用降级
        tools.py          web_search / note 系列工具
        events.py         SSE 事件构造与抽取
        nodes.py          节点实现
        builder.py        StateGraph 组装
        runner.py         面向 FastAPI 的调用封装
      services/
        search.py         多搜索后端 dispatch
        notes.py          文件式笔记存储
        text_processing.py 模型输出清洗
      main.py             FastAPI 入口
"""

__version__ = "0.1.0"
