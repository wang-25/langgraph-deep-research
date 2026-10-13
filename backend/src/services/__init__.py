"""服务层：搜索、笔记、文本清洗。

这一层刻意**不依赖 LangGraph**——它们是无状态的纯能力，
既能被工具（``graph/tools.py``）调用，也能被离线脚本单独测试。
"""

from .notes import Note, NoteStore
from .search import SearchOutcome, dispatch_search

__all__ = ["Note", "NoteStore", "SearchOutcome", "dispatch_search"]
