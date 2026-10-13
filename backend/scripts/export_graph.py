"""导出真实图结构（Mermaid），用于文档与 README。

图是**编译出来的**，不是手画的：改了边，文档跟着变，不会出现文档漂移。

跑法::

    cd backend-langgraph
    python scripts/export_graph.py            # 打印到 stdout
    python scripts/export_graph.py docs/graph.mmd   # 同时写入文件
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph.builder import get_research_graph  # noqa: E402


def main() -> int:
    graph = get_research_graph()
    mermaid = graph.get_graph().draw_mermaid()

    print(mermaid)
    print("\n--- 节点 ---")
    for node in sorted(graph.get_graph().nodes):
        print(f"  {node}")

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(mermaid + "\n", encoding="utf-8")
        print(f"\n已写入 {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
