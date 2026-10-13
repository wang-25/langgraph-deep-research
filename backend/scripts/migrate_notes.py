"""把旧的扁平笔记迁移成「一次研究一个目录 + 可读文件名 + 合法 YAML 头」。

背景（真实踩过的两个坑）
------------------------
1. 早期版本把所有笔记平铺在 ``notes/`` 下，文件名是 ``note-<时间戳>-<随机>.md``，
   一次查询会产生 5~6 个「没名字」的文件，几次查询之后就分不清哪份属于哪次。
2. 早期版本的 front matter 直接写 ``title: 任务 2: 学习准备路径``，
   值里的冒号让任何 YAML 解析器报 ``bad indentation of a mapping entry``，
   编辑器打开就是「加载失败」。

本脚本按「一次研究」分组（规则见下），把文件重命名并搬到子目录，同时重写 front matter。
原文件默认删除（内容已完整迁移），加 ``--keep-originals`` 可保留。

分组规则：按文件里的时间戳排序后，**遇到 ``conclusion`` 笔记即视为该次研究的结束**
（报告总是最后写的）——这样一次查询产生的任务笔记 + 报告会归到同一个目录。
目录名沿用运行时的命名规则：``<主题>-<YYYYMMDD-HHMMSS>``。

用法::

    python scripts/migrate_notes.py --dry-run     # 先看会怎么改
    python scripts/migrate_notes.py               # 真的迁移
    python scripts/migrate_notes.py --keep-originals
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from services.notes import Note, NoteStore, load_note, slugify  # noqa: E402

#: 旧版文件名：note-<14 位时间戳>-<4 位随机>.md
_LEGACY_NAME = re.compile(r"^note-(\d{14})-([0-9a-f]{2,8})\.md$", re.IGNORECASE)
#: 任务笔记标题：任务 2: 学习准备路径 / 任务2：学习准备路径
_TASK_TITLE = re.compile(r"^任务\s*(\d+)\s*[:：]\s*(.+)$")
#: 结论笔记标题：研究报告：xxx / 研究报告: xxx
_REPORT_TITLE = re.compile(r"^研究报告\s*[:：]\s*(.+)$")


def _timestamp_in_name(path: Path) -> str:
    match = _LEGACY_NAME.match(path.name)
    return match.group(1) if match else "00000000000000"


def group_runs(files: list[Path]) -> list[list[Path]]:
    """按「报告结束一次研究」分组。"""

    groups: list[list[Path]] = []
    current: list[Path] = []
    for path in sorted(files, key=_timestamp_in_name):
        current.append(path)
        try:
            note = load_note(path)
        except Exception:  # noqa: BLE001
            continue
        if note.note_type == "conclusion":
            groups.append(current)
            current = []
    if current:  # 没有报告的残组也保留，避免丢数据
        groups.append(current)
    return groups


def _run_folder_name(group: list[Path]) -> tuple[str, str]:
    """返回 (目录名, 主题)。"""

    topic = "legacy"
    for path in group:
        try:
            note = load_note(path)
        except Exception:  # noqa: BLE001
            continue
        if note.note_type == "conclusion":
            match = _REPORT_TITLE.match(note.title)
            topic = (match.group(1) if match else note.title).strip() or "legacy"
            break

    stamp = _timestamp_in_name(group[-1] if group else Path("00000000000000"))
    stamp = f"{stamp[:8]}-{stamp[8:]}" if len(stamp) == 14 else "unknown-time"
    return f"{slugify(topic, 30) or 'legacy'}-{stamp}", topic


def _target_for(note: Note, index: int, run_topic: str) -> tuple[str, str]:
    """决定 (新文件名 stem, 新 id)。"""

    if note.note_type == "conclusion":
        return "研究报告", "report"

    match = _TASK_TITLE.match(note.title)
    if match:
        number = int(match.group(1))
        return f"{number:02d}-{slugify(match.group(2), 40)}", f"task-{number:02d}"

    number = note.task_id or index
    title = note.title if note.title else run_topic
    return f"{number:02d}-{slugify(title, 40)}", f"task-{number:02d}"


def _write_index(folder: Path, topic: str, entries: list[tuple[str, str, str]]) -> None:
    """写一份 00-研究概要.md（迁移版：不含运行期统计）。

    走 NoteStore 而不是直接写文件，保证索引和运行期生成的索引格式完全一致
    （同样带合法 YAML 头、同样的 id/文件名约定）。
    """

    lines = [
        f"# 研究概要：{topic}",
        "",
        "> 本目录由 `scripts/migrate_notes.py` 从旧的扁平笔记迁移而来。",
        "",
        "## 任务清单",
        "",
        "| # | 任务 | 笔记 |",
        "| --- | --- | --- |",
    ]
    report_file = None
    for id_, title, filename in entries:
        if id_ == "report":
            report_file = filename
            continue
        lines.append(f"| {id_.split('-')[-1]} | {title} | [{filename}](<{filename}>) |")

    lines.extend(["", "## 产物", ""])
    lines.append(
        f"- 最终报告：[{report_file}](<{report_file}>)" if report_file else "- 最终报告：（缺失）"
    )

    store = NoteStore(folder)
    existing = store.read("index")
    if existing is not None and existing.path is not None:
        existing.path.unlink()
    store.create(
        title=f"研究概要：{topic}",
        content="\n".join(lines),
        note_type="index",
        tags=["deep_research", "index"],
        note_id="index",
        filename="00-研究概要.md",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移扁平笔记为按次分目录")
    parser.add_argument("--root", default=str(ROOT / "notes"), help="笔记根目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不改动文件")
    parser.add_argument("--keep-originals", action="store_true", help="保留原文件（复制而非移动）")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"目录不存在：{root}")
        return 1

    legacy = [p for p in root.glob("*.md") if _LEGACY_NAME.match(p.name)]
    already = [p for p in root.glob("*.md") if not _LEGACY_NAME.match(p.name)]
    print(f"根目录：{root}")
    print(f"待迁移（旧命名）：{len(legacy)} 个")
    if already:
        print(f"跳过（非旧命名，可能是新格式）：{[p.name for p in already]}")

    if not legacy:
        print("没有需要迁移的文件。")
        return 0

    groups = group_runs(legacy)
    moved = 0

    for group in groups:
        folder_name, topic = _run_folder_name(group)
        folder = root / folder_name
        print(f"\n=== 研究目录：{folder_name}（主题：{topic}）===")
        if not args.dry_run:
            folder.mkdir(parents=True, exist_ok=True)

        entries: list[tuple[str, str, str]] = []
        planned: set[Path] = set()

        for index, path in enumerate(group, start=1):
            note = load_note(path)
            stem, new_id = _target_for(note, index, topic)
            # 同目录内查重
            candidate, suffix = stem, 2
            while (folder / f"{candidate}.md") in planned:
                candidate = f"{stem}-{suffix}"
                suffix += 1
            planned.add(folder / f"{candidate}.md")

            note.id = new_id
            note.path = folder / f"{candidate}.md"
            print(f"  {path.name}  ->  {folder_name}/{candidate}.md   (id={new_id})")
            entries.append((new_id, note.title, f"{candidate}.md"))

            if not args.dry_run:
                note.path.write_text(note.to_markdown(), encoding="utf-8")
                if not args.keep_originals:
                    path.unlink()
            moved += 1

        if not args.dry_run:
            _write_index(folder, topic, entries)

    print(f"\n{'（dry-run）' if args.dry_run else ''}共处理 {moved} 个文件，{len(groups)} 个研究目录。")
    if not args.dry_run and not args.keep_originals:
        print("原文件已删除，内容已完整迁移到子目录；如发现异常请从 git 或备份恢复。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
