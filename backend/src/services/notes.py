"""文件式笔记存储（NoteStore）。

设计要点
--------
1. **一次研究 = 一个文件夹**：``scope_to_run(topic)`` 会在 ``notes/`` 下建一个
   ``<主题>-<时间戳>/`` 目录，该次研究的所有笔记都落在里面。
   好处：一个主题的 5 份任务笔记 + 1 份报告天然聚在一起，不会被后续查询冲散。

2. **ID 与文件名分离**：
   * ``id``（如 ``task-01`` / ``report``）是**稳定的 ASCII 标识**——它会被送给模型，
     ASCII 更不容易在工具调用里被写错；
   * 文件名是**给人看的**（如 ``01-岗位技能要求.md`` / ``研究报告.md``）。
   两者通过 front matter 里的 ``id`` 字段对应，读取时按 id 反查文件。

3. **front matter 必须是合法 YAML**：所有字符串值都用 JSON 双引号形式输出
   （JSON 是 YAML 的子集）。之前直接写 ``title: 任务 2: 学习准备路径``，
   值里的冒号会让任何 YAML 解析器报 ``bad indentation of a mapping entry``，
   结果就是编辑器「加载失败」——这是真实踩过的坑。

4. 头部解析**不依赖 PyYAML**（少一个运行依赖）：逐行解析 + 自己处理引号；
   ``tags`` 用 JSON 数组形式写出，解析时兼容数组与逗号串两种历史写法。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_HEADER_DELIMITER = "---"
#: 文件名里不允许出现的字符（Windows 比其他平台更严）
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_SLUG_WHITESPACE = re.compile(r"[\s_]+")
_SLUG_DASHES = re.compile(r"-{2,}")
#: 文件名里不允许出现的路径片段
_UNSAFE_ID_CHARS = re.compile(r"[\\/]|\.\.|\x00-\x1f")


@dataclass
class Note:
    """一条笔记。"""

    id: str
    title: str
    content: str
    note_type: str = "task_state"
    tags: list[str] = field(default_factory=list)
    task_id: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
    path: Optional[Path] = None

    @property
    def filename(self) -> str:
        return self.path.name if self.path else f"{self.id}.md"

    def to_markdown(self) -> str:
        """序列化成**合法 YAML** front matter + 正文。"""

        lines = [
            _HEADER_DELIMITER,
            f"id: {_yaml_str(self.id)}",
            f"title: {_yaml_str(self.title)}",
            f"note_type: {_yaml_str(self.note_type)}",
            f"tags: {json.dumps(self.tags, ensure_ascii=False)}",
        ]
        if self.task_id is not None:
            lines.append(f"task_id: {self.task_id}")
        lines.append(f"created_at: {_yaml_str(self.created_at)}")
        lines.append(f"updated_at: {_yaml_str(self.updated_at)}")
        lines.append(_HEADER_DELIMITER)
        lines.append("")
        lines.append(self.content.strip())
        lines.append("")
        return "\n".join(lines)

    def to_tool_text(self) -> str:
        """工具返回值：模型只能看到文本，所以这里要自解释。"""

        tags = ", ".join(self.tags) if self.tags else "-"
        return (
            f"笔记ID: {self.id}\n"
            f"文件名: {self.filename}\n"
            f"标题: {self.title}\n"
            f"类型: {self.note_type}\n"
            f"标签: {tags}\n"
            f"更新时间: {self.updated_at}\n"
            f"内容:\n{self.content.strip()}"
        )


class NoteStore:
    """一个目录下的 Markdown 笔记仓库（可被 ``scope_to_run`` 收窄到单次研究）。"""

    def __init__(
        self,
        workspace: str | Path,
        *,
        topic: Optional[str] = None,
        create: bool = True,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if create:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.topic = topic

    # ------------------------------------------------------------------
    # 运行级目录
    # ------------------------------------------------------------------
    def scope_to_run(self, topic: str, *, now: Optional[datetime] = None) -> "NoteStore":
        """为一次研究建独立子目录，并返回绑定该目录的新 store。

        目录名形如 ``LangGraph并行扇出实践-20261013-154942``：既能一眼看出主题，
        又能按时间排序、且同名主题的多次查询不会互相覆盖。
        """

        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        folder = self.workspace / f"{slugify(topic, 30) or 'research'}-{stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        logger.info("本次研究的笔记目录: %s", folder)
        return NoteStore(folder, topic=topic)

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        title: str,
        content: str = "",
        note_type: str = "task_state",
        tags: Iterable[str] | None = None,
        task_id: Optional[int] = None,
        note_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Note:
        """新建一条笔记并落盘。

        Args:
            note_id: 稳定 ASCII 标识（如 ``task-01``）；不给则从标题派生。
            filename: 人类可读的文件名（含扩展名）；不给则用 ``id.md``。
        """

        now = _utcnow()
        stem = _unique_stem(
            self.workspace, _safe_filename(filename) if filename else (note_id or slugify(title, 40))
        )
        note = Note(
            id=note_id or _id_from_filename(stem),
            title=title.strip() or "未命名笔记",
            content=content or "",
            note_type=note_type,
            tags=_normalise_tags(tags),
            task_id=task_id,
            created_at=now,
            updated_at=now,
        )
        note.path = self.workspace / f"{stem}.md"
        self._write(note)
        logger.info(
            "笔记已创建: id=%s file=%s type=%s", note.id, note.path.name, note.note_type
        )
        return note

    def update(
        self,
        note_id: str,
        *,
        content: Optional[str] = None,
        append: bool = False,
        title: Optional[str] = None,
        note_type: Optional[str] = None,
        tags: Iterable[str] | None = None,
        task_id: Optional[int] = None,
    ) -> Optional[Note]:
        """更新笔记；``append=True`` 时把 ``content`` 追加到末尾而不是替换。"""

        note = self.read(note_id)
        if note is None:
            return None

        if content is not None:
            if append and note.content.strip():
                note.content = f"{note.content.rstrip()}\n\n{content.strip()}"
            else:
                note.content = content
        if title:
            note.title = title.strip()
        if note_type:
            note.note_type = note_type
        if tags is not None:
            note.tags = _normalise_tags(tags)
        if task_id is not None:
            note.task_id = task_id

        note.updated_at = _utcnow()
        self._write(note)
        logger.info("笔记已更新: id=%s file=%s (append=%s)", note.id, note.filename, append)
        return note

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def read(self, note_id: str) -> Optional[Note]:
        """按 ID 读取；不存在时返回 ``None``（不抛异常）。

        解析顺序：**文件名命中** → **front matter 里的 id 命中**。
        后者让「文件名人类可读、id 保持 ASCII」这套设计成立。
        """

        if not note_id:
            return None

        direct = self._path_for(note_id)
        if direct and direct.exists():
            return load_note(direct)

        if _UNSAFE_ID_CHARS.search(note_id):
            logger.warning("非法笔记 ID: %r", note_id)
            return None

        for path in self.workspace.glob("*.md"):
            try:
                note = load_note(path)
            except Exception:  # noqa: BLE001 - 单个坏文件不该影响查找
                continue
            if note.id == note_id:
                return note

        logger.warning("笔记不存在: %s（目录 %s）", note_id, self.workspace)
        return None

    def list(self, *, tag: Optional[str] = None, limit: int = 100) -> list[Note]:
        """列出笔记，可按标签过滤，按更新时间倒序。"""

        notes: list[Note] = []
        for path in self.workspace.glob("*.md"):
            try:
                note = load_note(path)
            except Exception as exc:  # noqa: BLE001 - 单文件损坏不该拖垮整个列表
                logger.warning("跳过无法解析的笔记 %s: %s", path.name, exc)
                continue
            if tag and tag not in note.tags:
                continue
            notes.append(note)

        notes.sort(key=lambda item: item.updated_at, reverse=True)
        return notes[:limit]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _path_for(self, note_id: str) -> Optional[Path]:
        """把「文件名 stem」映射到路径，并挡住路径穿越。"""

        if not note_id:
            return None
        stem = note_id[:-3] if note_id.endswith(".md") else note_id
        if _UNSAFE_ID_CHARS.search(stem) or len(stem) > 120:
            logger.warning("非法笔记文件名: %r", note_id)
            return None
        candidate = (self.workspace / f"{stem}.md").resolve()
        if candidate.parent != self.workspace:
            logger.warning("拒绝越界的笔记路径: %s", candidate)
            return None
        return candidate

    def _write(self, note: Note) -> None:
        note.path = note.path or self._path_for(note.id)
        assert note.path is not None  # _path_for 已校验
        note.path.write_text(note.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# 模块级辅助
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _yaml_str(value: str) -> str:
    """把任意字符串安全地写进 YAML：JSON 双引号形式（JSON 是 YAML 的子集）。"""

    return json.dumps(str(value), ensure_ascii=False)


def slugify(text: str, max_len: int = 40) -> str:
    """把标题变成安全的文件名片段（保留中文，替换 Windows 非法字符）。"""

    cleaned = _UNSAFE_FILENAME_CHARS.sub(" ", (text or "").strip())
    cleaned = _SLUG_DASHES.sub("-", _SLUG_WHITESPACE.sub("-", cleaned)).strip("-.")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-.")
    return cleaned


def _safe_filename(filename: str) -> str:
    """清洗调用方给的文件名，去掉扩展名与非法字符。"""

    stem = filename[:-3] if filename.lower().endswith(".md") else filename
    return slugify(stem, 60) or "note"


def _unique_stem(workspace: Path, stem: str) -> str:
    """避免覆盖同名文件（同一次研究里出现同名任务时）。"""

    candidate = stem or "note"
    index = 2
    while (workspace / f"{candidate}.md").exists():
        candidate = f"{stem}-{index}"
        index += 1
    return candidate


def _id_from_filename(stem: str) -> str:
    """文件名 → ID：只保留 ASCII 安全的短标识，中文交给文件名表达。"""

    ascii_part = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-")
    return ascii_part[:60] or "note"


def _normalise_tags(tags: Iterable[str] | None) -> list[str]:
    if not tags:
        return []
    if isinstance(tags, str):  # 模型有时会传 "a,b" 而不是 ["a","b"]
        tags = [item.strip() for item in tags.split(",")]
    seen: list[str] = []
    for tag in tags:
        cleaned = str(tag).strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _unquote(value: str) -> str:
    """还原 ``_yaml_str`` 写出的引号值（同时兼容单引号）。"""

    text = value.strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text[1:-1]
    if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("''", "'")
    return text


def _parse_tags(raw: str) -> list[str]:
    """兼容三种历史写法：JSON 数组、``[a, b]`` 字面量、``a, b`` 逗号串。"""

    text = raw.strip()
    if not text or text in {"-", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            text = text[1:-1]
    return [item.strip().strip("\"'") for item in text.split(",") if item.strip()]


def load_note(path: Path) -> Note:
    """解析一条笔记文件（头部缺失时优雅降级为纯正文）。

    公开出来是为了让迁移脚本等外部工具能复用同一套解析逻辑，
    避免出现「两套解析规则互相打架」。
    """

    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = raw

    if raw.startswith(_HEADER_DELIMITER):
        parts = raw.split(_HEADER_DELIMITER, 2)
        if len(parts) == 3:
            header, body = parts[1], parts[2]
            for line in header.strip().splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()

    task_id_raw = _unquote(meta.get("task_id", ""))
    try:
        task_id = int(task_id_raw) if task_id_raw else None
    except (TypeError, ValueError):
        task_id = None

    return Note(
        id=_unquote(meta.get("id") or "") or path.stem,
        title=_unquote(meta.get("title") or "") or path.stem,
        content=body.strip(),
        note_type=_unquote(meta.get("note_type") or "") or "task_state",
        tags=_parse_tags(meta.get("tags", "")),
        task_id=task_id,
        created_at=_unquote(meta.get("created_at", "")),
        updated_at=_unquote(meta.get("updated_at", "")),
        path=path,
    )
