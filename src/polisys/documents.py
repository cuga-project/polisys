"""Read policy documents into the policy model (``model.py``).

Formats: Markdown and plain text are parsed directly. PDF, DOCX and HTML are converted to Markdown by docling
first (PDF page by page, so every statement keeps its page). The Markdown parser then finds:

- sections: numbered headings (``## 4. Invoice processing``), numbered paragraphs (``4.2 Order of steps ...``,
  parent ``4``), appendices (``## Appendix B``) and other headings; text before the first heading is ``preamble``;
- statements: numbered list items (kept as ``step`` N in the source), bullets, table rows (not the header or the
  separator) and the sentences of a paragraph.

``load_policy_set`` reads several documents with their metadata (trust, effective date, authority rank).
``document_from_text`` turns a system prompt or an online record (email, sheet row) into a document.

``LineSection`` and ``split_policy`` are the older line-based split that ``builder.py`` still uses.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .model import Document, PolicySet, Section, Statement, Trust

HEADING = re.compile(r"^(#{1,6})\s*\**\s*(.*?)\**\s*$")
SECTION_NO = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$|^(\d+(?:\.\d+)*)\.?$")
APPENDIX = re.compile(r"^(Appendix\s+[A-Z0-9]+)\b[.:]?\s*(.*)$", re.I)
NUMBERED_PARA = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\.?\s+(.+)$")
LIST_ITEM = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
BULLET = re.compile(r"^\s*[-*•]\s+(.+)$")
LABELLED = re.compile(
    r"^(?:\*\*)?([A-Z][^:]{1,70}?)(?:\*\*)?:(?:\*\*)?\s+(\S.+)$"
)  # "Generate workbook: On Jira ..."
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR = re.compile(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$")
INLINE_PARA = re.compile(
    r"\s+(?=(?<![§\w.])\d+\.\d+(?:\.\d+)?\s+(?:\*\*)?[A-Z])"
)  # "... 6.6 Payment reference."
INLINE_STEP = re.compile(r"(?<=[:.;])\s+(?=\d{1,2}\.\s+(?:\*\*)?[A-Z])")  # "... complete: 1. Confirm ..."
BOLD_LEAD = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$")
ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|no|U\.S|Inc|Ltd|Corp|approx|incl|Dr|Mr|Mrs|Ms|Jr|Sr|Prof)\.", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z§(\"'`])")
DOCLING_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".pptx"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def clean(text: str) -> str:
    """Remove Markdown emphasis and code marks; collapse spaces."""
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "").replace("\\_", "_")).strip()


def sentences(text: str) -> List[str]:
    protected = ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    return [s.replace("\x00", ".").strip() for s in SENTENCE.split(protected) if s.strip()]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "section"


TITLE_STOP = re.compile(
    r"[§\d]|\b(must|may|should|shall|do|does|not|never|is|are|will|can|this|these)\b", re.I
)


def _title_like(sentence: str) -> bool:
    """A short lead such as "Order of steps." or "Emergency payment." (a title, not a rule)."""
    return sentence.endswith(".") and len(sentence.split()) <= 4 and not TITLE_STOP.search(sentence)


class _Parser:
    """Markdown → sections and statements. State is kept across pages, so a section can span pages."""

    def __init__(self, doc_id: str, profile: Optional[Dict[str, Any]] = None):
        self.doc_id = doc_id
        self.patterns = [
            (re.compile(h["pattern"]), h["level"], h["kind"]) for h in (profile or {}).get("headings", [])
        ]
        self.levels: Dict[int, str] = {}  # level -> id of the latest section at that level
        self.title = ""
        self.sections: List[Section] = [Section(id="preamble", title="Preamble")]
        self.table_header: Optional[List[str]] = None
        self.in_table = False
        self.bullet_run: Optional[List[Any]] = (
            None  # [section id, items so far] of the current run of labelled bullets
        )

    @property
    def cur(self) -> Section:
        return self.sections[-1]

    def _section(self, sid: str, title: str, page: Optional[int], parent: Optional[str] = None) -> None:
        existing = next((s for s in self.sections if s.id == sid), None)
        if existing is not None and existing is self.cur:
            return  # a heading repeated on a new page
        if existing is not None:
            sid = f"{sid}#{sum(s.id.split('#')[0] == sid for s in self.sections) + 1}"
        self.sections.append(
            Section(
                id=sid,
                title=title,
                page=page,
                parent=parent,
                source={"document": self.doc_id, "section": sid, "page": page},
                quote=title,
            )
        )

    def _add(self, text: str, page: Optional[int], line: int, **extra: Any) -> None:
        text = clean(text)
        if not text or set(text) <= set("|-: "):
            return
        sec = self.cur
        sid = f"{self.doc_id}:{sec.id}:{len(sec.statements) + 1}"
        src = {"document": self.doc_id, "section": sec.id, "page": page, "line": line, **extra}
        sec.statements.append(Statement(id=sid, text=text, source=src, quote=text))

    def _heading(self, body: str, level: int, page: Optional[int], line: int) -> None:
        body = clean(body)
        if level == 1 and not self.title and not SECTION_NO.match(body) and not APPENDIX.match(body):
            self.title = body
            return
        m = SECTION_NO.match(body)
        if m:
            sid, rest = (m.group(1) or m.group(3)), (m.group(2) or "")
            if "." in sid:  # "3.1 Access by role:" is a numbered paragraph shown as a heading
                self._numbered(sid, rest, page, line)
            else:
                self._section(sid, rest, page)
            return
        a = APPENDIX.match(body)
        if a:
            self._section(a.group(1).title().replace("Appendix", "Appendix"), clean(a.group(2)), page)
            return
        self._section(_slug(body), body, page)

    def _numbered(self, sid: str, rest: str, page: Optional[int], line: int) -> None:
        """A numbered paragraph ("4.2 Order of steps. Process ..."): a subsection. A bold lead, or a first
        sentence of at most 5 words, is the section title; the other sentences are statements."""
        lead = BOLD_LEAD.match(rest)
        if lead:
            title, parts = clean(lead.group(1)).rstrip("."), sentences(clean(lead.group(2)))
        else:
            parts = sentences(clean(rest))
            short = parts and len(parts) > 1 and _title_like(parts[0])
            title, parts = (parts[0].rstrip("."), parts[1:]) if short else ("", parts)
        self._section(sid, title, page, parent=sid.rsplit(".", 1)[0])
        for s in parts:
            self._add(s, page, line)

    def _profiled(self, line: str, page: Optional[int], no: int) -> bool:
        """A line that matches one of the document's heading patterns (structure profile)."""
        for rx, level, kind in self.patterns:
            m = rx.match(line)
            if not m:
                continue
            sid = m.group("id").strip()
            if " " in sid:  # "ARTICLE II" -> "II": references use the number, not the keyword
                sid = sid.split()[-1]
            parent = self.levels.get(level - 1) if level > 1 else None
            if parent and not sid.startswith(parent + "."):
                sid = f"{parent}.{sid}"
            title = (m.groupdict().get("title") or "").strip() if kind == "heading" else ""
            if kind == "heading" and not title:
                title = clean(line[m.end() :])
            if kind == "heading" and (
                len(title.split()) > 8 or TITLE_STOP.search(re.sub(r"[§\d]", "", title))
            ):
                kind = "clause"  # the rest of the line reads like a rule, not a title: keep it as a statement
            self._section(sid, clean(title), page, parent=parent)
            self.levels[level] = self.cur.id
            for deeper in [lv for lv in self.levels if lv > level]:
                del self.levels[deeper]
            if kind == "clause":
                rest = m.groupdict().get("title") or line[m.end() :]
                for s in sentences(clean(rest)):
                    self._add(s, page, no)
            return True
        return False

    def feed(self, text: str, page: Optional[int] = None) -> None:
        for no, raw in enumerate(text.splitlines(), 1):
            line = html.unescape(raw.rstrip()).replace("\\_", "_")
            stripped = line.strip()
            if not stripped:
                continue  # blank lines do not end a table (some documents put one between rows)
            if stripped == "---":
                self.in_table = False
                continue
            if TABLE_ROW.match(stripped):
                cells = [clean(c) for c in stripped.strip("|").split("|")]
                if SEPARATOR.match(stripped):
                    continue
                if not self.in_table:
                    self.in_table, self.table_header = True, cells
                    if any(cells):  # a header row is not a statement
                        continue
                if any(cells):
                    self._add(" | ".join(c for c in cells), page, no, table_header=self.table_header)
                continue
            self.in_table = False
            body0 = HEADING.match(stripped).group(2) if stripped.startswith("#") else stripped
            if self.patterns and self._profiled(clean(body0), page, no):
                continue
            h = HEADING.match(stripped)
            if h and stripped.startswith("#"):
                self._heading(h.group(2), len(h.group(1)), page, no)
                continue
            b = BULLET.match(line)
            body = b.group(1).strip() if b else stripped  # docling writes "4.2 ..." paragraphs as bullets
            if b and self._labelled_step(body, page, no):
                continue
            self.bullet_run = None
            # PDF text can join several numbered paragraphs or list steps on one line: split them again
            for piece in (p for part in INLINE_PARA.split(body) for p in INLINE_STEP.split(part)):
                piece = piece.strip()
                np_ = NUMBERED_PARA.match(piece)
                li = LIST_ITEM.match(piece)
                if np_:
                    self._numbered(np_.group(1), np_.group(2), page, no)
                elif li:
                    self._add(li.group(2), page, no, step=int(li.group(1)))
                else:
                    for sent in sentences(piece):
                        self._add(sent, page, no)

    def _labelled_step(self, body: str, page: Optional[int], no: int) -> bool:
        """A bullet that starts with a short label ("Generate workbook: On Jira ...", "Verify approval (> $5,000):
        ...") is a step of a bulleted procedure; consecutive ones in a section are numbered 1, 2, 3 ... The
        procedures handler keeps such a run only when it is long enough and its steps map to tools."""
        m = LABELLED.match(body)
        if not m or NUMBERED_PARA.match(body) or LIST_ITEM.match(body):
            return False
        label = re.sub(r"\([^)]*\)", "", m.group(1)).split()
        if not 1 <= len(label) <= 6:
            return False
        run = self.bullet_run if self.bullet_run and self.bullet_run[0] == self.cur.id else [self.cur.id, 0]
        run[1] += 1
        self.bullet_run = run
        self._add(body, page, no, step=run[1], bullet=True)
        return True

    def sections_out(self) -> List[Section]:
        return [s for s in self.sections if s.statements or s.id != "preamble"]


_CONVERTER = None


def _converter():
    """docling converter, created once. PDF runs on CPU (the Mac GPU backend fails on float64) without OCR."""
    global _CONVERTER
    if _CONVERTER is None:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        )
        _CONVERTER = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _CONVERTER


# docling's HTML reader folds the en dash and the em dash to a hyphen (``_clean_unicode``), so a statement would no longer
# be the document's words. They are swapped for two private-use characters that it leaves alone, and swapped back in the
# Markdown. Nothing else changes, so statement ids stay as they were. Curly quotes stay folded: ``SENTENCE`` knows only
# straight quotes.
DASHES = {"\u2013": "\uf8f0", "\u2014": "\uf8f1"}
DASH_ENTITIES = {
    "&ndash;": "\uf8f0",
    "&#8211;": "\uf8f0",
    "&#x2013;": "\uf8f0",
    "&mdash;": "\uf8f1",
    "&#8212;": "\uf8f1",
    "&#x2014;": "\uf8f1",
}


def _hold_dashes(raw: str) -> str:
    for entity, mark in DASH_ENTITIES.items():
        raw = re.sub(re.escape(entity), mark, raw, flags=re.I)
    for dash, mark in DASHES.items():
        raw = raw.replace(dash, mark)
    return raw


def _restore_dashes(markdown: str) -> str:
    """The dashes of the document are back, except one that starts a line: docling's own folding made it a hyphen, which the
    Markdown parser reads as a list bullet, so it stays one."""
    markdown = re.sub(r"(?m)^(\s*)[\uf8f0\uf8f1]", r"\1-", markdown)
    for dash, mark in DASHES.items():
        markdown = markdown.replace(mark, dash)
    return markdown


CHARSET = re.compile(rb"charset\s*=\s*[\"']?([\w-]+)", re.I)


def _held_html(raw: bytes) -> Optional[bytes]:
    """The HTML with its dashes held, or None when it should be read as it is: a file that is not UTF-8, or declares another
    charset, would be misread with the placeholders in it. A UTF-8 file with no declaration gets one, so that docling does not
    guess."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    declared = CHARSET.search(raw[:4096])
    if declared and declared.group(1).decode("ascii", "replace").lower().replace("_", "-") not in (
        "utf-8",
        "utf8",
    ):
        return None
    if _hold_dashes(text) == text:
        return None  # no dash to keep
    held = _hold_dashes(text)
    if not declared:
        held = '<meta charset="utf-8">' + held
    return held.encode("utf-8")


def _pages_markdown(path: Path) -> List[Tuple[Optional[int], str]]:
    if path.suffix.lower() in (".html", ".htm"):
        held = _held_html(path.read_bytes())
        if held is not None:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                copy = Path(tmp) / path.name
                copy.write_bytes(held)
                doc = _converter().convert(str(copy)).document
            return [(None, _restore_dashes(doc.export_to_markdown()))]
    doc = _converter().convert(str(path)).document
    if path.suffix.lower() == ".pdf" and doc.pages:
        return [(p, doc.export_to_markdown(page_no=p)) for p in sorted(doc.pages)]
    return [(None, doc.export_to_markdown())]


def _meta_trust(value: Any) -> Trust:
    return value if isinstance(value, Trust) else Trust(value or "trusted")


def _make(
    doc_id: str,
    kind: str,
    pages: Iterable[Tuple[Optional[int], str]],
    meta: Dict[str, Any],
    source: Dict[str, Any],
    model: Optional[Callable[[str, str], str]] = None,
) -> Document:
    from .structure import learn_structure

    pages = list(pages)
    profile = meta.get("structure") or learn_structure("\n".join(t for _, t in pages), model)
    parser = _Parser(doc_id, profile)
    for page, text in pages:
        parser.feed(text, page)
    title = meta.get("title") or parser.title or doc_id
    return Document(
        id=doc_id,
        kind=kind,
        title=title,
        trust=_meta_trust(meta.get("trust")),
        effective_date=meta.get("effective_date"),
        authority_rank=int(meta.get("authority_rank", 0)),
        sections=parser.sections_out(),
        structure=profile,
        source=source,
        quote=title,
    )


def load_document(
    source: Union[str, Path],
    kind: str = "auto",
    meta: Optional[Dict[str, Any]] = None,
    model: Optional[Callable[[str, str], str]] = None,
) -> Document:
    """A file (Markdown, text, PDF, DOCX, HTML) or a text string → ``Document``.

    ``meta``: ``id``, ``title``, ``trust`` (trusted/untrusted), ``effective_date`` (ISO), ``authority_rank`` (int),
    ``structure`` (a ready structure profile). With a ``model``, the document's own heading and reference styles are
    learned first (``structure.learn_structure``, one call); without one, the code defaults are used.
    """
    meta = dict(meta or {})
    path = (
        Path(source)
        if isinstance(source, Path)
        or (isinstance(source, str) and "\n" not in source and len(source) < 500 and Path(source).exists())
        else None
    )
    if path is None:
        text = str(source)
        doc_id = meta.get("id") or "doc-" + hashlib.sha256(text.encode()).hexdigest()[:8]
        return _make(
            doc_id, "markdown" if kind == "auto" else kind, [(None, text)], meta, {"document": doc_id}, model
        )
    suffix = path.suffix.lower()
    kind = suffix.lstrip(".") if kind == "auto" else kind
    doc_id = meta.get("id") or path.stem
    src = {"document": doc_id, "file": str(path)}
    if suffix in DOCLING_SUFFIXES:
        return _make(doc_id, kind, _pages_markdown(path), meta, src, model)
    return _make(doc_id, kind, [(None, path.read_text(encoding="utf-8", errors="replace"))], meta, src, model)


def load_policy_set(
    sources: Sequence[Union[str, Path]],
    meta: Optional[Dict[str, Dict[str, Any]]] = None,
    model: Optional[Callable[[str, str], str]] = None,
) -> PolicySet:
    """Several documents in order. ``meta`` is keyed by document id (file stem) or by the source as given."""
    meta = meta or {}
    docs = []
    for src in sources:
        key = Path(src).stem if isinstance(src, Path) or Path(str(src)).exists() else str(src)
        docs.append(load_document(src, meta=meta.get(key) or meta.get(str(src)) or {}, model=model))
    return PolicySet(documents=docs)


def document_from_text(
    text: str,
    kind: str,
    source: Dict[str, Any],
    trust: Trust = Trust.TRUSTED,
    meta: Optional[Dict[str, Any]] = None,
) -> Document:
    """A system prompt or an online record (email, sheet row, request) → ``Document`` with its source kept."""
    meta = {**(meta or {}), "trust": trust}
    doc_id = (
        meta.get("id") or source.get("record_id") or "rec-" + hashlib.sha256(text.encode()).hexdigest()[:8]
    )
    doc = _make(doc_id, kind, [(None, text)], meta, {"document": doc_id, **source})
    for st in doc.statements:
        st.source.update(source)
    return doc


# -- older line-based split, used by builder.py ----------------------------------------------------------------------

LINE_HEADING = re.compile(r"^#{1,4}\s*\**\s*(\d+(?:\.\d+)*)\.?\s*(.*?)\**\s*$")


@dataclass
class LineSection:
    sid: str  # "preamble", "3.1"
    title: str
    lines: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.lines)

    @property
    def hash(self) -> str:
        return hashlib.sha256(f"{self.title}\n{self.text}".encode()).hexdigest()[:16]


def split_policy(text: str) -> List[LineSection]:
    sections = [LineSection("preamble", "Preamble")]
    for no, raw in enumerate(text.splitlines(), 1):
        m = LINE_HEADING.match(raw.strip())
        if m:
            sections.append(LineSection(m.group(1), clean(m.group(2))))
            continue
        if raw.strip() and raw.strip() not in ("---",) and not raw.strip().startswith("# "):
            sections[-1].lines.append((no, raw.rstrip()))
    return [s for s in sections if s.lines]
