"""Links between statements, sections and documents of a policy set, and their effect on rules.

Code finds the explicit links, in any document:

- references: ``§4.3``, ``§4.2 step 3``, "section 4.2", "Appendix B", "page 2", and a reference into another
  document by its code (``SS-01 §2``) or by its title;
- the link type from the words around the reference: "supersedes", "replaces", "takes precedence over" →
  ``supersedes``; "may skip", "except", "exception", "does not apply", "instead of", "unless" → ``exception_of``;
  otherwise ``refers_to``;
- defined terms: a row of a term/meaning table, or "X means ..." / "X is defined as ..." → ``defines`` links to
  every statement that uses the term.

A model may propose more ``exception_of``, ``supersedes`` and ``source_of_truth`` links. A proposed link is kept
only if both ends exist. A reference whose target is not in the set becomes a ``dangling_link`` coverage item.

``fold_exceptions`` adds each exception to the rules of the statement it changes (``rule["unless"]``), and
``drop_superseded`` removes rules whose statement lost to a superseding one (``model.resolve_conflict``).
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from ..llm import parse_json
from ..model import (
    CoverageItem,
    Document,
    Link,
    LinkType,
    PolicySet,
    Rule,
    Statement,
    StatementType,
    Trust,
    resolve_conflict,
)

REF = re.compile(
    r"(?:(?P<code>\b[A-Z]{2,}(?:-[A-Z0-9]+)+)\s+)?"
    r"(?:§\s?|\b[Ss]ection\s+)(?P<sec>\d+(?:\.\d+)*)(?:\s+step\s+(?P<step>\d+))?"
)
APPENDIX_REF = re.compile(r"\bAppendix\s+(?P<app>[A-Z0-9]+)\b")
PAGE_REF = re.compile(r"\bpage\s+(?P<page>\d+)\b", re.I)
DOC_CODE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")
TRUTH_WORDS = re.compile(
    r"\b(precedence|takes? priority|wins?|authoritative|source of truth|system of record|"
    r"latest|most recent|overrides?|instead of|rather than)\b",
    re.I,
)
SUPERSEDE_WORDS = re.compile(r"\b(supersedes?|replaces?|takes? precedence over|overrides?)\b", re.I)
EXCEPTION_WORDS = re.compile(
    r"\b(may skip|except|exception|does not apply|instead of|unless|waive[sd]?)\b", re.I
)
DEFINES = re.compile(r"^(?P<term>[A-Z][\w\- ]{1,40}?)\s+(?:means|is defined as|refers to)\s+", re.I)
MAX_TERM_USES = 50

SYSTEM = "You find links between company policy statements. Output JSON only."
USER = """Statements (id: text):
{statements}

Find only links that the text of statement A states in words. Kinds:
- exception_of: A says that B does not apply in some case (words like "except", "may skip", "unless", "does not apply")
- supersedes: A says it replaces or overrides B (words like "supersedes", "replaces", "overrides", "takes precedence")
- source_of_truth: A says which DATA SOURCE (a sheet, an email from someone, a system, the latest record) wins over
  another data source for some values. A list of allowed values is NOT a source of truth.
Do not link statements only because they mention the same field.
Return {{"links": [{{"type": "exception_of|supersedes|source_of_truth", "src": "<id of A>", "dst": "<id of B>", "why": "..."}}]}}"""


def _codes(doc: Document) -> List[str]:
    """Document codes mentioned at the top of a document (its own code and codes of related documents)."""
    head = " ".join([doc.title] + [s.text for s in doc.statements[:6]])
    return list(dict.fromkeys(DOC_CODE.findall(head)))


def _own_codes(doc: Document) -> set:
    """The document's own code: from an identity row ("Document | AP-SOP-04, version 3.2"), else its id."""
    for st in doc.statements[:8]:
        first = st.text.split("|")[0].strip().lower()
        if "|" in st.text and first in (
            "document",
            "document id",
            "doc id",
            "policy id",
            "code",
            "reference",
        ):
            return set(DOC_CODE.findall(st.text.split("|", 1)[1]))
    return {doc.id}


def _section_target(doc: Document, sec: str, step: Optional[str]) -> Optional[str]:
    section = next((s for s in doc.sections if s.id == sec), None)
    if section is None:
        return None
    if step:
        st = next((s for s in section.statements if str(s.source.get("step")) == step), None)
        return st.id if st else None
    return f"{doc.id}:{section.id}"


def _resolve_doc(ps: PolicySet, here: Document, code: Optional[str]) -> Optional[Document]:
    if not code:
        return here
    others = [d for d in ps.documents if d is not here]
    return next((d for d in others if code in _own_codes(d)), None) or next(
        (d for d in others if code == d.id), None
    )


def _own(st: Statement, dst: str) -> bool:
    """True if ``dst`` is the statement's own section or one of its parent sections."""
    sec = str(st.source.get("section", ""))
    doc = st.id.split(":")[0]
    parts = sec.split(".")
    own = {f"{doc}:{'.'.join(parts[:i])}" for i in range(1, len(parts) + 1)}
    return dst in own or dst.startswith(st.id)


def _profile_target(doc: Document, st: Statement, ref: str) -> Optional[str]:
    """Resolve a reference id: the exact section id, then within the statement's top-level part, then a unique
    section whose last id part is ``ref``."""
    ids = [s.id for s in doc.sections]
    top = str(st.source.get("section", "")).split(".")[0]
    for cand in (ref, f"{top}.{ref}"):
        if cand in ids:
            return f"{doc.id}:{cand}"
    tail = [i for i in ids if i.split(".")[-1] == ref]
    return f"{doc.id}:{tail[0]}" if len(tail) == 1 else None


def _kind(text: str) -> LinkType:
    if SUPERSEDE_WORDS.search(text):
        return LinkType.SUPERSEDES
    if EXCEPTION_WORDS.search(text):
        return LinkType.EXCEPTION_OF
    return LinkType.REFERS_TO


def _explicit(ps: PolicySet) -> Tuple[List[Link], List[CoverageItem]]:
    links: List[Link] = []
    cov: List[CoverageItem] = []
    for doc in ps.documents:
        own_codes = _own_codes(doc)
        for st in doc.statements:
            for m in REF.finditer(st.text):
                code = m.group("code")
                if code and code in own_codes:
                    code = None
                target_doc = _resolve_doc(ps, doc, code)
                dst = _section_target(target_doc, m.group("sec"), m.group("step")) if target_doc else None
                ref = m.group(0)
                if dst is None:
                    cov.append(
                        CoverageItem(
                            st.id,
                            "dangling_link",
                            f"no target for '{ref}' in the policy set",
                            st.source,
                            st.text,
                        )
                    )
                    continue
                if dst.startswith(st.id) or dst == f"{doc.id}:{st.source.get('section')}":
                    continue  # a statement pointing at its own section
                kind = _kind(st.text)
                if kind == LinkType.SUPERSEDES and target_doc is doc:
                    kind = LinkType.REFERS_TO  # a document does not supersede its own parts
                links.append(Link(type=kind, src=st.id, dst=dst, source=dict(st.source), quote=ref))
            for pattern in doc.structure.get("references", []):  # the document's own reference style
                try:
                    rx = re.compile(pattern, re.I)
                except re.error:
                    continue
                for m in rx.finditer(st.text):
                    dst = _profile_target(doc, st, m.group("ref"))
                    if dst is None:
                        cov.append(
                            CoverageItem(
                                st.id, "dangling_link", f"no target for '{m.group(0)}'", st.source, st.text
                            )
                        )
                    elif not _own(st, dst):
                        links.append(
                            Link(
                                type=_kind(st.text),
                                src=st.id,
                                dst=dst,
                                source=dict(st.source),
                                quote=m.group(0),
                            )
                        )
            for sec in doc.sections:  # a bare mention of an existing part id ("the rule in B.1")
                if (
                    len(sec.id) >= 3
                    and re.search(r"[.\d]", sec.id)
                    and re.search(rf"(?<![\w.§]){re.escape(sec.id)}(?![\w%]|\.\d)", st.text)
                ):
                    dst = f"{doc.id}:{sec.id}"
                    if not _own(st, dst):
                        links.append(
                            Link(
                                type=_kind(st.text), src=st.id, dst=dst, source=dict(st.source), quote=sec.id
                            )
                        )
            for sec in doc.sections:  # a part named by its title ("see the Escalation Procedure")
                if (
                    len(sec.title.split()) >= 2
                    and len(sec.title) >= 8
                    and sec.title.lower() in st.text.lower()
                ):
                    dst = f"{doc.id}:{sec.id}"
                    if not _own(st, dst):
                        links.append(
                            Link(
                                type=_kind(st.text),
                                src=st.id,
                                dst=dst,
                                source=dict(st.source),
                                quote=sec.title,
                            )
                        )
            for m in APPENDIX_REF.finditer(st.text):
                sec = f"Appendix {m.group('app')}"
                if str(st.source.get("section")) == sec:
                    continue
                dst = _section_target(doc, sec, None)
                if dst is None:
                    cov.append(
                        CoverageItem(
                            st.id, "dangling_link", f"no target for '{m.group(0)}'", st.source, st.text
                        )
                    )
                else:
                    links.append(
                        Link(
                            type=LinkType.REFERS_TO,
                            src=st.id,
                            dst=dst,
                            source=dict(st.source),
                            quote=m.group(0),
                        )
                    )
            for m in PAGE_REF.finditer(st.text):
                page = int(m.group("page"))
                sec = next((s for s in doc.sections if s.page == page), None)
                if sec is None:
                    cov.append(
                        CoverageItem(
                            st.id, "dangling_link", f"no page {page} in {doc.id}", st.source, st.text
                        )
                    )
                else:
                    links.append(
                        Link(
                            type=LinkType.REFERS_TO,
                            src=st.id,
                            dst=f"{doc.id}:{sec.id}",
                            source=dict(st.source),
                            quote=m.group(0),
                        )
                    )
            for other in ps.documents:  # another document named by its title
                if (
                    other is not doc
                    and other.title
                    and len(other.title) > 8
                    and other.title.lower() in st.text.lower()
                ):
                    links.append(
                        Link(
                            type=_kind(st.text),
                            src=st.id,
                            dst=other.id,
                            source=dict(st.source),
                            quote=other.title,
                        )
                    )
    return links, cov


def _terms(ps: PolicySet) -> List[Link]:
    links: List[Link] = []
    statements = [s for d in ps.documents for s in d.statements]
    for st in statements:
        term = None
        header = [str(h).lower() for h in (st.source.get("table_header") or [])]
        if header and ("meaning" in " ".join(header) or header[0] in ("term", "word")):
            term = st.text.split("|")[0].strip()
        else:
            m = DEFINES.match(st.text)
            term = m.group("term").strip() if m else None
        if not term or len(term) < 3:
            continue
        st.types = st.types or [StatementType.DEFINITION]
        pat = re.compile(rf"\b{re.escape(term)}s?\b", re.I)
        users = [u for u in statements if u is not st and pat.search(u.text)][:MAX_TERM_USES]
        links.extend(
            Link(type=LinkType.DEFINES, src=st.id, dst=u.id, source=dict(st.source), quote=term)
            for u in users
        )
    return links


def _proposed(ps: PolicySet, model: Callable[[str, str], str]) -> List[Link]:
    statements = [s for d in ps.documents for s in d.statements]
    ids = {s.id for s in statements} | {f"{d.id}:{sec.id}" for d in ps.documents for sec in d.sections}
    by_id = {s.id: s.text for s in statements}
    body = "\n".join(f"{s.id}: {s.text[:240]}" for s in statements)
    try:
        reply = parse_json(model(SYSTEM, USER.format(statements=body)))
    except Exception:
        return []
    out = []
    for raw in reply.get("links") or []:
        if not isinstance(raw, dict):
            continue
        try:
            kind = LinkType(raw.get("type"))
        except ValueError:
            continue
        if kind not in (LinkType.EXCEPTION_OF, LinkType.SUPERSEDES, LinkType.SOURCE_OF_TRUTH):
            continue
        src, dst = raw.get("src"), raw.get("dst")
        if src not in ids or dst not in ids or src == dst:
            continue
        text = by_id.get(src, "")
        evidence = {
            LinkType.EXCEPTION_OF: EXCEPTION_WORDS,
            LinkType.SUPERSEDES: SUPERSEDE_WORDS,
            LinkType.SOURCE_OF_TRUTH: TRUTH_WORDS,
        }[kind]
        if not evidence.search(text):
            continue  # the source statement does not say it: drop the proposal
        if kind == LinkType.SUPERSEDES and src.split(":")[0] == dst.split(":")[0]:
            kind = LinkType.REFERS_TO  # a document does not supersede its own parts
        out.append(Link(type=kind, src=src, dst=dst, quote=str(raw.get("why", ""))[:200]))
    return out


def find_links(
    policy_set: PolicySet, model: Optional[Callable[[str, str], str]] = None
) -> Tuple[List[Link], List[CoverageItem]]:
    links, cov = _explicit(policy_set)
    links += _terms(policy_set)
    if model is not None:
        links += _proposed(policy_set, model)
    toc = {
        f"{d.id}:{sec.id}"
        for d in policy_set.documents
        for sec in d.sections
        if any(
            re.fullmatch(r"(table[ -]of[ -])?contents", str(x or "").strip().lower())
            for x in (sec.title, sec.id)
        )
    }
    seen, unique = set(), []
    for lk in links:
        key = (lk.type, lk.src, lk.dst)
        if (
            ":".join(str(lk.src).split(":")[:2]) in toc
        ):  # a table of contents lists sections; it does not refer
            continue
        if key not in seen:
            seen.add(key)
            unique.append(lk)
    policy_set.links = unique
    seen_cov, cov_unique = set(), []
    for c in cov:
        if (c.statement_id, c.reason) not in seen_cov:
            seen_cov.add((c.statement_id, c.reason))
            cov_unique.append(c)
    return unique, cov_unique


def _covers(rule: Rule, target: str) -> bool:
    """True if the rule comes from the target statement or from a statement in the target section."""
    return any(sid == target or sid.startswith(target + ":") for sid in rule.statement_ids)


def fold_exceptions(
    rules: List[Rule], links: List[Link], policy_set: Optional[PolicySet] = None
) -> List[Rule]:
    """Add each ``exception_of`` link to the rules of the statement or section it changes, as ``rule["unless"]``."""
    by_id: Dict[str, Statement] = {
        s.id: s for d in (policy_set.documents if policy_set else []) for s in d.statements
    }
    for lk in links:
        if lk.type != LinkType.EXCEPTION_OF:
            continue
        doc = policy_set.document_of(lk.src) if policy_set else None
        if doc is not None and doc.trust != Trust.TRUSTED:
            continue  # an untrusted record never changes a trusted rule
        src = by_id.get(lk.src)
        for r in rules:
            if _covers(r, lk.dst) and lk.src not in r.statement_ids:
                cond = {
                    "condition": src.text if src else lk.quote,
                    "source": dict(src.source) if src else {"statement": lk.src},
                }
                if cond not in r.rule.setdefault("unless", []):
                    r.rule["unless"].append(cond)
    return rules


def drop_superseded(
    rules: List[Rule], links: List[Link], policy_set: PolicySet
) -> Tuple[List[Rule], List[CoverageItem]]:
    """Remove rules whose statement is superseded by a statement that wins (``resolve_conflict``)."""
    by_id = {s.id: s for d in policy_set.documents for s in d.statements}
    lost: Dict[str, str] = {}
    for lk in links:
        if lk.type != LinkType.SUPERSEDES or lk.src not in by_id:
            continue
        targets = [
            s
            for sid, s in by_id.items()
            if sid == lk.dst or sid.startswith(lk.dst + ":") or lk.dst == sid.split(":")[0]
        ]
        for t in targets:
            if t.id.split(":")[0] == lk.src.split(":")[0]:
                continue  # same document: not a replacement
            if t.id != lk.src and resolve_conflict(by_id[lk.src], t, policy_set) is by_id[lk.src]:
                lost[t.id] = lk.src
    kept = [r for r in rules if not (r.statement_ids and all(sid in lost for sid in r.statement_ids))]
    cov = [
        CoverageItem(sid, "superseded", f"replaced by {winner}", by_id[sid].source, by_id[sid].text)
        for sid, winner in lost.items()
    ]
    return kept, cov
