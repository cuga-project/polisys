"""SOP long-document discovery: find the procedure a case needs in a long SOP and expand it along its links.

Input: an offline intake build (``{"build": {...procedures, links...}, "policy_set": {...documents...}}``). Per case:
BM25 over the procedures, the top candidates re-ranked by one yes/no decision-model question each (optional), then the
chosen procedure expanded into a checklist: its section's statements, preconditions, exceptions, escalations and
templates, the statements one link away (``refers_to``, ``exception_of``, ``defines``, ...), the full text of the
templates it names, and the general rules that match the case. Nothing here is specific to a document or benchmark.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

from .model import step_text

STOP = set(
    "the and for with that this from must have has are not any all each into your when then than there their which "
    "been also only per its may can should shall will a an of to in on or by be is as at it if".split()
)
APPLIES_Q = "Does this SOP procedure apply to this case, given the case summary? Answer yes only if it is the procedure to follow."


SUFFIXES = ("ations", "ation", "ings", "ing", "ments", "ment", "ions", "ion", "ers", "er", "ed", "es", "s")


def stem(w: str) -> str:
    """Light suffix stripping so that "onboard" matches "onboarding" and "notify" matches "notified"."""
    for suf in SUFFIXES:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def toks(t: str) -> List[str]:
    t = (t or "").lower()
    special = re.findall(r"\[[a-z_ -]+\]|[\w.-]+@[\w.-]+|[\w-]+\.(?:xlsx|csv|json|docx|pdf|md)|#[\w-]+", t)
    return [stem(w) for w in re.findall(r"[a-z][a-z0-9_-]{2,}", t) if w not in STOP] + special


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [toks(d) for d in docs]
        self.avg = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.df = Counter(w for d in self.docs for w in set(d))
        self.n, self.k1, self.b = len(self.docs), k1, b
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> List[float]:
        q = toks(query)
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    norm = tf[w] + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / max(self.avg, 1))
                    s += idf * tf[w] * (self.k1 + 1) / norm
            out.append(s)
        return out


def _words(step: Any) -> str:
    """A step in words: a build's quote when it has one (as before), else ``step_text`` (the text of a package or graph, the
    plain string of a store graph, or the tool as the last resort)."""
    return str(step.get("quote") or step_text(step)) if isinstance(step, dict) else step_text(step)


def _section_key(source: Any) -> str:
    """``<document>:<section>`` of a ``source``, or "" when it names no document."""
    source = source if isinstance(source, dict) else {}
    return f"{source['document']}:{source.get('section')}" if source.get("document") else ""


def proc_text(p: Dict[str, Any]) -> str:
    parts = [p.get("name", "")] + [_words(s) for s in p.get("steps") or []]
    for k in ("preconditions", "exceptions", "escalations", "templates"):
        parts += [str(x.get("quote") or x.get("condition") or x.get("when") or "") for x in p.get(k) or []]
    return "\n".join(parts)


class SopIndex:
    def __init__(
        self, data: Dict[str, Any], ask: Optional[Callable[[str, Dict[str, str]], Dict[str, Any]]] = None
    ):
        build = data.get("build", data)
        ps = data.get("policy_set") or {}
        self.procs = [p for p in build.get("procedures") or [] if p.get("steps")]
        self.general = next((p for p in build.get("procedures") or [] if not p.get("steps")), None)
        self.links = build.get("links") or []
        self.sts = [
            s
            for d in ps.get("documents") or []
            for sec in d.get("sections") or []
            for s in sec.get("statements") or []
        ]
        self.secs = {
            f"{d['id']}:{sec['id']}": sec
            for d in ps.get("documents") or []
            for sec in d.get("sections") or []
        }
        self.parent = {
            k: (f"{k.split(':')[0]}:{v['parent']}" if v.get("parent") else None) for k, v in self.secs.items()
        }
        self.by_sec: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in self.sts:
            self.by_sec[self._sec(s)].append(s)
        self.by_id = {s["id"]: s for s in self.sts}
        self.pindex = BM25([proc_text(p) for p in self.procs]) if self.procs else None
        self.ask = ask

    @staticmethod
    def _sec(s: Dict[str, Any]) -> str:
        return f"{s['source']['document']}:{s['source'].get('section')}"

    def _ancestors(self, sid: Optional[str]) -> List[str]:
        out = []
        while sid:
            out.append(sid)
            sid = self.parent.get(sid)
        return out

    def rank(self, case: str, summary: str = "", top: int = 5) -> List[Dict[str, Any]]:
        """The procedures for a case, best first: the BM25 top candidates re-ranked by the decision model when one is
        given (a candidate the model does not answer for ranks last among them), then the rest in BM25 order. Empty when
        no procedure matches a word of the case."""
        if not self.pindex:
            return []
        q = f"{case}. {summary}"
        sc = self.pindex.scores(q)
        order = sorted(range(len(self.procs)), key=lambda i: -sc[i])
        if sc[order[0]] <= 0:
            return []
        if self.ask is None:
            return [self.procs[i] for i in order]
        ranked = []
        for pos, i in enumerate(order[:top]):
            p = self.procs[i]
            state = (
                f"Case: {case}\nSummary: {summary}\n\nSOP procedure: {p['name']}\nFirst steps: "
                + " / ".join(_words(s)[:120] for s in (p.get("steps") or [])[:4])
            )
            a = self.ask(state, {"applies": APPLIES_Q}).get("applies")
            ranked.append((a if a is not None else -1.0, -pos, i))
        return [self.procs[i] for _, _, i in sorted(ranked, reverse=True)] + [
            self.procs[i] for i in order[top:]
        ]

    def choose(self, case: str, summary: str = "", top: int = 5) -> Optional[Dict[str, Any]]:
        """The procedure for a case: BM25 top candidates, re-ranked by the decision model when one is given."""
        ranked = self.rank(case, summary, top)
        return ranked[0] if ranked else None

    def checklist(self, p: Dict[str, Any], case_q: str, max_chars: int = 9000) -> str:
        """The procedure and what it links to, as text for the agent."""
        # a package from a store graph has no source on a procedure or a step: it has no section either
        psec = {
            _section_key(x.get("source"))
            for x in [p] + [st for st in p.get("steps") or [] if isinstance(st, dict)]
        }
        psec.discard("")
        own = {s["id"] for s in self.sts if any(a in psec for a in self._ancestors(self._sec(s)))}
        linked: List[str] = []
        E = own | psec
        for lk in self.links:
            for a, b in ((lk["src"], lk["dst"]), (lk["dst"], lk["src"])):
                if a in E and b not in E:
                    linked += [s["id"] for s in self.by_sec.get(b, [])] if b in self.secs else [b]
        names = set(re.findall(r"Template\s+[A-Z]?-?\d+", proc_text(p)))
        templates = [
            "\n".join(s["text"] for s in self.by_sec.get(k, []))
            for k, sec in self.secs.items()
            if any(n.lower() in str(sec.get("title", "")).lower() for n in names)
        ]
        lines = [f"SOP procedure: {p.get('name')}", "Steps:"]
        lines += [f"{i}. {_words(s)}" for i, s in enumerate(p.get("steps") or [], 1)]
        for k, title in (
            ("preconditions", "Preconditions"),
            ("exceptions", "Exceptions"),
            ("escalations", "Escalations"),
        ):
            items = [str(x.get("quote") or "") for x in p.get(k) or [] if x.get("quote")]
            if items:
                lines += [f"{title}:"] + [f"- {q}" for q in items]
        if templates:
            lines += ["Templates (use the text exactly):"] + templates
        extra = [self.by_id[i]["text"] for i in dict.fromkeys(linked) if i in self.by_id]
        if extra:
            lines += ["Related SOP text (linked from this procedure):"] + [f"- {t}" for t in extra[:25]]
        if self.general:
            g = (
                (self.general.get("preconditions") or [])
                + (self.general.get("exceptions") or [])
                + (self.general.get("escalations") or [])
            )
            gq = [str(x.get("quote") or "") for x in g if x.get("quote")]
            if gq:
                sc = BM25(gq).scores(case_q)
                hits = [q for q, s in sorted(zip(gq, sc), key=lambda x: -x[1])[:5] if s > 0]
                if hits:
                    lines += ["General SOP rules that may apply:"] + [f"- {q}" for q in hits]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 20] + "\n...(cut)"
