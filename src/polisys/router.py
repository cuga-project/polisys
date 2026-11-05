"""Retrieval methods that can be swapped by name (design section 11).

Retrieval has two steps. Each is chosen in the config (``checkers.router.kind`` and ``checkers.router.expand``) and can be
replaced without touching the checks that use its output:

    Router.select(query, pkg)   -> procedure ids, best first
    Expander.expand(ids, pkg)   -> ``Scope``: the procedures and the rules that come along

Both receive the whole ``PolicyPackage``, so a method can walk its graph. ``ScopeReader`` runs the two steps for a gate or a
ledger: those read ``Scope`` and nothing else, so a new method is one registered class and one line of a profile.

Registered routers: ``agent_opened`` (the procedures the agent opened; measured live), ``bm25``, ``bm25_kev`` (BM25, then the
decision model re-ranks) and ``llm_cards`` (a model reads the procedure cards). Registered expanders: ``next_links`` (the
procedures a procedure links to with ``next``, and all escalation must-not rules; the scope of the gate as measured) and
``k_hops`` (``next_links``, then k hops along ``refers_to``, ``exception_of`` and ``supersedes``).

A new method: write the class, then ``register_router("name", Class, needs="")`` (``needs``: ``model`` for a JSON model,
``ask`` for the narrow-question model; the runtime gives it a traced one). An unknown name in a profile raises
``UnknownMethod`` and lists the names that exist. Nothing here names a domain or benchmark.
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .llm import parse_json
from .model import PolicyPackage, Procedure, step_text

# The links a walk follows. ``next`` is not one of them: ``next_links`` already brings it.
WALKED = ("refers_to", "exception_of", "supersedes")


class UnknownMethod(ValueError):
    """A router or an expander name that is not registered. The message lists the names that are."""


@dataclass(frozen=True)
class Query:
    """What a router is asked about. ``text``: a case, or a tool call with the data read recently (what a lexical router
    reads). ``action`` and ``context``: the same, apart: the acting call as ``tool(args)``, and the data read before it (what
    ``llm_cards`` reads). ``opened``: the procedure ids the agent opened, in the order it opened them (only ``agent_opened``
    reads it)."""

    text: str = ""
    opened: Tuple[str, ...] = ()
    action: str = ""
    context: str = ""


def as_query(query: Any) -> Query:
    return query if isinstance(query, Query) else Query(text=str(query or ""))


@dataclass
class ScopeRule:
    """A rule of a scope. ``owner``: the procedure id, or ``escalation``. ``kind``: ``must_not`` or ``decision`` (then
    ``decision`` holds the procedure's decision as the package has it: ``id``, ``rule``, ``then``, ``else``)."""

    owner: str
    kind: str
    text: str = ""
    decision: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scope:
    """What a check looks at: ``procedures`` (ids, in order) and ``rules`` (the must-not rules and decisions of those
    procedures, then the escalation must-not rules)."""

    procedures: List[str] = field(default_factory=list)
    rules: List[ScopeRule] = field(default_factory=list)


class Router:
    """Which procedures apply to a query. ``needs``: what the constructor takes besides its options (``""``, ``model`` or
    ``ask``)."""

    name = ""
    ranked = True  # the ids come best first; a router that returns a set says False, and recall@k then means little
    reads_opened = False  # it needs what the agent opened, so the procedure tools must be on
    options: Tuple[
        str, ...
    ] = ()  # the settings of ``checkers.router`` it reads (``top_k``, ``model``); a profile sets no other

    def select(self, query: Any, pkg: PolicyPackage) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError


class Expander:
    """The procedures and rules that come with the selected ids."""

    name = ""
    options: Tuple[str, ...] = ()  # the settings of ``checkers.router`` it reads (``hops``)

    def problem(self, pkg: PolicyPackage) -> str:
        """Why this expander changes nothing on this package, or ``""``. A run with such a setting would read "no gain" from
        a method that never ran, so the runtime refuses it."""
        return ""

    def expand(self, ids: Sequence[str], pkg: PolicyPackage) -> Scope:  # pragma: no cover - interface
        raise NotImplementedError


# -- registry ----------------------------------------------------------------------------------------------------------

ROUTERS: Dict[str, Tuple[Callable[..., Router], str]] = {}
EXPANDERS: Dict[str, Callable[..., Expander]] = {}


def register_router(name: str, factory: Callable[..., Router], needs: str = "") -> None:
    if needs not in ("", "model", "ask"):
        raise ValueError(f"needs is '', 'model' or 'ask', got {needs!r}")
    ROUTERS[name] = (factory, needs)


def register_expander(name: str, factory: Callable[..., Expander]) -> None:
    EXPANDERS[name] = factory


def router_names() -> List[str]:
    return sorted(ROUTERS)


def expander_names() -> List[str]:
    return sorted(EXPANDERS)


def router_class(name: str) -> Callable[..., Router]:
    router_needs(name)
    return ROUTERS[name][0]


def router_options(name: str) -> Tuple[str, ...]:
    return tuple(getattr(router_class(name), "options", ()))


def router_reads_opened(name: str) -> bool:
    """Whether the router needs the procedure tools. A name that is not registered says False: the build refuses it by its
    name, in a message of its own."""
    return name in ROUTERS and bool(getattr(ROUTERS[name][0], "reads_opened", False))


def expander_options(name: str) -> Tuple[str, ...]:
    check_names("agent_opened", name)
    return tuple(getattr(EXPANDERS[name], "options", ()))


def router_needs(name: str) -> str:
    """``""``, ``model`` or ``ask``: what ``make_router`` takes for this router. Raises ``UnknownMethod``."""
    if name not in ROUTERS:
        raise UnknownMethod(f"no router named {name!r}; the routers are: {', '.join(router_names())}")
    return ROUTERS[name][1]


def check_names(kind: str, expand: str) -> None:
    """Raises ``UnknownMethod`` for a router or an expander that is not registered."""
    router_needs(kind)
    if expand not in EXPANDERS:
        raise UnknownMethod(f"no expander named {expand!r}; the expanders are: {', '.join(expander_names())}")


def make_router(name: str, *, model: Any = None, ask: Any = None, **options: Any) -> Router:
    """The router of that name. ``model`` (a ``(system, user) -> text`` JSON model) or ``ask`` (the narrow-question
    model) goes to a router that needs it: the caller passes the traced one. Raises ``UnknownMethod``."""
    needs = router_needs(name)
    given = {"model": model, "ask": ask}
    if needs and given[needs] is None:
        raise ValueError(f"the router {name} needs a {needs}, and none was given")
    extra = {needs: given[needs]} if needs else {}
    return ROUTERS[name][0](**extra, **options)


def make_expander(name: str, **options: Any) -> Expander:
    check_names("agent_opened", name)
    return EXPANDERS[name](**options)


# -- reading a package -------------------------------------------------------------------------------------------------


def _procs(pkg: PolicyPackage) -> Dict[str, Procedure]:
    return {p.id: p for p in pkg.procedures}


def card_text(p: Procedure) -> str:
    """What a lexical router reads of a procedure: its name, when it applies, its steps and its must-not rules."""
    steps = " ".join(step_text(s) for s in p.steps)[:1500]
    return f"{p.name} {p.applies_when or ''} {steps} {' '.join(str(m) for m in p.must_not or [])}"


def _tokens(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]


def bm25_order(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[int]:
    """Indexes of ``docs``, best first by BM25 (ties keep the order of the documents)."""
    tokens = [_tokens(d) for d in docs]
    avg = sum(map(len, tokens)) / max(1, len(tokens))
    df = Counter(w for d in tokens for w in set(d))
    q = sorted(set(_tokens(query)))
    scores = []
    for i, d in enumerate(tokens):
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (len(tokens) - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append((-s, i))
    return [i for _, i in sorted(scores)]


# -- routers -----------------------------------------------------------------------------------------------------------


class AgentOpened(Router):
    """The procedures the agent opened (``policy__get_procedure``), for the whole run. Measured live."""

    name = "agent_opened"
    ranked = False
    reads_opened = True

    def __init__(self, **_: Any):
        pass

    def select(self, query: Any, pkg: PolicyPackage) -> List[str]:
        known = _procs(pkg)
        return list(dict.fromkeys(i for i in as_query(query).opened if i in known))


class Bm25(Router):
    """BM25 of the query over the cards of the procedures (``card_text``). Offline only; no model."""

    name = "bm25"
    options = ("top_k",)

    def __init__(self, top_k: Optional[int] = 3, **_: Any):
        self.top_k = top_k

    def select(self, query: Any, pkg: PolicyPackage) -> List[str]:
        procs = list(pkg.procedures)
        order = bm25_order(as_query(query).text, [card_text(p) for p in procs])
        ids = [procs[i].id for i in order]
        return ids if self.top_k is None else ids[: self.top_k]


class Bm25Kev(Router):
    """BM25 over the procedures (``SopIndex``), the best candidates re-ranked by one yes/no question each to the decision
    model (``ask``). Measured offline: 74% against 64% for BM25 alone (design section 11)."""

    name = "bm25_kev"
    options = ("top_k",)

    def __init__(self, ask: Callable[..., Any], top_k: Optional[int] = 3, candidates: int = 5, **_: Any):
        self.ask, self.top_k, self.candidates = ask, top_k, candidates
        self._index: Dict[int, Any] = {}

    def select(self, query: Any, pkg: PolicyPackage) -> List[str]:
        from .package import to_build
        from .sop_retrieval import SopIndex

        key = id(pkg)
        if key not in self._index:
            self._index[key] = (
                pkg,
                SopIndex(to_build(pkg), ask=self.ask),
            )  # keeps pkg alive: the id stays its own
        index = self._index[key][1]
        ids = [str(p.get("id")) for p in index.rank(as_query(query).text, "", self.candidates)]
        return ids if self.top_k is None else ids[: self.top_k]


LLM_SYSTEM = "You map an agent's action to the SOP procedure that governs it. Output JSON only."
LLM_USER = """SOP procedures (id | name | applies when):
{cards}

Recent data the agent read:
<<<
{context}
>>>

The agent's action:
{action}

Which procedures govern this action (the procedure whose steps this action carries out, or whose rules it must
follow)? Most relevant first, at most {top}. If no procedure governs it (e.g. a scratch computation), return [].
Return {{"procedures": ["<id>", ...]}}"""


class LlmCards(Router):
    """A model reads the procedure cards (id, name, when it applies) and ranks the ones that govern the query. Measured
    offline: 87.7% top-1, by agreement with another model's labels."""

    name = "llm_cards"
    options = ("top_k", "model")

    def __init__(self, model: Callable[[str, str], str], top_k: Optional[int] = 3, **_: Any):
        self.model, self.top_k = model, top_k or 3

    def select(self, query: Any, pkg: PolicyPackage) -> List[str]:
        """The prompt is the one of the measured router (87.7%): the cards, the data read, the action, at most 3 procedures.
        A query with only ``text`` has no separate action and data: the text is the action."""
        q = as_query(query)
        cards = "\n".join(f"{p.id} | {p.name} | {str(p.applies_when or '')[:300]}" for p in pkg.procedures)
        action = q.action or q.text
        reply = parse_json(
            self.model(
                LLM_SYSTEM,
                LLM_USER.format(cards=cards, context=q.context or "(none)", action=action, top=self.top_k),
            )
        )
        known = _procs(pkg)
        found = reply.get("procedures") if isinstance(reply, dict) else None
        return list(dict.fromkeys(str(i) for i in found or [] if str(i) in known))[: self.top_k]


register_router("agent_opened", AgentOpened)
register_router("bm25", Bm25)
register_router("bm25_kev", Bm25Kev, needs="ask")
register_router("llm_cards", LlmCards, needs="model")


# -- expanders ---------------------------------------------------------------------------------------------------------


def _next_of(p: Procedure) -> List[str]:
    return [lk["to"] for lk in p.links or [] if isinstance(lk, dict) and lk.get("type") == "next"]


def _scope_of(ids: Iterable[str], pkg: PolicyPackage) -> Scope:
    """The scope of procedures ``ids`` (in that order): their must-not rules and decisions, then the escalation must-nots."""
    procs = _procs(pkg)
    ordered = [i for i in dict.fromkeys(ids) if i in procs]
    rules: List[ScopeRule] = []
    for pid in ordered:
        p = procs[pid]
        rules += [ScopeRule(pid, "must_not", str(m)) for m in p.must_not or []]
        rules += [
            ScopeRule(pid, "decision", str(d.get("rule")), d)
            for d in p.decisions or []
            if isinstance(d, dict)
        ]
    escalation = pkg.global_.get("escalation") or {}
    rules += [ScopeRule("escalation", "must_not", str(m)) for m in escalation.get("must_not") or []]
    return Scope(ordered, rules)


def _with_next(ids: Iterable[str], procs: Dict[str, Procedure]) -> List[str]:
    out: List[str] = []
    for pid in ids:
        for x in [pid] + (_next_of(procs[pid]) if pid in procs else []):
            if x in procs and x not in out:
                out.append(x)
    return out


class NextLinks(Expander):
    """Each procedure and the procedures it links to with ``next``, and every escalation must-not rule. This is the scope
    the gate had before methods were swappable (``ProcedureStore._expand`` and ``StoreGate._rules``)."""

    name = "next_links"
    options = ()

    def __init__(self, **_: Any):
        pass

    def expand(self, ids: Sequence[str], pkg: PolicyPackage) -> Scope:
        return _scope_of(_with_next(ids, _procs(pkg)), pkg)


def procedure_graph(pkg: PolicyPackage) -> Dict[str, List[str]]:
    """The edges a k-hop walk follows, procedure to procedure. From a procedure's own ``links`` of a walked type (the
    procedure it names), and from the package's ``links`` between statements, sections and documents: a procedure owns its
    id, the sections of its source and of its steps, the statements of those sections and its document, and a link joins
    the owners of its two ends. ``refers_to`` and ``exception_of`` go both ways (the exception matters when the rule is
    open, and the rule when the exception is); ``supersedes`` goes only from the replaced procedure to the one that
    replaces it, because the rules of a replaced procedure no longer hold."""
    procs = _procs(pkg)
    edges: Dict[str, List[str]] = {pid: [] for pid in procs}

    def add(a: str, b: str) -> None:
        if a != b and a in edges and b not in edges[a]:
            edges[a].append(b)

    def join(kind: str, src: str, dst: str) -> None:
        if kind == "supersedes":
            add(dst, src)  # the replaced one reaches the one that replaces it, not the other way round
        else:
            add(src, dst)
            add(dst, src)

    for pid, p in procs.items():
        for lk in p.links or []:
            if isinstance(lk, dict) and lk.get("type") in WALKED and lk.get("to") in procs:
                join(lk["type"], pid, lk["to"])
    owners: Dict[str, Set[str]] = {pid: {pid} for pid in procs}

    def own(key: str, pid: str) -> None:
        if key:
            owners.setdefault(key, set()).add(pid)

    def key_of(source: Any) -> Tuple[str, str]:
        source = source if isinstance(source, dict) else {}
        doc = str(source.get("document") or "")
        return (f"{doc}:{source.get('section')}" if doc else ""), doc

    documents = [d.id for d in pkg.documents]
    for pid, p in procs.items():
        sources = [p.source] + [s.source for s in p.steps]
        if not any(isinstance(x, dict) and x.get("document") for x in sources):
            # A skill package has no source on a procedure: its id is the id of its section, in each document.
            for doc in documents:
                own(f"{doc}:{pid}", pid)
        for source in sources:
            section, doc = key_of(source)
            own(section, pid)
            own(doc, pid)
    for d in pkg.documents:
        for sec in d.sections:
            section = f"{d.id}:{sec.id}"
            for st in sec.statements:
                for pid in owners.get(section, ()):
                    own(st.id, pid)
    for lk in pkg.links:
        kind = lk.type.value if hasattr(lk.type, "value") else str(lk.type)
        if kind not in WALKED:
            continue
        for a in sorted(owners.get(lk.src, ())):
            for b in sorted(owners.get(lk.dst, ())):
                join(kind, a, b)
    return edges


class KHops(Expander):
    """``next_links``, then ``hops`` steps along ``refers_to``, ``exception_of`` and ``supersedes`` (see
    ``procedure_graph``). A walk keeps the procedures it has seen, so a cycle ends it. ``hops=0`` is ``next_links``."""

    name = "k_hops"
    options = ("hops",)

    def __init__(self, hops: int = 1, **_: Any):
        if hops < 0:
            raise ValueError(f"hops is 0 or more, got {hops}")
        self.hops = hops
        self._graphs: Dict[int, Any] = {}

    def _graph(self, pkg: PolicyPackage) -> Dict[str, List[str]]:
        key = id(pkg)
        if key not in self._graphs:
            self._graphs[key] = (pkg, procedure_graph(pkg))  # keeps pkg alive: the id stays its own
        return self._graphs[key][1]

    def problem(self, pkg: PolicyPackage) -> str:
        if self.hops == 0:
            return "hops is 0, which is next_links"
        if not any(self._graph(pkg).values()):
            return (
                "the package has no link to walk (a procedure's own refers_to, exception_of or supersedes, or a package link "
                "that reaches the sections of two procedures)"
            )
        return ""

    def expand(self, ids: Sequence[str], pkg: PolicyPackage) -> Scope:
        procs = _procs(pkg)
        start = _with_next(ids, procs)
        edges = self._graph(pkg)
        seen = list(start)
        frontier = deque((pid, 0) for pid in start)
        while frontier:
            pid, depth = frontier.popleft()
            if depth >= self.hops:
                continue
            for nxt in edges.get(pid, ()):
                for x in _with_next([nxt], procs):
                    if x not in seen:
                        seen.append(x)
                        frontier.append((x, depth + 1))
        return _scope_of(seen, pkg)


register_expander("next_links", NextLinks)
register_expander("k_hops", KHops)


# -- reading the two steps together ------------------------------------------------------------------------------------


class ScopeReader:
    """Router and expander over one package: what a gate or a ledger calls. ``store``: the ``ProcedureStore`` behind the
    agent's policy tools, which says what the agent opened (``store.calls``)."""

    def __init__(self, router: Router, expander: Expander, pkg: PolicyPackage, store: Any = None):
        self.router, self.expander, self.pkg, self.store = router, expander, pkg, store

    def opened(self) -> Tuple[str, ...]:
        calls = getattr(self.store, "calls", None) or []
        return tuple(str(i) for c in calls if c.get("tool") == "get_procedure" for i in c.get("ids") or [])

    def read(self, text: str = "", action: str = "", context: str = "") -> Scope:
        query = Query(text or f"{action}\n{context}".strip(), self.opened(), action, context)
        return self.expander.expand(self.router.select(query, self.pkg), self.pkg)

    @property
    def names(self) -> str:
        return f"{self.router.name} and {self.expander.name}"

    @property
    def is_default(self) -> bool:
        """The scope of the gate as measured. Only a method that is not this one shows in the record of a run."""
        return (self.router.name, self.expander.name) == ("agent_opened", "next_links")


def default_reader(pkg: PolicyPackage, store: Any = None) -> ScopeReader:
    """The scope of the gate as measured: the procedures the agent opened, with their ``next`` links."""
    return ScopeReader(make_router("agent_opened"), make_expander("next_links"), pkg, store)
