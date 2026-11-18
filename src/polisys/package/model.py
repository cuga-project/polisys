"""The policy model: what a policy is, once it has been read. See ``MODEL.md`` for the full description.

One model serves the offline builder (documents read before the run), online intake (policies found during the
run) and the runtime checks. Every object keeps its ``source`` (document, section, page, line or record id) and a
verbatim ``quote``, so every verdict can point back to the text it enforces.

Objects: ``PolicySet`` → ``Document`` → ``Section`` → ``Statement``. A statement has up to two ``StatementType``
values and is turned into ``Rule``s, ``Procedure`` steps or ``Definition``s. ``Link``s connect any two objects, also
across documents. ``Fact`` is data from a tool output; ``Verdict`` is the result of one check. ``PolicyBuild`` is
the output of a build: what the runtime loads. ``PolicyPackage`` is the one file format that holds a policy with its
rules, procedures and ``Obligation``s; ``package.py`` converts today's artifacts to it and back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class StatementType(str, Enum):
    """What a statement says. Decides which handler turns it into checks (``handlers/``)."""

    ALLOWED_VALUES = "allowed_values"  # "urgency must be one of 1, 2, 3"
    LOOKUP_MATRIX = "lookup_matrix"  # priority from impact x urgency
    STATE_TRANSITION = "state_transition"  # "new -> in progress -> resolved"
    REQUIRED_FIELD = "required_field"  # "every incident needs a caller"
    DERIVED_VALUE = "derived_value"  # "monthly charge = (cost - salvage) / life"
    THRESHOLD = "threshold"  # "refunds above $500 need approval"
    ROLE_PERMISSION = "role_permission"  # "only managers may close a change"
    APPROVAL = "approval"  # "CAB must approve a normal change"
    PROHIBITION = "prohibition"  # "do not send customer data outside"
    ELIGIBILITY_FILTER = "eligibility_filter"  # "skip units whose Notes contain an FSR- code"
    ROUTING = "routing"  # "network incidents go to NetOps"
    DEADLINE = "deadline"  # "respond within 24 hours"
    NOTIFICATION_DUTY = "notification_duty"  # "notify the owner when status changes"
    PROCEDURE = "procedure"  # "1. check, 2. approve, 3. close"
    EXCEPTION_OVERRIDE = "exception_override"  # "emergency payments may skip step 3"
    DEFINITION = "definition"  # "overdue = more than 30 days since inspection"
    FORMAT = "format"  # "IDs as INC_###"
    SOURCE_OF_TRUTH = "source_of_truth"  # "monitoring emails take precedence over the queue sheet"
    OPEN_JUDGMENT = "open_judgment"  # "use judgment for major incidents"


class Modality(str, Enum):
    MUST = "must"
    MUST_NOT = "must_not"
    MAY = "may"
    SHOULD = "should"


class LinkType(str, Enum):
    REFERS_TO = "refers_to"  # "see §4.3"
    DEFINES = "defines"  # a definition gives the meaning of a term used by the target
    EXCEPTION_OF = "exception_of"  # the source is an exception to the target rule
    SUPERSEDES = "supersedes"  # the source replaces the target
    DEPENDS_ON = "depends_on"  # the source needs the target to apply
    NEXT = "next"  # procedure order
    SOURCE_OF_TRUTH = "source_of_truth"  # the source data record wins over the target for a value


class Trust(str, Enum):
    TRUSTED = "trusted"  # the company's own staff or systems
    UNTRUSTED = "untrusted"  # an external sender or customer, or it asks to send data outside


MAX_TYPES = 2

RULE_LEVELS = ("observe", "remind", "block_once", "deny")  # what a rule does when it is broken, weakest first
RULE_CLASSES = ("hard", "judgment", "guidance")  # how a rule is decided (design section 4)

# Fields added for the policy package. They are left out of the JSON when empty, so builds and policy sets written by
# ``to_json`` stay as they were, and readers of those files see no change.
_PACKAGE_FIELD = {"omit_empty": True}


@dataclass
class Statement:
    """One sentence, bullet or table row that says something."""

    id: str
    text: str
    types: List[StatementType] = field(default_factory=list)
    modality: Optional[Modality] = None
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)

    def __post_init__(self) -> None:
        if len(self.types) > MAX_TYPES:
            raise ValueError(f"statement {self.id} has {len(self.types)} types; at most {MAX_TYPES}")


@dataclass
class Section:
    """A numbered or titled part of a document (``id`` like ``4.2`` or ``preamble``)."""

    id: str
    title: str = ""
    page: Optional[int] = None
    parent: Optional[str] = None
    statements: List[Statement] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)


@dataclass
class Document:
    """One policy source: a file, a system prompt, an email, a sheet, or a user message."""

    id: str
    kind: str
    title: str = ""
    trust: Trust = Trust.TRUSTED
    effective_date: Optional[str] = None  # ISO date; later wins when two statements conflict
    authority_rank: int = 0  # higher wins when dates are equal
    sections: List[Section] = field(default_factory=list)
    structure: Dict[str, Any] = field(default_factory=dict)  # heading and reference patterns (structure.py)
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    priority_class: Optional[str] = field(default=None, metadata=_PACKAGE_FIELD)  # design section 10
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)

    @property
    def statements(self) -> List[Statement]:
        return [s for sec in self.sections for s in sec.statements]

    @property
    def hash(self) -> str:
        body = "\n".join(
            f"{sec.id}|{sec.title}|" + "\n".join(s.text for s in sec.statements) for sec in self.sections
        )
        return hashlib.sha256(f"{self.kind}|{self.title}|{body}".encode()).hexdigest()[:16]


@dataclass
class Link:
    """A typed edge between two objects (statement, section, rule or document ids), also across documents."""

    type: LinkType
    src: str
    dst: str
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)


@dataclass
class PolicySet:
    """All policy that applies to one agent or thread. Document order is the order of ``documents``."""

    documents: List[Document] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""

    @property
    def version(self) -> str:
        return hashlib.sha256("|".join(d.hash for d in self.documents).encode()).hexdigest()[:16]

    def document_of(self, statement_id: str) -> Optional[Document]:
        for d in self.documents:
            if any(s.id == statement_id for s in d.statements):
                return d
        return None


@dataclass
class Rule:
    """A checkable form of one or more statements. ``rule`` is the typed rule dict that ``ruleset.py`` runs.

    ``cls`` (JSON key ``class``): hard, judgment or guidance. ``level``: observe, remind, block_once or deny. ``package.py``
    derives both for converted artifacts. ``RuleSetHooks`` reads ``level`` (see ``RuleSetHooks._level``); nothing at
    runtime reads ``cls`` yet.
    """

    id: str
    type: str
    rule: Dict[str, Any]
    statement_ids: List[str] = field(default_factory=list)
    checker: str = "code"
    may_deny: bool = True
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    cls: str = field(default="", metadata={**_PACKAGE_FIELD, "json": "class"})
    level: str = field(default="", metadata=_PACKAGE_FIELD)
    why: str = field(
        default="", metadata=_PACKAGE_FIELD
    )  # why this type: a ``verify`` rule says why no typed rule fits
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)


@dataclass
class Step:
    """One step of a procedure: a tool call (or ``reply``) that is required or optional, maybe with a condition.

    ``text`` is the step in words: a store graph has only that (its steps are plain strings), a build has ``action``.
    """

    id: str = ""
    action: str = ""
    required: bool = True
    condition: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    text: str = field(default="", metadata=_PACKAGE_FIELD)
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)


def step_text(step: Any) -> str:
    """A procedure step in words, from a ``Step``, a step dict of either format, or the plain string of a store graph.
    A graph step has ``text``, a build step has ``quote``; a step with neither gives its tool (``action``)."""
    if isinstance(step, str):
        return step
    read = step.get if isinstance(step, dict) else lambda key: getattr(step, key, "")
    return read("text") or read("quote") or read("action") or ""


@dataclass
class Procedure:
    """An SOP as a step graph. ``edges`` are ``next`` links between step ids, maybe with a condition."""

    id: str
    name: str = ""
    trigger: Dict[str, Any] = field(default_factory=dict)
    steps: List[Step] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    # filled by handlers/procedure_rules.py, each item with a verbatim "quote":
    preconditions: List[Dict[str, Any]] = field(
        default_factory=list
    )  # must hold before the procedure starts for a case
    exceptions: List[Dict[str, Any]] = field(
        default_factory=list
    )  # cases where it must not run, stops, or changes
    templates: List[Dict[str, Any]] = field(default_factory=list)  # exact text a step's output must follow
    escalations: List[Dict[str, Any]] = field(default_factory=list)  # who to notify when a condition holds
    # from the store graph (procedure_store.py); a build does not have them:
    applies_when: str = field(default="", metadata=_PACKAGE_FIELD)
    inputs: List[Dict[str, Any]] = field(default_factory=list, metadata=_PACKAGE_FIELD)  # {fact, source}
    decisions: List[Dict[str, Any]] = field(
        default_factory=list, metadata=_PACKAGE_FIELD
    )  # {id, rule, then, else}
    must_not: List[str] = field(default_factory=list, metadata=_PACKAGE_FIELD)
    deadlines: List[str] = field(default_factory=list, metadata=_PACKAGE_FIELD)
    links: List[Dict[str, Any]] = field(default_factory=list, metadata=_PACKAGE_FIELD)  # {type, to, when}
    notes: List[str] = field(default_factory=list, metadata=_PACKAGE_FIELD)
    priority: Any = field(default=None, metadata=_PACKAGE_FIELD)
    category: str = field(default="", metadata=_PACKAGE_FIELD)
    src: str = field(
        default="", metadata=_PACKAGE_FIELD
    )  # section reference as written in the graph, e.g. "§11.1"
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)  # fields not modelled here


@dataclass
class Definition:
    term: str
    meaning: str
    used_by: List[str] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""


@dataclass
class Fact:
    """Data from a tool output that a check uses."""

    record: Dict[str, Any]
    tool: str
    time: Optional[str] = None
    trust: Trust = Trust.TRUSTED
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""


@dataclass
class Verdict:
    """The result of one check. ``outcome``: allow, deny, remind or follow_up."""

    outcome: str
    rule_id: Optional[str] = None
    checker: Optional[str] = None
    reason: str = ""
    remedy: str = ""
    latency_ms: Optional[float] = None
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""


@dataclass
class CoverageItem:
    """A statement and what became of it, or why nothing did (``kind``: rule, step, definition, guidance,
    out_of_scope, dangling_link, conflict, classify_fallback, ...)."""

    statement_id: str
    kind: str
    reason: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    extra: Dict[str, Any] = field(default_factory=dict, metadata=_PACKAGE_FIELD)


@dataclass
class PolicyBuild:
    """What a build produces and the runtime loads. ``generator``: {"kind": "pipeline"|"skill", "harness", "model"}."""

    version: str
    rules: List[Rule] = field(default_factory=list)
    procedures: List[Procedure] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    definitions: List[Definition] = field(default_factory=list)
    coverage: List[CoverageItem] = field(default_factory=list)
    validation: List[Dict[str, Any]] = field(default_factory=list)
    generator: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    # {statement_ids, question, default_level}: what the build could not decide (``PolicyPackage.policy_questions``)
    policy_questions: List[Dict[str, Any]] = field(default_factory=list, metadata=_PACKAGE_FIELD)


@dataclass
class Obligation:
    """Work the policy requires, matched against what the agent did (design section 5). Status lives in the ledger.

    ``origin``: package, request, data or duty. ``effect``: ``{"kind": write|file|message|reply, "tools", "match"}``.
    ``values``: name → ``{"from": literal|template|computed|case_fact|request, ...}``; a value with no origin is not
    asserted. ``check``: code, kev or model. ``level``: observe, show or follow_up.
    """

    id: str
    origin: str = "package"
    scope: Dict[str, Any] = field(default_factory=dict)  # {"procedure": ..., "when": {...}}
    effect: Dict[str, Any] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    check: str = "code"
    level: str = "observe"
    source: Dict[str, Any] = field(default_factory=dict)  # {"section": ..., "quote": ...}


PACKAGE_SCHEMA = "policy-package/1"


@dataclass
class PolicyPackage:
    """One policy in one file (design section 4). ``package.py`` loads, saves and converts it.

    ``global_`` (JSON key ``global``): ``global_rules``, ``triage``, ``routing``, ``housekeeping``, ``priority``,
    ``escalation``, ``templates``, ``calendar`` as in a store graph, and ``definitions`` as a list of
    ``{term, meaning, ...}``. ``extra``: what a source artifact has and the package does not model.
    """

    schema: str = PACKAGE_SCHEMA
    version: str = ""
    policy_hash: str = ""
    generator: Dict[str, Any] = field(default_factory=dict)
    documents: List[Document] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    procedures: List[Procedure] = field(default_factory=list)
    obligations: List[Obligation] = field(default_factory=list)
    global_: Dict[str, Any] = field(default_factory=dict, metadata={"json": "global"})
    links: List[Link] = field(default_factory=list)
    coverage: List[CoverageItem] = field(default_factory=list)
    validation: List[Dict[str, Any]] = field(default_factory=list)
    policy_questions: List[Dict[str, Any]] = field(
        default_factory=list
    )  # {statement_ids, question, default_level}
    tests: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


# -- precedence ------------------------------------------------------------------------------------------------------


def _linked(ps: PolicySet, src_doc: Document, src: Statement, dst_doc: Document, dst: Statement) -> bool:
    """True if ``src`` supersedes ``dst``: a link between the statements, their sections or their documents."""
    src_ids = {src.id, f"{src_doc.id}:{src.source.get('section')}", src_doc.id}
    dst_ids = {dst.id, f"{dst_doc.id}:{dst.source.get('section')}", dst_doc.id}
    return any(lk.type == LinkType.SUPERSEDES and lk.src in src_ids and lk.dst in dst_ids for lk in ps.links)


def priority_ranks(priority_classes: Sequence[str]) -> Dict[str, int]:
    """The authority rank of each priority class of the config, which lists them from the highest: for
    ``[regulatory, company, team]`` it is regulatory 3, company 2, team 1. A class not listed has no rank here."""
    classes = list(priority_classes)
    return {c: len(classes) - i for i, c in enumerate(classes)}


def authority_of(doc: Document, priority_classes: Sequence[str] = ()) -> int:
    """The authority rank of a document. With a list of ``priority_classes``, a document whose ``priority_class`` is in it
    has the rank of that class (1 for the lowest), and the class replaces its own ``authority_rank``; any other document
    keeps its ``authority_rank``. So a classed document outranks one with rank 0, and the ranks of classes and the
    ``authority_rank`` of documents with no class share one scale: give both only on purpose."""
    return priority_ranks(priority_classes).get(doc.priority_class or "", doc.authority_rank)


def resolve_conflict(
    a: Statement, b: Statement, policy_set: PolicySet, priority_classes: Sequence[str] = ()
) -> Optional[Statement]:
    """Which of two conflicting statements wins, or None if nothing decides (then both are advisory only).

    Order: a trusted source beats an untrusted one; then an explicit ``supersedes`` link; then the newer effective
    date; then the higher authority rank (``authority_of``: the rank of the document's priority class, when the config
    lists one). Document order alone does not decide (2026-11-01 ruling: an unresolved conflict must never deny), and
    a tie on date and rank is never decided here.
    """
    da, db = policy_set.document_of(a.id), policy_set.document_of(b.id)
    if da is None or db is None:
        return None
    if da.trust != db.trust:
        return a if da.trust == Trust.TRUSTED else b
    a_over_b, b_over_a = _linked(policy_set, da, a, db, b), _linked(policy_set, db, b, da, a)
    if a_over_b != b_over_a:
        return a if a_over_b else b
    if da.effective_date and db.effective_date and da.effective_date != db.effective_date:
        return a if da.effective_date > db.effective_date else b
    ra, rb = authority_of(da, priority_classes), authority_of(db, priority_classes)
    if ra != rb:
        return a if ra > rb else b
    return None


# -- JSON ------------------------------------------------------------------------------------------------------------


def json_key(f: dataclasses.Field) -> str:
    """The JSON key of a model field: its name, or the Python keyword it stands for (``class``, ``global``)."""
    return f.metadata.get("json", f.name)


def to_json(obj: Any) -> Any:
    """Any model object (or list/dict of them) → plain JSON data. Enums become their values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            json_key(f): to_json(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if not (f.metadata.get("omit_empty") and getattr(obj, f.name) in (None, "", [], {}))
        }
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    return obj


def _decode(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Union:  # Optional[X]
        inner = [a for a in args if a is not type(None)]
        return _decode(inner[0], value) if len(inner) == 1 else value
    if origin in (list, List):
        if not isinstance(value, list):
            return value
        return [_decode(args[0], v) for v in value] if args else list(value)
    if origin in (dict, Dict):
        return dict(value) if isinstance(value, dict) else value
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(value)
    if tp is Step and isinstance(value, str):  # store graph steps are plain strings
        return Step(text=value)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return from_json(tp, value)
    return value


def from_json(cls: type, data: Dict[str, Any]) -> Any:
    """Plain JSON data → a model object of class ``cls`` (nested objects and enums restored). Keys the class does not
    have go to its ``extra``, if it has one; other classes ignore them."""
    hints = typing.get_type_hints(cls)
    kwargs, known = {}, set()
    for f in dataclasses.fields(cls):
        key = json_key(f) if json_key(f) in data else f.name
        known.add(key)
        if key in data:
            kwargs[f.name] = _decode(hints[f.name], data[key])
    obj = cls(**kwargs)
    unknown = {k: v for k, v in data.items() if k not in known}
    if unknown and hasattr(obj, "extra"):
        obj.extra = {**(obj.extra or {}), **unknown}
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(to_json(obj), indent=1, ensure_ascii=False)
