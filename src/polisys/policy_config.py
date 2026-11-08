"""Policy config: one YAML file sets one deployment or experiment (design spec, section 12).

A profile holds facts the deployment knows and method choices. It never holds rules: rules live in the policy package.

    policy:
      package: packages/<bench>/{policy}/<source>/package.json   # none = rules only from data
      sources: {data: true, discovery: true, request: false}
      checkers: {rules: on, gate: {mode: on, model: gcp/gemini-3.8-flash}}
      organizers: {guide: on, loop: {mode: on, model: azure/DeepSeek-V4-Pro}, sop_tracker: off}
      ...

``load_config(path)`` reads a file and returns a ``PolicyConfig``. Rules of the format:

- Every key must be one this module knows. An unknown key raises ``PolicyConfigError`` that names it with its full path
  (``policy.checkers.gate.modle``), at every nesting level. So does a value of the wrong type or outside its choices.
- A duplicate key in the file raises too (YAML would keep the last one).
- A component is ``auto``, ``on`` or ``off``. ``auto`` lets the control level decide (``levels.py``); a measured profile
  lists every component as ``on`` or ``off``, and an explicit ``on`` or ``off`` always wins. A section whose fields include ``mode`` can be written as the mode alone
  (``guide: on``). YAML reads a bare ``on`` or ``off`` as a boolean; the loader accepts both forms.
- A text field that can be empty takes ``null`` or ``none``.
- ``package`` is a path relative to the folder that holds ``profiles/``. It may hold the placeholder ``{policy}``: one
  profile serves several policies, and the harness fills it in with the id of the task's policy. Nothing here opens it.
"""

from __future__ import annotations

import collections.abc
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

KEV = "jaredpalmer/kev-4b"  # decision model for narrow yes/no questions (spec section 12)

Mode = Literal["auto", "on", "off"]
Level = Literal[0, 1, 2, 3, "by_model"]


class _AtLeast:
    """Marker for ``Annotated``: the whole number may not be below ``floor``."""

    def __init__(self, floor: int):
        self.floor = floor


Count = Annotated[int, _AtLeast(0)]  # a number of blocks or follow-ups: 0 or more


class PolicyConfigError(ValueError):
    """The config has an unknown key, a value of the wrong type or a value outside its choices. Names the key."""


@dataclass
class Sources:
    """Where rules and requirements may come from."""

    data: bool = False  # rules found in tool results while the agent runs (online intake)
    discovery: bool = False  # the layer searches the message sources for rules at the start
    listing: bool = False  # the layer reads every record the host lists at the start (listing.py)
    request: bool = False  # the user's request is read as a source of requirements


@dataclass
class Authority:
    company_domains: List[str] = field(default_factory=list)  # sender domains whose messages are trusted
    may_change_rules: List[str] = field(default_factory=list)  # roles that may change a rule
    supersedes: bool = False  # a rule replaced by a newer one leaves the current list


@dataclass
class Risk:
    level3_tools: List[str] = field(
        default_factory=list
    )  # tool name patterns that get level-3 checks for any model


@dataclass
class Control:
    """The control level (spec section 7): 0 observe, 1 advise, 2 check, 3 control. ``levels.py`` reads it.

    ``by_model`` is also the table of model strength: a lower level means a stronger model (a strong agent needs less
    control), and ``gate.min_strength`` compares the gate's model with the agent's by it. A model that is not in the table
    takes ``default_level``, and the run says so in its audit. These settings are read only where a level decides something (a
    component left to ``auto``, ``write_checks``, a ledger with no ``item_level``, level 0): on a profile that lists every
    component, ``build_runtime`` raises on them. At level 0 every check runs where it is built, and so does the ledger's binding
    and checklist call, with nothing sent to the agent."""

    level: Level = "by_model"  # 0 to 3, or by_model: the level comes from the table below
    by_model: Dict[str, Literal[0, 1, 2, 3]] = field(default_factory=dict)  # agent model -> default level
    default_level: Literal[0, 1, 2, 3] = (
        1  # a model that is not in by_model: advise (hard rules and the brief); auto turns on no check that calls a model
    )
    risk: Risk = field(default_factory=Risk)


@dataclass
class Router:
    """Which procedures are in scope for the gate and the ledger (spec section 11; ``router.py`` registers the names)."""

    kind: str = "agent_opened"  # which procedures apply; a registered router
    expand: str = "next_links"  # which procedures and rules come along; a registered expander
    model: Optional[str] = (
        None  # the JSON model of a router that reads cards (llm_cards); bm25_kev uses narrow_questions
    )
    top_k: Count = (
        3  # procedures a searching router returns (agent_opened returns every one the agent opened)
    )
    hops: Count = 1  # k_hops: steps along refers_to, exception_of and supersedes


@dataclass
class Gate:
    """Model check before an acting call, over the rules in scope; blocks once with quoted evidence."""

    mode: Mode = "auto"
    model: Optional[str] = None
    scope: Literal["active_procedures", "whole_sop"] = "active_procedures"
    min_strength: Literal["agent", "none"] = "agent"  # none: the gate model may be weaker than the agent
    seed_request: bool = False  # the user's request counts as data the agent read
    max_tokens: Count = 4000  # one old runner gave gpt-oss models 16000
    missing_needs_read: bool = False  # a "something is missing" block needs a read of the record that should hold it


@dataclass
class Verifier:
    """Model check on policy statements; reminds unless ``may_deny``.

    ``scope``: ``all_statements`` sends every statement that did not become a typed rule (every trusted statement that
    did not compile, or every ``verify`` rule); ``specific_statements`` sends only those that name one tool and one
    argument (spec section 7, principle 8). The measured runs used ``all_statements``.
    """

    mode: Mode = "auto"
    model: Optional[str] = None
    scope: Literal["specific_statements", "all_statements"] = "specific_statements"
    may_deny: bool = False


@dataclass
class Checkers:
    rules: Mode = "auto"  # typed rules enforced by code (and Cedar for roles)
    compile: Optional[str] = None  # model that turns policy text found in data into rules
    narrow_questions: Optional[str] = None  # model for narrow yes/no questions
    detector: Optional[str] = None  # model that marks which records are policies
    router: Router = field(default_factory=Router)
    gate: Gate = field(default_factory=Gate)
    verifier: Verifier = field(default_factory=Verifier)
    guard_code: Literal["off", "remind", "deny"] = "off"  # generated code checks per statement and write tool
    read_before_write: Literal["off", "remind", "block_once"] = "off"  # a referred policy source is unread
    method_choice: Optional[str] = None  # path of a file with the measured checker for each rule type
    # One hook that runs the write checks in order and stops at the first block (spec section 6): the typed rules, then the
    # gate, then the verifier. off: separate hooks, as measured. auto: on when two or more of the three are on and the level
    # switch turned at least one of them on, or a tool is in ``control.risk.level3_tools``; a profile that lists every
    # component keeps its separate hooks. A stage the profile says ``on`` runs for every call, whatever the level.
    write_checks: Mode = "auto"


@dataclass
class Guide:
    """Procedure guide: stops for steps and preconditions of the procedure the agent is in."""

    mode: Mode = "auto"
    evidence_model: Optional[str] = (
        None  # a model checks hard preconditions against evidence; none = no check
    )
    stop_first_write: bool = True  # stop the first write of a procedure
    stop_templates: bool = True  # stop on a template mismatch


@dataclass
class ProcedurePush:
    """The policy system picks the procedures that govern the data just read and attaches them to the tool result."""

    mode: Mode = "off"
    model: Optional[str] = None  # the llm_cards router's model
    top_k: Count = 2
    max_selections: Count = 15  # router calls per run


@dataclass
class Loop:
    """SOP loop: explore, then one scoped turn per case."""

    mode: Mode = "auto"
    model: Optional[str] = None
    style: Literal["control", "advisory", "discovery"] = "control"
    rules_from_data: bool = True  # with no procedures, the loop asks the model for the rules in the data


@dataclass
class SopTracker:
    """A progress note with each customer turn."""

    mode: Mode = "auto"
    model: Optional[str] = None


@dataclass
class Navigator:
    """Hand-off check: a premature hand-off is blocked and the open actions are named."""

    mode: Mode = "auto"
    semantic_evidence: bool = False
    max_blocks: Count = 2
    handoff_text: List[str] = field(default_factory=list)  # phrases in a reply that hand the customer off
    handoff_tools: List[str] = field(default_factory=list)  # tools that hand the customer off


@dataclass
class RequirementsTracker:
    """A checklist of what the request and the policy require, checked at the reply."""

    mode: Mode = "auto"
    model: Optional[str] = None
    wrong_items: Literal["off", "verify", "fix"] = "off"  # what to do with an item judged wrong
    read_system_prompt: bool = False  # the checklist also reads the system prompt as policy text


@dataclass
class Organizers:
    procedure_tools: Mode = (
        "auto"  # the agent reads the procedures through tools; what it opens scopes the gate
    )
    guide: Guide = field(default_factory=Guide)
    procedure_push: ProcedurePush = field(default_factory=ProcedurePush)
    loop: Loop = field(default_factory=Loop)
    sop_tracker: SopTracker = field(default_factory=SopTracker)
    navigator: Navigator = field(default_factory=Navigator)
    requirements_tracker: RequirementsTracker = field(default_factory=RequirementsTracker)


@dataclass
class Obligations:
    # Empty: no ledger. Any producer turns the obligation ledger on, and then only the ledger sends follow-ups (the requirements
    # tracker is refused; the procedure monitor and the duty rules only record).
    producers: List[Literal["package", "request", "duty"]] = field(default_factory=list)
    max_follow_ups: Count = 1
    wrong_values: Literal["never", "known_origin_only"] = "known_origin_only"
    model: Optional[str] = (
        None  # model that lists the checklist of the request, and that binds the ledger's cases
    )
    # The level of every item of the ledger. None: ``build_runtime`` uses observe, so the level written in an obligation is not
    # read in a run. ``show`` is accepted and not built: only follow_up items are sent, so it acts as observe. Packages say
    # observe until a measurement chooses.
    item_level: Optional[Literal["observe", "show", "follow_up"]] = None


@dataclass
class Decisions:
    """Narrow-question write checks and what a verdict does."""

    mode: Mode = "auto"
    effect: Literal["remind", "block_once"] = "remind"
    after_repeated_denials: Literal["allow", "skip"] = "allow"
    require_evidence: bool = False  # a stop needs quoted evidence about the case
    record_checks: bool = False  # record signals, missing facts, content match
    retry_effect: Literal["allow", "hold"] = "allow"  # what a recheck of a blocked call does


@dataclass
class Followups:
    """What a check may say to the agent at the reply (``followups.py``)."""

    voice: Literal["order", "check"] = "order"  # check: says it is automated, and the checker sees the agent's reply
    evidence: Literal["off", "on"] = "off"  # on: an action goes to the agent only if a check finds it really required


@dataclass
class Notes:
    reminders: Literal["tool_result", "system"] = "tool_result"  # where a reminder goes
    brief: Literal["system", "message", "off"] = "system"  # where the brief goes (agent setting), or off
    # The brief of the package's rules also lists the guidance rules that name no tool (the measured runs never showed them).
    brief_guidance: bool = False
    # General rules for SOP work on many records, in the system prompt (work_method.py); "readback" adds one read-back turn
    # before the final answer. Empty: none.
    work_method: List[Literal["worksheet", "actors", "request_scope", "cells", "idempotent", "readback"]] = field(
        default_factory=list
    )


Shape = Literal[
    "procedures", "data", "flow", "rules"
]  # spec section 8: how a source of policy reaches the agent


@dataclass
class PolicyConfig:
    package: Optional[str] = None
    # The shapes of the policy sources of this run; empty: inferred from the package and the sources (levels.shapes_of).
    shapes: List[Shape] = field(default_factory=list)
    sources: Sources = field(default_factory=Sources)
    authority: Authority = field(default_factory=Authority)
    priority_classes: List[str] = field(default_factory=list)
    control: Control = field(default_factory=Control)
    checkers: Checkers = field(default_factory=Checkers)
    organizers: Organizers = field(default_factory=Organizers)
    obligations: Obligations = field(default_factory=Obligations)
    decisions: Decisions = field(default_factory=Decisions)
    followups: Followups = field(default_factory=Followups)
    notes: Notes = field(default_factory=Notes)
    trace: bool = True

    @classmethod
    def from_dict(cls, data: Any, path: str = "policy") -> "PolicyConfig":
        return _load(cls, data, path)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _kind(value: Any) -> str:
    return type(value).__name__


def _load(cls: Any, data: Any, path: str) -> Any:
    fields_ = {f.name: f for f in dataclasses.fields(cls)}
    if not isinstance(data, Mapping):
        if "mode" in fields_ and isinstance(data, (bool, str)):  # `guide: on` means `guide: {mode: on}`
            data = {"mode": data}
        else:
            raise PolicyConfigError(
                f"'{path}' must be a mapping with the keys {', '.join(fields_)}; got {_kind(data)}"
            )
    for key in data:
        if key not in fields_:
            raise PolicyConfigError(f"unknown key '{path}.{key}' (allowed in '{path}': {', '.join(fields_)})")
    hints = get_type_hints(cls, include_extras=True)
    return cls(**{key: _convert(value, hints[key], f"{path}.{key}") for key, value in data.items()})


def _convert(value: Any, tp: Any, path: str) -> Any:
    origin = get_origin(tp)
    if origin is Annotated:
        inner, *marks = get_args(tp)
        out = _convert(value, inner, path)
        for mark in marks:
            if isinstance(mark, _AtLeast) and out < mark.floor:
                raise PolicyConfigError(f"'{path}' must be {mark.floor} or more; got {out}")
        return out
    if dataclasses.is_dataclass(tp):
        return _load(tp, value, path)
    if origin is Union:  # Optional[X]
        inner = [a for a in get_args(tp) if a is not type(None)][0]
        if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
            return None
        return _convert(value, inner, path)
    if origin is Literal:
        choices = get_args(tp)
        if isinstance(value, bool):  # YAML reads a bare on / off as a boolean
            word = "on" if value else "off"
            if word in choices:
                return word
        elif type(value) in (int, str) and value in choices:
            return value
        raise PolicyConfigError(
            f"'{path}' must be one of {', '.join(repr(c) for c in choices)}; got {value!r}"
        )
    if origin is list:
        if not isinstance(value, list):
            raise PolicyConfigError(f"'{path}' must be a list; got {_kind(value)}")
        return [_convert(item, get_args(tp)[0], f"{path}[{i}]") for i, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise PolicyConfigError(f"'{path}' must be a mapping; got {_kind(value)}")
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PolicyConfigError(f"'{path}' has a key that is not text: {key!r}")
            out[key] = _convert(item, get_args(tp)[1], f"{path}.{key}")
        return out
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is str:
        if isinstance(value, str):
            return value
    else:  # a type this module declares but cannot read: a mistake in the schema, not in the file
        raise TypeError(f"policy_config cannot read the type {tp} of '{path}'")
    raise PolicyConfigError(f"'{path}' must be {tp.__name__}; got {_kind(value)} {value!r}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML keeps the last of two equal keys without a word; here it is an error."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Dict[Any, Any]:
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, collections.abc.Hashable):
                if key in seen:
                    raise PolicyConfigError(f"duplicate key '{key}' (line {key_node.start_mark.line + 1})")
                seen.add(key)
        return super().construct_mapping(node, deep)


def load_config(path: Union[str, Path]) -> PolicyConfig:
    """Read a profile file. The only top-level key is ``policy``."""
    path = Path(path)
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise PolicyConfigError(f"{path}: cannot read the file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise PolicyConfigError(f"{path}: the file is not UTF-8 text: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PolicyConfigError(f"{path}: not valid YAML: {exc}") from exc
    except PolicyConfigError as exc:
        raise PolicyConfigError(f"{path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise PolicyConfigError(
            f"{path}: the file must hold a mapping with the key 'policy'; got {_kind(data)}"
        )
    for key in data:
        if key != "policy":
            raise PolicyConfigError(f"{path}: unknown key '{key}' (the only top-level key is 'policy')")
    if "policy" not in data:
        raise PolicyConfigError(f"{path}: the key 'policy' is missing")
    try:
        return PolicyConfig.from_dict(data["policy"])
    except PolicyConfigError as exc:
        raise PolicyConfigError(f"{path}: {exc}") from exc
