"""Policy runtime trace and audit.

Two things, both wired by ``runtime.build_runtime`` for every benchmark (``PolicyConfig.trace``, default on):

**The trace** (``PolicyTrace``): one event per model, decision-model and detector call that the policy layer makes, for
debugging: the component, the purpose, the prompt or state, the answer, the time and the tokens. The wrappers keep the
``usage`` and ``model_name`` attributes the runners read, call the model with the same arguments and return what it
returned, so the agent sees what it saw without the trace. Events stay in memory (``trace.events``).

**The audit** (``AuditLog``): one line in ``audit.jsonl`` for every decision of every component, in one shape, so that a
later reader (a replay, the fidelity report of a package) needs no knowledge of the component. A decision is anything a
component records as a verdict today (its ``records``), plus the few decisions that had no record. Fields, all strings, a
missing value is an empty string and never a missing key:

    time            UTC, ISO 8601, to the millisecond
    component       the name in ``runtime.COMPONENTS``
    task            the task id of the run
    hook            the hook, or ""
    tool            the tool of the call, or ""
    rule_id         the id of the rule; when the component has no ids, the name of the check (``procedure_gate``); for
                    the gate and for the statements that intake found in the data, the text of the rule; ``model_call``,
                    ``kev_call``, ``detector_call`` and ``hook_failed`` for a failure (see below)
    class           hard, judgment or guidance: the class of a package rule, ``judgment`` for the gate and for what a
                    model or the decision model says about a statement
    level           the level the package gives the rule (observe, remind, block_once, deny), or ""
    checker         who decided: code, cedar, verifier, kev, model, monitor, brief
    verdict         what the component did: allow, deny, remind, block, observe, followup, ..., ``error`` when the check
                    could not be made
    evidence        why: the reason, or the quoted data; cut to ``EVIDENCE_CHARS``
    source          where the rule comes from in the policy text (a section), or ""
    package_version the run's package version (``<version>+<file hash>``), or "" with no package

Every component with a ``records`` list gets its lines from that list with no change of its own: ``AuditLog.attach``
replaces the list by a ``Records`` list that reports each appended record. A component must finish a record before it
appends it; a change after the append is not in the line. A decision that no record holds is reported with ``emit``. The
mapping from a record to the fields is in ``NORMALIZERS``: one function per component whose records the generic one reads
wrongly. A component that is not in the table is read by the generic one.

**A check that fails open is a line.** The hooks run inside a registry that swallows an exception and goes on, a model
client answers with no value when it gives up, and a component keeps its default when a question fails. Each of these is a
line with the verdict ``error`` and the rule id of the failure: the three wrappers of the trace report a call that raised
or that a decision-model client gave up on (``model_call``, ``kev_call``, ``detector_call``); ``audited_failure`` reports an
exception of a hook (``hook_failed``) and raises it again; a component reports the check it skipped (the guide's evidence
check, the navigator's hand-off with no chapter, the loop's case with a failed question). The outcome for the run is not
changed by any of them. ``PolicyTrace.summary()`` counts the failed calls in ``errors``.

The audit never changes a run. It reads records, it never changes them, and an error in it (a folder that cannot be
written, a record it cannot read) is logged once and the run goes on. It never holds a key: the values of secret-looking
environment variables and key-shaped strings are replaced. It never holds a prompt: evidence is cut to ``EVIDENCE_CHARS``.
"""

from __future__ import annotations

import functools
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

AUDIT_FILE = "audit.jsonl"
AUDIT_FIELDS = (
    "time",
    "component",
    "task",
    "hook",
    "tool",
    "rule_id",
    "class",
    "level",
    "checker",
    "verdict",
    "evidence",
    "source",
    "package_version",
)
HOOK_FAILED = "hook_failed"  # the rule id of a line for a hook that raised
EVIDENCE_CHARS = 600  # one field of an audit line: a longer text is cut and says how long it was

# purpose from the first words of a known system prompt or user prompt (the rest is kept as text)
PURPOSES = (
    ("You read records that an assistant fetched", "extract_policies"),
    ("You check one planned action", None),  # verify or reply check: decided from the user prompt below
    ("You list what a user request asks", "deliverables"),
    ("You read a work request and the data", "bind_cases"),
)
USER_PURPOSES = (("Planned action:", "verify_write"), ("Write actions done", "reply_missing_check"))


def _purpose(system: str, user: str) -> str:
    for prefix, name in PURPOSES:
        if system.startswith(prefix):
            if name:
                return name
            for marker, uname in USER_PURPOSES:
                if marker in user:
                    return uname
    return (system.strip().splitlines() or ["model_call"])[0][:60]


class PolicyTrace:
    def __init__(self, max_chars: int = 20000, audit: Optional["AuditLog"] = None):
        """``audit``: where a failed call is reported (see ``AuditLog.failed_call``); none: events only."""
        self.max_chars = max_chars
        self.audit = audit
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    def _add(self, ev: Dict[str, Any]) -> None:
        ev["t_s"] = round(time.monotonic() - self._t0, 2)
        with self._lock:
            ev["n"] = len(self.events) + 1
            self.events.append(ev)

    def _cut(self, x: Any) -> Any:
        if isinstance(x, str) and len(x) > self.max_chars:
            return x[: self.max_chars] + f"...(cut, {len(x)} chars)"
        return x

    def _failed(self, kind: str, component: str, purpose: str, error: Optional[str]) -> None:
        """A call that raised, or that a decision-model client gave up on after its retries, is a decision with no answer:
        the component then goes on with its default. The audit says so."""
        if error and self.audit is not None:
            self.audit.failed_call(kind, component, purpose, error)

    @staticmethod
    def _kev_errors(model: Any, before: Dict[str, Any]) -> int:
        """Calls that the client of a decision model gave up on (it counts them in ``usage["errors"]`` and returns no answer,
        so the call does not raise)."""
        usage = getattr(model, "usage", None)
        return (
            ((usage or {}).get("errors", 0) or 0) - (before.get("errors", 0) or 0)
            if isinstance(usage, dict)
            else 0
        )

    @staticmethod
    def _keep_attrs(src: Any, dst: Callable) -> Callable:
        for a in ("usage", "model_name"):
            if hasattr(src, a):
                setattr(dst, a, getattr(src, a))
        return dst

    def wrap_model(self, model: Callable[[str, str], str], component: str) -> Callable[[str, str], str]:
        def call(system: str, user: str, *a: Any, **kw: Any) -> str:
            usage = getattr(model, "usage", None)
            before = dict(usage) if isinstance(usage, dict) else {}
            t = time.monotonic()
            err = None
            out = ""
            try:
                out = model(system, user, *a, **kw)
                return out
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                after = dict(getattr(model, "usage", None) or {})
                self._add(
                    {
                        "kind": "llm",
                        "component": component,
                        "purpose": _purpose(system or "", user or ""),
                        "model": getattr(model, "model_name", None),
                        "ms": round((time.monotonic() - t) * 1000),
                        "tokens_in": (after.get("input_tokens", 0) or 0)
                        - (before.get("input_tokens", 0) or 0),
                        "tokens_out": (after.get("output_tokens", 0) or 0)
                        - (before.get("output_tokens", 0) or 0),
                        "system": self._cut(system),
                        "prompt": self._cut(user),
                        "answer": self._cut(out),
                        "error": err,
                    }
                )
                self._failed("llm", component, _purpose(system or "", user or ""), err)

        return self._keep_attrs(model, call)

    def wrap_ask(self, ask: Callable[[str, Dict[str, str]], Dict[str, Any]], component: str) -> Callable:
        def call(state: str, questions: Dict[str, str]) -> Dict[str, Any]:
            t = time.monotonic()
            before = dict(getattr(ask, "usage", None) or {})
            err = None
            out: Dict[str, Any] = {}
            try:
                out = ask(state, questions)
                return out
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if not err and self._kev_errors(ask, before):
                    err = "the decision model gave no answer after its retries"
                self._add(
                    {
                        "kind": "decision",
                        "component": component,
                        "purpose": ",".join(questions),
                        "model": getattr(ask, "model_name", None),
                        "ms": round((time.monotonic() - t) * 1000),
                        "questions": questions,
                        "state": self._cut(state),
                        "answers": out,
                        "error": err,
                    }
                )
                self._failed("kev", component, ",".join(questions), err)

        return self._keep_attrs(ask, call)

    def wrap_detector(
        self, detect: Callable[[str, Any], List[float]], component: str = "detector"
    ) -> Callable:
        def call(request: str, texts: Any) -> List[float]:
            t = time.monotonic()
            before = dict(getattr(detect, "usage", None) or {})
            err = None
            scores: List[float] = []
            try:
                scores = detect(request, texts)
                return scores
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                failed = self._kev_errors(detect, before)
                if not err and failed:
                    err = f"the detector gave no score for {failed} record(s) after its retries, so they count as policies"
                self._add(
                    {
                        "kind": "detector",
                        "component": component,
                        "purpose": "is_policy",
                        "model": getattr(detect, "model_name", None),
                        "ms": round((time.monotonic() - t) * 1000),
                        "records": [self._cut(str(x))[:600] for x in texts],
                        "scores": scores,
                        "error": err,
                    }
                )
                self._failed("detector", component, "is_policy", err)

        return self._keep_attrs(detect, call)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """By kind and purpose: calls, time, tokens, and how many of the calls failed."""
        out: Dict[str, Dict[str, float]] = {}
        for e in self.events:
            k = f"{e['kind']}:{e['purpose']}"
            s = out.setdefault(k, {"calls": 0, "errors": 0, "ms": 0, "tokens_in": 0, "tokens_out": 0})
            s["calls"] += 1
            s["errors"] += 1 if e.get("error") else 0
            s["ms"] += e.get("ms", 0)
            s["tokens_in"] += e.get("tokens_in", 0) or 0
            s["tokens_out"] += e.get("tokens_out", 0) or 0
        return out


# -- the audit -----------------------------------------------------------------------------------------------------------

# A normalizer gives the fields of one line, several lines (a list), or None for a record that decides nothing.
Normalizer = Callable[[Dict[str, Any], Any], Union[None, Dict[str, Any], List[Dict[str, Any]]]]
# Who decides when a component's records do not say. A component that is not here: nothing is claimed.
DEFAULT_CHECKER = {
    "requirements_tracker": "model",
    "sop_tracker": "model",
    "procedure_monitor": "monitor",
    "guide": "code",
}
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.I)
_SECRET_SHAPE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,}")
REDACTED = "[redacted]"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:  # noqa: BLE001 - a value that cannot be written is still a value
        return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Records(list):
    """A list of records of one component that also reports each record it receives. In every other way a list: the
    component, the runner and the replay tools read it as they did. ``clear`` (and ``fresh``) keep the tap."""

    def __init__(self, items: Iterable[Any] = (), sink: Optional[Callable[[Any], None]] = None):
        super().__init__(items)
        self._sink = sink

    def _report(self, rec: Any) -> None:
        if self._sink is not None:
            try:
                self._sink(rec)
            except Exception:  # noqa: BLE001 - a record that cannot be audited never fails the component
                pass

    def append(self, rec: Any) -> None:
        super().append(rec)
        self._report(rec)

    def extend(self, items: Iterable[Any]) -> None:
        items = list(items)
        super().extend(items)
        for rec in items:
            self._report(rec)

    def insert(self, index: Any, rec: Any) -> None:
        super().insert(index, rec)
        self._report(rec)

    def __iadd__(self, items: Iterable[Any]) -> "Records":
        self.extend(items)
        return self

    def __reduce__(self) -> Any:
        return (list, (list(self),))  # a copy of the records is not a second tap: no line is written twice


def fresh(records: Any) -> List[Any]:
    """The record list of a new session: the same list, emptied, when it is tapped, so that its tap stays; else a new list."""
    if isinstance(records, Records):
        records.clear()
        return records
    return []


def emit(owner: Any, rec: Dict[str, Any]) -> None:
    """Report a decision that no record holds. Nothing happens when the run has no audit."""
    sink = getattr(owner, "_audit_sink", None)
    if sink is not None:
        try:
            sink(rec)
        except Exception:  # noqa: BLE001
            pass


def hook_failure(owner: Any, hook: str, args: Tuple[Any, ...], exc: BaseException) -> None:
    """One line (``rule_id`` ``hook_failed``, verdict ``error``) for an exception of a hook of ``owner``. The one who catches
    it goes on without the hook, so without this line the audit would show nothing for what the hook did not do. An exception
    that passes several of these is one line. Never raises; nothing happens when the owner has no audit."""
    try:
        if getattr(owner, "_audit_sink", None) is None or getattr(exc, "_hook_failed_line", False):
            return
        tool = args[0] if hook in ("before_tool_call", "after_tool_call", "tool_blocked") and args else ""
        emit(
            owner,
            {
                "hook": hook,
                "tool": tool,
                "rule": HOOK_FAILED,
                "outcome": "error",
                "reason": f"{type(exc).__name__}: {exc} (the run goes on without this check)",
            },
        )
        exc._hook_failed_line = True  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - the audit never changes a run
        pass


def audited_failure(hook: Callable[..., Any]) -> Callable[..., Any]:
    """For a hook that decides: an exception in it is reported by ``hook_failure`` and raised again, also where no registry
    catches it (a replay, a direct call)."""

    @functools.wraps(hook)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return hook(self, *args, **kwargs)
        except Exception as exc:
            from polisys.hooks import HookBlocked

            if not isinstance(exc, HookBlocked):
                hook_failure(self, hook.__name__, args, exc)
            raise

    return wrapper


class AuditLog:
    """The lines of one run. ``path``: the file; ``None`` keeps them in ``lines`` only. ``start`` creates the file empty, so
    that a run with a policy and no decision has a file and a baseline run has none."""

    def __init__(
        self,
        path: Optional[Path] = None,
        task: str = "",
        package_version: str = "",
        evidence_chars: int = EVIDENCE_CHARS,
    ):
        self.path = Path(path) if path else None
        self.task, self.package_version, self.evidence_chars = task, package_version, evidence_chars
        self.lines: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._attached: set = set()
        self._warned: set = set()
        # Secrets of this process, longest first. Read once: a key does not change during a run.
        values = {v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= 8}
        self._secrets = sorted(values, key=len, reverse=True)

    def _warn(self, key: str, message: str) -> None:
        """One warning for each kind of failure in a run: a folder that cannot be written would warn on every line."""
        if key in self._warned:
            return
        self._warned.add(key)
        try:
            from loguru import logger

            logger.warning("policy audit: {}", message)
        except Exception:  # noqa: BLE001 - the log is not worth a failure either
            pass

    def start(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")
        except OSError as exc:
            self._warn("file", f"cannot write {self.path}: {exc}; the run goes on with no audit file")
            self.path = None

    def attach(self, component: str, obj: Any) -> None:
        """Tap the record lists of ``obj`` (``records``, and ``log`` of the loop) and give it the sink of ``emit``. Lists
        that are properties are not tapped: the object that owns the list is."""
        if obj is None or id(obj) in self._attached:
            return
        self._attached.add(id(obj))

        def sink(rec: Any, _owner: Any = obj) -> None:
            self.record(component, _owner, rec)

        try:
            obj._audit_sink = sink
            obj._report_failure = functools.partial(hook_failure, obj)  # read by the hook registry
        except Exception:  # noqa: BLE001 - an object that takes no attribute has no decision to emit
            pass
        holder = getattr(obj, "__dict__", {})
        for attr in ("records", "log"):
            current = holder.get(attr)
            if isinstance(current, list) and not isinstance(current, Records):
                holder[attr] = Records(current, sink)

    def _clean(self, value: Any) -> str:
        text = _text(value)
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        text = _SECRET_SHAPE.sub(REDACTED, text)
        if len(text) > self.evidence_chars:
            text = text[: self.evidence_chars] + f"...(cut, {len(text)} chars)"
        return text

    def _build(self, component: str, fields: Dict[str, Any]) -> Dict[str, str]:
        line = {name: "" for name in AUDIT_FIELDS}
        line.update(component=component, task=self.task, package_version=self.package_version)
        line["time"] = _now()
        if not fields.get("checker"):
            fields = {**fields, "checker": DEFAULT_CHECKER.get(component, "")}
        for name, value in fields.items():
            if name in line and name not in ("time", "component", "task", "package_version"):
                line[name] = self._clean(value)
        return line

    def lines_of(self, component: str, owner: Any, rec: Any) -> List[Dict[str, str]]:
        """The lines of one record: none, one, or several (a verdict on each of the statements that a write broke)."""
        if not isinstance(rec, dict):
            return []
        # a failed hook is no decision of its component: its function would read the record as one (the gate as an allow)
        read = generic if rec.get("rule") == HOOK_FAILED else NORMALIZERS.get(component, generic)
        fields = read(rec, owner)
        if fields is None:
            return []
        return [self._build(component, f) for f in (fields if isinstance(fields, list) else [fields])]

    def _write(self, lines: List[Dict[str, str]]) -> None:
        with self._lock:
            self.lines.extend(lines)
            if self.path is not None and lines:
                try:
                    # a lone surrogate in a text of the run must not stop the line: it is written as an escape
                    with open(self.path, "a", encoding="utf-8", errors="backslashreplace") as f:
                        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
                except OSError as exc:
                    self._warn("file", f"cannot write {self.path}: {exc}; the run goes on with no audit file")
                    self.path = None

    def record(self, component: str, owner: Any, rec: Any) -> None:
        """One decision. Never raises."""
        try:
            self._write(self.lines_of(component, owner, rec))
        except Exception as exc:  # noqa: BLE001 - the audit never changes a run
            self._warn("record", f"a record could not be audited ({type(exc).__name__}: {exc})")

    def failed_call(self, kind: str, component: str, purpose: str, error: str) -> None:
        """A model, decision-model or detector call that failed. The component goes on with no answer, so this is a line of
        its own, whatever the component does with it. Never raises."""
        try:
            self._write(
                [
                    self._build(
                        component,
                        {
                            "rule_id": {
                                "llm": "model_call",
                                "kev": "kev_call",
                                "detector": "detector_call",
                            }.get(kind, "call"),
                            "checker": {"llm": "model"}.get(kind, "kev"),
                            "verdict": "error",
                            "evidence": f"{purpose}: {error}",
                        },
                    )
                ]
            )
        except Exception as exc:  # noqa: BLE001
            self._warn("record", f"a failed call could not be audited ({type(exc).__name__}: {exc})")


# -- from a record to the fields -------------------------------------------------------------------------------------------

# keys a record uses for a field of the line, or for timing and advice that is not evidence
_RECORD_KEYS = {
    "hook",
    "tool",
    "rule",
    "rule_id",
    "outcome",
    "verdict",
    "verifier",
    "type",
    "source",
    "check_ms",
    "reason",
    "remedy",
    "checker",
}


def _rule_of(owner: Any, rule_id: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """The rule dict behind an id, and the rule set that holds it: the owner, or the rule set inside an intake."""
    if rule_id:
        for holder in (owner, getattr(owner, "enforcer", None)):
            for rule in getattr(holder, "rules", None) or []:
                if isinstance(rule, dict) and rule.get("id") == rule_id:
                    return holder, rule
    return None, None


def _rule_fields(owner: Any, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Class, level and checker of the package rule that a record names (a record with ``rule_id``, as a rule set writes
    it). Empty when it names none."""
    if "rule_id" not in rec:
        return {}
    holder, rule = _rule_of(owner, _text(rec.get("rule_id")))
    rule_type = _text(rec.get("type") or (rule or {}).get("type"))
    if not rule_type:
        return {}
    from .handlers.prose import CHECKER
    from .package import rule_class

    checker = ""
    if rule is not None:
        checker = "verifier" if rule.get("routed_from") else ""
        if not checker:
            try:
                checker = holder._choice(rule).get("checker") or ""
            except Exception:  # noqa: BLE001 - a rule set without method choice
                checker = ""
    return {
        "class": rule_class(rule_type),
        "level": (rule or {}).get("level") or "",
        "checker": checker or CHECKER.get(rule_type, "code"),
    }


def generic(rec: Dict[str, Any], owner: Any) -> Dict[str, Any]:
    """For a record with an ``outcome`` (every component that has no function of its own)."""
    rule = rec.get("rule")
    reason = rec.get("reason")
    rest = {k: v for k, v in rec.items() if k not in _RECORD_KEYS}
    verdict = rec.get("outcome") or rec.get("verdict") or rec.get("verifier")
    if rule == "deny_cap":  # the rule set's legacy name of the cap, one name in the audit
        verdict = "deny_after_cap" if verdict == "deny" else "allow_after_cap"
    if isinstance(reason, str) and reason.startswith("verifier_error"):
        verdict = "error"  # the verifier failed and the write was let through: not a normal allow
    fields = {
        **_rule_fields(owner, rec),
        "hook": rec.get("hook"),
        "tool": rec.get("tool"),
        "rule_id": rec.get("rule_id") or (rule if isinstance(rule, str) else "") or rec.get("procedure"),
        "verdict": verdict,
        "evidence": reason if isinstance(reason, str) and reason else (rest or ""),
        "source": rec.get("source"),
    }
    if rec.get("checker"):
        fields["checker"] = rec["checker"]
    return fields


def _gate(rec: Dict[str, Any], owner: Any) -> Dict[str, Any]:
    """The gate's record says what the model found; ``gate_verdict`` says what the gate did with it."""
    from .store_gate import gate_verdict

    base = {"hook": "before_tool_call", "tool": rec.get("tool"), "checker": "model", "level": "block_once"}
    if rec.get("gate") == "error":
        return {**base, "verdict": "error", "evidence": rec.get("error")}
    verdict = gate_verdict(rec)
    scope = (
        f"scope {rec['scope']}" if rec.get("scope") else ""
    )  # only when the method is not the measured default
    if verdict == "allow":
        return {**base, "class": "judgment", "verdict": "allow", **({"evidence": scope} if scope else {})}
    rule = _text(rec.get("rule"))
    tag = re.match(r"\s*\[([^\]]+)\]", rule)
    return {
        **base,
        "class": "judgment",
        "rule_id": rule,
        "verdict": verdict,
        "evidence": f"{rec.get('case')}: {rec.get('evidence')}" + (f" [{scope}]" if scope else ""),
        "source": tag.group(1) if tag else "",
    }


def _navigator(rec: Dict[str, Any], owner: Any) -> Dict[str, Any]:
    """A hand-off check (``SopNavigator.check_handoff``). It blocks when steps of the chapter are still open: the
    navigator then gives a note, and the guard does not forward the hand-off."""
    from .sop_navigator import holds

    if "outcome" in rec:  # the navigator's or the guard's own decision (``emit``)
        return generic(rec, owner)
    pending = rec.get("pending")
    exits = rec.get("exits") or {}
    base = {"checker": "kev", "source": rec.get("chapter"), "hook": rec.get("hook"), "tool": rec.get("tool")}
    if pending:
        return {
            **base,
            "rule_id": ",".join(str(p) for p in pending),
            "verdict": "block",
            "evidence": f"open steps of the chapter: {', '.join(str(p) for p in pending)}",
        }
    if holds(rec.get("solved")):
        return {
            **base,
            "verdict": "allow",
            "evidence": f"the customer's problem is solved (p={rec['solved']})",
        }
    if any(holds(v) for v in exits.values()):
        return {**base, "verdict": "allow", "evidence": "an escalation condition of the SOP holds"}
    return {**base, "verdict": "allow", "evidence": "no open step in the chapter"}


def _sop_tracker(rec: Dict[str, Any], owner: Any) -> Dict[str, Any]:
    """What the tracker told the agent before a turn: an escalation, the steps not done yet, or nothing."""
    base = {"checker": "model", "source": ", ".join(str(s) for s in rec.get("sections") or [])}
    if "error" in rec:
        return {**base, "verdict": "error", "evidence": rec["error"]}
    pending = [s for s in rec.get("steps") or [] if isinstance(s, dict) and s.get("status") == "pending"]
    esc = rec.get("escalation")
    if esc:
        return {**base, "verdict": "escalate", "evidence": f"{esc.get('quote')} | {esc.get('evidence')}"}
    if pending:
        return {**base, "verdict": "remind", "evidence": "; ".join(str(s.get("action")) for s in pending)}
    return {**base, "verdict": "allow"}


def _loop(rec: Dict[str, Any], owner: Any) -> Union[None, Dict[str, Any]]:
    """``SopLoop.log`` (a ``phase``) holds the decision for each case; ``LoopHooks.records`` hold the stops of writes. A
    question that failed leaves the case at its default (follow) or the plan with no case: the line says ``error``, and
    the run goes on as it did."""
    if "phase" not in rec:
        return {**generic(rec, owner), "checker": "code"}
    errors = rec.get("errors") or []
    if (
        rec.get("phase") == "plan"
    ):  # the plan lists the cases and decides nothing about a rule, unless it could not be made
        if not errors:
            return None
        return {
            "checker": "model",
            "rule_id": "questions",
            "verdict": "error",
            "evidence": f"{len(errors)} question(s) of the plan failed ({errors[0]}): no case was checked",
        }
    if rec.get("phase") != "case":
        return None
    reasons = (
        "; ".join(str(r) for r in (rec.get("reasons") or rec.get("per_item") or [])) or "no reason given"
    )
    if errors:
        reasons = (
            f"{len(errors)} question(s) failed ({errors[0]}), so the decision {rec.get('decision')} rests on the rest: "
            + reasons
        )
    decision = rec.get("decision")
    # with no procedure chosen, a decision that is not "follow" came from the general rules of the SOP
    general = "general_rules" if decision not in ("follow", "discovery") and not rec.get("procedure") else ""
    return {
        "checker": "model",
        "rule_id": rec.get("procedure") or general,
        "verdict": "error" if errors else decision,
        "evidence": f"{rec.get('case')}: {reasons}",
    }


def _statement_id(item: Dict[str, Any], rule: Any) -> str:
    """The id of an online statement: its text (it has no other id; ``n`` is a place in the list of one call)."""
    return _text(item.get("text") or rule or f"statement {item.get('statement')}")


def _intake(rec: Dict[str, Any], owner: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Intake's decisions, from the checks it makes itself: the decision model, the generated code, the read-before-write
    check, and the model that verifies a write against the statements found. A write that broke several statements gives
    a line for each. The online rule set's own checks (``rule_id``) and the cap are read as a rule set's."""
    if "rule_id" in rec or rec.get("rule") == "deny_cap":
        return generic(rec, owner)
    rule = rec.get("rule")
    named = {"decisions": "kev", "guard_code": "code", "read_before_write": "code"}
    base = generic(rec, owner)
    items = rec.get("violations") if "violations" in rec else rec.get("missing")
    base["checker"] = named.get(rule) or ("model" if items is not None else "")
    if rule == "decisions":
        base["class"] = "judgment"
    if not items:
        return base
    return [
        {
            **base,
            "rule_id": _statement_id(i, rule),
            "class": "hard" if rule == "guard_code" else "judgment",
            "evidence": i.get("reason") or i.get("action"),
            "source": i.get("source"),
        }
        for i in items
        if isinstance(i, dict)
    ]


def _deliverables(rec: Dict[str, Any], owner: Any) -> Dict[str, Any]:
    """The list of what the request asks is made by a model; the checks against the writes are code."""
    extracted = rec.get("outcome") in ("deliverables", "deliverables_error")
    return {**generic(rec, owner), "checker": "model" if extracted else "code"}


def _obligations(rec: Dict[str, Any], owner: Any) -> Dict[str, Any]:
    """One item of the ledger, or one decision about a follow-up. The status is the verdict."""
    return {**generic(rec, owner), "level": rec.get("level")}


def _guide(rec: Dict[str, Any], owner: Any) -> Dict[str, Any]:
    return {**generic(rec, owner), "checker": "model" if rec.get("rule") == "procedure_evidence" else "code"}


NORMALIZERS: Dict[str, Normalizer] = {
    "gate": _gate,
    "guide": _guide,
    "loop": _loop,
    "intake": _intake,
    "deliverables": _deliverables,
    "obligations": _obligations,
    "navigator": _navigator,
    "sop_tracker": _sop_tracker,
}
