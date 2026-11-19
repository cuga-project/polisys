"""The checks that code decides (part of ``RuleSetHooks``)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from polisys.values import _condition, _joined, _number, _when, canon, norm
from polisys.checks.rules.rule_types import Check, FREE_TEXT, ID_VALUE
from polisys.checks.rules.seen import _stem


class _CodeChecks:
    """One method per rule type that code decides: ``_check_<type>(rule, tool, args)`` gives a ``Check``, or None when the
    rule does not decide this call. ``RuleSetHooks._checks`` calls them by the type of the rule. They read ``self.facts``,
    ``self.called`` and ``self.specs``."""

    def _check_record_condition(self, r, tool, args) -> Optional[Check]:
        if not _when(r.get("when"), args):
            return None
        from polisys.checks.facts import call_tokens  # identifiers in the call's arguments

        fld, op, value = r.get("field", ""), r.get("op", "contains"), r.get("value")
        seen_any = False
        for tok in call_tokens(args):
            for rec in self.facts.named(tok):
                key = next((k for k in rec if k.lower() == str(fld).lower()), None)
                if key is None:
                    continue
                seen_any = True
                if _condition(rec.get(key), op, value):
                    return Check(
                        r,
                        False,
                        f"record {tok} has {key} {op} {value!r}",
                        f"do not act on {tok}; leave it out (policy condition on {key})",
                    )
        return Check(r, True, f"named records do not meet {fld} {op} {value!r}") if seen_any else None

    def _check_threshold(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        num = _number(args.get(arg))
        if num is None or not _condition(num, r.get("op", "gt"), r.get("value")):
            return None if num is None else Check(r, True, f"{arg} within the limit")
        if r.get("then") == "requires_approval" and r.get("approval_tool") in self.called:
            return Check(r, True, f"{arg} above the limit, approval requested before")
        if r.get("then") == "requires_approval":
            return Check(
                r,
                False,
                f"`{arg}={args.get(arg)}` needs approval ({r.get('op')} {r.get('value')})",
                "request approval first"
                + (f" with `{r['approval_tool']}`" if r.get("approval_tool") else ""),
            )
        return Check(
            r,
            False,
            f"`{arg}={args.get(arg)}` is over the limit ({r.get('op')} {r.get('value')})",
            f"keep {arg} within the limit or stop and explain",
        )

    def _check_format(self, r, tool, args) -> Optional[Check]:
        arg, pattern = r.get("arg"), r.get("pattern", "")
        if arg not in args or args[arg] in (None, ""):
            return None
        try:  # free text (a message, a note) must contain the pattern; a value field must match it whole
            text, pattern = canon(args[arg]), canon(pattern)
            free = bool(FREE_TEXT.search(arg)) or len(text.split()) > 3
            ok = (
                re.search(pattern, text) if free else re.fullmatch(pattern, text, flags=re.DOTALL)
            ) is not None
        except re.error:
            return None
        return (
            Check(r, True, f"{arg} format ok")
            if ok
            else Check(
                r,
                False,
                f"`{arg}={args[arg]}` does not have the required format",
                f"write {arg} in the form {pattern}",
            )
        )

    def _check_required_args(self, r, tool, args) -> Optional[Check]:
        if not _when(r.get("when"), args):
            return None
        missing = [a for a in r.get("args", []) if args.get(a) in (None, "", [])]
        if missing:
            return Check(
                r,
                False,
                f"missing mandatory field(s) {_joined(', ', missing)}",
                f"look up the value(s) and pass {_joined(', ', missing)}; if the data does not exist, stop and "
                f"state which field is required",
            )
        return Check(r, True, "mandatory fields present")

    def _check_allowed_values(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        if arg not in args or not _when(r.get("when"), args):
            return None
        allowed = {norm(str(v).lstrip("#@")) for v in r.get("values", [])}
        if norm(str(args[arg]).lstrip("#@")) not in allowed and not self._names_allowed(args[arg], allowed):
            if r.get("schema_conflict"):
                return Check(
                    r,
                    False,
                    f"the policy lists {_joined(', ', r.get('values'))} for `{arg}`, but the "
                    f"tool accepts other values",
                    f"keep a value the tool accepts for {arg}; do not drop {arg}",
                )
            return Check(
                r,
                False,
                f"`{arg}={args[arg]}` is not allowed here; allowed: {_joined(', ', r.get('values'))}",
                f"use one of the allowed values for {arg}",
            )
        return Check(r, True, f"{arg} allowed")

    def _names_allowed(self, value: Any, allowed: set) -> bool:
        """An id whose record carries an allowed name ("C_SALES_ALERTS" with name "sales-alerts") is allowed."""
        for rec in self.facts.named(str(value)):
            for k, v in rec.items():
                if (
                    isinstance(v, str)
                    and re.search(r"name|title|label|handle", k, re.I)
                    and norm(v.lstrip("#@")) in allowed
                ):
                    return True
        return False

    def _check_derived_value(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        if arg not in args:
            return None
        inputs = r.get("inputs", [])
        vals = []
        for i in inputs:
            v = args.get(i)
            if v is None:
                rec = self._record_for(args)
                v = rec.get(i) if rec else None
            vals.append(norm(v) if v is not None else None)
        if None in vals:
            return Check(
                r,
                False,
                f"`{arg}` is computed from {_joined(', ', inputs)}, which are not known for this call",
                f"do not set {arg}; the system computes it",
            )
        want = (r.get("table") or {}).get("|".join(vals))
        if want is not None and norm(args[arg]) != norm(want):
            return Check(
                r,
                False,
                f"`{arg}={args[arg]}` does not match the policy table ({' x '.join(vals)} → {want})",
                f"do not set {arg}, or set it to {want}",
            )
        return Check(r, True, f"{arg} matches the table")

    def _record_for(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for k, v in args.items():
            if isinstance(v, str) and (k == "id" or k.endswith("_id")) and v in self.facts.records:
                return self.facts.records[v]
        return None

    def _check_transition(self, r, tool, args) -> Optional[Check]:
        arg, id_arg = r.get("arg"), r.get("id_arg")
        if arg not in args:
            return None
        rec = self.facts.records.get(str(args.get(id_arg))) if id_arg else None
        current = rec.get(arg) if rec else None
        if current is None:
            return Check(
                r,
                False,
                f"the current `{arg}` of {args.get(id_arg)} is not known, so the change to "
                f"`{args[arg]}` cannot be checked",
                f"read the record first, then change {arg}",
            )
        edges = {norm(k): {norm(x) for x in v} for k, v in (r.get("edges") or {}).items()}
        if norm(current) == norm(args[arg]):
            return Check(r, True, "no change")
        if norm(args[arg]) not in edges.get(norm(current), set()):
            allowed = ", ".join(sorted(edges.get(norm(current), set()))) or "none"
            return Check(
                r,
                False,
                f"`{arg}` cannot go from {current} to {args[arg]}; allowed from {current}: {allowed}",
                "use an allowed transition (possibly through an intermediate state)",
            )
        return Check(r, True, f"{current} → {args[arg]} allowed")

    def _check_reference_status(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        if arg not in args:
            return None
        rec = self.facts.records.get(str(args[arg]))
        status = rec.get(r.get("field")) if rec else None
        if status is None:
            return None  # unknown: id_provenance covers lookups
        if norm(status) not in {norm(v) for v in r.get("allowed", [])}:
            return Check(
                r,
                False,
                f"{args[arg]} has {r.get('field')}={status}; allowed: {_joined(', ', r.get('allowed'))}",
                f"do not use {args[arg]} here",
            )
        return Check(r, True, f"{arg} status allowed")

    def _check_membership(self, r, tool, args) -> Optional[Check]:
        """The record named by ``arg`` (e.g. the assignee) must be a member of the one named by ``group_arg``, as the
        data read shows: allowed when one record links both; denied when the data lists other members of that group
        and none is this one; not decided when the group's members were never read."""
        member, group = args.get(r.get("arg")), args.get(r.get("group_arg"))
        if member in (None, "") or group in (None, ""):
            return None
        m, g = str(member), str(group)

        def values(n: Dict[str, Any]) -> List[str]:
            out = []
            for v in n.values():
                out += (
                    [str(x) for x in v]
                    if isinstance(v, list)
                    else [str(v)]
                    if isinstance(v, (str, int))
                    else []
                )
            return out

        names_group = [n for n in self.facts.nodes if g in values(n)]
        if any(m in values(n) for n in names_group):
            return Check(r, True, f"{m} is a member of {g}")
        others = [
            n
            for n in names_group
            if any(
                ("user" in k.lower() or "member" in k.lower()) and v not in (None, "", m)
                for k, v in n.items()
            )
        ]
        if not others:
            return None
        return Check(
            r,
            False,
            f"the members of {g} that were read do not include {m}",
            f"choose a member of {g}, or add {m} to the group first if the request asks for it",
        )

    def _check_grounded_values(self, r, tool, args, demanded) -> Optional[Check]:
        props = (self.specs.get(tool, {}).get("inputSchema") or {}).get("properties") or {}
        required = set((self.specs.get(tool, {}).get("inputSchema") or {}).get("required") or [])
        bad = []
        for k, v in args.items():
            if k in required or k in demanded or isinstance(v, (bool, dict, list)) or v in (None, ""):
                continue
            prop = props.get(k, {})
            if not prop.get("enum") and (FREE_TEXT.search(k) or len(str(v).split()) > 3):
                continue  # free text is written by the agent
            if ID_VALUE.match(str(v)):
                continue  # IDs are checked by id_provenance
            if not self.facts.seen(v) and not (prop.get("enum") and self._stem_seen(v)):
                bad.append(f"{k}={v}")
        if bad:
            return Check(
                r,
                False,
                f"optional value(s) {', '.join(bad)} do not come from the request or from any "
                f"tool result (no assumed data)",
                "leave these arguments out, or use values stated in the request or returned by a tool",
            )
        return Check(r, True, "optional values grounded")

    def _stem_seen(self, value: Any) -> bool:
        """A tool option counts as stated when its word stem appears in the request or data ("retire" -> "retired")."""
        target = _stem(norm(value))
        return bool(target) and target in {_stem(w) for w in re.findall(r"[a-z]+", self.facts.text)}

    def _check_id_provenance(self, r, tool, args, demanded) -> Optional[Check]:
        bad = [
            f"{k}={v}"
            for k, v in args.items()
            if isinstance(v, str) and ID_VALUE.match(v) and not self.facts.seen(v)
        ]
        if bad:
            return Check(
                r,
                False,
                f"ID(s) {', '.join(bad)} were not returned by any tool and are not in the request",
                "look the record up first and use the ID it returns",
            )
        return Check(r, True, "IDs come from data")
