"""The rules found so far (part of ``OnlinePolicyIntake``)."""

from __future__ import annotations

import json
import re
from concurrent.futures import Future, wait
from typing import Any, Dict, List, Optional, Tuple

from polisys.compile.handlers.prose import prose_rules
from polisys.compile.handlers.tables import table_rules
from polisys.package.model import Document, Section, Statement, StatementType, Trust
from polisys.sources.intake.prompts import (
    EXTRACT_SYSTEM, EXTRACT_USER, EXTRACT_USER_SUPERSEDES, MAX_BATCH_CHARS, MAX_KNOWN_RULES, MAX_RECORD_CHARS,
    _TYPE_VALUES, _json, _norm_statements, _norm_text, _row_statements,
)


class _FoundRules:
    """The statements in force (``_statements``), a rule that replaces another, the extraction of rules from records by the
    model (``_extract``), and compiling them into the enforcer (``_take``). Extraction runs in a background thread;
    ``_drain`` waits for it."""

    def _statements(self, trusted_only: bool = True) -> List[Dict[str, Any]]:
        """Statements for the verifier and the brief; source-of-truth statements first (they decide which data wins)."""
        items = []
        for p in self.policies:
            if trusted_only and p.get("trust") != "trusted":
                continue
            types = p.get("types") or [[] for _ in p.get("statements") or []]
            for s, t in zip(p.get("statements") or [], types):
                if trusted_only and s in self.compiled:
                    continue  # enforced by a code rule
                if s in self.superseded:
                    continue  # replaced by a later rule
                items.append((0 if "source_of_truth" in (t or []) else 1, s, p))
        items.sort(key=lambda x: x[0])
        return [
            {
                "n": i,
                "text": s,
                "source": p.get("title") or p.get("tool"),
                "trust": p.get("trust"),
                "types": (p.get("types") or [[]] * len(p.get("statements") or []))[
                    (p.get("statements") or []).index(s)
                ]
                if s in (p.get("statements") or [])
                else [],
            }
            for i, (_, s, p) in enumerate(items, 1)
        ]

    @staticmethod
    def _fmt(statements: List[Dict[str, Any]]) -> str:
        return "\n".join(f"{s['n']}. [{s['source']}] {s['text']}" for s in statements)

    def _current_rules(self) -> List[Dict[str, Any]]:
        """Trusted statements not replaced so far, with their source (shown to the extraction call as K1, K2, ...)."""
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for p in self.policies:
            if p.get("trust") != "trusted":
                continue
            for s in p.get("statements") or []:
                if s not in self.superseded and s not in seen:
                    seen.add(s)
                    out.append({"text": s, "source": p.get("title") or p.get("tool")})
        return out

    def _mark_replaced(self, text: str, by: str) -> None:
        if not text or text in self.superseded:
            return
        self.superseded[text] = by
        dropped = self.enforcer.remove_rules_for(text) if self.enforcer is not None else 0
        self.records.append(
            {
                "hook": "after_tool_call",
                "outcome": "superseded",
                "statement": text,
                "by": by,
                "code_rules_dropped": dropped,
            }
        )

    def _apply_replacements(self, found: List[Dict[str, Any]], known: List[Dict[str, Any]]) -> None:
        """Mark replaced statements from one extraction answer, rule by rule. Only trusted records replace rules. An
        entry names the replaced statements by their text; a bare record or K number replaces only when the replaced
        side has one statement, so one changed rule never takes the other rules of its record with it."""

        def num(x: Any) -> Optional[int]:
            m = re.search(r"\d+", str(x))
            return int(m.group(0)) if m else None

        def pick(entry: Any, key: str, pool: List[str]) -> Tuple[Optional[int], List[str]]:
            """The number in the entry and the statements of ``pool`` it names (exact text, spaces normalized)."""
            if isinstance(entry, dict):
                wanted = {_norm_text(str(t)) for t in entry.get("statements") or []}
                return num(entry.get(key)), [s for s in pool if _norm_text(s) in wanted]
            return num(entry), (list(pool) if len(pool) == 1 else [])

        for p in found:
            if p.get("trust") != "trusted":
                continue
            own = list(dict.fromkeys(p.get("statements") or []))
            by = str(p.get("title") or p.get("tool"))
            for x in p.get("replaces_known") or []:
                n = num(x)
                if n and 1 <= n <= len(known) and known[n - 1]["text"] not in own:
                    self._mark_replaced(known[n - 1]["text"], by)
            for x in p.get("replaces_records") or []:
                n = num(x.get("record")) if isinstance(x, dict) else num(x)
                for q in found:
                    if q is p or n is None or num(q.get("record")) != n:
                        continue
                    _, hit = pick(x, "record", [s for s in q.get("statements") or [] if s not in own])
                    if not hit:
                        self._note_unclear(by, f"record {n}")
                    for s in hit:
                        self._mark_replaced(s, by)
            for x in p.get("replaced_by_known") or []:
                n, hit = pick(x, "known", own)
                if not (n and 1 <= n <= len(known)):
                    continue
                k = known[n - 1]
                hit = [s for s in hit if s != k["text"]]
                if not hit:
                    self._note_unclear(by, f"K{n}")
                for s in hit:
                    self._mark_replaced(s, f"{k['source']}: {k['text']}"[:240])
            if p.get("archived") is True:
                for s in own:
                    self._mark_replaced(s, "its own record says it is archived or no longer in effect")

    def _note_unclear(self, by: str, target: str) -> None:
        """A replacement that names no statement of a record with several rules: nothing is marked (the unknown path)."""
        self.records.append(
            {"hook": "after_tool_call", "outcome": "replacement_unclear", "by": by, "target": target}
        )

    def _drain(self, timeout: Optional[float]) -> None:
        if self.pending:
            wait(self.pending, timeout=timeout)
            self.pending = [f for f in self.pending if not f.done()]

    def _extract(self, tool: str, records: List[Any]) -> Dict[str, Any]:
        texts = [json.dumps(r, default=str)[:MAX_RECORD_CHARS] for r in records]
        body = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))[:MAX_BATCH_CHARS]
        types = ", ".join(t.value for t in StatementType)
        known: List[Dict[str, Any]] = []
        if self.track_supersedes:
            known = self._current_rules()[:MAX_KNOWN_RULES]
            listing = (
                "\n".join(f"K{i}. [{k['source']}] {k['text']}" for i, k in enumerate(known, 1))
                or "(none yet)"
            )
            prompt = EXTRACT_USER_SUPERSEDES.format(
                request=self.request, tool=tool, records=body, types=types, known=listing
            )
        else:
            prompt = EXTRACT_USER.format(request=self.request, tool=tool, records=body, types=types)
        data = _json(self.model(EXTRACT_SYSTEM, prompt))
        out = {
            "tool": tool,
            "n_records": len(records),
            "policies": data.get("policies"),
            "records": records,
            "known": known,
        }
        if self.enforcer is not None:
            out["compiled"] = self._compile(tool, records, data.get("policies") or [])
        return out

    def _compile(self, tool: str, records: List[Any], policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trusted policies → code rules: rule tables (rows of a list of records) by code, prose by the model.
        Returns the compiled rule dicts; their statements leave the verifier's list."""
        rules: List[Dict[str, Any]] = []
        statements: List[Statement] = []
        for p in policies:
            if not isinstance(p, dict) or p.get("trust") != "trusted":
                continue
            for i, s in enumerate(_norm_statements(p.get("statements")), 1):
                types = [StatementType(t) for t in s["types"] if t in _TYPE_VALUES][:2]
                statements.append(
                    Statement(
                        id=f"{tool}:{p.get('record')}:{i}",
                        text=s["text"],
                        types=types,
                        source={
                            "document": f"{tool}#{p.get('record')}",
                            "section": str(p.get("record")),
                            "tool": tool,
                        },
                        quote=s["text"],
                    )
                )
        rows = [r for r in records if isinstance(r, list)] + [r for r in records if isinstance(r, dict)]
        table_sts = _row_statements(tool, rows)
        if table_sts:
            t_rules, _ = table_rules(table_sts, self.tool_specs)
            rules += [r.rule for r in t_rules]
        if statements:
            p_rules, _ = prose_rules(statements, self.tool_specs, self.model, skip_types=())
            for r in p_rules:
                if r.type not in ("verify", "guidance"):
                    rules.append({**r.rule, "statement_text": r.quote})
        return rules

    def _done(self, fut: Future) -> None:
        try:
            data = fut.result()
        except Exception as exc:  # intake failure must not stop the agent
            self.records.append({"hook": "after_tool_call", "outcome": "intake_error", "reason": str(exc)})
            return
        self._take(data)

    def _take(self, data: Dict[str, Any]) -> None:
        found = [p for p in data.get("policies") or [] if isinstance(p, dict) and p.get("statements")]
        for p in found:
            p["tool"] = data["tool"]
            norm = _norm_statements(p.get("statements"))
            p["statements"], p["types"] = [s["text"] for s in norm], [s["types"] for s in norm]
            self.policies.append(p)
            trust = Trust.TRUSTED if p.get("trust") == "trusted" else Trust.UNTRUSTED
            self.overlay.documents.append(
                Document(
                    id=f"{data['tool']}#{p.get('record')}#{len(self.overlay.documents) + 1}",
                    kind=data["tool"],
                    title=str(p.get("title") or ""),
                    trust=trust,
                    sections=[
                        Section(
                            id="1",
                            statements=[
                                Statement(
                                    id=f"{data['tool']}#{p.get('record')}:{i}",
                                    text=s["text"],
                                    types=[StatementType(t) for t in s["types"] if t in _TYPE_VALUES][:2],
                                )
                                for i, s in enumerate(norm, 1)
                            ],
                        )
                    ],
                    source={"tool": data["tool"], "record": p.get("record")},
                )
            )
            self.records.append(
                {
                    "hook": "after_tool_call",
                    "outcome": "policy_found",
                    "tool": data["tool"],
                    "intake_ms": data.get("ms"),
                    "record": p.get("record"),
                    "title": p.get("title"),
                    "trust": p.get("trust"),
                    "trust_reason": p.get("trust_reason"),
                    "statements": list(p["statements"]),
                    "supersedes": p.get("supersedes"),
                }
            )
        compiled = data.get("compiled") or []
        if compiled and data["tool"] == "user_request":  # the user's own words: remind, never deny
            for r in compiled:
                r["may_deny"] = False
        if compiled and self.enforcer is not None:
            added = self.enforcer.add_rules(compiled)
            self.compiled |= {r.get("statement_text") for r in compiled if r.get("statement_text")}
            self.records.append(
                {
                    "hook": "after_tool_call",
                    "outcome": "compiled",
                    "tool": data["tool"],
                    "rules": added,
                    "types": sorted({r["type"] for r in compiled}),
                }
            )
        if len(found) < data["n_records"]:
            self.records.append(
                {
                    "hook": "after_tool_call",
                    "outcome": "not_policy",
                    "tool": data["tool"],
                    "records": data["n_records"] - len(found),
                }
            )
        if self.track_supersedes and found:
            self._apply_replacements(found, data.get("known") or [])
