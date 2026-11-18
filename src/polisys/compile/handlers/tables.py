"""Rules that code reads from tables and lists, with no model call.

- "Table.Field | Allowed Values" tables → ``allowed_values`` rules (unless the tool schema already says the same);
- input → output tables (two or more input columns that are arguments of a write tool) → ``derived_value`` rules;
- transition lists ("* `new` → `in_progress` / `on_hold`") whose states are all listed values → ``transition`` rules.

``code_rules`` works on line sections (``documents.split_policy``). ``table_rules`` does the same for statements of
the policy model: it rebuilds the table lines from the statements and maps every rule back to its statement ids.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

from ..documents import LineSection as Section
from ..documents import clean as _clean
from ..model import Rule, Statement
from ..ruleset import is_write_tool, norm

ARROW = re.compile(r"[`'\"]?([\w-]+)[`'\"]?\s*(?:→|->)\s*(.+)$")


def _tables(sec: Section) -> List[Tuple[List[str], List[List[str]], List[int]]]:
    """Markdown tables in a section: (header, rows, line numbers)."""
    out, cur, nos = [], [], []
    for no, line in sec.lines + [(0, "")]:
        if line.strip().startswith("|"):
            cells = [_clean(c) for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                cur.append(cells)
            nos.append(no)
        elif cur:
            out.append((cur[0], cur[1:], nos))
            cur, nos = [], []
    return out


def _values(cell: str) -> List[str]:
    return [v.strip(" '\"") for v in re.split(r",|/", cell) if v.strip(" '\"")]


def _props(spec: Dict[str, Any]) -> Dict[str, Any]:
    return (spec.get("inputSchema") or {}).get("properties") or {}


def _required(spec: Dict[str, Any]) -> List[str]:
    return list((spec.get("inputSchema") or {}).get("required") or [])


def _rid(*parts: Any) -> str:
    return "r_" + hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:8]


ALIASES = {"state": "status", "status": "state"}  # common synonyms between policy text and tool arguments


def _tools_for_field(table: str, fld: str, specs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """(write tool, argument) pairs for a policy field: the tool name mentions the table and has the argument
    (incident.status → update_incident.status); a common synonym is used when the exact name is missing."""
    stem = table.split("_")[0].lower().rstrip("s")
    out = []
    for t in specs:
        if not is_write_tool(t["name"]) or stem not in t["name"].lower():
            continue
        arg = fld if fld in _props(t) else ALIASES.get(fld) if ALIASES.get(fld) in _props(t) else None
        if arg:
            out.append((t, arg))
    return out


def code_rules(
    sections: List[Section], specs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """Rules read by code from tables and transition lists. Returns rules and {line: how it was consumed}."""
    rules: List[Dict[str, Any]] = []
    used: Dict[int, str] = {}
    enums: Dict[
        str, List[str]
    ] = {}  # "incident.status" -> values; read first, transition lists refer to them
    for sec in sections:
        for header, rows, _ in _tables(sec):
            if len(header) == 2 and "allowed" in norm(header[1]):
                enums.update({row[0]: _values(row[1]) for row in rows if "." in row[0]})
    for sec in sections:
        for header, rows, nos in _tables(sec):
            h = [norm(c) for c in header]
            if len(h) == 2 and "." in "".join(r[0] for r in rows) and "allowed" in h[1]:
                for row in rows:
                    if "." not in row[0]:
                        continue
                    table, fld = row[0].split(".", 1)
                    enums[row[0]] = _values(row[1])
                    for t, arg in _tools_for_field(table, fld, specs):
                        schema_enum = [norm(v) for v in _props(t)[arg].get("enum") or []]
                        if schema_enum and set(schema_enum) == {norm(v) for v in enums[row[0]]}:
                            continue  # the tool schema already says the same thing
                        rules.append(
                            {
                                "id": _rid("enum", t["name"], arg, enums[row[0]]),
                                "type": "allowed_values",
                                "tool": t["name"],
                                "arg": arg,
                                "values": enums[row[0]],
                                "source": f"§{sec.sid} L{nos[0]}",
                                "quote": f"{row[0]} | {row[1]}",
                                "by": "code",
                            }
                        )
                for n in nos:
                    used[n] = "enumeration table"
            elif len(h) >= 3 and rows and all(len(r) == len(h) for r in rows):
                inputs, out = [c.replace(" ", "_") for c in h[:-1]], h[-1].replace(" ", "_")
                cands = [
                    t
                    for t in specs
                    if is_write_tool(t["name"]) and out in _props(t) and all(i in _props(t) for i in inputs)
                ]
                if not cands:
                    continue
                table = {"|".join(norm(c) for c in r[:-1]): r[-1] for r in rows}
                for t in cands:
                    rules.append(
                        {
                            "id": _rid("matrix", t["name"], out, table),
                            "type": "derived_value",
                            "tool": t["name"],
                            "arg": out,
                            "inputs": inputs,
                            "table": table,
                            "source": f"§{sec.sid} L{nos[0]}",
                            "quote": " | ".join(header),
                            "by": "code",
                        }
                    )
                for n in nos:
                    used[n] = "matrix table"
        # transition lists: "* `new` → `in_progress` / `on_hold`", kept only when every state is a listed value
        for key, vals in enums.items():
            known = {v for v in vals}
            edges: Dict[str, List[str]] = {}
            enos = []
            for no, line in sec.lines:
                m = ARROW.search(_clean(line).lstrip("* "))
                if not m or line.strip().startswith("|"):
                    continue
                targets = [
                    t for t in (re.sub(r"\(.*", "", v).strip(" `'\"") for v in m.group(2).split("/")) if t
                ]
                if m.group(1) in known and targets and set(targets) <= known:
                    edges.setdefault(m.group(1), []).extend(targets)
                    enos.append(no)
            if len(edges) < 2:
                continue
            table, fld = key.split(".", 1)
            for t, arg in _tools_for_field(table, fld, specs):
                if not t["name"].startswith("update"):
                    continue
                id_arg = next((a for a in _required(t) if a.endswith("_id")), None)
                rules.append(
                    {
                        "id": _rid("trans", t["name"], arg, edges),
                        "type": "transition",
                        "tool": t["name"],
                        "arg": arg,
                        "id_arg": id_arg,
                        "edges": edges,
                        "source": f"§{sec.sid} L{enos[0]}",
                        "quote": "Valid transitions",
                        "by": "code",
                    }
                )
            for n in enos:
                used[n] = "transition list"
    unique = {r["id"]: r for r in rules}  # the same table may appear in two sections
    return list(unique.values()), used


def tool_catalog(specs: List[Dict[str, Any]]) -> str:
    lines = []
    for t in sorted(specs, key=lambda s: s["name"]):
        req = set(_required(t))
        args = []
        for a, p in _props(t).items():
            enum = p.get("enum")
            args.append(f"{a}{'*' if a in req else ''}" + (f"[{'|'.join(map(str, enum))}]" if enum else ""))
        lines.append(f"{t['name']}({', '.join(args)})")
    return "\n".join(lines)


def _line_sections(statements: List[Statement]) -> Tuple[List[Section], Dict[int, str]]:
    """Statements → line sections that ``code_rules`` can read; also {line number: statement id}."""
    secs: Dict[str, Section] = {}
    line_of: Dict[int, str] = {}
    last_header: Dict[str, Any] = {}
    no = 0
    for st in statements:
        sid = str(st.source.get("section", "x"))
        sec = secs.setdefault(sid, Section(sid, sid))
        header = st.source.get("table_header")
        if header:
            if last_header.get(sid) != header:
                no += 1
                sec.lines.append((no, "| " + " | ".join(h or "" for h in header) + " |"))
                no += 1
                sec.lines.append((no, "|" + "---|" * len(header)))
                last_header[sid] = header
            no += 1
            sec.lines.append((no, "| " + " | ".join(c.strip() for c in st.text.split("|")) + " |"))
        else:
            last_header[sid] = None
            no += 1
            sec.lines.append((no, "* " + st.text))
        line_of[no] = st.id
    return list(secs.values()), line_of


def _rows_of(rule: Dict[str, Any], statements: List[Statement]) -> List[str]:
    """The statements a code rule came from: the enum row, every row of the matrix (in every section where the same
    table appears), or the transition lines."""
    t = rule.get("type")
    if t == "allowed_values":
        q = norm(_clean(rule.get("quote", "")))
        return [st.id for st in statements if norm(_clean(st.text)) == q]
    if t == "derived_value":
        header = [norm(h) for h in rule.get("quote", "").split("|")]
        keys = {k for k in (rule.get("table") or {})}
        out = []
        for st in statements:
            h = [norm(x) for x in (st.source.get("table_header") or []) if x is not None]
            cells = [norm(c) for c in st.text.split("|")]
            if h == header and "|".join(cells[:-1]) in keys:
                out.append(st.id)
        return out
    if t == "transition":
        starts = {norm(k) for k in (rule.get("edges") or {})}
        out = []
        for st in statements:
            m = ARROW.search(_clean(st.text).lstrip("* "))
            if m and not st.source.get("table_header") and norm(m.group(1)) in starts:
                out.append(st.id)
        return out
    return []


def table_rules(statements: List[Statement], specs: List[Dict[str, Any]]) -> Tuple[List[Rule], List[str]]:
    """Code-read rules for the statements, as ``Rule`` objects. Returns (rules, ids of the statements consumed)."""
    sections, line_of = _line_sections(statements)
    raw, used = code_rules(sections, specs)
    by_id = {st.id: st for st in statements}
    consumed = sorted({line_of[n] for n in used if n in line_of})
    rules = []
    for r in raw:
        ids = _rows_of(r, statements)
        if not ids:  # fall back to the rule's own section
            sec = r.get("source", "").split()[0].lstrip("§") if r.get("source") else ""
            ids = [i for i in consumed if str(by_id[i].source.get("section")) == sec] or consumed
        first = by_id[ids[0]] if ids else None
        rules.append(
            Rule(
                id=r["id"],
                type=r["type"],
                rule=r,
                statement_ids=ids,
                checker="code",
                source=dict(first.source) if first else {},
                quote=r.get("quote", ""),
            )
        )
    return rules, consumed
