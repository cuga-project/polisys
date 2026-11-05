"""Handlers: code rules from tables, model slot filling for prose, and the new rule types at run time."""

import json
from pathlib import Path

from polisys.documents import load_document, split_policy
from polisys.handlers.prose import prose_rules
from polisys.handlers.tables import code_rules, table_rules
from polisys.model import StatementType
from polisys.ruleset import RuleSetHooks

ITSM = Path(__file__).resolve().parents[5] / "docs" / "examples" / "policy_samples"

SPECS = [
    {
        "name": "create_refund",
        "inputSchema": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "amount": {"type": "number"},
                "reference": {"type": "string"},
            },
            "required": ["invoice_id", "amount"],
        },
    },
    {
        "name": "schedule_inspection",
        "inputSchema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "start": {"type": "string"}},
            "required": ["summary"],
        },
    },
    {
        "name": "request_approval",
        "inputSchema": {"type": "object", "properties": {"invoice_id": {"type": "string"}}},
    },
]


class Fake:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def __call__(self, system, user):
        self.calls += 1
        return json.dumps(self.reply)


def _doc(text):
    doc = load_document(text)
    for st in doc.statements:
        st.types = [StatementType.THRESHOLD]
    return doc


def test_table_rules_match_code_rules_on_line_sections():
    text = (
        "## 2. Values\n\n| Field | Allowed Values |\n|---|---|\n| refund.reason | damaged, late |\n\n"
        "## 3. Priority\n\n| amount | reference | invoice_id |\n|---|---|---|\n| 1 | a | x |\n"
    )
    raw, _ = code_rules(split_policy(text), SPECS)
    rules, consumed = table_rules(load_document(text).statements, SPECS)
    assert sorted(r["type"] for r in raw) == sorted(r.type for r in rules)
    assert all(r.statement_ids for r in rules)


def test_record_condition_from_filter_statement():
    doc = _doc("## 4. Holds\n\n- Do not schedule inspections for units whose Notes contain an FSR- code.")
    model = Fake(
        {
            "statements": [
                {
                    "s": 1,
                    "class": "rule",
                    "rules": [
                        {
                            "type": "record_condition",
                            "tools": ["schedule_inspection"],
                            "field": "Notes",
                            "op": "contains",
                            "value": "FSR-",
                            "effect": "forbid",
                            "quote": "Do not schedule inspections for units whose Notes contain an FSR- code",
                        }
                    ],
                }
            ]
        }
    )
    rules, cov = prose_rules(doc.statements, SPECS, model)
    assert [r.type for r in rules] == ["record_condition"]
    assert (
        rules[0].rule["field"] == "Notes"
        and rules[0].rule["op"] == "contains"
        and rules[0].rule["value"] == "FSR-"
    )


def test_threshold_rule_slots_and_bad_rule_becomes_verify():
    doc = _doc("## 5. Refunds\n\n- Refunds above $500 need approval.\n- Refunds go to the original card.")
    model = Fake(
        {
            "statements": [
                {
                    "s": 1,
                    "class": "rule",
                    "rules": [
                        {
                            "type": "threshold",
                            "tool": "create_refund",
                            "arg": "amount",
                            "op": "gt",
                            "value": 500,
                            "then": "requires_approval",
                            "approval_tool": "request_approval",
                            "quote": "Refunds above $500 need approval",
                        }
                    ],
                },
                {
                    "s": 2,
                    "class": "rule",
                    "rules": [
                        {
                            "type": "allowed_values",
                            "tool": "create_refund",
                            "arg": "card",
                            "values": ["original"],
                            "quote": "Refunds go to the original card",
                        }
                    ],
                },
            ]
        }
    )
    rules, cov = prose_rules(doc.statements, SPECS, model)
    assert [r.type for r in rules] == ["threshold", "verify"]
    assert rules[1].rule["statement"] == "Refunds go to the original card." and rules[1].rule["tools"] == [
        "create_refund"
    ]
    assert [c.kind for c in cov] == ["rule", "verify"]


def test_format_rule_pattern_must_compile():
    doc = _doc("## 6. References\n\n- References are AP-<id>.")
    model = Fake(
        {
            "statements": [
                {
                    "s": 1,
                    "class": "rule",
                    "rules": [
                        {
                            "type": "format",
                            "tool": "create_refund",
                            "arg": "reference",
                            "pattern": "AP-[",
                            "quote": "References are AP-<id>",
                        }
                    ],
                }
            ]
        }
    )
    rules, _ = prose_rules(doc.statements, SPECS, model)
    assert [r.type for r in rules] == ["verify"]


def test_new_rule_types_enforced_by_ruleset():
    rules = [
        {
            "id": "t",
            "type": "threshold",
            "tool": "create_refund",
            "arg": "amount",
            "op": "gt",
            "value": 500,
            "then": "requires_approval",
            "approval_tool": "request_approval",
            "source": "§5",
        },
        {
            "id": "f",
            "type": "format",
            "tool": "create_refund",
            "arg": "reference",
            "pattern": r"AP-\d+",
            "source": "§6",
        },
        {
            "id": "c",
            "type": "record_condition",
            "tools": ["schedule_inspection"],
            "field": "Notes",
            "op": "contains",
            "value": "FSR-",
            "effect": "forbid",
            "source": "§4",
        },
    ]
    hooks = RuleSetHooks(rules, SPECS)
    hooks.after_tool_call(
        "get_units",
        {},
        {
            "rows": [
                {"cells": {"Unit ID": "FE-103", "Notes": ""}},
                {"cells": {"Unit ID": "FE-111", "Notes": "under FSR-2026-008 review"}},
            ]
        },
    )
    assert hooks.before_tool_call(
        "create_refund", {"invoice_id": "I1", "amount": 900}
    ).message  # needs approval
    hooks.after_tool_call("request_approval", {"invoice_id": "I1"}, {"ok": True})
    assert not hooks.before_tool_call(
        "create_refund", {"invoice_id": "I1", "amount": 900, "reference": "AP-7"}
    ).message
    assert hooks.before_tool_call(
        "create_refund", {"invoice_id": "I1", "amount": 10, "reference": "X-7"}
    ).message
    assert not hooks.before_tool_call("schedule_inspection", {"summary": "Inspection - FE-103"}).message
    assert (
        "FE-111" in hooks.before_tool_call("schedule_inspection", {"summary": "Inspection - FE-111"}).message
    )
