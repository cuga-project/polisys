"""Run time: verifier fallback with the records the call names, method choice routing, remind-only types, errors."""

import asyncio
import json

import pytest

from polisys.facts import FactStore
from polisys.ruleset import RuleSetHooks
from polisys.verifier import Verifier

SPECS = [
    {
        "name": "schedule_inspection",
        "inputSchema": {"type": "object", "properties": {"summary": {"type": "string"}}},
    },
    {"name": "send_report", "inputSchema": {"type": "object", "properties": {"body": {"type": "string"}}}},
]
UNITS = {
    "rows": [
        {"cells": {"Unit ID": f"FE-{100 + i}", "Notes": "x" * 300 if i != 3 else ""}} for i in range(1, 40)
    ]
}
HOLD = {
    "id": "v1",
    "type": "verify",
    "statement": "Do not schedule units whose Notes contain an FSR- code.",
    "tools": ["schedule_inspection"],
    "types": ["eligibility_filter"],
    "source": "§4",
}


class Model:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def __call__(self, system, user):
        self.prompts.append(user)
        if self.error:
            raise self.error
        return json.dumps(self.reply)


def test_fact_store_puts_named_records_first():
    store = FactStore()
    store.add("get_units", UNITS)
    text = store.for_call({"summary": "Inspection - FE-103"}, budget=2000)
    assert text.startswith("Records that mention values in this call") and '"Unit ID": "FE-103"' in text


def test_verify_rule_uses_records_named_in_call():
    model = Model({"violations": []})
    hooks = RuleSetHooks([HOLD], SPECS, verifier=Verifier(model))
    hooks.after_tool_call("get_units", {}, UNITS)
    assert hooks.before_tool_call("schedule_inspection", {"summary": "Inspection - FE-103"}).message is None
    assert '"Unit ID": "FE-103"' in model.prompts[0]


def test_verifier_error_allows_and_records():
    hooks = RuleSetHooks([HOLD], SPECS, verifier=Verifier(Model(error=TimeoutError("slow"))))
    res = hooks.before_tool_call("schedule_inspection", {"summary": "Inspection - FE-103"})
    assert res.message is None and any("verifier_error" in r["reason"] for r in hooks.records)


def test_remind_only_type_goes_to_note_not_deny():
    model = Model({"violations": [{"statement": 1, "reason": "FE-111 is on hold", "fix": "skip it"}]})
    hooks = RuleSetHooks(
        [HOLD],
        SPECS,
        verifier=Verifier(model),
        method_choice={"eligibility_filter": {"checker": "verifier", "may_deny": False}},
    )
    res = hooks.before_tool_call("schedule_inspection", {"summary": "Inspection - FE-111"})
    # a before_tool_call message blocks the call, so a reminder must not be one: it reaches the next model call
    assert res.message is None and hooks.records[-1]["outcome"] == "remind"
    brief = hooks.before_llm_call(None).message
    assert (
        "POLICY NOTE about your `schedule_inspection` call (it was run)" in brief
        and "FE-111 is on hold" in brief
    )
    assert "FE-111 is on hold" not in (hooks.before_llm_call(None).message or "")  # shown once


def test_reminder_does_not_block_the_hooked_tool_but_a_deny_does():
    from langchain_core.tools import StructuredTool

    from polisys.hooks import HookBlocked, HookRegistry, wrap_tools_with_hooks

    ran = []

    async def schedule_inspection(summary: str):
        ran.append(summary)
        return {"ok": True}

    tool = StructuredTool.from_function(
        coroutine=schedule_inspection, name="schedule_inspection", description="x"
    )
    remind = RuleSetHooks(
        [HOLD],
        SPECS,
        verifier=Verifier(Model({"violations": [{"statement": 1, "reason": "on hold"}]})),
        method_choice={"eligibility_filter": {"checker": "verifier", "may_deny": False}},
    )
    (hooked,) = wrap_tools_with_hooks([tool], HookRegistry([remind]))
    asyncio.run(hooked.coroutine(summary="Inspection - FE-111"))
    assert ran == ["Inspection - FE-111"]
    deny = RuleSetHooks(
        [HOLD],
        SPECS,
        verifier=Verifier(Model({"violations": [{"statement": 1, "reason": "on hold"}]})),
        verifier_may_deny=True,
    )
    (hooked,) = wrap_tools_with_hooks([tool], HookRegistry([deny]))
    with pytest.raises(HookBlocked):
        asyncio.run(hooked.coroutine(summary="Inspection - FE-112"))
    assert ran == ["Inspection - FE-111"]


def test_method_choice_routes_a_code_type_to_the_verifier():
    rule = {
        "id": "f1",
        "type": "format",
        "tool": "send_report",
        "arg": "body",
        "pattern": r"REP-\d+",
        "quote": "Reports start with REP-<number>",
        "source": "§6",
    }
    model = Model({"violations": []})
    hooks = RuleSetHooks(
        [rule],
        SPECS,
        verifier=Verifier(model),
        method_choice={"format": {"checker": "verifier", "code_may_deny": False}},
    )
    assert (
        hooks.before_tool_call("send_report", {"body": "hello"}).message is None
    )  # code would deny; verifier decides
    assert "Reports start with REP-<number>" in model.prompts[0]


def test_global_verify_rule_applies_to_every_write():
    rule = {
        "id": "g",
        "type": "verify",
        "statement": "Never send customer card numbers.",
        "tools": [],
        "source": "§1",
    }
    model = Model({"violations": [{"statement": 1, "reason": "card number in body", "fix": "remove it"}]})
    hooks = RuleSetHooks([rule], SPECS, verifier=Verifier(model), verifier_may_deny=True)
    assert "card number" in hooks.before_tool_call("send_report", {"body": "4111 1111"}).message
    default = RuleSetHooks([rule], SPECS, verifier=Verifier(model))  # model verdicts only remind by default
    assert default.before_tool_call("send_report", {"body": "4111 1111"}).message is None
    assert (
        default.records[-1]["outcome"] == "remind" and "card number" in default.before_llm_call(None).message
    )


def test_write_tools_include_camel_case_and_unknown_verbs():
    from polisys.ruleset import is_write_tool

    assert is_write_tool("recruitee_candidateCreate") and is_write_tool("xero_email_invoice")
    assert (
        is_write_tool("twitter_like_tweet")
        and not is_write_tool("email_list")
        and not is_write_tool("gmail_find_email")
    )


def test_record_condition_names_records_by_their_own_id_not_by_owner():
    rule = {
        "id": "r",
        "type": "record_condition",
        "tools": ["salesforce_task_create"],
        "field": "Notes",
        "op": "contains",
        "value": "secondment",
        "effect": "forbid",
        "source": "§1",
    }
    hooks = RuleSetHooks([rule], [{"name": "salesforce_task_create"}])
    hooks.after_tool_call(
        "salesforce_query",
        {},
        {
            "results": [
                {"Id": "cont_1", "OwnerId": "owner_1", "Notes": ""},
                {"Id": "cont_5", "OwnerId": "owner_1", "Notes": "temporary secondment to BrightPath"},
            ]
        },
    )
    assert (
        hooks.before_tool_call(
            "salesforce_task_create", {"owner_id": "owner_1", "contact_id": "cont_1"}
        ).message
        is None
    )
    assert hooks.before_tool_call(
        "salesforce_task_create", {"owner_id": "owner_1", "contact_id": "cont_5"}
    ).message


def test_allowed_value_accepts_an_id_with_an_allowed_name():
    rule = {
        "id": "a",
        "type": "allowed_values",
        "tool": "slack_send_channel_message",
        "arg": "channel",
        "values": ["#sales-alerts"],
        "source": "§2",
    }
    hooks = RuleSetHooks([rule], [{"name": "slack_send_channel_message"}])
    hooks.after_tool_call(
        "slack_find_channel", {}, {"channels": [{"id": "C_SALES_ALERTS", "name": "sales-alerts"}]}
    )
    assert hooks.before_tool_call("slack_send_channel_message", {"channel": "C_SALES_ALERTS"}).message is None
    assert hooks.before_tool_call("slack_send_channel_message", {"channel": "C_RANDOM"}).message


def test_code_rule_keeps_deny_when_only_the_verifier_side_is_remind_only():
    rule = {
        "id": "t",
        "type": "threshold",
        "tool": "send_report",
        "arg": "body",
        "op": "gt",
        "value": 5,
        "then": "forbid",
        "source": "§1",
    }
    hooks = RuleSetHooks(
        [rule],
        SPECS,
        method_choice={"threshold": {"checker": "verifier", "may_deny": False, "code_may_deny": True}},
    )
    assert hooks.before_tool_call("send_report", {"body": 9}).message.startswith("POLICY:")
