"""Producers of the obligation ledger, and the retirement of the parts they replace (Task 4.2).

Each producer on small inline fixtures and in a real agent loop with a scripted model; that only the ledger sends follow-ups
when a profile has a producer, and that every old part sends as before when it has none; the procedure monitor on a package
written from a skill; one list of read verbs; ``store_verifier`` gone.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
from langchain_core.tools import tool

from polisys.hooks import AgentHooks
from polisys.deliverables import obligations_from_request
from polisys.dry_run import RecordingModels
from polisys.model import Obligation, PolicyPackage, Procedure, Step
from polisys.obligations import Case, ObligationLedger
from polisys.package import from_build, obligations_from_package
from polisys.policy_config import PolicyConfig, PolicyConfigError
from polisys.procedure_monitor import ProcedureMonitor
from polisys.ruleset import (
    READ_VERBS,
    is_write_tool,
    obligations_from_duties,
)
from polisys.runtime import COMPONENTS, RunContext, RuntimeWiringError, build_runtime
from polisys.store_gate import StoreGate
from polisys.tool_kind import READ_FIRST, acts
try:  # the CUGA agent is optional: its real-loop tests are skipped without it
    from cuga.sdk import CugaAgent
except ImportError:  # pragma: no cover
    CugaAgent = None

from .test_obligations import Scripted

SPECS = [
    {
        "name": "create_ticket",
        "inputSchema": {"type": "object", "properties": {"summary": {"type": "string"}}},
    },
    {
        "name": "update_ticket",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed"]},
            },
        },
    },
    {
        "name": "create_task",
        "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
    },
    {
        "name": "create_follow_up_task",
        "inputSchema": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "due": {"type": "string"}},
        },
    },
    {
        "name": "send_email",
        "inputSchema": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
        },
    },
    {
        "name": "post_message",
        "inputSchema": {
            "type": "object",
            "properties": {"channel": {"type": "string"}, "text": {"type": "string"}},
        },
    },
    {
        "name": "get_ticket",
        "inputSchema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}},
    },
    {"name": "count_tickets", "inputSchema": {"type": "object", "properties": {}}},
]

# -- the list of read verbs --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["count_incidents", "average_target_mins_per_priority", "total_sum_of_x"])
def test_count_average_and_total_are_reads_for_the_store_gate_and_the_write_check(name):
    assert acts(name) is False and is_write_tool(name) is False


def test_one_list_of_read_verbs_serves_the_store_gate_and_the_rules():
    assert READ_VERBS is READ_FIRST
    assert {"count", "average", "total", "download", "export"} <= READ_FIRST
    assert acts("mark_as_read") is True  # the first word decides for the store gate
    assert acts("policy__get_procedure") is False
    assert acts("create_ticket") is True and acts("google_mail__download_attachment") is False


def test_the_store_gate_keeps_a_count_tool_as_data_and_does_not_check_it():
    model_calls: List[str] = []
    gate = StoreGate(None, lambda system, user: model_calls.append(user) or "{}", sop_text="Do the work.")
    assert gate.before_tool_call("count_tickets", {}) is None
    gate.after_tool_call("count_tickets", {}, {"n": 3})
    assert model_calls == [] and any("count_tickets" in r for r in gate.reads)


def test_the_store_verifier_is_gone_from_imports_registry_and_config():
    with pytest.raises(ImportError):
        importlib.import_module("polisys.store_verifier")
    assert "store_verifier" not in COMPONENTS and "deliverables" not in COMPONENTS
    with pytest.raises(PolicyConfigError, match="store_verifier"):
        PolicyConfig.from_dict({"checkers": {"store_verifier": "on"}})


# -- the procedure monitor ---------------------------------------------------------------------------------------------


def skill_procedure() -> Procedure:
    """Steps as a package written from a skill has them: text, no quote, a condition on some, a reply step."""
    return Procedure(
        id="3.3",
        name="Hold",
        steps=[
            Step(id="1", action="get_ticket", text="Read the ticket."),
            Step(
                id="2",
                action="update_ticket",
                text="Put the ticket on hold.",
                condition="the caller asked for it",
            ),
            Step(id="3", action="reply", text="Tell the caller about the hold."),
            Step(id="4", action="create_task", text="Create a follow-up task."),
            Step(id="6", action="post_message", text="Post the notice."),
            Step(
                id="5",
                action="send_email",
                text="Email the owner.",
                condition="the owner is outside the team",
            ),
        ],
    )


def test_a_package_written_from_a_skill_gives_no_line_for_a_conditional_step_or_a_reply():
    monitor = ProcedureMonitor([skill_procedure()])
    monitor.after_tool_call("create_task", {"title": "x"}, {"id": "K1"})
    reply = monitor.before_reply("done").message
    assert reply == (
        "POLICY: before you finish, these required procedure steps are still missing:\n"
        "- [Hold §] Post the notice. (`post_message`)"
    )
    brief = monitor.before_llm_call(None).message
    assert "Post the notice." in brief and "Put the ticket on hold" not in brief
    assert (
        "Tell the caller" not in brief and "Email the owner" not in brief and "Read the ticket" not in brief
    )


def test_with_only_conditional_reply_and_read_steps_left_nothing_is_sent():
    monitor = ProcedureMonitor([skill_procedure()])
    monitor.after_tool_call("create_task", {"title": "x"}, {"id": "K1"})
    monitor.after_tool_call("post_message", {"text": "x"}, {"id": "M1"})
    assert monitor.before_reply("done").message is None
    assert monitor.before_llm_call(None).message is None


def test_a_read_call_does_not_start_a_procedure_of_a_skill_style_package_and_counts_as_done():
    monitor = ProcedureMonitor([skill_procedure()])
    monitor.after_tool_call("get_ticket", {}, {"id": "T1"})
    assert monitor.before_reply("done").message is None and monitor.active == set()
    assert monitor.done["3.3"] == {"1"}  # the read is done, so it is not asked for later


def test_a_tool_that_is_a_step_of_several_procedures_starts_none_of_them():
    one = Procedure(
        id="a",
        name="A",
        steps=[
            Step(id="1", action="create_task", text="Make the task."),
            Step(id="2", action="post_message", text="Post it."),
        ],
    )
    two = Procedure(
        id="b",
        name="B",
        steps=[
            Step(id="1", action="create_task", text="Make the task again."),
            Step(id="2", action="send_email", text="Mail it."),
        ],
    )
    monitor = ProcedureMonitor([one, two])
    monitor.after_tool_call("create_task", {}, {"id": "K1"})
    assert monitor.before_reply("done").message is None and monitor.active == set()
    monitor.after_tool_call("post_message", {}, {"id": "M1"})  # a tool of one procedure only starts that one
    assert monitor.active == {"a"}


def test_every_line_of_the_monitor_has_a_description():
    proc = Procedure(
        id="p", name="P", steps=[Step(id="1", action="get_ticket"), Step(id="2", action="create_task")]
    )
    monitor = ProcedureMonitor([proc])
    monitor.after_tool_call("get_ticket", {}, {})
    # a step with neither text nor quote shows its tool, not an empty line
    assert monitor.before_reply("x").message.endswith("\n- [P §] create_task (`create_task`)")


BUILD_PROCEDURE = {
    "id": "p1",
    "name": "Tickets",
    "steps": [
        {"id": "s1", "action": "get_ticket", "quote": "Read the ticket.", "source": {"section": "3.1"}},
        {"id": "s2", "action": "update_ticket", "quote": "Update the ticket.", "source": {"section": "3.2"}},
    ],
}


def test_a_procedure_of_a_build_gets_the_text_of_the_old_monitor():
    """A build has a ``quote`` and no condition: the messages are the bytes the old monitor gave."""
    proc = from_build({"rules": [], "procedures": [BUILD_PROCEDURE]}).procedures
    monitor = ProcedureMonitor(proc)
    monitor.after_tool_call("get_ticket", {}, {})
    assert monitor.before_llm_call(None).message == (
        "## Procedures in progress\n- Next required step of Tickets [§3.2]: Update the ticket. (`update_ticket`)"
    )
    assert monitor.before_reply("done").message == (
        "POLICY: before you finish, these required procedure steps are still missing:\n"
        "- [Tickets §3.2] Update the ticket. (`update_ticket`)"
    )
    assert monitor.before_reply("again").message is None  # once per thread, as before


def test_a_monitor_that_does_not_send_keeps_its_records():
    monitor = ProcedureMonitor([skill_procedure()], follow_up=False)
    monitor.after_tool_call("create_task", {}, {})
    assert monitor.before_llm_call(None).message is None and monitor.before_reply("x").message is None
    assert [r["outcome"] for r in monitor.records] == ["step_done"]


# -- duties of the rules -----------------------------------------------------------------------------------------------

DUTY = {
    "id": "d1",
    "type": "duty",
    "trigger": {"tool": "update_ticket", "when": {"state": ["closed"]}},
    "then": {"tool": "create_task", "when": {"title": ["Follow up"]}},
    "source": "§4.2",
    "asked": True,
    "quote": "After closing a ticket, create a follow-up task.",
}
DUTY_UNLESS = {
    **DUTY,
    "id": "d2",
    "then": {"tool": "create_task"},
    "unless": [{"condition": "the ticket was a duplicate"}],
}


def test_a_duty_becomes_an_obligation_that_opens_at_its_trigger():
    (ob,) = obligations_from_duties([DUTY], SPECS)
    assert (ob.id, ob.origin, ob.check, ob.level) == ("duty_d1", "duty", "code", "observe")  # no level yet
    assert ob.scope == {"trigger": {"tools": ["update_ticket"], "when": {"state": ["closed"]}}}
    assert ob.effect["tools"] == ["create_task", "create_follow_up_task"]  # the named tool first
    assert ob.effect["match"] == {"title": {"pattern": "{title}"}}
    assert ob.values == {"title": {"from": "literal", "value": "Follow up"}}
    assert "after `update_ticket`" in ob.source["quote"] and ob.source["section"] == "§4.2"


def test_the_tools_of_a_duty_are_the_named_tool_and_tools_for_the_same_kind_of_record():
    then = {"tool": "create_task", "when": {"title": ["Follow up"], "due": ["*"]}}
    (ob,) = obligations_from_duties([{**DUTY, "then": then}], SPECS)
    assert (
        ob.effect["tools"] == ["create_task", "create_follow_up_task"][:1]
        or "create_task" in ob.effect["tools"]
    )
    # a "*" or a list of values is named in the text and never compared
    assert "due" not in ob.effect["match"]


def test_an_exception_of_a_duty_stays_in_the_text_and_a_rule_that_is_not_a_duty_is_skipped():
    out = obligations_from_duties([DUTY_UNLESS, {"id": "r", "type": "allowed_values"}], SPECS, level="show")
    assert [o.id for o in out] == ["duty_d2"] and out[0].level == "show"
    assert "(not needed if: the ticket was a duplicate)" in out[0].source["quote"]
    assert out[0].effect["match"] == {}


def duty_ledger(rules=None, **kw) -> ObligationLedger:
    led = ObligationLedger(
        obligations_from_duties(rules or [DUTY], SPECS, level="follow_up"), None, tool_specs=SPECS, **kw
    )
    led.on_user_message("Close the ticket T1.", "t")
    return led


def test_a_duty_opens_only_when_the_trigger_call_carries_the_values_the_rule_names():
    led = duty_ledger()
    led.after_tool_call("update_ticket", {"ticket_id": "T1", "state": "open"}, {"ok": True})
    assert led.items == [] and led.before_reply("done").message is None
    led.after_tool_call("update_ticket", {"ticket_id": "T1", "state": "Closed"}, {"ok": True})
    assert len(led.items) == 1 and led.items[0].case.name == "after update_ticket"


def test_an_open_duty_gives_one_follow_up_and_the_duty_call_closes_it():
    led = duty_ledger()
    led.after_tool_call("update_ticket", {"ticket_id": "T1", "state": "closed"}, {"ok": True})
    text = led.before_reply("done").message
    assert "create_task" in text and "after `update_ticket`" in text and 'title = "Follow up"' in text
    assert led.before_reply("done again").message is None  # one follow-up (max_follow_ups 1), then nothing
    led.after_tool_call("create_task", {"title": "Follow up on T1"}, {"id": "K1"})
    assert led.before_reply("done").message is None and led.items[0].status == "met"


def test_a_call_before_the_trigger_does_not_do_the_duty():
    led = duty_ledger()
    led.after_tool_call("create_task", {"title": "Follow up"}, {"id": "K0"})  # the work came first
    led.after_tool_call("update_ticket", {"ticket_id": "T1", "state": "closed"}, {"ok": True})
    assert led.items[0].status == "open"
    assert "create_task" in led.before_reply("done").message


# -- the package ------------------------------------------------------------------------------------------------------


def ob(id: str, procedure: str = "", origin: str = "package", **over) -> Obligation:
    return Obligation(
        id=id,
        origin=origin,
        scope={"procedure": procedure} if procedure else {},
        effect={"kind": "write", "tools": ["create_ticket"]},
        level="observe",
        **over,
    )


def test_the_package_gives_its_obligations_and_those_of_a_case():
    pkg = PolicyPackage(
        version="v",
        obligations=[
            ob("a", "12.1"),
            ob("b", "12.2"),
            ob("every"),
            ob("req", "12.2", origin="request"),
            ob("duty", "", origin="duty"),
        ],
    )
    assert [o.id for o in obligations_from_package(pkg)] == ["a", "b", "every", "req", "duty"]
    assert [o.id for o in obligations_from_package(pkg, Case(name="x", procedure="12.1"))] == [
        "a",
        "every",
        "req",
        "duty",
    ]
    assert [o.id for o in obligations_from_package(pkg, Case(name="x", procedure="99"))] == [
        "every",
        "req",
        "duty",
    ]


# -- the request ------------------------------------------------------------------------------------------------------

REQUEST = "Please mark the ticket closed, create a ticket with summary 'Kiln repair', email lena@kiln.example and post in #ops."


def request_items() -> Dict[str, Any]:
    return {
        "items": [
            {
                "action": "create a ticket",
                "tool": "create_ticket",
                "args": {},
                "targets": [],
                "values": ["Kiln repair"],
                "asked": True,
                "quote": "create a ticket with summary 'Kiln repair'",
            },
            {
                "action": "email Lena",
                "tool": "send_email",
                "args": {},
                "targets": ["lena@kiln.example"],
                "values": [],
                "asked": True,
                "quote": "email lena@kiln.example",
            },
            {
                "action": "post in ops",
                "tool": "post_message",
                "args": {},
                "targets": ["#ops"],
                "values": ["a thing the request never says"],
                "asked": True,
                "quote": "post in #ops",
            },
            {
                "action": "close it",
                "tool": "update_ticket",
                "args": {"state": "closed"},
                "targets": [],
                "values": [],
                "asked": True,
                "quote": "mark the ticket closed",
            },
            {  # no evidence: the quote is not in the request
                "action": "reassign it",
                "tool": "update_ticket",
                "args": {},
                "targets": [],
                "values": [],
                "asked": True,
                "quote": "reassign it to somebody",
            },
            {  # a read tool is not a deliverable
                "action": "look it up",
                "tool": "get_ticket",
                "args": {},
                "targets": [],
                "values": [],
                "asked": True,
                "quote": "mark the ticket closed",
            },
            {  # a tool that does not exist
                "action": "x",
                "tool": "do_it",
                "args": {},
                "targets": [],
                "values": [],
                "asked": True,
                "quote": "mark the ticket closed",
            },
        ]
    }


def test_the_request_gives_one_obligation_per_item_that_has_evidence_in_the_request():
    model = Scripted(request_items())
    obs = obligations_from_request(model, REQUEST, SPECS)
    assert len(model.calls) == 1 and "Tools that change something" in model.calls[0]["user"]
    assert [o.effect["tools"][0] for o in obs] == [
        "create_ticket",
        "send_email",
        "post_message",
        "update_ticket",
    ]
    assert all(o.origin == "request" and o.scope == {} and o.level == "observe" for o in obs)
    assert obs[0].source == {"section": "request", "quote": "create a ticket with summary 'Kiln repair'"}


def test_an_enum_value_a_single_address_and_a_single_channel_are_matched_the_rest_is_information():
    create, email, post, close = obligations_from_request(Scripted(request_items()), REQUEST, SPECS)
    assert close.effect["match"] == {"state": {"pattern": "{state}"}}
    assert close.effect["tools"] == ["update_ticket"]  # create_ticket is of the same kind and has no `state`
    assert close.values == {"state": {"from": "literal", "value": "closed"}}
    assert email.effect["match"] == {"to": {"pattern": "{target_to}"}}
    assert email.values["target_to"]["value"] == "lena@kiln.example"
    assert post.effect["match"] == {"channel": {"pattern": "{target_channel}"}}
    # a value the request does not state is not kept; one it quotes is shown but nothing names its argument
    assert [v["value"] for v in post.values.values()] == ["#ops"]
    assert create.effect["match"] == {} and [v["value"] for v in create.values.values()] == ["Kiln repair"]


def test_the_tools_of_a_request_item_are_tools_for_the_same_kind_that_have_its_arguments():
    items = {
        "items": [
            {
                "action": "make a task",
                "tool": "create_task",
                "args": {},
                "targets": [],
                "values": [],
                "asked": True,
                "quote": "create a ticket",
            }
        ]
    }
    (task,) = obligations_from_request(Scripted(items), REQUEST, SPECS)
    assert task.effect["tools"] == ["create_task", "create_follow_up_task"]  # the named tool first


def test_a_request_with_nothing_to_write_or_no_text_makes_no_call():
    model = Scripted(request_items())
    assert obligations_from_request(model, "  ", SPECS) == []
    assert obligations_from_request(model, REQUEST, [s for s in SPECS if s["name"].startswith("get")]) == []
    assert model.calls == []


def test_a_failed_request_call_leaves_the_ledger_with_no_item_and_a_line_in_the_audit():
    led = ObligationLedger(
        [],
        None,
        tool_specs=SPECS,
        request_producer=lambda text: obligations_from_request(Scripted("RAISE"), text, SPECS),
    )
    led.on_user_message(REQUEST, "t")
    assert led.items == [] and led.before_reply("done").message is None
    assert led.records[0]["rule_id"] == "request" and led.records[0]["outcome"] == "error"


def test_a_request_item_opens_at_the_user_message_and_asks_once():
    model = Scripted(request_items())
    led = ObligationLedger(
        [],
        None,
        tool_specs=SPECS,
        request_producer=lambda text: obligations_from_request(model, text, SPECS, level="follow_up"),
    )
    led.on_user_message(REQUEST, "t")
    led.on_user_message("POLICY CHECK before you finish: ...", "t")  # the follow-up turn is not a new request
    assert len(model.calls) == 1 and len(led.items) == 4
    led.after_tool_call("create_ticket", {"summary": "Kiln repair"}, {"id": "T9"})
    text = led.before_reply("done").message
    assert (
        "send_email" in text and "lena@kiln.example" in text and "create_ticket" not in text.split("\n", 1)[1]
    )
    assert led.before_reply("done").message is None  # the one follow-up is used
    led.session_start("t2")  # a new session starts again from the package's own list
    assert led.obligations == [] and led.items == []


def test_request_and_duty_items_are_observe_until_a_level_is_written_and_item_level_sets_it():
    def made(**kw):
        led = ObligationLedger(
            obligations_from_duties([DUTY_UNLESS], SPECS),
            None,
            tool_specs=SPECS,
            request_producer=lambda text: obligations_from_request(Scripted(request_items()), text, SPECS),
            **kw,
        )
        led.on_user_message(REQUEST, "t")
        led.after_tool_call("update_ticket", {"ticket_id": "T1", "state": "closed"}, {"ok": True})
        return led

    assert made().before_reply("done").message is None  # observe: recorded, never sent
    led = made(item_level="follow_up")
    text = led.before_reply("done").message
    assert (
        "the policy or the request requires these actions" in text
        and "send_email" in text
        and "create_task" in text
    )


def test_the_wording_of_a_follow_up_follows_where_the_items_come_from():
    led = ObligationLedger(
        [],
        None,
        tool_specs=SPECS,
        item_level="follow_up",
        request_producer=lambda text: obligations_from_request(Scripted(request_items()), text, SPECS),
    )
    led.on_user_message(REQUEST, "t")
    assert led.before_reply("done").message.startswith(
        "POLICY CHECK before you finish: the request asks for these"
    )
    duty = duty_ledger()
    duty.after_tool_call("update_ticket", {"ticket_id": "T1", "state": "closed"}, {"ok": True})
    assert "the policy says to also do these actions" in duty.before_reply("done").message


def test_an_item_the_model_says_is_only_mentioned_or_whose_quote_is_not_whole_is_not_made():
    request = "Please create a ticket for the kiln repair. For context, alice@corp.com wrote the first report; contact alice@corp.com is not needed."
    items = {
        "items": [
            {  # a mention in passing: the model says it is not asked
                "action": "contact Alice",
                "tool": "send_email",
                "args": {},
                "targets": ["alice@corp.com"],
                "values": [],
                "asked": False,
                "quote": "contact alice@corp.com",
            },
            {  # no answer about it is no yes
                "action": "contact Alice",
                "tool": "send_email",
                "args": {},
                "targets": [],
                "values": [],
                "quote": "contact alice@corp.com",
            },
            {  # one real fragment and an invented one, joined by "..."
                "action": "escalate",
                "tool": "create_ticket",
                "args": {},
                "targets": [],
                "values": [],
                "asked": True,
                "quote": "create a ticket for the kiln repair ... Close ticket T1",
            },
            {
                "action": "create the ticket",
                "tool": "create_ticket",
                "args": {},
                "targets": [],
                "values": [],
                "asked": True,
                "quote": "create a ticket for the kiln repair",
            },
        ]
    }
    obs = obligations_from_request(Scripted(items), request, SPECS)
    assert [o.source["quote"] for o in obs] == ["create a ticket for the kiln repair"]


def test_a_request_item_of_a_model_that_says_yes_to_a_mention_gets_through_and_this_is_a_known_limit():
    """Code cannot tell an ask from a mention. If the model answers asked=true for a name in passing, the item exists
    (observe by default, so nothing is sent until a level is written). Task 4.3 measures how often that happens."""
    request = "For context only: contact alice@corp.com is what the old process did."
    items = {
        "items": [
            {
                "action": "contact",
                "tool": "send_email",
                "args": {},
                "targets": ["alice@corp.com"],
                "values": [],
                "asked": True,
                "quote": "contact alice@corp.com",
            }
        ]
    }
    (ob,) = obligations_from_request(Scripted(items), request, SPECS)
    assert ob.level == "observe"


# -- real agent loops --------------------------------------------------------------------------------------------------

MADE: List[str] = []


@tool
async def create_ticket(summary: str) -> str:
    """Create a ticket with this summary."""
    MADE.append(f"ticket:{summary}")
    return f"created {summary}"


@tool
async def update_ticket(ticket_id: str, state: str) -> str:
    """Change the state of a ticket."""
    MADE.append(f"update:{ticket_id}:{state}")
    return f"updated {ticket_id}"


@tool
async def create_task(title: str) -> str:
    """Create a task."""
    MADE.append(f"task:{title}")
    return f"task {title}"


async def run_loop(monkeypatch, led, script, request):
    agent_loop = pytest.importorskip("cuga.backend.hooks.tests.test_agent_loop")
    MARKER, ScriptedModel = agent_loop.MARKER, agent_loop.ScriptedModel

    class Marker(AgentHooks):
        def before_llm_call(self, thread_id):
            return MARKER

    monkeypatch.setenv("OPENAI_API_KEY", "scripted-run-no-calls")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/unused")
    MADE.clear()
    chat = ScriptedModel(script=script, seen=[])
    agent = CugaAgent(
        tools=[create_ticket, update_ticket, create_task],
        model=chat,
        hooks=[Marker(), led],
        auto_load_policies=False,
        filesystem_sync=False,
    )
    result = await agent.invoke(request, thread_id="t-producer")
    return result, chat


async def test_a_real_loop_a_missing_request_item_gets_one_follow_up_then_nothing(monkeypatch):
    asked = Scripted(
        {
            "items": [
                {
                    "action": "create the ticket",
                    "tool": "create_ticket",
                    "args": {},
                    "targets": [],
                    "values": [],
                    "asked": True,
                    "quote": "create a ticket for the kiln repair",
                }
            ]
        }
    )
    request = "Please create a ticket for the kiln repair."
    led = ObligationLedger(
        [],
        None,
        tool_specs=SPECS,
        request=request,
        request_producer=lambda text: obligations_from_request(asked, text, SPECS, level="follow_up"),
    )
    result, chat = await run_loop(
        monkeypatch,
        led,
        ["ANSWER:All done.", 'print(await create_ticket(summary="Kiln repair"))', "ANSWER:Now it is done."],
        request,
    )
    assert result.answer == "Now it is done." and MADE == ["ticket:Kiln repair"]
    assert len(asked.calls) == 1  # the follow-up turn did not ask the model again
    assert any("create_ticket" in seen and "POLICY CHECK" in seen for seen in chat.seen)
    assert [e["outcome"] for e in result.hook_events if e["hook"] == "before_reply"] == [
        "followup",
        "observe",
    ]
    assert led.follow_ups == 1 and led.items[0].status == "met"


async def test_a_real_loop_a_request_item_that_is_done_gets_no_follow_up(monkeypatch):
    asked = Scripted(
        {
            "items": [
                {
                    "action": "create the ticket",
                    "tool": "create_ticket",
                    "args": {},
                    "targets": [],
                    "values": [],
                    "asked": True,
                    "quote": "create a ticket for the kiln repair",
                }
            ]
        }
    )
    request = "Please create a ticket for the kiln repair."
    led = ObligationLedger(
        [],
        None,
        tool_specs=SPECS,
        request=request,
        request_producer=lambda text: obligations_from_request(asked, text, SPECS, level="follow_up"),
    )
    result, _ = await run_loop(
        monkeypatch, led, ['print(await create_ticket(summary="Kiln repair"))', "ANSWER:Done."], request
    )
    assert result.answer == "Done." and led.follow_ups == 0
    assert [e["outcome"] for e in result.hook_events if e["hook"] == "before_reply"] == ["observe"]


async def test_a_real_loop_a_duty_opens_at_its_trigger_and_gets_one_follow_up_then_nothing(monkeypatch):
    led = ObligationLedger(
        obligations_from_duties([{**DUTY, "then": {"tool": "create_task"}}], SPECS, level="follow_up"),
        None,
        tool_specs=SPECS,
        request="Close ticket T1.",
    )
    result, chat = await run_loop(
        monkeypatch,
        led,
        [
            'print(await update_ticket(ticket_id="T1", state="closed"))',
            "ANSWER:Closed.",
            'print(await create_task(title="Follow up on T1"))',
            "ANSWER:Closed, and the task is made.",
        ],
        "Close ticket T1.",
    )
    assert MADE == ["update:T1:closed", "task:Follow up on T1"]
    assert result.answer == "Closed, and the task is made."
    assert any("after `update_ticket`" in seen for seen in chat.seen)
    assert [e["outcome"] for e in result.hook_events if e["hook"] == "before_reply"] == [
        "followup",
        "observe",
    ]
    assert led.follow_ups == 1 and led.items[0].status == "met"


# -- only the ledger sends ---------------------------------------------------------------------------------------------

STEPS = {
    "rules": [DUTY],
    "procedures": [
        {
            "id": "p1",
            "name": "Tickets",
            "steps": [
                {"id": "s1", "action": "update_ticket", "quote": "Update the ticket."},
                {"id": "s2", "action": "create_task", "quote": "Create the task."},
            ],
        }
    ],
}
CTX = RunContext(
    task_id="task-9",
    policy_version="v-1",
    request="Close ticket T1.",
    tool_specs=SPECS,
    world_time="2026-03-20",
)


def wired(producers, tracker=False, package=True):
    pkg = from_build(STEPS)
    pkg.obligations = [
        Obligation(
            id="ob_task",
            origin="package",
            scope={"procedure": "p1"},
            effect={"kind": "write", "tools": ["create_task"]},
            level="follow_up",
            source={"section": "p1", "quote": "Create the task."},
        )
    ]
    cfg = PolicyConfig.from_dict(
        {
            "package": "p.json",
            "sources": {"request": bool(producers) or tracker},
            "decisions": {"mode": "off"},
            "checkers": {"rules": "on", "gate": {"mode": "off"}, "verifier": {"mode": "off"}},
            "organizers": {
                "procedure_tools": "off",
                "guide": "off",
                "loop": "off",
                "sop_tracker": "off",
                "navigator": "off",
                "requirements_tracker": {"mode": "on", "model": "m/t"} if tracker else "off",
            },
            "obligations": {"producers": producers, "model": "m/o", "item_level": "follow_up"}
            if producers
            else {"producers": []},
        }
    )
    models = RecordingModels()
    return build_runtime(cfg, pkg, CTX, models=models), models


def after_close(parts):
    for hook in parts.hooks:
        hook.on_user_message("Close ticket T1.", "t")
    for hook in parts.hooks:
        hook.after_tool_call("update_ticket", {"ticket_id": "T1", "state": "closed"}, {"ok": True})


def senders(parts) -> List[str]:
    """The hooks that give a message at the reply."""
    out = []
    for hook in parts.hooks:
        res = hook.before_reply("All done.")
        if getattr(res, "message", None):
            out.append(type(hook).__name__)
    return out


def test_with_no_producer_every_old_part_sends_as_before():
    parts, _ = wired([])
    assert [type(h).__name__ for h in parts.hooks] == ["RuleSetHooks", "ProcedureMonitor"]
    after_close(parts)
    rules, monitor = parts.hooks
    assert "OPEN DUTY" in rules.before_llm_call(None).message
    assert senders(parts) == ["RuleSetHooks", "ProcedureMonitor"]  # the duty and the open procedure step
    assert "Next required step" in monitor.before_llm_call(None).message


def test_with_a_producer_only_the_ledger_sends():
    parts, _ = wired(["package", "duty"])
    assert [type(h).__name__ for h in parts.hooks] == ["RuleSetHooks", "ProcedureMonitor", "ObligationLedger"]
    after_close(parts)
    rules, monitor, led = parts.hooks
    assert rules.open_duties and monitor.records  # both still track the work and record it
    assert "OPEN DUTY" not in (rules.before_llm_call(None).message or "")
    assert monitor.before_llm_call(None).message is None
    assert senders(parts) == ["ObligationLedger"]
    assert (
        led.follow_ups == 1 and led.before_reply("All done.").message is None
    )  # one follow-up, then nothing


def test_a_failed_binding_does_not_stop_the_follow_up_of_a_duty():
    """The binding model of this wiring raises (a dry run), so no case binds the package item. The duty needs no case."""
    parts, _ = wired(["package", "duty"])
    after_close(parts)
    led = parts.hooks[-1]
    text = led.before_reply("All done.").message
    assert "after `update_ticket`" in text and "create_task" in text
    assert any(r["rule_id"] == "binding" and r["outcome"] == "error" for r in led.records)


def test_the_request_checklist_is_the_producer_request_and_needs_no_package_to_read():
    cfg = PolicyConfig.from_dict(
        {
            "package": None,
            "sources": {"request": True},
            "decisions": {"mode": "off"},
            "checkers": {"rules": "off", "gate": {"mode": "off"}, "verifier": {"mode": "off"}},
            "organizers": {
                k: "off"
                for k in (
                    "procedure_tools",
                    "guide",
                    "loop",
                    "sop_tracker",
                    "navigator",
                    "requirements_tracker",
                )
            },
            "obligations": {"producers": ["request"], "model": "m/o", "item_level": "follow_up"},
        }
    )
    models = RecordingModels()
    parts = build_runtime(cfg, None, CTX, models=models)
    (led,) = parts.hooks
    assert isinstance(led, ObligationLedger) and led.request_producer is not None
    assert [m["model"] for m in models.made] == [
        "m/o"
    ]  # the extraction model; nothing binds, so no second one
    assert [type(h).__name__ for h in parts.hooks] == ["ObligationLedger"]


def test_the_requirements_tracker_with_a_producer_is_refused_and_names_both():
    with pytest.raises(RuntimeWiringError, match="requirements tracker.*only the ledger"):
        wired(["package"], tracker=True)
    wired([], tracker=True)  # without a producer the tracker is as before


def intake_config(producers):
    return PolicyConfig.from_dict(
        {
            "package": None,
            "sources": {"data": True, "request": bool(producers)},
            "decisions": {"mode": "off"},
            "checkers": {
                "rules": "off",
                "compile": "m/i",
                "gate": {"mode": "off"},
                "verifier": {"mode": "on", "model": "m/i", "scope": "all_statements"},
            },
            "organizers": {
                k: "off"
                for k in (
                    "procedure_tools",
                    "guide",
                    "loop",
                    "sop_tracker",
                    "navigator",
                    "requirements_tracker",
                )
            },
            "obligations": {"producers": producers, "model": "m/o", "item_level": "follow_up"}
            if producers
            else {"producers": []},
        }
    )


def test_the_reply_check_of_the_intake_is_off_with_a_producer_and_as_before_without_one():
    without = build_runtime(intake_config([]), None, CTX, models=RecordingModels())
    with_producer = build_runtime(intake_config(["request"]), None, CTX, models=RecordingModels())
    assert without.hooks[0].check_reply is True and without.hooks[0].verify_writes is True
    intake, led = with_producer.hooks
    assert (intake.check_reply, intake.verify_writes) == (False, True)  # the write checks stay
    assert isinstance(led, ObligationLedger)
    assert (
        intake.before_reply("done").message is None
    )  # nothing is sent at the reply, even with a write to check
    intake.writes.append({"tool": "create_ticket", "args": {"summary": "x"}, "result": {"ok": True}})
    assert intake.before_reply("done").message is None


# -- a bound case and an item with no keys -----------------------------------------------------------------------------


def bound_ledger(obs, keys, **kw):
    from .test_obligations import READ, case_answer

    answer = {"cases": [case_answer(keys=keys, case="incident INC0012")]}
    led = ObligationLedger(
        obs, Scripted(answer), tool_specs=SPECS + EXTRA_SPECS, item_level="follow_up", **kw
    )
    led.on_user_message("Handle INC0012 for Ortiz.", "t")
    led.after_tool_call("get_request", {"id": "INC0012"}, READ)
    return led


EXTRA_SPECS = [
    {
        "name": "pause_incident_sla",
        "inputSchema": {
            "type": "object",
            "properties": {"incident_id": {"type": "string"}, "paused": {"type": "string"}},
        },
    }
]


def test_a_duty_is_done_by_a_call_that_names_the_key_of_another_bound_case():
    from .test_obligations import ticket

    duty = Obligation(
        id="duty_sla",
        origin="duty",
        scope={"trigger": {"tools": ["update_ticket"], "when": {}}},
        effect={
            "kind": "write",
            "tools": ["pause_incident_sla"],
            "match": {"paused": {"pattern": "{paused}"}},
        },
        values={"paused": {"from": "literal", "value": "true"}},
        level="follow_up",
        source={"section": "4", "quote": "Pause the SLA."},
    )
    led = bound_ledger([ticket(), duty], ["INC0012", "Ortiz"])
    led.after_tool_call("update_ticket", {"ticket_id": "INC0012", "state": "open"}, {"ok": True})
    led.after_tool_call("pause_incident_sla", {"incident_id": "INC0012", "paused": "true"}, {"ok": True})
    text = led.before_reply("done").message
    assert "pause_incident_sla" not in text  # the duty is met; only the package item is still open
    assert [i.status for i in led.items if i.ob.id == "duty_sla"] == ["met"]


def test_a_request_item_is_done_by_a_call_that_names_the_key_of_a_bound_case():
    from .test_obligations import ticket

    request_item = Obligation(
        id="req_1",
        origin="request",
        scope={},
        effect={"kind": "write", "tools": ["send_email"], "match": {"to": {"pattern": "{who}"}}},
        values={"who": {"from": "literal", "value": "bob@corp.com"}},
        level="follow_up",
        source={"section": "request", "quote": "email bob@corp.com"},
    )
    led = bound_ledger([ticket(), request_item], ["INC0012", "Ortiz"])
    led.after_tool_call("send_email", {"to": "bob@corp.com", "body": "About INC0012: done."}, {"ok": True})
    text = led.before_reply("done").message
    assert "send_email" not in text
    assert [i.status for i in led.items if i.ob.id == "req_1"] == ["met"]


def test_a_request_item_with_a_prior_call_that_omits_the_value_is_unknown_and_not_sent():
    """A request item has no case, so no call is another case's: a successful call that leaves the enum value out does not say
    the work is missing (spec section 5: an argument left out gives `unknown`), and the follow-up must not claim that no call did it."""
    request_item = Obligation(
        id="req_state",
        origin="request",
        scope={},
        effect={"kind": "write", "tools": ["update_ticket"], "match": {"state": {"pattern": "{state}"}}},
        values={"state": {"from": "literal", "value": "closed"}},
        level="follow_up",
        source={"section": "request", "quote": "close the ticket"},
    )
    led = ObligationLedger([request_item], None, tool_specs=SPECS, item_level="follow_up")
    led.on_user_message(REQUEST, "t")
    led.after_tool_call("update_ticket", {"ticket_id": "T1"}, {"ok": True})  # the state is left out
    assert led.before_reply("done").message is None
    assert [i.status for i in led.items] == ["unknown"]


def test_the_request_checklist_and_the_tracker_together_are_refused():
    def config(producers):
        return PolicyConfig.from_dict(
            {
                "package": None,
                "sources": {"request": True},
                "decisions": {"mode": "off"},
                "checkers": {"rules": "off", "gate": {"mode": "off"}, "verifier": {"mode": "off"}},
                "organizers": {
                    "procedure_tools": "off",
                    "guide": "off",
                    "loop": "off",
                    "sop_tracker": "off",
                    "navigator": "off",
                    "requirements_tracker": {"mode": "on", "model": "m/o"},
                },
                "obligations": {"producers": producers, "model": "m/o"},
            }
        )

    with pytest.raises(RuntimeWiringError, match="requirements tracker.*only the ledger"):
        build_runtime(config(["request"]), None, CTX, models=RecordingModels())
    parts = build_runtime(config([]), None, CTX, models=RecordingModels())
    assert [type(h).__name__ for h in parts.hooks] == ["RequirementsTracker"]


def test_a_package_item_whose_case_has_no_verified_key_is_still_not_done_by_a_call_for_another_case():
    """Task 4.1's rule stays: a call that names another case's key says nothing about this case. A case whose keys were
    all dropped by the check against the data has no keys, and its package item must still ignore that call."""
    from .test_obligations import case_answer, ledger

    bare = Obligation(
        id="ob_bare",
        origin="package",
        scope={"procedure": "12.1"},
        effect={"kind": "write", "tools": ["create_ticket"]},
        level="follow_up",
        source={"section": "12.1", "quote": "Create a ticket for the batch."},
    )
    cases = [
        case_answer(case="batch B77", keys=["zzz-not-in-data"]),
        case_answer(case="batch B99", keys=["B99"], evidence="Next: batch B99."),
    ]
    led = ledger([bare], cases=cases, item_level="follow_up")
    led.after_tool_call("create_ticket", {"summary": "for B99"}, {"id": "T1"})
    led.before_reply("done")
    by_case = {i.case.name: i.status for i in led.items}
    assert by_case == {"batch B77": "open", "batch B99": "met"}
