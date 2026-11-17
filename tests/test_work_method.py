"""notes.work_method: general rules for SOP work on many records in the system prompt, and one read-back turn."""

import pytest

from polisys.policy_config import PolicyConfig, PolicyConfigError
from polisys.work_method import METHODS, READBACK, ReadbackHooks, note
from tests.test_runtime_assembly import CTX, build, config


def test_only_the_listed_rules_reach_the_note():
    text = note(["actors", "readback"])
    assert METHODS["actors"] in text and METHODS["worksheet"] not in text and READBACK not in text
    assert note([]) == "" and note(["readback"]) == ""


def test_readback_is_sent_once_per_run():
    h = ReadbackHooks()
    assert h.before_reply("done").message == READBACK
    assert h.before_reply("done again") is None
    h.session_start("t2")
    assert h.before_reply("done").message == READBACK


def test_an_unknown_rule_name_is_refused():
    with pytest.raises(PolicyConfigError):
        PolicyConfig.from_dict({"notes": {"work_method": ["guess"]}})


def test_the_runtime_adds_the_note_and_the_readback_hook():
    parts, _ = build(config(notes={"work_method": ["worksheet", "readback"]}), ctx=CTX)
    assert METHODS["worksheet"] in parts.system_note
    assert any(isinstance(h, ReadbackHooks) for h in parts.hooks)
    off, _ = build(config(), ctx=CTX)
    assert METHODS["worksheet"] not in (off.system_note or "")
    assert not any(isinstance(h, ReadbackHooks) for h in off.hooks)


def test_gate_missing_needs_read_adds_the_rule_only_when_on():
    from polisys.store_gate import MISSING_NEEDS_READ, USER, StoreGate
    from tests.test_runtime_assembly import GATE_ON, GRAPH, from_store_graph

    args = dict(label="L", rules="R", dates="", data="D", action="A")
    assert "show it. Return" in USER.format(**args, extra="")  # off: the measured prompt
    assert MISSING_NEEDS_READ in USER.format(**args, extra=MISSING_NEEDS_READ)
    on = {"gate": {**GATE_ON["gate"], "missing_needs_read": True}}
    for gate_cfg, want in ((GATE_ON, ""), (on, MISSING_NEEDS_READ)):
        parts, _ = build(config(gate_cfg, {"procedure_tools": "on"}, package="p.json"), from_store_graph(GRAPH))
        gate = next(h for h in parts.hooks if isinstance(h, StoreGate))
        assert gate.extra == want


def test_readback_sop_shows_the_opened_procedures_and_needs_one_form():
    from polisys.package import from_store_graph
    from polisys.procedure_store import ProcedureStore
    from polisys.work_method import READBACK_SOP
    from tests.test_runtime_assembly import GRAPH

    store = ProcedureStore.from_package(from_store_graph(GRAPH))
    h = ReadbackHooks(store=store)
    assert h.before_reply("done").message == READBACK_SOP  # nothing opened yet
    store.get_procedure(["1.1"])
    h.session_start("t2")
    msg = h.before_reply("done").message
    assert msg.startswith(READBACK_SOP) and "Open a ticket." in msg
    with pytest.raises(Exception):
        build(config(notes={"work_method": ["readback", "readback_sop"]}), ctx=CTX)


def test_gate_trigger_check_is_added_next_to_missing_needs_read():
    from polisys.store_gate import MISSING_NEEDS_READ, TRIGGER_CHECK, StoreGate
    from tests.test_runtime_assembly import GATE_ON, GRAPH, from_store_graph

    both = {"gate": {**GATE_ON["gate"], "missing_needs_read": True, "trigger_check": True}}
    parts, _ = build(config(both, {"procedure_tools": "on"}, package="p.json"), from_store_graph(GRAPH))
    gate = next(h for h in parts.hooks if isinstance(h, StoreGate))
    assert gate.extra == MISSING_NEEDS_READ + TRIGGER_CHECK
    assert METHODS["sop_literal"] in note(["sop_literal"])


# -- lookup_retry and finish_check ------------------------------------------------------------------------------------------


def _is_write(tool):
    return tool.startswith(("create_", "update_"))


def test_finish_check_is_sent_once_and_is_one_of_the_reply_turns():
    from polisys.work_method import FINISH_CHECK, FinishCheckHooks

    h = FinishCheckHooks()
    assert h.before_reply("done").message == FINISH_CHECK
    assert h.before_reply("done again") is None
    with pytest.raises(Exception, match="list one"):
        build(config(notes={"work_method": ["readback", "finish_check"]}), ctx=CTX)


def test_a_failed_lookup_of_a_name_the_request_gives_gets_one_note():
    from polisys.work_method import LookupRetryHooks

    h = LookupRetryHooks("Add the staff of the Northern Depot Team to the roster.", _is_write)
    miss = {"error": {"detail": "not found"}}
    note = h.after_tool_call("find_team", {"name": "Northern Depot Team"}, miss)
    assert note is not None and '"Northern Depot Team"' in note.message
    assert h.after_tool_call("find_team", {"name": "northern depot team"}, miss) is None  # one note per name
    assert h.records[0]["outcome"] == "lookup_retry_note"


def test_no_lookup_note_for_a_success_a_write_a_short_value_or_a_value_the_request_does_not_hold():
    from polisys.work_method import LookupRetryHooks, named_in

    h = LookupRetryHooks("Add the staff of the Northern Depot Team to the roster.", _is_write)
    miss = {"error": "not found"}
    assert h.after_tool_call("find_team", {"name": "Northern Depot Team"}, {"team": {"id": 3}}) is None
    assert h.after_tool_call("create_team", {"name": "Northern Depot Team"}, miss) is None
    assert h.after_tool_call("find_team", {"name": "Southern Depot"}, miss) is None
    assert h.after_tool_call("find_team", {"name": "the"}, miss) is None
    assert not named_in("Depot Tea", "the Northern Depot Team")  # whole words only
    assert not named_in(7, "team 7")


def test_the_lookup_notes_stop_at_the_limit():
    from polisys.work_method import LookupRetryHooks

    h = LookupRetryHooks("Find Alpha Unit, Beta Unit, Gamma Unit and Delta Unit.", _is_write, max_notes=3)
    miss = {"error": "not found"}
    sent = [h.after_tool_call("find_unit", {"name": f"{n} Unit"}, miss) for n in ("Alpha", "Beta", "Gamma", "Delta")]
    assert [s is not None for s in sent] == [True, True, True, False]


def test_the_runtime_adds_the_two_new_hooks_only_when_listed():
    from polisys.work_method import FinishCheckHooks, LookupRetryHooks

    parts, _ = build(config(notes={"work_method": ["lookup_retry", "finish_check"]}), ctx=CTX)
    kinds = {type(h) for h in parts.hooks}
    assert FinishCheckHooks in kinds and LookupRetryHooks in kinds
    off, _ = build(config(notes={"work_method": ["readback"]}), ctx=CTX)
    assert not {FinishCheckHooks, LookupRetryHooks} & {type(h) for h in off.hooks}


def test_the_finish_check_says_that_the_tools_list_is_in_force_only_when_the_deployment_says_so():
    from polisys.work_method import FINISH_CHECK, FINISH_CHECK_TOOL_LISTS, FinishCheckHooks

    assert FinishCheckHooks().before_reply("done").message == FINISH_CHECK
    assert FinishCheckHooks("tool").before_reply("done").message == FINISH_CHECK + FINISH_CHECK_TOOL_LISTS
    parts, _ = build(config(notes={"work_method": ["finish_check"]}, authority={"value_lists": "tool"}), ctx=CTX)
    assert parts.hooks[0].message.endswith(FINISH_CHECK_TOOL_LISTS)

