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
