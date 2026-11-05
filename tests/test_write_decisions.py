"""Write decisions (decision-model questions), block-once model verdicts, rule scope, denial cap, policy trace."""

import json

from polisys.decisions import Decisions, has_limit
from polisys.intake import OnlinePolicyIntake
from polisys.ruleset import MAX_SAME_DENY, RuleSetHooks
from polisys.trace import PolicyTrace

RULES = {
    "subject": "RTO policy",
    "body": "Policy: employees must be in the office 3 days a week; remote staff exempt.",
}
RULE = "Employees must be in the office 3 days a week; remote staff are exempt."


class Model:
    """Extraction finds RULE; the verifier answers with ``violations``."""

    def __init__(self, violations=None):
        self.violations = violations or []

    def __call__(self, system, user):
        if "Planned action" in user:
            return json.dumps({"violations": self.violations})
        if "Records (returned by tool" in user and "RTO policy" in user:
            return json.dumps(
                {"policies": [{"record": 1, "title": "RTO", "trust": "trusted", "statements": [RULE]}]}
            )
        return json.dumps({"policies": []})


def ask_with(answers):
    calls = []

    def ask(state, questions):
        calls.append((state, dict(questions)))
        return {n: answers.get(n) for n in questions}

    ask.calls = calls
    return ask


def _hooks(model, **kw):
    h = OnlinePolicyIntake(model, check_reply=False, read_before_write=False, **kw)
    h.on_user_message("send RTO reminders", None)
    h.after_tool_call("gmail_find_email", {"query": "policy"}, {"messages": [RULES]})
    h._drain(5)
    return h


SEND = ("gmail_send_email", {"to": "alice@x.com", "body": "Please come in 3 days."})


def test_excluded_record_is_stopped_once_then_the_same_call_goes_through():
    ask = ask_with({"excluded": 0.9})
    h = _hooks(Model(), decisions=Decisions(ask))
    first = h.before_tool_call(*SEND)
    assert first.message and "was not called" in first.message
    assert h.before_tool_call(*SEND).message is None  # repeated unchanged: allowed
    assert "over_limit" not in ask.calls[0][1]  # no limit in the rules: not asked


def test_limit_question_only_when_a_rule_states_a_limit():
    assert has_limit("maximum 2 posts per channel per week")
    assert not has_limit("Employees must be in the office.")


def test_no_decisions_by_default_keeps_old_behaviour():
    h = _hooks(Model())
    assert h.before_tool_call(*SEND).message is None


def test_model_violation_blocks_once_when_configured():
    v = [{"statement": 1, "reason": "Alice is remote", "fix": "skip Alice"}]
    h = _hooks(Model(v), model_effect="block_once")
    assert "was not called" in (h.before_tool_call(*SEND).message or "")
    assert h.before_tool_call(*SEND).message is None
    old = _hooks(Model(v))  # default: note after the write, never a block
    assert old.before_tool_call(*SEND).message is None


SPECS = [{"name": "buffer_add_to_queue", "inputSchema": {"properties": {"text": {"type": "string"}}}}]
FMT = {
    "id": "f1",
    "type": "format",
    "tool": "buffer_add_to_queue",
    "arg": "text",
    "pattern": r"\[THOUGHT-LEAD\]",
    "source": "§1",
    "quote": "LinkedIn thought-leadership posts must include the [THOUGHT-LEAD] tag.",
}


def test_code_rule_out_of_scope_does_not_deny():
    h = RuleSetHooks([FMT], SPECS, scope_check=lambda rule, act: 0.1)
    assert h.before_tool_call("buffer_add_to_queue", {"text": "Twitter post"}).message is None
    h2 = RuleSetHooks([FMT], SPECS, scope_check=lambda rule, act: 0.9)
    assert "was not called" in (
        h2.before_tool_call("buffer_add_to_queue", {"text": "Twitter post"}).message or ""
    )


def test_cap_skip_keeps_blocking_after_repeated_denials():
    h = RuleSetHooks([FMT], SPECS, cap_effect="skip")
    msgs = [
        h.before_tool_call("buffer_add_to_queue", {"text": "x"}).message for _ in range(MAX_SAME_DENY + 1)
    ]
    assert all(msgs) and "Do not try this action again" in msgs[-1]
    allow = RuleSetHooks([FMT], SPECS)  # default: the call after the cap goes through
    last = [
        allow.before_tool_call("buffer_add_to_queue", {"text": "x"}).message for _ in range(MAX_SAME_DENY + 1)
    ]
    assert last[-1] is None


def test_rule_change_without_policy_words_is_extracted():
    seen = []

    class M(Model):
        def __call__(self, system, user):
            seen.append(user)
            return super().__call__(system, user)

    ask = ask_with({"changes": 0.95})
    adjust = {
        "subject": "Engineering adjustment",
        "body": "Hi HR team, I agreed an adjustment: Engineering staff only need 2 days in office from now. Approved by the COO.",
    }
    detector = lambda req, texts: [0.4 for _ in texts]  # noqa: E731 - below the 0.47 threshold
    h = OnlinePolicyIntake(M(), check_reply=False, detector=detector, decisions=Decisions(ask))
    h.on_user_message("send RTO reminders", None)
    h.after_tool_call("gmail_find_email", {"query": "office"}, {"messages": [RULES, adjust]})
    h._drain(5)
    assert any("Engineering staff only need 2 days" in u for u in seen if "Records (returned by tool" in u)
    assert any(r.get("outcome") == "rule_change_detected" for r in h.records)


def test_trace_records_llm_and_decision_calls():
    trace = PolicyTrace()
    model = trace.wrap_model(Model([{"statement": 1, "reason": "r"}]), "policy_layer")
    ask = trace.wrap_ask(ask_with({"excluded": 0.1}), "decisions")
    h = _hooks(model, decisions=Decisions(ask))
    h.before_tool_call(*SEND)
    kinds = [(e["kind"], e["purpose"]) for e in trace.events]
    assert (
        ("llm", "extract_policies") in kinds
        and ("decision", "excluded") in kinds
        and ("llm", "verify_write") in kinds
    )
    assert all("ms" in e for e in trace.events) and trace.summary()


# -- record checks (record signals, missing facts), hold on recheck, records found by content -----------------------


def test_record_signal_is_checked_even_without_rules():
    from polisys.intake import OnlinePolicyIntake as Intake

    ask = ask_with({"record_says_skip": 0.9})
    h = Intake(
        Model(), check_reply=False, read_before_write=False, decisions=Decisions(ask), record_checks=True
    )
    h.on_user_message("sync the tickets", None)
    h.after_tool_call(
        "zendesk_find_ticket", {"query": "x"}, {"tickets": [{"subject": "Internal test ticket"}]}
    )
    res = h.before_tool_call("freshdesk_create_ticket", {"subject": "Internal test ticket"})
    assert res.message and "own data" in res.message
    assert set(ask.calls[0][1]) == {"record_says_skip"}  # no rules: only the record question
    assert "Internal test ticket" in ask.calls[0][0]  # the source record found by its subject


def test_hold_after_recheck_keeps_a_flagged_call_stopped():
    ask = ask_with({"excluded": 0.9})
    h = _hooks(Model(), decisions=Decisions(ask), retry_effect="hold")
    assert "was not called" in (h.before_tool_call(*SEND).message or "")
    held = h.before_tool_call(*SEND).message
    assert held and "Do not try this action again" in held
    assert any(r.get("outcome") == "hold_after_recheck" for r in h.records)


def test_missing_facts_flag_never_holds_the_retry():
    ask = ask_with({"needs_data": 0.9})
    h = _hooks(Model(), decisions=Decisions(ask), retry_effect="hold", record_checks=True)
    assert "facts you have not read" in (h.before_tool_call(*SEND).message or "")
    assert h.before_tool_call(*SEND).message is None


def test_model_violation_twice_is_held_with_hold_effect():
    v = [{"statement": 1, "reason": "Alice is remote"}]
    h = _hooks(Model(v), model_effect="block_once", retry_effect="hold")
    assert h.before_tool_call(*SEND).message
    assert "Do not try this action again" in (h.before_tool_call(*SEND).message or "")


def test_fact_store_finds_records_by_short_text_values_only_when_enabled():
    from polisys.facts import FactStore, call_phrases

    rows = {"tickets": [{"subject": "Internal test ticket", "status": "open"}, {"subject": "Login fails"}]}
    on, off = FactStore(match_phrases=True), FactStore()
    for st in (on, off):
        st.add("find_ticket", rows)
    args = {"subject": "Internal test ticket", "body": "long text " * 30}
    assert call_phrases(args) == ["internal test ticket"]
    assert [r["node"]["subject"] for r in on.records_for(args)] == ["Internal test ticket"]
    assert off.records_for(args) == []
