"""The narrow write checks of the online intake: the request's conditions as a statement, proof from the data shown, a
summary that names skipped items, and a call that undoes a message of this run."""

import json

import pytest

from polisys.runtime.context import RuntimeWiringError
from polisys.sources.intake.online_intake import OnlinePolicyIntake
from polisys.sources.intake.prompts import PROVEN_ONLY, REPORT_SYSTEM, UNDO_SYSTEM

from .test_runtime_assembly import CTX, build, config

RULE = "Send the renewal notice only to customers whose contract has expired."
POLICY = {"subject": "Renewal policy", "body": "Policy: " + RULE}
SUMMARY = (
    "Renewal notices were sent to Dana Okafor and Lee Marsh. Priya Nand was skipped because her contract runs to 2027, "
    "and Tom Reyes was excluded as a partner account."
)
SEND = ("mail_send_email", {"to": "ops@example.com", "subject": "Renewals done", "body": SUMMARY})
NOTICE = ("mail_send_email", {"to": "dana@example.com", "body": "Your contract expired."})


class Model:
    """Answers by the kind of call: extraction, the verifier, the report question, the tool question."""

    def __init__(self, violations=None, report=None, undo=True, policies=True):
        self.violations, self.report, self.undo, self.policies = violations or [], report or {}, undo, policies
        self.prompts = {"verify": [], "report": [], "undo": []}

    def __call__(self, system, user):
        if system == REPORT_SYSTEM:
            self.prompts["report"].append(user)
            return json.dumps(self.report)
        if system == UNDO_SYSTEM:
            self.prompts["undo"].append(user)
            return json.dumps({"undo": self.undo})
        if "Planned action" in user:
            self.prompts["verify"].append(user)
            return json.dumps({"violations": self.violations})
        if self.policies and "Records (returned by tool" in user and "Renewal policy" in user:
            return json.dumps(
                {"policies": [{"record": 1, "title": "Renewals", "trust": "trusted", "statements": [RULE]}]}
            )
        return json.dumps({"policies": []})


def _hooks(model, **kw):
    kw.setdefault("check_reply", False)
    kw.setdefault("read_before_write", False)
    h = OnlinePolicyIntake(model, **kw)
    h.on_user_message("Send renewal notices to customers with expired contracts.", None)
    h.after_tool_call("mail_find_email", {"query": "policy"}, {"messages": [POLICY]})
    h._drain(5)
    return h


# -- the verifier: request conditions, proof from the data --------------------------------------------------------------


def test_request_conditions_adds_one_statement_and_proven_only_one_sentence():
    model = Model()
    h = _hooks(model, request_conditions=True, proven_only=True)
    h.before_tool_call(*NOTICE)
    prompt = model.prompts["verify"][-1]
    assert f"1. [Renewals] {RULE}" in prompt
    assert "2. [user request] Do the requested action only for records that meet the conditions" in prompt
    assert prompt.endswith(PROVEN_ONLY)
    assert "Arguments this tool accepts: to, body" in prompt  # no tool spec: the call's own argument names


def test_without_the_settings_the_verifier_prompt_is_as_before():
    model = Model()
    _hooks(model).before_tool_call(*NOTICE)
    prompt = model.prompts["verify"][-1]
    assert "[user request]" not in prompt and PROVEN_ONLY not in prompt
    assert "Arguments this tool accepts: (unknown)" in prompt


def test_request_conditions_checks_a_write_when_no_rule_was_found():
    broken = [{"statement": 1, "reason": "the contract runs to 2027", "fix": "skip this customer"}]
    model = Model(broken, policies=False)
    h = _hooks(model, request_conditions=True, verifier_may_deny=True)
    res = h.before_tool_call(*NOTICE)
    assert "1. [user request]" in model.prompts["verify"][-1]
    assert "was not called" in res.message and "[user request]" in res.message
    quiet = Model(broken, policies=False)
    assert _hooks(quiet).before_tool_call(*NOTICE).message is None and quiet.prompts["verify"] == []


def test_reply_check_off_keeps_the_write_verifier_and_drops_the_question_at_the_reply():
    verifier = {"mode": "on", "model": "m/i", "scope": "all_statements"}

    def intake(**more):
        cfg = config({"compile": "m/i", "verifier": {**verifier, **more}}, sources={"data": True})
        return next(h for h in build(cfg, None, CTX)[0].hooks if type(h).__name__ == "OnlinePolicyIntake")

    assert intake().check_reply and intake().verify_writes
    narrow = intake(reply_check=False, request_conditions=True, proven_only=True)
    assert narrow.verify_writes and not narrow.check_reply
    assert narrow.request_conditions and narrow.proven_only


def test_a_narrow_setting_that_nothing_reads_raises():
    for checkers in (
        {"unasked_report": "deny"},
        {"own_message_undo": "block_once"},
        {"compile": "m/i", "verifier": {"mode": "off", "request_conditions": True}},
    ):
        data = "compile" in checkers
        with pytest.raises(RuntimeWiringError, match="nothing reads it"):
            build(config(checkers, sources={"data": data}), None, CTX)


# -- unasked_report ------------------------------------------------------------------------------------------------------

FLAG = {
    "names_skipped": True,
    "skipped_items": ["Priya Nand was skipped", "Tom Reyes was excluded as a partner account"],
    "asked": False,
    "asked_by": "",
    "summary": True,
}


def test_a_summary_that_names_skipped_items_is_stopped_until_it_changes():
    model = Model(report=FLAG)
    h = _hooks(model, verify_writes=False, unasked_report="deny")
    res = h.before_tool_call(*SEND)
    assert "was not called" in res.message and '"Priya Nand was skipped"' in res.message
    assert "Company rules:\n1. [Renewals]" in model.prompts["report"][-1] and SUMMARY in model.prompts["report"][-1]
    assert h.records[-1]["rule"] == "unasked_report" and h.records[-1]["outcome"] == "deny"
    model.report = {**FLAG, "names_skipped": False, "skipped_items": []}
    clean = {**SEND[1], "body": "Renewal notices were sent to Dana Okafor and Lee Marsh, the two customers due this month."}
    assert h.before_tool_call(SEND[0], clean).message is None
    assert h.records[-1]["outcome"] == "allow"


@pytest.mark.parametrize(
    "change",
    [
        {"asked": True, "asked_by": "list who was left out"},  # the request or a rule asks for the list
        {"summary": False},  # a message to one person, not a report of the work
        {"skipped_items": ["Omar Haddad was skipped"]},  # the copy is not in the message
        {"skipped_items": []},
        {"names_skipped": "yes"},  # not a clear true
    ],
)
def test_the_report_check_lets_a_write_through_unless_every_answer_is_clear(change):
    h = _hooks(Model(report={**FLAG, **change}), verify_writes=False, unasked_report="deny")
    assert h.before_tool_call(*SEND).message is None


def test_the_report_check_reads_only_writes_with_a_long_text_and_is_off_by_default():
    model = Model(report=FLAG)
    h = _hooks(model, verify_writes=False, unasked_report="deny")
    assert h.before_tool_call(*NOTICE).message is None  # a short body
    assert h.before_tool_call("mail_find_email", {"query": SUMMARY}).message is None  # a read
    assert model.prompts["report"] == []
    assert _hooks(model, verify_writes=False).before_tool_call(*SEND).message is None and model.prompts["report"] == []


def test_the_report_check_gives_way_after_three_denials_and_a_model_error_stops_nothing():
    h = _hooks(Model(report=FLAG), verify_writes=False, unasked_report="deny")
    outcomes = [bool(h.before_tool_call(*SEND).message) for _ in range(4)]
    assert outcomes == [True, True, True, False] and h.records[-1]["outcome"] == "allow_after_cap"

    def failing(system, user):
        if system == REPORT_SYSTEM:
            raise TimeoutError("no answer")
        return json.dumps({"policies": []})

    h = _hooks(failing, verify_writes=False, unasked_report="deny")
    assert h.before_tool_call(*SEND).message is None and h.records[-1]["outcome"] == "error"


def test_block_once_stops_the_report_once_and_the_same_call_goes_through():
    model = Model(report=FLAG)
    h = _hooks(model, verify_writes=False, unasked_report="block_once")
    assert "make the same call again unchanged" in h.before_tool_call(*SEND).message
    assert h.before_tool_call(*SEND).message is None and len(model.prompts["report"]) == 1


# -- own_message_undo ----------------------------------------------------------------------------------------------------

DOCS = {"mail_trash_email": "Move an email to the trash.", "mail_add_label": "Add a label to an email."}


def _sent(model, **kw):
    h = _hooks(model, verify_writes=False, own_message_undo="block_once", tool_docs=DOCS, **kw)
    h.after_tool_call("mail_find_email", {"query": "renewal"}, {"messages": [{"id": "msg-1107ab", "thread": "thr-5520"}]})
    h.after_tool_call(*SEND, {"id": "msg-8842cd", "thread": "thr-5520", "status": "sent"})
    return h


def test_a_call_that_undoes_a_message_of_this_run_is_stopped_once():
    model = Model()
    h = _sent(model)
    assert h.own_ids == {"msg-8842cd"}  # not the thread (read before), not the short status
    first = h.before_tool_call("mail_trash_email", {"message_id": "msg-8842cd"})
    assert "was not called" in first.message and "msg-8842cd" in first.message
    assert h.records[-1]["rule"] == "own_message_undo" and h.records[-1]["outcome"] == "block_once"
    assert h.before_tool_call("mail_trash_email", {"message_id": "msg-8842cd"}).message is None
    assert h.records[-1]["outcome"] == "allow_after_block"
    assert "Tool description: Move an email to the trash." in model.prompts["undo"][0]


def test_only_own_messages_and_only_tools_that_undo():
    model = Model()
    h = _sent(model)
    assert h.before_tool_call("mail_trash_email", {"message_id": "msg-1107ab"}).message is None  # a message it read
    assert model.prompts["undo"] == []  # no own id in the call: the tool question is not asked
    model.undo = False
    assert h.before_tool_call("mail_add_label", {"message_id": "msg-8842cd", "label": "done"}).message is None
    h.before_tool_call("mail_add_label", {"message_id": "msg-8842cd", "label": "sent"})
    assert len(model.prompts["undo"]) == 1  # asked once per tool


def test_a_write_with_no_long_text_gives_no_own_id_and_the_check_is_off_by_default():
    h = _hooks(Model(), verify_writes=False, own_message_undo="block_once", tool_docs=DOCS)
    h.after_tool_call(*NOTICE, {"id": "msg-9001ef"})
    assert h.own_ids == set()
    off = _hooks(Model(), verify_writes=False)
    off.after_tool_call(*SEND, {"id": "msg-8842cd"})
    assert off.own_ids == set() and off.before_tool_call("mail_trash_email", {"message_id": "msg-8842cd"}).message is None


def test_the_checks_have_a_checker_in_the_audit():
    from polisys.runtime.trace import _intake

    h = _sent(Model(report=FLAG), unasked_report="deny")
    h.before_tool_call("mail_trash_email", {"message_id": "msg-8842cd"})
    h.before_tool_call(*SEND)
    lines = {r["rule"]: _intake(r, h) for r in h.records if r.get("rule") in ("own_message_undo", "unasked_report")}
    assert lines["own_message_undo"]["checker"] == "code" and lines["own_message_undo"]["verdict"] == "block_once"
    assert lines["unasked_report"]["checker"] == "model" and lines["unasked_report"]["verdict"] == "deny"
