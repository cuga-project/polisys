"""Replacement tracking in online intake: a replaced rule leaves the current list, in either reading order."""

import json

from polisys.intake import OnlinePolicyIntake

SOP = {
    "subject": "Conversion SOP",
    "body": "Policy: send every closed deal to Google Ads (minimum value $0).",
}
UPDATE = {
    "subject": "Threshold change",
    "body": "Policy update: only send deals of $10,000 or more. This supersedes the $0 minimum in the SOP.",
}
OLD_RULE = "Send every closed deal to Google Ads (minimum $0)."
NEW_RULE = "Only send deals of $10,000 or more."


class ScriptedModel:
    """Answers extraction calls in order; records the prompts."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, system, user):
        self.prompts.append(user)
        return json.dumps(self.replies.pop(0) if self.replies else {"policies": []})


def _policy(statement, **extra):
    return {
        "record": 1,
        "title": extra.pop("title", "p"),
        "trust": "trusted",
        "statements": [statement],
        **extra,
    }


def _read(hooks, record):
    hooks.after_tool_call("gmail_find_email", {"query": "policy"}, {"messages": [record]})
    hooks._drain(5)


def _hooks(replies, on=True):
    model = ScriptedModel(replies)
    hooks = OnlinePolicyIntake(model, verify_writes=False, check_reply=False, track_supersedes=on)
    hooks.on_user_message("send the closed deals to Google Ads", None)
    return hooks, model


def test_update_read_after_the_old_rule_replaces_it():
    hooks, model = _hooks(
        [
            {"policies": [_policy(OLD_RULE, title="SOP")]},
            {"policies": [_policy(NEW_RULE, title="Update", replaces_known=["K1"])]},
        ]
    )
    _read(hooks, SOP)
    _read(hooks, UPDATE)
    current = [s["text"] for s in hooks._statements(True)]
    assert current == [NEW_RULE]
    assert "K1. [SOP] " + OLD_RULE in model.prompts[1]  # the second call saw the rule found first
    brief = hooks.before_llm_call(None).message
    assert "Replaced rules" in brief and OLD_RULE in brief


def test_old_rule_read_after_the_update_is_marked_replaced():
    hooks, _ = _hooks(
        [
            {"policies": [_policy(NEW_RULE, title="Update")]},
            {"policies": [_policy(OLD_RULE, title="SOP", replaced_by_known=["K1"])]},
        ]
    )
    _read(hooks, UPDATE)
    _read(hooks, SOP)
    assert [s["text"] for s in hooks._statements(True)] == [NEW_RULE]
    assert hooks.superseded[OLD_RULE].startswith("Update: ")


def test_untrusted_records_cannot_replace_rules():
    hooks, _ = _hooks(
        [
            {"policies": [_policy(OLD_RULE, title="SOP")]},
            {"policies": [{**_policy(NEW_RULE, replaces_known=["K1"]), "trust": "untrusted"}]},
        ]
    )
    _read(hooks, SOP)
    _read(hooks, UPDATE)
    assert OLD_RULE in [s["text"] for s in hooks._statements(True)]
    assert not hooks.superseded


def test_switch_off_keeps_the_original_prompt_and_both_rules():
    hooks, model = _hooks(
        [
            {"policies": [_policy(OLD_RULE)]},
            {"policies": [_policy(NEW_RULE, replaces_known=["K1"])]},
        ],
        on=False,
    )
    _read(hooks, SOP)
    _read(hooks, UPDATE)
    assert "Rules already found" not in model.prompts[1]
    assert sorted(s["text"] for s in hooks._statements(True)) == sorted([OLD_RULE, NEW_RULE])


CFO = {"subject": "Treasury policy", "body": "Sweep above target. Investments need committee approval."}
VP = {"subject": "This week", "body": "For this week only, lower the operating target."}
SWEEP = "Sweep the excess when operating exceeds the target by more than $50,000."
COMMITTEE = "Investment decisions require Investment Committee approval."
TARGET = "For this week only, the operating target is $150,000."


def test_a_change_to_one_rule_keeps_the_other_rules_of_that_record():
    two = {"record": 1, "title": "Treasury", "trust": "trusted", "statements": [SWEEP, COMMITTEE]}
    hooks, _ = _hooks(
        [
            {
                "policies": [
                    two,
                    {
                        **_policy(TARGET, title="VP"),
                        "record": 2,
                        "replaces_records": [{"record": 1, "statements": [SWEEP]}],
                    },
                ]
            },
        ]
    )
    hooks.after_tool_call("gmail_find_email", {"query": "policy"}, {"messages": [CFO, VP]})
    hooks._drain(5)
    current = [s["text"] for s in hooks._statements(True)]
    assert COMMITTEE in current and TARGET in current and SWEEP not in current


def test_a_bare_reference_to_a_record_with_several_rules_marks_nothing():
    two = {"record": 1, "title": "Treasury", "trust": "trusted", "statements": [SWEEP, COMMITTEE]}
    hooks, _ = _hooks([{"policies": [two, {**_policy(TARGET, title="VP"), "record": 2, "replaces_records": [1]}]}])
    hooks.after_tool_call("gmail_find_email", {"query": "policy"}, {"messages": [CFO, VP]})
    hooks._drain(5)
    assert not hooks.superseded
    assert any(r.get("outcome") == "replacement_unclear" for r in hooks.records)


def test_the_agents_own_sent_text_is_not_read_as_a_rule():
    hooks, model = _hooks([])
    report = "Per the treasury policy, excess cash above the target must be invested in the bond fund every day."
    hooks.after_tool_call(
        "gmail_send_email", {"to": "treasury@x.example", "body": report}, {"id": "m9", "body": report}
    )
    hooks.after_tool_call("gmail_find_email", {"query": "sweep"}, {"messages": [{"id": "m9", "body": report}]})
    hooks._drain(5)
    assert model.prompts == []  # neither the write result nor the read-back of the sent mail reached extraction
    other = {"id": "m10", "body": "Policy: all sweeps above $100,000 need the CFO's sign-off."}
    hooks.after_tool_call("gmail_find_email", {"query": "policy"}, {"messages": [other]})
    hooks._drain(5)
    assert len(model.prompts) == 1  # a record someone else wrote still goes
