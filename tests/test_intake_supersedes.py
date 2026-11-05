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
