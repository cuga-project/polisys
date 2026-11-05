"""What a check says at the reply: voice and evidence (followups.py)."""

import json

from polisys.followups import CHECK_HEADER, required_actions
from polisys.tracker import RequirementsTracker

ITEM = {"kind": "effect", "text": "Send the report to the team.", "quote": "send the report", "when": "", "check": {}}


def _model(status="missing", required=None, seen=None):
    def call(system, user):
        if seen is not None:
            seen.append(user)
        if "list what the finished work must contain" in system:
            return json.dumps({"items": [ITEM]})
        if "finished work against a checklist" in system:
            return json.dumps({"items": [{"id": "R1", "status": status, "why": "no call"}]})
        return json.dumps({"items": [{"n": 1, "required": required, "why": "general"}]})

    return call


def _reply(model, **kw):
    t = RequirementsTracker(model, **kw)
    t.on_user_message("Please send the report.", None)
    return t, t.before_reply("I left it out because the report is on hold.")


def test_default_keeps_the_order_voice_and_shows_no_reply_to_the_checker():
    seen = []
    _, res = _reply(_model(seen=seen))
    assert "do these now" in res.message
    assert CHECK_HEADER not in res.message
    assert not any("on hold" in u for u in seen)


def test_check_voice_says_it_is_automated_and_the_checker_reads_the_reply():
    seen = []
    _, res = _reply(_model(seen=seen), voice="check")
    assert res.message.startswith(CHECK_HEADER)
    assert "do these now" not in res.message
    assert any("on hold" in u for u in seen)


def test_evidence_sends_only_an_action_found_required():
    t, res = _reply(_model(required="unclear"), evidence="on")
    assert not res.message
    assert t.records[-1]["left_out"][0]["required"] == "unclear"
    _, res = _reply(_model(required="yes"), evidence="on")
    assert "Send the report" in res.message


def test_a_failed_evidence_check_leaves_the_items():
    def model(system, user):
        if "really required" in system:
            raise RuntimeError("gateway")
        return _model()(system, user)

    t, res = _reply(model, evidence="on")
    assert "Send the report" in res.message
    assert any(r["outcome"] == "evidence_error" for r in t.records)


def test_required_actions_treats_no_answer_as_not_required():
    keep, left_out = required_actions(lambda s, u: '{"items": []}', ["a"], "r", "", "", "")
    assert keep == [] and left_out[0]["required"] == "no answer"
