"""Policy discovery: the layer searches message sources for written rules at the request."""

import json

from polisys.intake import DISCOVERY_QUERIES, OnlinePolicyIntake, discovery_tools

MAIL = [
    {"id": "m1", "subject": "Alert policy update", "body": "Policy: only alert on pages with CVR below 3%."},
    {"id": "m2", "subject": "Lunch", "body": "Pizza on Friday."},
]


def gmail_find_email(world=None, query: str = "", max_results: int = 10):
    return json.dumps({"messages": [m for m in MAIL if query.lower() in json.dumps(m).lower()]})


def gmail_send_email(world=None, to: str = "", body: str = ""):
    return "{}"


def slack_find_message(world=None, query: str = ""):
    return json.dumps({"messages": []})


class Model:
    def __init__(self):
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(user)
        found = "CVR below 3%" in user
        return json.dumps(
            {
                "policies": [
                    {
                        "record": 1,
                        "title": "Alert policy update",
                        "trust": "trusted",
                        "statements": ["Alert only below 3% CVR."],
                    }
                ]
                if found
                else []
            }
        )


def test_discovery_tools_are_message_searches_with_a_query():
    assert discovery_tools([gmail_find_email, gmail_send_email, slack_find_message]) == [
        "gmail_find_email",
        "slack_find_message",
    ]


def test_discovery_finds_a_rule_the_agent_never_read():
    calls = []

    def search(q):
        calls.append(q)
        return gmail_find_email(query=q)

    hooks = OnlinePolicyIntake(
        Model(), verify_writes=False, check_reply=False, discover=[("gmail_find_email", search)]
    )
    hooks.on_user_message("alert the team about weak landing pages", None)
    hooks._drain(5)
    assert calls == list(DISCOVERY_QUERIES)
    assert [s["text"] for s in hooks._statements(True)] == ["Alert only below 3% CVR."]
    assert "Alert only below 3% CVR." in hooks.before_llm_call(None).message
    assert hooks.called == []  # discovery searches are not agent calls
    assert any(r.get("outcome") == "discovery" for r in hooks.records)


def test_no_discovery_by_default():
    hooks = OnlinePolicyIntake(Model(), verify_writes=False, check_reply=False)
    hooks.on_user_message("alert the team", None)
    hooks._drain(5)
    assert hooks._statements(True) == []
