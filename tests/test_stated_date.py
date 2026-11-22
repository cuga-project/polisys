"""The date a request states as today, in digits or in words, and its use by the checks of the online intake."""

import json

import pytest

from polisys.sources.intake.online_intake import OnlinePolicyIntake
from polisys.sources.intake.prompts import stated_today

RULE = "Send the renewal notice only to customers whose contract has expired."
POLICY = {"subject": "Renewal policy", "body": "Policy: " + RULE}


class Verifier:
    """Extraction finds RULE; the verifier's prompts are kept."""

    def __init__(self):
        self.verify = []

    def __call__(self, system, user):
        if "Planned action" in user:
            self.verify.append(user)
            return json.dumps({"violations": []})
        if "Records (returned by tool" in user and "Renewal policy" in user:
            return json.dumps(
                {"policies": [{"record": 1, "title": "Renewals", "trust": "trusted", "statements": [RULE]}]}
            )
        return json.dumps({"policies": []})


@pytest.mark.parametrize(
    "text, date",
    [
        ("Today is 2026-03-03. Close the overdue tickets.", "2026-03-03"),
        ("Today is March 3, 2026. Close the overdue tickets.", "2026-03-03"),
        ("today is Tuesday, March 3rd 2026", "2026-03-03"),
        ("Today's date is 3 March 2026.", "2026-03-03"),
        ("Today is Sept. 9, 2025", "2025-09-09"),
        ("Today is the deadline for March 3, 2026 invoices.", ""),
        ("Today is Marble 3, 2026", ""),
        ("Today is February 30, 2026", ""),
        ("Close the overdue tickets.", ""),
    ],
)
def test_the_date_a_text_states_as_today(text, date):
    assert stated_today(text) == date


def test_the_checks_judge_dates_by_the_date_the_request_states_in_words():
    model = Verifier()
    h = OnlinePolicyIntake(model, check_reply=False, read_before_write=False)
    h.on_user_message("Today is March 3, 2026. Send renewal notices to customers with expired contracts.", None)
    h.after_tool_call("mail_find_email", {"query": "policy"}, {"messages": [POLICY]})
    h._drain(5)
    h.before_tool_call("mail_send_email", {"to": "dana@example.com", "body": "Your contract expired."})
    assert model.verify[-1].startswith("Today's date: 2026-03-03\n\n")
