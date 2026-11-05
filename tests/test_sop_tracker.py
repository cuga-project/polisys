"""SOP tracker: chapters stay chosen across turns; steps without an SOP quote, or "done" without conversation
evidence, are dropped; an escalation needs both quotes."""

import json

from polisys.sop_tracker import SopTracker

SOP = """# Service
## SIM
Use check_sim_status to check the SIM. If it shows Locked, escalate to technical support.
# Data
## Roaming
If the user is abroad, enable roaming on the line and ask the user to turn data roaming on.
"""


def model_for(chapters, steps, esc=None):
    def model(system, user):
        if "SOP chapters" in user:
            return json.dumps({"chapters": chapters})
        return json.dumps({"steps": steps, "escalation": esc or {"met": False}})

    return model


def test_quotes_are_checked_and_chapters_kept():
    steps = [
        {
            "action": "check SIM",
            "quote": "Use check_sim_status to check the SIM.",
            "status": "done",
            "evidence": "SIM: Locked",
        },
        {"action": "invented", "quote": "Offer a discount to the user.", "status": "pending", "evidence": ""},
        {
            "action": "roaming",
            "quote": "enable roaming on the line",
            "status": "done",
            "evidence": "never said",
        },
    ]
    esc = {
        "met": True,
        "quote": "If it shows Locked, escalate to technical support.",
        "evidence": "SIM: Locked",
    }
    t = SopTracker(SOP, model_for(["1"], steps, esc))
    note = t.note([{"role": "customer", "text": "No service. My check says SIM: Locked"}])
    assert "escalate to technical support" in note and "Offer a discount" not in note
    assert [s["action"] for s in t.records[-1]["steps"]] == ["check SIM"]
    t.model = model_for(["2"], [])
    t.note([{"role": "customer", "text": "I am abroad"}])
    assert t.kept == ["Service", "Data"]
