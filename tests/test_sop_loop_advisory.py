"""SOP loop: evidence-gated decisions and advisory mode (one work turn for all cases)."""

import asyncio
import json

from polisys.sop_loop import LoopHooks, SopLoop, _grounded

PROC = {
    "id": "p1",
    "name": "Onboarding",
    "steps": [{"id": "s1", "action": "send_email", "quote": "Step 1: Send the welcome email."}],
    "exceptions": [
        {"when": "the start date is postponed", "then": "do not onboard yet", "quote": "If postponed..."}
    ],
}


class Model:
    def __init__(self, evidence):
        self.evidence = evidence

    def __call__(self, system, user):
        if "Which one procedure" in user:
            return json.dumps({"procedure": 1, "evidence": "new hire"})
        if "Does the condition hold" in user:
            return json.dumps({"answer": "yes", "evidence": self.evidence})
        if "already done" in user:
            return json.dumps({"answer": "no", "evidence": ""})
        if "General SOP rules" in user:
            return json.dumps({"apply": []})
        return json.dumps(
            {"cases": [{"case": "Onboard Ann", "keys": ["Ann"]}, {"case": "Onboard Bo", "keys": ["Bo"]}]}
        )


class Agent:
    def __init__(self):
        self.messages = []

    async def invoke(self, msg, thread_id=None):
        self.messages.append(msg)

        class R:
            receipt = None

        return R()


def _run(mode, evidence, require):
    agent, rec = Agent(), LoopHooks()
    rec.after_tool_call("read_sheet", {}, {"rows": [{"name": "Ann", "start": "2026-04-20"}, {"name": "Bo"}]})
    loop = SopLoop(Model(evidence), [PROC], mode=mode, require_evidence=require)
    asyncio.run(loop.run(agent, "Onboard the new hires.", "t1", rec, 60))
    return agent, loop


def test_grounded_needs_a_quote_from_the_data():
    data = 'Hi HR, the start date for Ann is postponed to June. - Manager'
    assert _grounded('"the start date for Ann is postponed"', data)
    assert not _grounded("the manager seems to want a delay", data)


def test_ungrounded_exception_does_not_change_the_decision():
    _, loop = _run("control", "the onboarding seems postponed", require=True)
    cases = [e for e in loop.log if e.get("phase") == "case"]
    assert all(e["decision"] == "follow" for e in cases) and all(e.get("ungrounded") for e in cases)
    _, old = _run("control", "the onboarding seems postponed", require=False)
    assert all(e["decision"] == "exception" for e in old.log if e.get("phase") == "case")


def test_advisory_mode_runs_one_work_turn_for_all_cases():
    agent, _ = _run("advisory", "x", require=True)
    assert len(agent.messages) == 3  # explore, one work turn, finish
    assert "Onboard Ann" in agent.messages[1] and "Onboard Bo" in agent.messages[1]
    assert "Handle only this case" not in agent.messages[1]
    control, _ = _run("control", "x", require=True)
    assert len(control.messages) == 4  # explore, two case turns, finish


# -- discovery mode: retrieval + expanded checklist, no model decisions ---------------------------------------------

BUILD = {
    "build": {
        "procedures": [
            {
                "id": "d:2",
                "name": "Onboarding",
                "source": {"document": "d", "section": "2"},
                "steps": [
                    {
                        "id": "d:2:1",
                        "quote": "Step 1: Send the welcome email using Template 1.",
                        "source": {"document": "d", "section": "2"},
                    }
                ],
                "exceptions": [
                    {"when": "start postponed", "then": "wait", "quote": "If the start is postponed, wait."}
                ],
            },
            {
                "id": "d:3",
                "name": "Offboarding",
                "source": {"document": "d", "section": "3"},
                "steps": [
                    {
                        "id": "d:3:1",
                        "quote": "Step 1: Revoke access after written authorization.",
                        "source": {"document": "d", "section": "3"},
                    }
                ],
            },
        ],
        "links": [{"type": "refers_to", "src": "d:2:1", "dst": "d:9"}],
    },
    "policy_set": {
        "documents": [
            {
                "id": "d",
                "sections": [
                    {
                        "id": "2",
                        "title": "Onboarding",
                        "parent": None,
                        "statements": [
                            {
                                "id": "d:2:1",
                                "text": "Step 1: Send the welcome email using Template 1.",
                                "source": {"document": "d", "section": "2"},
                            }
                        ],
                    },
                    {
                        "id": "9",
                        "title": "Email rules",
                        "parent": None,
                        "statements": [
                            {
                                "id": "d:9:1",
                                "text": "All emails use the subject format [HR] <name>.",
                                "source": {"document": "d", "section": "9"},
                            }
                        ],
                    },
                    {
                        "id": "t1",
                        "title": "Template 1: Welcome email",
                        "parent": None,
                        "statements": [
                            {
                                "id": "d:t1:1",
                                "text": "Dear [Name], welcome to the team.",
                                "source": {"document": "d", "section": "t1"},
                            }
                        ],
                    },
                ],
            }
        ]
    },
}


def test_retrieval_picks_the_procedure_and_expands_links_and_templates():
    from polisys.sop_retrieval import SopIndex

    idx = SopIndex(BUILD)
    p = idx.choose("Onboard Ann", "new hire onboarding, welcome email")
    assert p["name"] == "Onboarding"
    text = idx.checklist(p, "Onboard Ann")
    assert "Dear [Name], welcome to the team." in text  # template text by name
    assert "subject format [HR]" in text  # one link away
    assert "If the start is postponed, wait." in text


def test_discovery_mode_shows_checklists_in_one_turn_without_decisions():
    from polisys.sop_retrieval import SopIndex

    agent, rec = Agent(), LoopHooks()
    rec.after_tool_call("read_sheet", {}, {"rows": [{"name": "Ann"}, {"name": "Bo"}]})
    loop = SopLoop(Model("x"), BUILD["build"]["procedures"], mode="discovery", index=SopIndex(BUILD))
    asyncio.run(loop.run(agent, "Onboard the new hires.", "t1", rec, 60))
    assert len(agent.messages) == 3  # explore, one discovery turn, finish
    assert "Template" in agent.messages[1] and "Onboarding" in agent.messages[1]
    assert "decisions stand" not in agent.messages[2]
