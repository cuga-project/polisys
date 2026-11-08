"""SOP loop: a general rule that applies to a case is a rule to keep while doing the work; it never stops the case."""

import asyncio
import json

from polisys.sop_loop import LoopHooks, SopLoop

PROC = {
    "id": "p1",
    "name": "Onboarding",
    "steps": [{"id": "s1", "action": "send_email", "quote": "Step 1: Send the welcome email."}],
}
GENERAL = {
    "id": "g",
    "name": "General rules",
    "exceptions": [{"when": "the message is about a case", "then": "do not use direct messages", "quote": "..."}],
}


class Model:
    def __call__(self, system, user):
        if "Which one procedure" in user:
            return json.dumps({"procedure": 1, "evidence": "new hire"})
        if "General SOP rules" in user:
            return json.dumps({"apply": [{"n": 1, "evidence": "please start onboarding for Ann"}]})
        if "already done" in user:
            return json.dumps({"answer": "no", "evidence": ""})
        return json.dumps({"cases": [{"case": "Onboard Ann", "keys": ["Ann"]}]})


class Agent:
    def __init__(self):
        self.messages = []

    async def invoke(self, msg, thread_id=None):
        self.messages.append(msg)

        class R:
            receipt = None

        return R()


def test_general_rule_is_kept_as_a_rule_and_the_procedure_still_runs():
    agent, rec = Agent(), LoopHooks()
    rec.after_tool_call("read_mail", {}, {"body": "Hi HR, please start onboarding for Ann on 2026-04-20."})
    loop = SopLoop(Model(), [PROC, GENERAL], mode="control", require_evidence=True)
    asyncio.run(loop.run(agent, "Process the new hire emails.", "t1", rec, 60))
    case = next(e for e in loop.log if e.get("phase") == "case")
    assert case["decision"] == "follow" and not case["reasons"]
    assert case["constraints"] and "do not use direct messages" in case["constraints"][0]
    turn = next(m for m in agent.messages if "Case 1 of 1" in m)
    assert "FOLLOW the procedure" in turn and "do not use direct messages" in turn and "Steps:" in turn
    assert not rec.held
