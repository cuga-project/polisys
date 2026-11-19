"""The quoted-value guard, the policy's default values, the run ledger, value coverage and policy lines: what code decides."""

import json

from polisys.experimental.checks.default_values import DefaultValueHooks, build_defaults, check_defaults
from polisys.experimental.guidance.work_method_notes import LiteralValueHooks, altered_quote, quoted_values

REQUEST = "Open a ticket for Ada's team. Use the title 'Pump 7: seal worn, replace before Friday' and add the note \"Parts are in bay 3.\""
WRITE = lambda name: name.startswith(("create_", "update_", "add_"))  # noqa: E731


def test_quoted_values_are_the_texts_between_quotation_marks_and_an_apostrophe_opens_none():
    assert quoted_values(REQUEST) == ["Pump 7: seal worn, replace before Friday", "Parts are in bay 3."]
    assert quoted_values("Set it to 'short'.") == []  # under the length limit


def test_a_changed_copy_differs_only_at_its_ends_and_an_exact_copy_or_other_text_is_left_alone():
    q = quoted_values(REQUEST)
    assert altered_quote("Pump 7: seal worn, replace before Friday.", q) == q[0]
    assert altered_quote(" Parts are in bay 3", q) == q[1]
    assert altered_quote(q[0], q) is None
    assert altered_quote("Pump 7: seal worn", q) is None and altered_quote("pump 7: seal worn, replace before friday", q) is None
    assert altered_quote(7, q) is None


def test_the_guard_blocks_a_write_once_per_argument_and_never_a_read():
    hook = LiteralValueHooks(REQUEST, WRITE)
    bad = {"title": "Pump 7: seal worn, replace before Friday.", "team": "Ada"}
    assert hook.before_tool_call("find_ticket", bad) is None
    first = hook.before_tool_call("create_ticket", bad)
    assert first and "'Pump 7: seal worn, replace before Friday'" in first.message
    assert hook.before_tool_call("create_ticket", bad) is None  # the same call again: the agent may have a reason
    other = hook.before_tool_call("create_ticket", {**bad, "team": "Bo"})
    assert other is not None  # another call of the same turn with the same changed text is blocked too
    assert hook.before_tool_call("create_ticket", {"title": "Pump 7: seal worn, replace before Friday"}) is None
    assert [r["outcome"] for r in hook.records] == ["literal_value_block"] * 2


POLICY = "Tickets. Urgency: if the request gives none, the default is normal. Titles are 30 to 120 characters long."
SPECS = [
    {"name": "create_ticket", "description": "Create.", "inputSchema": {"properties": {"urgency": {"enum": ["low", "normal", "high"]}, "title": {}}}},
    {"name": "find_ticket", "description": "Find.", "inputSchema": {"properties": {"urgency": {}}}},
]
LINE = "Urgency: if the request gives none, the default is normal."
GOOD = {"tool": "create_ticket", "argument": "urgency", "value": "normal", "quote": LINE}


def test_a_default_is_kept_only_for_a_write_tool_its_argument_a_quoted_line_and_a_value_the_tool_accepts():
    raw = [
        GOOD,
        {**GOOD, "tool": "find_ticket"},
        {**GOOD, "argument": "owner"},
        {**GOOD, "quote": "The default urgency is normal."},
        {**GOOD, "value": "medium"},
        {**GOOD, "argument": "title", "value": "standard"},
    ]
    kept, dropped = check_defaults(raw, POLICY, SPECS, WRITE)
    assert kept == [GOOD] and len(dropped) == 5
    title = {"tool": "create_ticket", "argument": "title", "quote": "Titles are 30 to 120 characters long."}
    both, _ = check_defaults([{**title, "value": "30"}, {**title, "value": "120"}, GOOD], POLICY, SPECS, WRITE)
    assert both == [GOOD]  # two defaults for one argument: neither is enforced


def test_a_call_without_the_argument_is_blocked_once_with_the_default_and_a_given_value_passes(tmp_path):
    calls = []

    def model(system, user):
        calls.append(user)
        return json.dumps({"defaults": [GOOD]})

    hook = DefaultValueHooks(lambda: build_defaults(POLICY, SPECS, model, WRITE, str(tmp_path)))
    assert hook.before_tool_call("create_ticket", {"title": "x", "urgency": "high"}) is None
    block = hook.before_tool_call("create_ticket", {"title": "x"})
    assert block and "urgency=normal" in block.message and LINE in block.message
    assert hook.before_tool_call("create_ticket", {"title": "x"}) is None
    assert hook.before_tool_call("create_ticket", {"title": "y"}) is not None  # another call with the same gap
    assert "find_ticket" not in calls[0] and len(calls) == 1
    again = DefaultValueHooks(lambda: build_defaults(POLICY, SPECS, model, WRITE, str(tmp_path)))
    assert again.before_tool_call("create_ticket", {}) and len(calls) == 1  # the second run reads the cache


from polisys.experimental.guidance.policy_lines import HEADER as LINES_HEADER, PolicyLines, build_lines, check_lines  # noqa: E402
from polisys.experimental.guidance.work_method_notes import FinishLedgerHooks, LiteralCoverageHooks, request_literals  # noqa: E402


def test_the_ledger_lists_each_write_with_its_outcome_and_counts_the_reads():
    hook = FinishLedgerHooks("policy", WRITE)
    hook.after_tool_call("find_ticket", {"q": "pump"}, {"error": "none found"})
    hook.after_tool_call("create_ticket", {"title": "Pump 7"}, {"id": 4})
    hook.after_tool_call("update_ticket", {"id": 9}, {"error": "no such ticket"})
    sent = hook.before_reply("done").message
    assert '- create_ticket({"title": "Pump 7"}) -> ok' in sent and "- update_ticket" in sent and "FAILED: no such ticket" in sent
    assert "Read calls: 1, of which 1 failed." in sent and "find_ticket(" not in sent
    assert hook.before_reply("done") is None


def test_the_values_of_the_request_are_addresses_codes_and_quoted_texts():
    text = "Move HRX0000714 to ada@plant.example (account AB-789-012, type STANDARD) with the note 'Parts are in bay 3 today'. In 2026."
    assert request_literals(text) == ["ada@plant.example", "HRX0000714", "AB-789-012", "Parts are in bay 3 today"]


def test_a_value_of_the_request_seen_in_no_call_and_no_result_is_named_once():
    hook = LiteralCoverageHooks("Close TK-0042 and TK-0077 and tell ada@plant.example.")
    hook.after_tool_call("find_ticket", {"code": "tk-0042"}, {"owner": "ADA@plant.example"})
    sent = hook.before_reply("done")
    assert sent and "'TK-0077'" in sent.message and "TK-0042" not in sent.message and "ada@" not in sent.message
    assert hook.before_reply("done") is None
    full = LiteralCoverageHooks("Close TK-0042.")
    full.after_tool_call("update_ticket", {"code": "TK-0042"}, {"ok": True})
    assert full.before_reply("done") is None and full.records[0]["missing"] == 0


def test_a_policy_line_is_kept_only_word_for_word_and_for_a_write_tool_and_goes_into_its_description():
    raw = [
        {"tool": "create_ticket", "quote": LINE},
        {"tool": "create_ticket", "quote": "Urgency defaults to normal."},
        {"tool": "find_ticket", "quote": LINE},
        {"tool": "create_ticket", "quote": LINE},
    ]
    assert check_lines(raw, POLICY, SPECS, WRITE) == {"create_ticket": [LINE]}
    from langchain_core.tools import StructuredTool

    tools = [StructuredTool.from_function(func=lambda **k: "", name=s["name"], description=s["description"]) for s in SPECS]
    lines = PolicyLines(lambda: build_lines(POLICY, SPECS, lambda s, u: json.dumps({"lines": raw}), WRITE))
    create, find = lines.wrap_tools(tools)
    assert LINES_HEADER in create.description and f'- "{LINE}"' in create.description and find.description == "Find."
    assert lines.records[-1]["lines"] == 1


WHEN_POLICY = POLICY + " When a ticket gets an owner, its urgency must be high."
WHEN_SPECS = [{**SPECS[0], "inputSchema": {"properties": {**SPECS[0]["inputSchema"]["properties"], "owner": {}}}}, SPECS[1]]
WHEN = {"tool": "create_ticket", "argument": "urgency", "value": "high", "kind": "required_when", "when_argument": "owner",
        "quote": "When a ticket gets an owner, its urgency must be high."}


def test_a_value_required_with_another_argument_is_kept_only_in_scope_stated_and_with_that_argument():
    kept, _ = check_defaults([WHEN, GOOD], WHEN_POLICY, WHEN_SPECS, WRITE, stated=True)
    assert [k.get("kind") for k in kept] == ["required_when", None]
    assert check_defaults([WHEN], WHEN_POLICY, WHEN_SPECS, WRITE)[0] == []  # scope defaults: as before
    assert check_defaults([{**WHEN, "when_argument": "team"}], WHEN_POLICY, WHEN_SPECS, WRITE, stated=True)[0] == []


def test_the_required_value_is_asked_for_only_when_the_other_argument_is_set_and_the_value_differs():
    data = {"defaults": check_defaults([WHEN], WHEN_POLICY, WHEN_SPECS, WRITE, stated=True)[0]}
    hook = DefaultValueHooks(lambda: data)
    assert hook.before_tool_call("create_ticket", {"title": "x", "urgency": "low"}) is None  # no owner in the call
    assert hook.before_tool_call("create_ticket", {"title": "x", "owner": "ada", "urgency": "High"}) is None
    block = hook.before_tool_call("create_ticket", {"title": "x", "owner": "ada", "urgency": "low"})
    assert block and "urgency=high" in block.message and "sets owner" in block.message
    assert hook.records[-1]["outcome"] == "required_value_block"
