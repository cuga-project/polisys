"""notes.tool_notes: notes on each tool from the policy text, checked in code, cached, and shown in the tool descriptions."""

import json

from langchain_core.tools import StructuredTool

from polisys.tool_notes import HEADER, ToolNotes, build_notes, check_notes, describe, note_kind, quoted_in
from tests.test_runtime_assembly import build, config
from polisys.runtime import RunContext

POLICY = (
    "A shipment may be released only to a carrier that is on the approved list (if it is not, add it to the list first).\n"
    "**Label:** when the request gives no size, the default size is `medium`.\n"
    "After a shipment is released, the customer must be told."
)
SPECS = [
    {
        "name": "release_shipment",
        "description": "Release.",
        "inputSchema": {"properties": {"carrier": {}, "size": {"enum": ["small", "medium", "pallet"]}}},
    },
    {"name": "add_carrier", "description": "Add a carrier.", "inputSchema": {"properties": {"name": {}}}},
    {"name": "find_carrier", "description": "Find a carrier.", "inputSchema": {"properties": {"name": {}}}},
]
GOOD = {
    "tool": "release_shipment",
    "kind": "before",
    "text": "Check that the carrier is on the approved list. If it is not, add it with add_carrier first.",
    "quote": "released only to a carrier that is on the approved list",
    "fix_tool": "add_carrier",
}
SIZE = {
    "tool": "release_shipment",
    "kind": "values",
    "text": "Pass size as medium when the request gives none.",
    "quote": "the default size is `medium`",
    "argument": "size",
}
TELL = {
    "tool": "release_shipment",
    "kind": "after",
    "text": "After a release, tell the customer with add_carrier.",
    "quote": "After a shipment is released, the customer must be told.",
    "then_tool": "add_carrier",
}


def fake_model(notes, calls=None):
    def call(system, user):
        if calls is not None:
            calls.append(user)
        return json.dumps({"notes": notes})

    call.model_name = "m/notes"
    call.usage = {"calls": 0}
    return call


def test_a_quote_is_matched_word_for_word_apart_from_case_spacing_and_markdown():
    assert quoted_in("the default size is medium", POLICY)  # the policy has `medium` in backticks
    assert quoted_in("Label: when the request gives no size", POLICY)
    assert not quoted_in("the default size is large", POLICY)
    assert not quoted_in("medium", POLICY)  # too short to be evidence


def test_a_note_is_dropped_for_an_unknown_tool_a_quote_not_in_the_policy_an_unknown_kind_or_a_long_text():
    raw = [
        GOOD,
        {**GOOD, "tool": "delete_everything"},
        {**GOOD, "quote": "a carrier must be insured"},
        {**GOOD, "kind": "always"},
        {**GOOD, "text": "x" * 500},
    ]
    kept, dropped = check_notes(raw, POLICY, SPECS)
    assert [n.text for n in kept] == [GOOD["text"]]
    assert [d["why"] for d in dropped] == [
        "tool not in the run",
        "quote not in the policy text",
        "unknown kind",
        "no text or too long",
    ]


def test_a_note_only_helps_each_kind_is_kept_only_with_the_tool_or_argument_it_points_to():
    raw = [
        GOOD,
        SIZE,
        TELL,
        {**GOOD, "fix_tool": ""},  # a condition with nothing that fixes it is a restriction: the policy text has it already
        {**GOOD, "fix_tool": "call_the_manager"},
        {**GOOD, "fix_tool": "find_carrier"},  # a tool that only reads checks the condition; it cannot make it true
        {**SIZE, "argument": "weight"},
        {**TELL, "then_tool": "send_fax"},
    ]
    kept, dropped = check_notes(raw, POLICY, SPECS)
    assert [(n.kind, n.ref) for n in kept] == [("before", "add_carrier"), ("values", "size"), ("after", "add_carrier")]
    assert [d["why"] for d in dropped] == [
        "no write tool of the run fixes the condition",
        "no write tool of the run fixes the condition",
        "no write tool of the run fixes the condition",
        "not an argument of this tool",
        "no tool of the run does the next step",
    ]


def test_a_remedy_the_policy_does_not_state_is_kept_only_when_the_deployment_says_fix_with_tool():
    inferred = {**GOOD, "inferred": True}
    assert check_notes([inferred], POLICY, SPECS)[0] == []
    assert len(check_notes([inferred], POLICY, SPECS, "fix_with_tool")[0]) == 1


def test_a_conflict_of_lists_is_written_by_code_and_only_when_the_tools_list_is_in_force():
    policy = POLICY + "\nAllowed sizes: small, medium."
    conflict = {"tool": "release_shipment", "argument": "size", "quote": "Allowed sizes: small, medium."}
    stands = check_notes([], policy, SPECS, conflicts=[conflict])
    assert stands == ([], [{"tool": "release_shipment", "why": "the policy's list stands"}])
    kept, _ = check_notes([], policy, SPECS, value_lists="tool", conflicts=[conflict])
    assert kept[0].conflict and kept[0].kind == "values" and "small, medium, pallet" in kept[0].text


def test_a_conflict_needs_a_tool_list_a_quote_from_the_policy_and_a_tool_value_the_quote_lacks():
    policy = POLICY + "\nAllowed sizes: small, medium.\nSizes are small, medium and pallet."

    def why(item):
        return check_notes([], policy, SPECS, value_lists="tool", conflicts=[item])[1][0]["why"]

    base = {"tool": "release_shipment", "argument": "size", "quote": "Allowed sizes: small, medium."}
    assert why({**base, "argument": "carrier"}) == "the tool has no list for this argument"
    assert why({**base, "quote": "Allowed sizes: tiny."}) == "quote not in the policy text"
    same = {**base, "quote": "Sizes are small, medium and pallet."}
    assert why(same) == "the quoted policy list holds every value of the tool"


def test_the_notes_are_cached_by_policy_text_and_tool_set(tmp_path):
    calls = []
    first = build_notes(POLICY, SPECS, fake_model([GOOD], calls), cache_dir=str(tmp_path))
    second = build_notes(POLICY, SPECS, fake_model([], calls), cache_dir=str(tmp_path))
    assert len(calls) == 1 and not first["cached"] and second["cached"]
    assert second["notes"] == first["notes"]
    other = build_notes(POLICY + " More.", SPECS, fake_model([], calls), cache_dir=str(tmp_path))
    assert len(calls) == 2 and other["notes"] == []


def test_the_notes_go_into_the_description_of_their_tool_only():
    tools = [StructuredTool.from_function(func=lambda **k: "", name=s["name"], description=s["description"]) for s in SPECS]
    notes = ToolNotes(lambda: build_notes(POLICY, SPECS, fake_model([GOOD])))
    release, add, _find = notes.wrap_tools(tools)
    assert HEADER in release.description and "Before calling: Check that the carrier" in release.description
    assert release.description.startswith("Release.") and add.description == "Add a carrier."
    assert tools[0].description == "Release."  # the benchmark's own tool is not changed
    assert notes.records[0]["outcome"] == "notes_shown" and notes.records[0]["tools"] == 1


def test_describe_orders_the_kinds_and_cuts_at_the_limit():
    many = [{**TELL, "text": "A" * 300}] * 8 + [{**SIZE, "text": "first"}]
    text = describe(many)["release_shipment"]
    assert text.index("Values: first") < text.index("After it returns")
    assert len(text) <= 1400


def test_show_keeps_only_the_kinds_asked_for_and_a_conflict_is_a_kind_of_its_own():
    conflict = {**SIZE, "text": "The tool's list is the one in force.", "conflict": True}
    notes = [TELL, SIZE, conflict]
    assert [note_kind(n) for n in notes] == [TELL["kind"], "values", "conflict"]
    assert describe(notes) == describe(notes, ["before", "values", "after", "conflict"])  # the default shows every kind
    only = describe(notes, ["conflict"])["release_shipment"]
    assert "in force" in only and SIZE["text"] not in only and TELL["text"] not in only
    assert describe(notes, ["before"]) == {}  # no note of that kind: the tool keeps its own description


def test_the_record_counts_the_notes_shown_and_the_notes_written():
    tools = [StructuredTool.from_function(func=lambda **k: "", name=s["name"], description=s["description"]) for s in SPECS]
    notes = ToolNotes(lambda: build_notes(POLICY, SPECS, fake_model([GOOD])), ["values"])
    notes.wrap_tools(tools)
    rec = notes.records[0]
    assert rec["written"] >= rec["notes"] and rec["notes"] == sum(1 for n in notes.data["notes"] if note_kind(n) == "values")


def test_an_empty_show_list_is_refused_when_the_notes_are_on():
    import pytest

    with pytest.raises(Exception, match="no kind of note"):
        build(config(notes={"tool_notes": {"mode": "on", "model": "agent", "show": []}}), ctx=CTX)


CTX = RunContext(task_id="t", request="Release it.", system_prompt=POLICY, tool_specs=SPECS, agent_model="m/agent")


def test_the_runtime_fills_wrap_tools_only_when_the_setting_is_on_and_wiring_asks_no_model():
    off, _ = build(config(), ctx=CTX)
    assert off.wrap_tools is None
    on, models = build(config(notes={"tool_notes": {"mode": "on", "model": "agent"}}), ctx=CTX)
    assert on.wrap_tools is not None
    assert [m["model"] for m in models.made] == ["m/agent"]  # "agent" is the agent's model; it is not called at wiring


def test_an_answer_with_no_list_of_notes_is_asked_again_and_is_never_cached_as_empty(tmp_path):
    import pytest

    from polisys.tool_notes import NotesError

    answers = ['{"notes": [{"tool": "release_shipment", "kind": "before", "te', json.dumps({"notes": [GOOD]})]

    def cut_then_whole(system, user):
        return answers.pop(0)

    cut_then_whole.model_name = "m/notes"
    data = build_notes(POLICY, SPECS, cut_then_whole, cache_dir=str(tmp_path))
    assert len(data["notes"]) == 1 and answers == []

    def always_cut(system, user):
        return '{"notes": [{"tool": "rel'

    always_cut.model_name = "m/other"
    with pytest.raises(NotesError):
        build_notes(POLICY, SPECS, always_cut, cache_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*.json"))) == 1  # only the good answer is in the cache


def test_a_run_whose_notes_fail_goes_on_without_notes_and_says_so():
    def broken():
        raise RuntimeError("no model")

    tools = [StructuredTool.from_function(func=lambda **k: "", name="release_shipment", description="Release.")]
    notes = ToolNotes(broken)
    assert notes.wrap_tools(tools)[0].description == "Release."
    assert [r["outcome"] for r in notes.records] == ["notes_failed", "notes_shown"]


def test_the_tool_list_of_a_prompt_gives_each_argument_with_its_allowed_values():
    from polisys.tool_notes import tool_lines

    spec = {
        "name": "set_size",
        "description": "Set the size.",
        "inputSchema": {"properties": {"size": {"description": "Label size", "enum": ["small", "medium"]}}, "required": ["size"]},
    }
    assert tool_lines([spec]) == "- set_size: Set the size.\n    size (required): Label size; allowed: small, medium"

