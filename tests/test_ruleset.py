"""Typed rule set: builder code steps and RuleSetHooks checks, without a model."""

from polisys.builder import build, split_policy, validate
from polisys.ruleset import RuleSetHooks

POLICY = """# Policy
Never assume or fabricate IDs. The same is the case of optional or default arguments.
## **1. Tickets**
* **Mandatory Inputs:** `caller_id` and `impact`.
| Impact | Urgency | Priority |
| --- | --- | --- |
| low | low | low |
| high | high | critical |
| high | low | moderate |
| low | high | moderate |
* `new` → `open` / `done`
* `open` → `done`
## **2. Lists**
| Table.Field | Allowed Values |
| :--- | :--- |
| **ticket.status** | `new`, `open`, `done` |
"""


def _spec(name, props, required=()):
    return {
        "name": name,
        "inputSchema": {
            "type": "object",
            "required": list(required),
            "properties": {
                p: ({"type": "string", "enum": e} if e else {"type": "string"}) for p, e in props.items()
            },
        },
    }


SPECS = [
    _spec(
        "create_ticket",
        {
            "caller_id": None,
            "impact": None,
            "urgency": None,
            "priority": None,
            "status": ["new", "open", "done", "x"],
        },
        ["caller_id"],
    ),
    _spec(
        "update_ticket",
        {"ticket_id": None, "status": ["new", "open", "done", "x"], "urgency": None},
        ["ticket_id"],
    ),
    _spec("get_ticket", {"ticket_id": None}, ["ticket_id"]),
]


def test_split_keeps_preamble_and_sections():
    assert [s.sid for s in split_policy(POLICY)] == ["preamble", "1", "2"]


def test_code_rules_from_tables_and_transitions():
    res = build(POLICY, SPECS, model=None)
    types = res.stats["by_type"]
    assert types.get("derived_value") == 1 and types.get("transition") == 1
    assert types.get("allowed_values") == 2  # policy list is narrower than the schema enum ("x")
    trans = next(r for r in res.rules if r["type"] == "transition")
    assert trans["tool"] == "update_ticket" and trans["id_arg"] == "ticket_id"


def test_validate_rejects_unknown_names_and_loose_quotes():
    by = {s["name"]: s for s in SPECS}
    stmt = "Mandatory Inputs: caller_id and impact."
    ok = {
        "type": "required_args",
        "tool": "create_ticket",
        "args": ["caller_id", "impact"],
        "quote": "caller_id and impact",
    }
    assert validate(ok, by, stmt) is None
    assert "unknown argument" in validate({**ok, "args": ["nope"]}, by, stmt)
    assert "unknown tool" in validate({**ok, "tool": "make_ticket"}, by, stmt)
    assert "verbatim" in validate({**ok, "quote": "callers and impacts"}, by, stmt)


def _hooks(extra=(), role="agent"):
    rules = build(POLICY, SPECS, model=None).rules + [
        {"id": "g1", "type": "grounded_values", "source": "preamble", "may_deny": True},
        {"id": "g2", "type": "id_provenance", "source": "preamble", "may_deny": True},
        {
            "id": "req",
            "type": "required_args",
            "tool": "create_ticket",
            "args": ["caller_id", "impact"],
            "source": "§1",
        },
        *extra,
    ]
    h = RuleSetHooks(rules, SPECS, principal={"id": "U_1", "role": role})
    h.on_user_message("Open a ticket for USER_7 with high impact and high urgency", None)
    return h


def test_grounding_ids_required_and_matrix():
    h = _hooks()
    assert h.before_tool_call("get_ticket", {"ticket_id": "TCK_9"}).message is None  # reads are not checked
    msg = h.before_tool_call(
        "create_ticket", {"caller_id": "USER_7", "impact": "high", "urgency": "low"}
    ).message
    assert "urgency=low" in msg  # not in the request: assumed value
    msg = h.before_tool_call("create_ticket", {"caller_id": "USER_8", "impact": "high"}).message
    assert "USER_8" in msg  # ID not seen anywhere
    msg = h.before_tool_call("create_ticket", {"caller_id": "USER_7", "urgency": "high"}).message
    assert "impact" in msg  # mandatory field missing
    msg = h.before_tool_call(
        "create_ticket", {"caller_id": "USER_7", "impact": "high", "urgency": "high", "priority": "moderate"}
    ).message
    assert "priority" in msg  # matrix says critical
    assert (
        h.before_tool_call(
            "create_ticket", {"caller_id": "USER_7", "impact": "high", "urgency": "high"}
        ).message
        is None
    )


def test_transition_uses_known_state():
    h = _hooks()
    h.after_tool_call("get_ticket", {"ticket_id": "TCK_1"}, {"ticket_id": "TCK_1", "status": "open"})
    assert (
        "cannot go from open to new"
        in h.before_tool_call("update_ticket", {"ticket_id": "TCK_1", "status": "new"}).message
    )
    assert h.before_tool_call("update_ticket", {"ticket_id": "TCK_1", "status": "done"}).message is None


def test_role_rule_via_cedar_and_duty_followup():
    role_rule = {
        "id": "role1",
        "type": "role_permission",
        "tools": ["update_ticket"],
        "roles": ["reporter"],
        "effect": "forbid",
        "source": "§2",
    }
    duty = {
        "id": "d1",
        "type": "duty",
        "trigger": {"tool": "update_ticket", "when": {"status": ["done"]}},
        "then": {"tool": "create_ticket"},
        "source": "§1",
    }
    h = _hooks([role_rule], role="reporter")
    h.after_tool_call("get_ticket", {"ticket_id": "TCK_1"}, {"ticket_id": "TCK_1", "status": "open"})
    assert (
        "role `reporter`"
        in h.before_tool_call("update_ticket", {"ticket_id": "TCK_1", "status": "done"}).message
    )
    h = _hooks([duty])
    h.after_tool_call("get_ticket", {"ticket_id": "TCK_1"}, {"ticket_id": "TCK_1", "status": "open"})
    h.after_tool_call("update_ticket", {"ticket_id": "TCK_1", "status": "done"}, {"ok": True})
    assert "still open" in h.before_reply("done").message
    h.after_tool_call("create_ticket", {"caller_id": "USER_7"}, {"ticket_id": "TCK_2"})
    assert h.before_reply("done").message is None


def test_cedar_role_rule_lets_a_permitted_role_through():
    role_rule = {
        "id": "role1",
        "type": "role_permission",
        "tools": ["update_ticket"],
        "roles": ["reporter"],
        "effect": "forbid",
        "source": "§2",
    }
    call = ("update_ticket", {"ticket_id": "TCK_1", "status": "done"})
    for role, allowed in (("manager", True), ("agent", True), ("reporter", False)):
        h = _hooks([role_rule], role=role)
        assert h.cedar is not None  # Cedar decides here; without cedarpy the call would pass for every role
        h.after_tool_call("get_ticket", {"ticket_id": "TCK_1"}, {"ticket_id": "TCK_1", "status": "open"})
        message = h.before_tool_call(*call).message
        assert (message is None) == allowed, (role, message)
    other = _hooks([role_rule], role="reporter")  # the rule names update_ticket only
    assert (
        other.before_tool_call(
            "create_ticket", {"caller_id": "USER_7", "impact": "high", "urgency": "high"}
        ).message
        is None
    )


def test_repeated_denial_cap():
    h = _hooks()
    calls = [
        h.before_tool_call("create_ticket", {"caller_id": "USER_8", "impact": "high"}).message
        for _ in range(4)
    ]
    assert calls[2] is not None and calls[3] is None


def test_duty_not_opened_when_its_tool_is_missing():
    duty = {
        "id": "d1",
        "type": "duty",
        "trigger": {"tool": "update_ticket"},
        "then": {"tool": "notify_owner"},
        "source": "§1",
    }
    h = _hooks([duty])
    h.after_tool_call("update_ticket", {"ticket_id": "TCK_1", "status": "done"}, {"ok": True})
    assert h.before_reply("done").message is None


def test_no_assumed_data_rules_are_observed_unless_measured():
    from polisys.ruleset import RuleSetHooks

    rules = [{"id": "g1", "type": "grounded_values", "source": "preamble"}]
    h = RuleSetHooks(rules, SPECS)
    h.on_user_message("Create a ticket for USER_7 with high impact.", None)
    assert (
        h.before_tool_call(
            "create_ticket", {"caller_id": "USER_7", "impact": "high", "urgency": "low"}
        ).message
        is None
    )
    assert any(r["outcome"] == "observe" and r["type"] == "grounded_values" for r in h.records)
    assert h.before_llm_call(None).message is None  # recorded for audit, not shown to the agent
    measured = RuleSetHooks(rules, SPECS, method_choice={"grounded_values": {"code_may_deny": True}})
    measured.on_user_message("Create a ticket for USER_7 with high impact.", None)
    assert (
        "urgency=low"
        in measured.before_tool_call(
            "create_ticket", {"caller_id": "USER_7", "impact": "high", "urgency": "low"}
        ).message
    )


def test_unicode_dashes_do_not_split_values_or_patterns():
    from polisys.ruleset import RuleSetHooks

    specs = [
        {"name": "gmail_send_email", "inputSchema": {"properties": {"to": {"type": "string"}}}},
        {"name": "slack_send_channel_message", "inputSchema": {"properties": {"text": {"type": "string"}}}},
    ]
    rules = [
        {
            "id": "a1",
            "type": "allowed_values",
            "tool": "gmail_send_email",
            "arg": "to",
            "values": ["product‑voc@company.example.com"],
            "source": "§1",
        },
        {
            "id": "a2",
            "type": "allowed_values",
            "tool": "gmail_send_email",
            "arg": "to",
            "values": ["product-voc@company.example.com"],
            "source": "§1",
        },
        {
            "id": "f1",
            "type": "format",
            "tool": "slack_send_channel_message",
            "arg": "text",
            "pattern": "VOC‑XREF‑W\\d+‑\\d{4}",
            "source": "§1",
        },
    ]
    h = RuleSetHooks(rules, specs)
    assert h.before_tool_call("gmail_send_email", {"to": "product-voc@company.example.com"}).message is None
    assert h.before_tool_call("gmail_send_email", {"to": "product‑voc@company.example.com"}).message is None
    assert (
        h.before_tool_call(
            "slack_send_channel_message", {"text": "Summary. Tracking Code: VOC-XREF-W4-2026"}
        ).message
        is None
    )


def test_policy_tool_conflict_is_observed_not_shown():
    from polisys.ruleset import RuleSetHooks

    rule = {
        "id": "c1",
        "type": "allowed_values",
        "tool": "create_ticket",
        "arg": "urgency",
        "values": ["p1", "p2"],
        "schema_conflict": True,
        "may_deny": False,
        "source": "§3",
    }
    h = RuleSetHooks([rule], SPECS)
    assert h.before_tool_call("create_ticket", {"caller_id": "USER_7", "urgency": "high"}).message is None
    assert h.records[-1]["outcome"] == "observe" and h.before_llm_call(None).message is None


def _spec(name, props):
    return {
        "name": name,
        "description": "",
        "input_schema": {"type": "object", "properties": {k: {"type": "string"} for k in props}},
    }


def test_membership_denies_only_when_the_group_members_were_read():
    rules = [
        {"id": "m1", "type": "membership", "tool": "update_case", "arg": "assigned_to", "group_arg": "group"}
    ]
    h = RuleSetHooks(
        rules,
        [_spec("update_case", ["case_id", "assigned_to", "group"])],
        principal={"id": "u", "role": "agent"},
    )
    assert not h.before_tool_call(
        "update_case", {"case_id": "C1", "assigned_to": "U9", "group": "G1"}
    ).message  # not read
    h.after_tool_call(
        "list_group_members",
        {"group": "G1"},
        [{"group_id": "G1", "user_id": "U1"}, {"group_id": "G1", "user_id": "U2"}],
    )
    res = h.before_tool_call("update_case", {"case_id": "C1", "assigned_to": "U9", "group": "G1"})
    assert res.message and "do not include U9" in res.message
    assert not h.before_tool_call(
        "update_case", {"case_id": "C1", "assigned_to": "U2", "group": "G1"}
    ).message


def test_when_star_means_the_argument_is_given():
    rules = [
        {
            "id": "a1",
            "type": "allowed_values",
            "tool": "update_problem",
            "arg": "status",
            "values": ["fix_in_progress"],
            "when": {"workaround": ["*"]},
        }
    ]
    h = RuleSetHooks(
        rules,
        [_spec("update_problem", ["problem_id", "status", "workaround"])],
        principal={"id": "u", "role": "agent"},
    )
    assert h.before_tool_call(
        "update_problem", {"problem_id": "P1", "status": "new", "workaround": "restart"}
    ).message
    assert not h.before_tool_call("update_problem", {"problem_id": "P1", "status": "new"}).message


# -- dates in a record condition ------------------------------------------------------------------------------------------


def test_two_iso_dates_are_compared_as_dates_for_each_operator():
    from polisys.ruleset import _condition

    cases = [  # (record value, op, rule value, result)
        ("2025-12-31", "le", "2025-02-25", False),  # the first number says 2025 <= 2025
        ("2025-02-24", "ge", "2025-02-25", False),
        ("2025-02-24", "lt", "2025-02-25", True),
        ("2025-02-25", "lt", "2025-02-25", False),
        ("2025-02-25", "le", "2025-02-25", True),
        ("2025-02-25", "ge", "2025-02-25", True),
        ("2025-02-26", "gt", "2025-02-25", True),
        ("2025-02-25", "gt", "2025-02-25", False),
        ("2024-12-31", "lt", "2025-02-25", True),
        ("2026-06-30", "gt", "2025-02-25", True),
        ("2025-02-25T12:08:00-05:00", "lt", "2025-02-25", False),  # 17:08 UTC is after midnight
        ("2025-02-25T12:08:00-05:00", "gt", "2025-02-25T12:00:00Z", True),
        ("2025-02-25 12:08", "gt", "2025-02-25", True),
    ]
    for actual, op, value, expected in cases:
        assert _condition(actual, op, value) is expected, (actual, op, value)


def test_numbers_and_other_forms_keep_the_first_number_reading():
    from polisys.ruleset import _condition

    assert _condition(5, "gt", 3) and not _condition("5", "lt", "3") and _condition("1,200", "gt", 1000)
    assert _condition("02/25/2025", "lt", "03/01/2025")  # not ISO: the first number, 2 < 3, as before
    assert not _condition(
        "2025-02-24", "le", "Feb 25 2025"
    )  # a mixed pair: the first numbers, 2025 <= 25, as before
    assert not _condition(
        "2025-13-45", "lt", "2025-02-25"
    )  # not a date: the first number 2025 < 2025 is False
    assert not _condition("n/a", "lt", "2025-02-25")


def test_a_record_condition_on_a_contract_end_date_denies_only_a_contract_that_has_ended():
    rule = {
        "id": "r",
        "type": "record_condition",
        "tools": ["resume_line"],
        "field": "contract_end_date",
        "op": "lt",
        "value": "2025-02-25",
        "effect": "forbid",
        "level": "deny",
    }
    specs = [{"name": "resume_line", "inputSchema": {"properties": {"line_id": {"type": "string"}}}}]
    outcomes = {}
    for end in ("2024-12-31", "2025-02-24", "2025-02-25", "2026-06-30"):
        hooks = RuleSetHooks([rule], specs)
        hooks.after_tool_call("get_line", {"id": "L1003"}, {"line_id": "L1003", "contract_end_date": end})
        hooks.before_tool_call("resume_line", {"line_id": "L1003"})
        outcomes[end] = [r["outcome"] for r in hooks.records if r["rule_id"] == "r"][-1]
    assert outcomes == {
        "2024-12-31": "deny",
        "2025-02-24": "deny",
        "2025-02-25": "allow",
        "2026-06-30": "allow",
    }
