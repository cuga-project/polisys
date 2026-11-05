"""Generated guard code: safety check, execution limits, records, and the intake path (remind and deny)."""

import json

import pytest

from polisys.guard_code import compile_check, flatten_records, is_safe, run_check

CHECK = '''def check(call, data, today):
    out = []
    for r in records(data):
        start = date(r.get("Start Date"))
        if start and start > date(today) and mentions(call["args"].get("body", ""), r.get("Item")):
            out.append(f"{r['Item']} starts {start}, after today; do not amortize it")
    return out
'''
ROWS = {
    "rows": [
        {"cells": {"Item": "Annual Insurance", "Start Date": "2025-07-01"}},
        {"cells": {"Item": "Office Lease Deposit", "Start Date": "2026-04-01"}},
    ]
}


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef check(call, data, today):\n    return []",
        "def check(call, data, today):\n    return open('/etc/passwd').read()",
        "def check(call, data, today):\n    return call.__class__.__bases__",
        "def check(call, data, today):\n    return __import__('os').listdir('/')",
        "def other(a):\n    return []",
    ],
)
def test_unsafe_or_wrong_code_is_rejected(code):
    assert not is_safe(code)[0]
    with pytest.raises(ValueError):
        compile_check(code)


def test_check_runs_on_records_with_helpers():
    fn = compile_check(CHECK)
    data = [{"tool": "get_rows", "args": {}, "records": flatten_records(ROWS)}]
    out = run_check(
        fn, {"tool": "send", "args": {"body": "Total: Office Lease Deposit $750"}}, data, "2026-02-20"
    )
    assert len(out) == 1 and "Office Lease Deposit" in out[0]
    assert (
        run_check(fn, {"tool": "send", "args": {"body": "Annual Insurance $2,000"}}, data, "2026-02-20") == []
    )


def test_runaway_check_is_stopped():
    fn = compile_check("def check(call, data, today):\n    while True:\n        pass\n")
    with pytest.raises(Exception):
        run_check(fn, {"tool": "t", "args": {}}, [], "2026-01-01")


def test_intake_reminds_by_default_and_denies_in_deny_mode():
    from polisys.intake import OnlinePolicyIntake

    def model(system, user):
        if "Python check" in system:
            return json.dumps({"applies": True, "code": CHECK})
        return json.dumps({"violations": []})

    for mode in ("remind", "deny"):
        hooks = OnlinePolicyIntake(
            model, check_reply=False, read_before_write=False, guard_code=mode, today="2026-02-20"
        )
        hooks.on_user_message("Process this month's amortization.", None)
        hooks.policies = [
            {
                "title": "Amortization rules",
                "trust": "trusted",
                "statements": ["Only amortize items whose coverage period has started."],
            }
        ]
        hooks.after_tool_call("get_rows", {"spreadsheet": "ss_prepaids"}, ROWS)
        res = hooks.before_tool_call("gmail_send_email", {"body": "Office Lease Deposit $750"})
        rec = [r for r in hooks.records if r.get("rule") == "guard_code"][-1]
        if mode == "remind":
            assert res.message is None and rec["outcome"] == "remind"
            assert "Office Lease Deposit" in hooks.before_llm_call(None).message
        else:
            assert "Office Lease Deposit" in res.message and rec["outcome"] == "deny"


def test_allowed_module_imports_work_and_others_fail():
    fn = compile_check(
        "import datetime\nfrom math import floor\ndef check(call, data, today):\n"
        "    return [] if datetime.date(2026, 1, 1) and floor(1.5) == 1 else ['x']\n"
    )
    assert run_check(fn, {"tool": "t", "args": {}}, [], "2026-01-01") == []
    assert not is_safe("import os\ndef check(call, data, today):\n    return []")[0]
