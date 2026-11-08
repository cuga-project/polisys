"""sources.listing: every listed record is sorted by the model; rule records and tables reach extraction."""

import json
import re

from polisys import listing as L
from polisys.intake import OnlinePolicyIntake

NOTE = {"id": "m1", "subject": "Refund handling", "body": "Refunds over 300 need a manager's approval."}
CHAT = {"id": "m2", "subject": "Lunch", "body": "Pizza on Friday."}
TABLE = {
    "id": "s1/w1",
    "worksheet": "Vendors",
    "rows": [{"id": "r1", "Vendor": "Acme", "Note": ""}, {"id": "r2", "Vendor": "Birch", "Note": "on hold"}],
}


class Model:
    """Sorting calls get the labels; extraction calls return one trusted statement per call and record the records."""

    def __init__(self, labels):
        self.labels, self.extracted = labels, []

    def __call__(self, system, user):
        if "sort business records" in system:
            out = []
            for m in re.finditer(r"^\[(\d+)\] (.*)$", user, re.M):
                out.append(
                    {
                        "n": int(m.group(1)),
                        "label": next((v for k, v in self.labels.items() if k in m.group(2)), "data"),
                    }
                )
            return json.dumps({"labels": out})
        self.extracted.append(user)
        return json.dumps({"policies": [{"record": 1, "title": "t", "trust": "trusted", "statements": ["S"]}]})


def _hooks(model, by_source):
    hooks = OnlinePolicyIntake(model, verify_writes=False, check_reply=False, listing=lambda: by_source)
    hooks.on_user_message("process the refunds", None)
    hooks._drain(10)
    return hooks


def test_rule_record_and_table_reach_extraction_data_does_not():
    model = Model({"Refund handling": "rule", "Lunch": "data"})
    hooks = _hooks(model, {"gmail": [NOTE, CHAT], "sheets": [TABLE]})
    sent = "\n".join(model.extracted)
    assert "Refunds over 300" in sent and "on hold" in sent  # the table goes although the model said data
    assert "Pizza on Friday" not in sent
    listing = [r for r in hooks.records if r.get("outcome") == "listing"][0]
    assert listing["records"] == 3 and listing["kept"] == 2 and listing["tables"] == 1


def test_a_record_without_a_label_is_kept():
    class Broken(Model):
        def __call__(self, system, user):
            if "sort business records" in system:
                return "not json"
            return super().__call__(system, user)

    model = Broken({})
    _hooks(model, {"gmail": [CHAT]})
    assert any("Pizza on Friday" in u for u in model.extracted)


def test_records_the_listing_read_skip_the_online_intake():
    model = Model({"Refund handling": "rule"})
    hooks = _hooks(model, {"gmail": [NOTE]})
    n = len(model.extracted)
    hooks.after_tool_call("gmail_find_email", {"query": "refund"}, {"messages": [dict(NOTE, extra="x")]})
    hooks._drain(5)
    assert len(model.extracted) == n  # same ID: already read
    hooks.after_tool_call("gmail_find_email", {"query": "x"}, {"messages": [{"id": "m9", "body": "New policy: x."}]})
    hooks._drain(5)
    assert len(model.extracted) == n + 1  # a new record still goes


def test_long_table_is_split_into_parts_of_whole_rows():
    rows = [{"id": f"r{i}", "text": "x" * 500} for i in range(30)]
    parts = L.split_table({"id": "t", "worksheet": "W", "rows": rows}, limit=6000)
    assert len(parts) > 1 and all(p["worksheet"] == "W" for p in parts)
    assert [r["id"] for p in parts for r in p["rows"]] == [r["id"] for r in rows]


def test_records_sort_oldest_first_within_a_source():
    new = {"id": "b", "date": "2026-02-01T09:00:00Z", "body": "rule v2"}
    old = {"id": "a", "date": "2026-01-05T09:00:00Z", "body": "rule v1"}
    assert [r["id"] for _, r in L.flatten({"gmail": [new, old]})] == ["a", "b"]


def test_a_part_of_a_listed_record_without_an_id_is_known():
    listed = L.Listed(L.flatten({"sheets": [{"id": "t", "rows": [{"id": "r1", "cells": {"Note": "hold, policy"}}]}]}))
    assert L.unlisted([{"Note": "hold, policy"}, {"Note": "other"}], listed) == [{"Note": "other"}]


CORR = {"id": "s9", "text": "Correction: the East amount should be $35,000, not $38,000."}


def test_listing_data_keeps_corrections_for_the_brief_and_not_for_extraction():
    model = Model({"Refund handling": "rule", "Correction": "correction", "Lunch": "data"})
    hooks = OnlinePolicyIntake(
        model,
        verify_writes=False,
        check_reply=False,
        listing=lambda: {"gmail": [NOTE, CHAT], "slack": [CORR]},
        listing_data=True,
    )
    hooks.on_user_message("process the refunds", None)
    hooks._drain(10)
    assert not any("35,000" in u for u in model.extracted)
    brief = hooks.before_llm_call(None).message
    assert "[correction] (slack)" in brief and "35,000" in brief
    off = Model({"Refund handling": "rule", "Correction": "correction"})
    _hooks(off, {"slack": [CORR]})
    assert any("35,000" in u for u in off.extracted)  # without the flag a correction is not sorted out as data
