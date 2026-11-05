"""``cli counts``: the numbers behind a row of the coverage matrix."""

import json

from polisys import cli
from polisys.counts import package_counts
from polisys.documents import load_document
from polisys.model import CoverageItem, PolicyPackage, Rule
from polisys.package import save_package


def _package():
    doc = load_document(
        "## 1. Tickets\n\n- Status must be open or closed.\n- Never delete a ticket.\n- Tickets are kept.\n",
        meta={"id": "d"},
    )
    for st, modality in zip(doc.sections[0].statements, ("must", "must_not", None)):
        st.modality = modality
    allowed = Rule(
        id="r1",
        type="allowed_values",
        rule={
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "status",
            "values": ["open", "closed"],
        },
        statement_ids=["d:1:1"],
        cls="hard",
        level="deny",
    )
    verify = Rule(
        id="r2",
        type="verify",
        rule={"type": "verify", "tools": ["delete_ticket"], "statement": "Never delete a ticket."},
        statement_ids=["d:1:2"],
        cls="judgment",
        level="remind",
    )
    return PolicyPackage(
        version="v",
        documents=[doc],
        rules=[allowed, verify],
        coverage=[CoverageItem("d:1:3", "out_of_scope", "not for the agent")],
        policy_questions=[{"statement_ids": ["d:1:3"], "question": "?", "default_level": "remind"}],
        tests=[{"tool": "update_ticket", "args": {}, "expected": "deny"}, {"note": "not a case"}],
    )


def test_counts_of_a_package():
    c = package_counts(_package())
    assert (c["statements"], c["must_statements"], c["must_without_outcome"]) == (3, 2, 0)
    assert c["by_modality"] == {"must": 1, "must_not": 1, "none": 1}
    assert c["by_outcome"] == {"hard_rule": 1, "judgment_rule": 1, "out_of_scope": 1, "question": 1}
    assert (c["rules"], c["verify_rules"], c["tests"], c["policy_questions"]) == (2, 1, 1, 1)
    assert c["rules_by_class"] == {"hard": 1, "judgment": 1} and c["rules_by_level"] == {
        "deny": 1,
        "remind": 1,
    }


def test_the_counts_command_prints_json_for_one_and_for_several_packages(tmp_path, capsys):
    path = tmp_path / "p.json"
    save_package(_package(), path)
    assert cli.main(["counts", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["rules"] == 2
    other = tmp_path / "q.json"
    save_package(_package(), other)
    assert cli.main(["counts", str(path), str(other)]) == 0
    assert list(json.loads(capsys.readouterr().out)) == [str(path), str(other)]
    assert cli.main(["counts", str(tmp_path / "missing.json")]) == 1
