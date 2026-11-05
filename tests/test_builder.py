"""The offline pipeline end to end, with a fake model: rules, repair, cache, regression against the old builder, CLI."""

import pytest
import json
import subprocess
import sys
from pathlib import Path

from polisys.builder import build, build_policy_set
from polisys.documents import load_document, load_policy_set
from polisys.model import PolicySet, to_json

FIX = Path(__file__).parent / "fixtures"
SPECS = [
    {
        "name": "create_refund",
        "inputSchema": {
            "type": "object",
            "properties": {"amount": {"type": "number"}, "reference": {"type": "string"}},
        },
    }
]
POLICY = "## 1. Refunds\n\n- Refunds above $500 are not allowed.\n- References look like RF-123.\n"


class Scripted:
    """Answers classify and prose prompts; the first prose answer has a bad argument name."""

    model_name = "scripted"

    def __init__(self, bad_first=True):
        self.bad_first, self.prose_calls, self.usage = bad_first, 0, {"calls": 0}

    def __call__(self, system, user):
        self.usage["calls"] += 1
        if "classify" in system:
            return json.dumps(
                {"statements": [{"n": 1, "types": ["threshold"]}, {"n": 2, "types": ["format"]}]}
            )
        if "typed rules" in system:
            self.prose_calls += 1
            arg = "amt" if self.bad_first and self.prose_calls == 1 else "amount"
            rules = []
            if "S1" in user and "References" not in user.split("S1")[1][:60]:
                rules.append(
                    {
                        "s": 1,
                        "class": "rule",
                        "rules": [
                            {
                                "type": "threshold",
                                "tool": "create_refund",
                                "arg": arg,
                                "op": "gt",
                                "value": 500,
                                "then": "forbid",
                                "quote": "Refunds above $500 are not allowed",
                            }
                        ],
                    }
                )
            if "References look like" in user:
                n = 2 if "S2" in user else 1
                rules.append(
                    {
                        "s": n,
                        "class": "rule",
                        "rules": [
                            {
                                "type": "format",
                                "tool": "create_refund",
                                "arg": "reference",
                                "pattern": r"RF-\d+",
                                "quote": "References look like RF-123",
                            }
                        ],
                    }
                )
            return json.dumps({"statements": rules})
        return json.dumps({"links": []})


def _ps():
    return PolicySet(documents=[load_document(POLICY, meta={"id": "p"})])


def test_build_end_to_end_with_repair():
    model = Scripted()
    b = build_policy_set(_ps(), SPECS, model)
    assert sorted(r.type for r in b.rules) == ["format", "threshold"]
    assert model.prose_calls == 2  # first answer rejected (unknown argument amt), repaired once
    assert b.validation == [] and b.generator["kind"] == "pipeline"


def test_rebuild_hits_cache(tmp_path):
    first = Scripted(bad_first=False)
    build_policy_set(_ps(), SPECS, first, cache_dir=tmp_path)
    second = Scripted(bad_first=False)
    b = build_policy_set(_ps(), SPECS, second, cache_dir=tmp_path)
    assert second.usage["calls"] == 0 and b.stats["cache_hits"] >= 2


@pytest.mark.skipif(not (FIX / "itsm_policy.md").exists(), reason="needs a real policy fixture that this repo does not ship")
def test_code_rules_match_the_older_builder_on_a_real_policy():
    specs = json.loads((FIX / "itsm_tools.json").read_text())
    text = (FIX / "itsm_policy.md").read_text()
    new = build_policy_set(PolicySet(documents=[load_document(text, meta={"id": "i"})]), specs)
    old = build(text, specs)
    key = lambda t, r: (t, r.get("tool"), r.get("arg"))  # noqa: E731
    assert sorted(key(r.type, r.rule) for r in new.rules if r.rule.get("by") == "code") == sorted(
        key(r["type"], r) for r in old.rules
    )
    assert not [e for e in new.validation if e["code"] != "schema_conflict"]


def test_cli_validate_exit_codes(tmp_path):
    ps = load_policy_set([FIX / "ap_set" / "update_2026.md"])
    (tmp_path / "set.json").write_text(json.dumps(to_json(ps)))
    (tmp_path / "tools.json").write_text(json.dumps(SPECS))

    def run(*a):
        return subprocess.run(
            [sys.executable, "-m", "polisys.cli", *a], capture_output=True, text=True, cwd=tmp_path
        )

    assert run("build", "set.json", "--tools", "tools.json", "-o", "build.json").returncode == 0
    assert run("validate", "build.json", "--tools", "tools.json", "--set", "set.json").returncode == 0
    data = json.loads((tmp_path / "build.json").read_text())
    data["rules"].append(
        {
            "id": "x",
            "type": "format",
            "rule": {"type": "format", "tool": "nope", "arg": "a", "pattern": "x", "quote": "x"},
            "statement_ids": ["missing"],
        }
    )
    (tmp_path / "bad.json").write_text(json.dumps(data))
    assert run("validate", "bad.json", "--tools", "tools.json").returncode == 1
