"""Policy config: the loader names a bad key or value, and defaults load."""

import re
from pathlib import Path

import pytest

from polisys import policy_config
from polisys.policy_config import (
    KEV,
    PolicyConfig,
    PolicyConfigError,
    load_config,
)

SPEC_EXAMPLE = """
policy:
  package: packages/example/package.json
  sources: {data: true, discovery: true, request: true}
  authority: {company_domains: [company.example.com], may_change_rules: [VP, COO, HR]}
  priority_classes: [regulatory, company, department, team]
  control:
    level: by_model
    by_model: {openai/gpt-oss-120b: 3, deepseek/deepseek-v4-pro: 2, gcp/gemini-3.8-flash: 2}
    risk: {level3_tools: ["*_payment*", "*_delete*", send_email_external]}
  checkers:
    compile: openai/gpt-oss-120b
    narrow_questions: jaredpalmer/kev-4b
    router: {kind: agent_opened, expand: next_links}
    gate: {model: gcp/gemini-3.8-flash, scope: active_procedures, min_strength: agent}
    verifier: {model: openai/gpt-oss-120b, scope: specific_statements, may_deny: false}
  organizers: {procedure_tools: auto, guide: auto, loop: auto, sop_tracker: auto, navigator: auto, requirements_tracker: auto}
  obligations: {producers: [package, request, duty], max_follow_ups: 1, wrong_values: known_origin_only}
  decisions: {require_evidence: true, after_repeated_denials: skip}
  notes: {reminders: tool_result, brief: system}
  trace: true
"""


def write(tmp_path, text, name="profile.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def paths_with_auto(value, prefix=""):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in paths_with_auto(v, f"{prefix}.{k}" if prefix else k)]
    return [prefix] if value == "auto" else []


# --- the loader --------------------------------------------------------------------------------------------------


def test_defaults_load(tmp_path):
    config = load_config(write(tmp_path, "policy: {}\n"))
    assert config == PolicyConfig()
    assert config.package is None and config.trace is True
    assert config.control.level == "by_model" and config.control.by_model == {}
    assert config.checkers.router.kind == "agent_opened" and config.checkers.router.expand == "next_links"
    assert not config.sources.data and not config.sources.discovery and not config.sources.request
    # every component the config can force is auto until a profile says otherwise
    assert sorted(paths_with_auto(config.to_dict())) == sorted(
        [
            "checkers.rules",
            "checkers.gate.mode",
            "checkers.verifier.mode",
            "checkers.write_checks",
            "organizers.procedure_tools",
            "organizers.guide.mode",
            "organizers.loop.mode",
            "organizers.sop_tracker.mode",
            "organizers.navigator.mode",
            "organizers.requirements_tracker.mode",
            "decisions.mode",
        ]
    )


def test_the_example_of_the_spec_loads(tmp_path):
    config = load_config(write(tmp_path, SPEC_EXAMPLE))
    assert config.package == "packages/example/package.json"
    assert config.sources.request is True
    assert config.authority.may_change_rules == ["VP", "COO", "HR"]
    assert config.priority_classes == ["regulatory", "company", "department", "team"]
    assert config.control.by_model["deepseek/deepseek-v4-pro"] == 2
    assert config.control.risk.level3_tools == ["*_payment*", "*_delete*", "send_email_external"]
    assert config.checkers.narrow_questions == KEV
    assert (config.checkers.gate.model, config.checkers.gate.scope) == (
        "gcp/gemini-3.8-flash",
        "active_procedures",
    )
    assert config.checkers.verifier.may_deny is False
    assert config.organizers.guide.mode == "auto"
    assert config.obligations.producers == ["package", "request", "duty"]
    assert config.decisions.after_repeated_denials == "skip" and config.decisions.require_evidence is True
    assert (config.notes.reminders, config.notes.brief) == ("tool_result", "system")


@pytest.mark.parametrize(
    "text, key",
    [
        ("policy: {}\nextra: 1\n", "extra"),
        ("policy: {colour: red}\n", "policy.colour"),
        ("policy: {sources: {dataa: true}}\n", "policy.sources.dataa"),
        ("policy: {checkers: {gate: {modle: x}}}\n", "policy.checkers.gate.modle"),
        ("policy: {checkers: {router: {kind: a, expands: b}}}\n", "policy.checkers.router.expands"),
        ("policy: {organizers: {planner: on}}\n", "policy.organizers.planner"),
        ("policy: {organizers: {guide: {mode: on, evidence: x}}}\n", "policy.organizers.guide.evidence"),
        ("policy: {organizers: {loop: {style: control, mdoe: on}}}\n", "policy.organizers.loop.mdoe"),
        ("policy: {control: {risk: {level3: []}}}\n", "policy.control.risk.level3"),
    ],
)
def test_an_unknown_key_raises_and_names_it(tmp_path, text, key):
    with pytest.raises(PolicyConfigError) as error:
        load_config(write(tmp_path, text))
    assert f"unknown key '{key}'" in str(error.value)


@pytest.mark.parametrize(
    "text, key",
    [
        ("policy: {sources: {data: 'yes'}}\n", "policy.sources.data"),
        ("policy: {sources: {data: 1}}\n", "policy.sources.data"),
        ("policy: {trace: 1}\n", "policy.trace"),
        ("policy: {package: 5}\n", "policy.package"),
        ("policy: {priority_classes: company}\n", "policy.priority_classes"),
        ("policy: {priority_classes: [company, 3]}\n", "policy.priority_classes[1]"),
        ("policy: {control: {by_model: {m: 7}}}\n", "policy.control.by_model.m"),
        ("policy: {control: {by_model: {m: true}}}\n", "policy.control.by_model.m"),
        ("policy: {control: {by_model: [1, 2]}}\n", "policy.control.by_model"),
        ("policy: {control: {level: true}}\n", "policy.control.level"),
        ("policy: {control: {level: '2'}}\n", "policy.control.level"),
        ("policy: {checkers: {gate: {scope: everywhere}}}\n", "policy.checkers.gate.scope"),
        ("policy: {checkers: {gate: 7}}\n", "policy.checkers.gate"),
        ("policy: {checkers: {rules: maybe}}\n", "policy.checkers.rules"),
        ("policy: {organizers: {guide: 2}}\n", "policy.organizers.guide"),
        ("policy: {organizers: {navigator: {max_blocks: two}}}\n", "policy.organizers.navigator.max_blocks"),
        ("policy: {organizers: {navigator: {max_blocks: true}}}\n", "policy.organizers.navigator.max_blocks"),
        ("policy: {organizers: {navigator: {max_blocks: -3}}}\n", "policy.organizers.navigator.max_blocks"),
        ("policy: {obligations: {max_follow_ups: -1}}\n", "policy.obligations.max_follow_ups"),
        ("policy: {obligations: {producers: [ledger]}}\n", "policy.obligations.producers[0]"),
        ("policy: {notes: {brief: 'no'}}\n", "policy.notes.brief"),
        ("policy: {checkers: {gate: {seed_request: 1}}}\n", "policy.checkers.gate.seed_request"),
        ("policy: {checkers: {gate: {max_tokens: -1}}}\n", "policy.checkers.gate.max_tokens"),
        (
            "policy: {organizers: {requirements_tracker: {read_system_prompt: 'yes'}}}\n",
            "policy.organizers.requirements_tracker.read_system_prompt",
        ),
        ("policy: {sources: yes}\n", "policy.sources"),
        ("policy: [a]\n", "policy"),
        ("policy:\n", "policy"),
    ],
)
def test_a_wrong_type_or_choice_raises_and_names_the_key(tmp_path, text, key):
    with pytest.raises(PolicyConfigError) as error:
        load_config(write(tmp_path, text))
    assert f"'{key}'" in str(error.value)


def test_the_fields_that_differ_between_the_old_runners_load_and_default_to_the_common_value(tmp_path):
    default = PolicyConfig()
    assert default.checkers.gate.seed_request is False and default.checkers.gate.max_tokens == 4000
    assert default.organizers.requirements_tracker.read_system_prompt is False
    config = load_config(
        write(
            tmp_path,
            "policy:\n  checkers: {gate: {seed_request: true, max_tokens: 16000}}\n"
            "  organizers: {requirements_tracker: {read_system_prompt: true}}\n",
        )
    )
    assert config.checkers.gate.seed_request is True and config.checkers.gate.max_tokens == 16000
    assert config.organizers.requirements_tracker.read_system_prompt is True


def test_on_and_off_are_read_as_modes_although_yaml_reads_them_as_booleans(tmp_path):
    config = load_config(
        write(
            tmp_path,
            "policy:\n  checkers: {rules: on, gate: off, verifier: {mode: on}}\n  organizers: {guide: on, loop: off}\n"
            "  notes: {brief: off}\n",
        )
    )
    assert config.checkers.rules == "on" and config.checkers.gate.mode == "off"
    assert config.checkers.verifier.mode == "on" and config.organizers.guide.mode == "on"
    assert config.organizers.loop.mode == "off" and config.notes.brief == "off"
    quoted = load_config(write(tmp_path, 'policy: {organizers: {guide: "on", loop: {mode: "auto"}}}\n'))
    assert quoted.organizers.guide.mode == "on" and quoted.organizers.loop.mode == "auto"


def test_a_text_field_takes_null_or_none(tmp_path):
    config = load_config(
        write(tmp_path, "policy: {package: none, checkers: {compile: ~, narrow_questions: null}}\n")
    )
    assert (
        config.package is None
        and config.checkers.compile is None
        and config.checkers.narrow_questions is None
    )


def test_a_section_may_be_written_as_a_mapping_or_as_its_mode(tmp_path):
    short = load_config(write(tmp_path, "policy: {organizers: {guide: on}}\n"))
    long = load_config(write(tmp_path, "policy: {organizers: {guide: {mode: on}}}\n"))
    assert short == long


def test_a_duplicate_key_raises(tmp_path):
    with pytest.raises(PolicyConfigError, match="duplicate key 'data'"):
        load_config(write(tmp_path, "policy:\n  sources:\n    data: true\n    data: false\n"))


def test_the_file_must_be_a_policy_mapping(tmp_path):
    for text in ("", "- a\n", "sources: {data: true}\n"):
        with pytest.raises(PolicyConfigError):
            load_config(write(tmp_path, text))
    with pytest.raises(PolicyConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "policy: {sources: [\n"))
    with pytest.raises(PolicyConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")


def test_a_file_that_is_not_utf8_text_raises_and_names_the_file(tmp_path):
    path = tmp_path / "latin.yaml"
    path.write_bytes("policy: {package: caf\xe9}\n".encode("latin-1"))
    with pytest.raises(PolicyConfigError, match="latin.yaml.*not UTF-8"):
        load_config(path)


def test_a_count_of_zero_is_allowed(tmp_path):
    config = load_config(
        write(
            tmp_path, "policy: {organizers: {navigator: {max_blocks: 0}}, obligations: {max_follow_ups: 0}}\n"
        )
    )
    assert config.organizers.navigator.max_blocks == 0 and config.obligations.max_follow_ups == 0


def test_an_error_names_the_file(tmp_path):
    path = write(tmp_path, "policy: {colour: red}\n", "mine.yaml")
    with pytest.raises(PolicyConfigError, match="mine.yaml"):
        load_config(path)


def test_the_module_names_no_benchmark():
    text = Path(policy_config.__file__).read_text()
    found = sorted(
        {
            m.group(0).lower()
            for m in re.finditer(
                r"automationbench|itsm|handbook|tau2|τ|enterpriseops|appworld|telecom|crestwood|human.agent|transfer_to",
                text,
                re.I,
            )
        }
    )
    assert found == []


def test_the_module_reads_no_environment_and_has_no_function_of_the_old_switches():
    text = Path(policy_config.__file__).read_text()
    assert "environ" not in text and "getenv" not in text
    for name in ("config_from_env", "migration_table"):
        assert not hasattr(policy_config, name)
