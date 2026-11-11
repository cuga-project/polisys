# Components

The parts of polisys, module by module. The policy model is described in [MODEL.md](../src/polisys/MODEL.md). For the
big picture start with the [README](../README.md) and the [design](design.md).

## Two ways a policy comes in

| | Offline (before the run) | Online (during the run) |
|---|---|---|
| Source | Documents: PDF, DOCX, HTML, Markdown, text; one or several | Tool output (emails, sheets, tickets) and the user's request |
| Code | `documents.py` → `builder.build_policy_set` (or the skill) | `intake.OnlinePolicyIntake` |
| Output | `PolicyBuild` (rules, step graphs, links, coverage) | overlay rules for this thread |

## Offline: build a policy

```python
from polisys.documents import load_policy_set
from polisys.builder import build_policy_set
from polisys.llm import openai_json_model

ps = load_policy_set(["sop.pdf", "update_memo.md"], {"update_memo": {"effective_date": "2026-10-01"}})
build = build_policy_set(ps, tool_specs, openai_json_model("openai/gpt-oss-120b"), cache_dir=".policy_cache")
print(build.stats, build.validation)
```

Or from the shell (also used by the policy compiler skill, `skill/SKILL.md`, in any agent harness):

```bash
python -m polisys.cli convert sop.pdf update_memo.md -o set.json --meta meta.json
python -m polisys.cli build set.json --tools tools.json -o build.json --model openai/gpt-oss-120b
python -m polisys.cli validate build.json --tools tools.json --set set.json
# a policy package written by a person or an agent (the skill): draft.json + set.json -> package.json
python -m polisys.cli pack set.json draft.json -o package.json --harness claude-code --model <model> --tools tools.json
python -m polisys.cli validate package.json --tools tools.json
```

Pipeline: read documents (docling for PDF/DOCX/HTML) → learn each document's structure (heading and reference
styles; `structure.py`) → classify statement types (`classify.py`) → links (`handlers/links.py`) → rules from tables
by code (`handlers/tables.py`) → step graphs (`handlers/procedures.py`) → rules from prose by a small model
(`handlers/prose.py`) → repair rejected rules → fold exceptions, drop superseded rules → repair the rule set
(`rule_repair.py`: tools for `verify` rules, typed rules for `verify` rules, role rules from grant tables at `remind`,
`when` values as lists, guidance for a `verify` rule no write can trigger) → validate (`validate.py`).

## Run time

A profile (a YAML file, `policy_config.py`) sets every component, and a policy package holds the rules. `build_runtime`
gives the hooks, the tools and the notes for one task. Policy switches in the environment do not exist; only secrets and
the gateway address come from there.

```python
from polisys.package import load_package
from polisys.policy_config import load_config
from polisys.runtime import RunContext, build_runtime

cfg = load_config("profiles/my-profile.yaml")
parts = build_runtime(cfg, load_package("packages/my-policy/package.json"), RunContext(task_id="t1", tool_specs=tool_specs))
agent = MyAgent(tools=tools + parts.tools, hooks=parts.hooks)  # any host that runs the hook contract, see docs/hooks.md
```

| Module | What it does |
|---|---|
| `model.py`, `MODEL.md` | The policy model and its documentation |
| `package.py` | The policy package (one JSON file per policy): load, save, and convert builds and store graphs to it and back. Components read it with `from_package(pkg, ...)` |
| `documents.py`, `structure.py` | Read documents; sections, statements, structure profiles |
| `classify.py` | Statement types (code for tables, a model for prose) |
| `handlers/` | Rules from tables and prose, links, procedures |
| `validate.py` | Rule and build checks; build-twice diff |
| `builder.py` | The offline pipeline (and the older single-document `build`) |
| `ruleset.py` | Typed rules enforced at the hooks (code and Cedar checkers) |
| `procedure_monitor.py` | Step graphs during the run |
| `obligations.py` | The ledger: at the final reply, code checks what the policy requires against the calls that succeeded; one follow-up with known values, never for blocked work |
| `intake.py`, `detectors.py`, `facts.py` | Online intake: detection (code, decision model), extraction, write verifier |
| `llm.py` | Small-model clients (any OpenAI-compatible gateway) with usage counts |
| `cli.py`, `skill/` | Command line and the harness-neutral policy compiler skill |

Nothing in this package is specific to a benchmark or a document type.

## SOP adherence (added 2026-11-02)

Measured on one benchmark with written SOPs, 3 public tasks, paired with the agent alone, in-sample (details are in the
private evaluation repo).

| Part | File | What it does | Status |
|---|---|---|---|
| Procedure intake | `handlers/procedures.py`, `handlers/procedure_rules.py` | SOP → procedures ("Step N:" text or numbered lists; table of contents skipped); per procedure: preconditions, exceptions, templates, escalations (verbatim quotes); rules outside procedures into a "General rules" entry | Built, tested |
| Procedure guide | `procedure_guide.py` | Stops the first write of a procedure once with its preconditions/exceptions; template check; evidence check of hard preconditions (authorization, approval, written, signed) on the data read, at most 2 stops; general-rules stop at the first write linked to no procedure | Built, tested. DeepSeek agent: 82 → 114 of 153 rubric items |
| SOP-aware loop | `sop_loop.py` | Outer controller: explore (writes stopped) → cases → per case typed questions (procedure, preconditions, exceptions, general rules, already handled) → decision in code → one scoped agent turn per case → finish turn | Spike. +40 of 306 items over 3 rounds on that benchmark; neutral or harmful on the benchmarks without an SOP |
| SOP planner | `sop_planner.py` | Per-case plan from a second model | Spike, dropped (+2/+4 items; case decisions on partial data added errors) |
| Requirements tracker | `requirements.py`, `tracker.py` | Checklist of required effects at the reply + follow-up | Neutral on the benchmarks without an SOP, harmful on the SOP benchmark when items were wrong |
| Request checklist | `deliverables.py` | Request deliverables checked at the reply; now the producer `request` of the obligation ledger (`obligations_from_request`) | Within noise |
