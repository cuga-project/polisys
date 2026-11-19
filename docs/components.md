# Components

The parts of polisys, module by module. The policy model is described in [MODEL.md](../src/polisys/MODEL.md). For the
big picture start with the [README](../README.md) and the [design](design.md).

## Two ways a policy comes in

| | Offline (before the run) | Online (during the run) |
|---|---|---|
| Source | Documents: PDF, DOCX, HTML, Markdown, text; one or several | Tool output (emails, sheets, tickets) and the user's request |
| Code | `compile/documents.py` → `builder.build_policy_set` (or the skill) | `intake.OnlinePolicyIntake` |
| Output | `PolicyBuild` (rules, step graphs, links, coverage) | overlay rules for this thread |

## Offline: build a policy

```python
from polisys.compile.documents import load_policy_set
from polisys.compile.builder import build_policy_set
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
styles; `compile/structure.py`) → classify statement types (`compile/classify.py`) → links (`compile/handlers/links.py`) → rules from tables
by code (`compile/handlers/tables.py`) → step graphs (`compile/handlers/procedures.py`) → rules from prose by a small model
(`compile/handlers/prose.py`) → repair rejected rules → fold exceptions, drop superseded rules → repair the rule set
(`compile/rule_repair.py`: tools for `verify` rules, typed rules for `verify` rules, role rules from grant tables at `remind`,
`when` values as lists, guidance for a `verify` rule no write can trigger) → validate (`compile/validate/`).

## Run time

A profile (a YAML file, `runtime/profile.py`) sets every component, and a policy package holds the rules. `build_runtime`
gives the hooks, the tools and the notes for one task. The components are listed in `runtime/registry.py`; each one's settings
(`Settings`) and its `build` function are in its own file. Policy switches in the environment do not exist; only secrets and
the gateway address come from there.

```python
from polisys.package import load_package
from polisys.runtime.profile import load_config
from polisys.runtime import RunContext, build_runtime

cfg = load_config("profiles/my-profile.yaml")
parts = build_runtime(cfg, load_package("packages/my-policy/package.json"), RunContext(task_id="t1", tool_specs=tool_specs))
agent = MyAgent(tools=tools + parts.tools, hooks=parts.hooks)  # any host that runs the hook contract, see docs/hooks.md
```

| Module | What it does |
|---|---|
| `package/model.py`, `MODEL.md` | The policy model and its documentation |
| `package/files.py` | The policy package (one JSON file per policy): load, save, and convert builds and store graphs to it and back. Components read it with `from_package(pkg, ...)` |
| `compile/documents.py`, `compile/structure.py` | Read documents; sections, statements, structure profiles |
| `compile/classify.py` | Statement types (code for tables, a model for prose) |
| `compile/handlers/` | Rules from tables and prose, links, procedures |
| `compile/validate/` | Rule and build checks; build-twice diff |
| `compile/builder.py` | The offline pipeline (and the older single-document `build`) |
| `checks/rules.py` | Typed rules enforced at the hooks (code and Cedar checkers) |
| `checks/procedure_monitor.py` | Step graphs during the run |
| `experimental/finish/ledger.py` | The ledger: at the final reply, code checks what the policy requires against the calls that succeeded; one follow-up with known values, never for blocked work |
| `sources/intake.py`, `sources/detectors.py`, `checks/facts.py` | Online intake: detection (code, decision model), extraction, write verifier |
| `llm.py` | Small-model clients (any OpenAI-compatible gateway) with usage counts |
| `cli.py`, `skill/` | Command line and the harness-neutral policy compiler skill |

Nothing in this package is specific to a benchmark or a document type.

## Core and experimental components (2026-11-19)

A component is **core** when it is on in a profile whose gain over the agent alone was shown on a held-out task set, by a
decision rule written before the run. The evidence belongs to the profile: no run measured one of its components alone.
Every other component is **experimental**: a profile can turn it on, and it is not a recommended default.

Profiles with a held-out gain:

| Profile | Components on | Held-out result (agent alone → with the profile) |
|---|---|---|
| Rules from data | typed rules, online intake with discovery, verifier, write decisions, read-before-write reminder, requirements tracker | AutomationBench, 42 tasks x 3 trials, gpt-oss-120b: strict 4.0 → 11.9 (p 0.032), partial 38.4 → 48.0 |
| Listing and read-back | online intake with listing, superseded rules dropped, read-back note | Same set, gpt-oss-120b: strict 4.0 → 15.9 (p 0.004), partial 38.4 → 54.4. Haiku 5.5: no gain shown, no harm |
| Read-back | read-back note | EnterpriseOps-Gym HR, 80 tasks x 3 trials, Haiku 5.5: strict 15.8 → 19.6 (p 0.025), partial 52.5 → 56.5 |

| Component | Profile key | Status | Why |
|---|---|---|---|
| Typed rules, procedure monitor | `checkers.rules` | Core | In "Rules from data". Role rules made from grant tables failed the no-harm check on two service-desk domains: keep them at `remind` |
| Online intake, detectors, listing | `sources.data`, `sources.listing` | Core | In both AutomationBench profiles |
| Verifier, write decisions | `checkers.verifier`, `decisions` | Core | In "Rules from data" |
| Requirements tracker | `organizers.requirements_tracker` | Core | In "Rules from data". Alone it was neutral, and harmful when its items were wrong |
| Read-back note | `notes.work_method: [readback]` | Core | Held-out gain on HR |
| Other work-method notes (finish check, lookup retry, quoted values) | `notes.work_method` | Experimental | Tuning gain on 5 HR tasks; held-out gain not shown; no harm |
| Gate | `checkers.gate` | Experimental | Gain only in-sample (3 tasks with written SOPs) |
| Procedure guide, procedure tools, procedure push | `organizers.guide`, `organizers.procedure_tools`, `organizers.procedure_push` | Experimental | Guide: gain only in-sample. Tools alone: no gain |
| Obligation ledger, request checklist | `obligations` | Experimental | Held-out gain not shown; no harm |
| Tool notes, default values, policy lines, code guards | `notes.tool_notes`, `checkers.default_values`, `notes.policy_lines`, `checkers.guard_code` | Experimental | Held-out gain not shown; no harm |
| Read-first note | `notes.read_first` | Experimental | In-sample: fewer guardrail breaks and a lower strict score |
| SOP loop | `organizers.loop` | Experimental | Gain only in-sample; neutral or harmful on tasks without an SOP |
| SOP tracker | `organizers.sop_tracker` | Experimental | τ² Telecom: +12 points, not significant (p 0.15); harm on the airline domain |
| Navigator | `organizers.navigator` | Experimental | Offline replay only |
| SOP planner, procedure compiler | none | Experimental | Nothing imports them; the planner is a dropped spike |

## SOP adherence (added 2026-11-02)

Measured on one benchmark with written SOPs, 3 public tasks, paired with the agent alone, in-sample (details are in the
private evaluation repo).

| Part | File | What it does | Status |
|---|---|---|---|
| Procedure intake | `compile/handlers/procedures.py`, `compile/handlers/procedure_rules.py` | SOP → procedures ("Step N:" text or numbered lists; table of contents skipped); per procedure: preconditions, exceptions, templates, escalations (verbatim quotes); rules outside procedures into a "General rules" entry | Built, tested |
| Procedure guide | `experimental/guidance/procedure_guide.py` | Stops the first write of a procedure once with its preconditions/exceptions; template check; evidence check of hard preconditions (authorization, approval, written, signed) on the data read, at most 2 stops; general-rules stop at the first write linked to no procedure | Built, tested. DeepSeek agent: 82 → 114 of 153 rubric items |
| SOP-aware loop | `experimental/organizers/sop_loop.py` | Outer controller: explore (writes stopped) → cases → per case typed questions (procedure, preconditions, exceptions, general rules, already handled) → decision in code → one scoped agent turn per case → finish turn | Spike. +40 of 306 items over 3 rounds on that benchmark; neutral or harmful on the benchmarks without an SOP |
| SOP planner | `experimental/organizers/sop_planner.py` | Per-case plan from a second model | Spike, dropped (+2/+4 items; case decisions on partial data added errors) |
| Requirements tracker | `finish/requirements.py`, `finish/requirements_tracker.py` | Checklist of required effects at the reply + follow-up | Neutral on the benchmarks without an SOP, harmful on the SOP benchmark when items were wrong |
| Request checklist | `experimental/finish/request_checklist.py` | Request deliverables checked at the reply; now the producer `request` of the obligation ledger (`obligations_from_request`) | Within noise |
