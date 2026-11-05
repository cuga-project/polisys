# polisys

**Policy and SOP adherence for tool-using agents.** polisys turns a company policy (a handbook, an SOP, a rule list) into
checks that run while an agent works: rules that stop a wrong tool call, procedures that guide the agent through a case,
and obligations that find work the agent never did. It is a library with a hook interface, so it runs next to an agent
and does not contain one.

> **Status: experimental, private.** The code is stable and tested (the suite has more than 3,400 tests), and the measured
> results are on tuning or already-seen task sets only; nothing here is a held-out result. Read [Evidence](#evidence) before
> you quote a number.

## What it does

| Part | What it gives the agent | Checked by |
|---|---|---|
| **Typed rules** | A wrong call is stopped before it runs, with a message that says what to do instead. 15 rule types, for example required arguments, allowed values, record state, role permission, thresholds | Code (and Cedar for hard rules). No model call |
| **Procedures** | The steps, preconditions, exceptions and must-not rules of the SOP section that applies to the case, as tools and a brief | Code, plus a small model for judgement checks |
| **Obligations** | At the final reply: "these required writes did not happen", with the values the policy prescribes. Never for work that a hook blocked | Code first; the ledger stays silent when unsure |
| **Control levels** | Strong models get fewer checks than weak ones; one switch sets which checks run | `levels.py` |
| **Audit** | One line per decision: rule, source, evidence, package version, and each failure of a check | `audit.jsonl` |

Everything is chosen by a **profile** (one YAML file). Policy switches in the environment do not exist. A profile never holds
rules; rules live in the policy **package**.

## How it fits together

```
 policy documents ──► compile ──► package.json ──► build_runtime(profile, package, context)
 (PDF, Word, HTML,    skill or      one file:             │
  Markdown)           pipeline      rules, procedures,    ▼
                                    obligations,       hook objects, tools, brief
                                    questions, tests      │
                                                          ▼
                                       agent host ◄── nine lifecycle hooks (polisys.hooks)
                                       (tool calls, replies)
```

- **Compile** reads documents into statements and writes the package. Two routes: a Claude **skill**
  ([`skill/SKILL.md`](src/polisys/skill/SKILL.md), the better route in every comparison so far) and an open-model
  **pipeline** (`builder.py`).
- **Package** is one JSON file per policy. [`MODEL.md`](src/polisys/MODEL.md) describes it; `polisys validate` checks it
  against the tool schemas.
- **Runtime** is `build_runtime`. It refuses wiring that can have no effect (a profile that needs a package and has none, a
  check turned on at a level where it never runs) instead of running without the policy.
- **Hooks** are the only link to an agent. See [docs/hooks.md](docs/hooks.md).

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"              # add [cedar], [llm] or [documents] as needed, or [all]

# the example is a made-up Facilities Desk policy
cd examples/facilities_desk
polisys convert policy.md -o set.json                                    # document -> statements
polisys pack set.json draft.json -o package.json --harness example --model hand-written \
        --tools tools.json --tests tests.json                            # statements + draft -> one package
polisys validate package.json --tools tools.json                         # 0 errors, 0 warnings
polisys counts package.json                                              # what the package holds, in numbers
cd ../..

python examples/quickstart.py        # enforce the package at tool-call time, no agent and no model
```

The quickstart runs two profiles over the same three calls. With `baseline` (policy off) nothing is added and every call
runs. With `rules_only` a call with no requester and a call with `priority=whenever` are stopped, and the message says what
to do instead:

```
create_ticket({'priority': 'normal'})
  -> BLOCKED: POLICY: `create_ticket` was not called.
     - missing mandatory field(s) requester. Do instead: look up the value(s) and pass requester; ...
```

## Using it with an agent

```python
from polisys.package import load_package
from polisys.policy_config import load_config
from polisys.runtime import RunContext, build_runtime

cfg = load_config("profile.yaml")
parts = build_runtime(cfg, load_package("package.json"), RunContext(task_id="t1", tool_specs=tool_specs))
agent = MyAgent(tools=tools + parts.tools, hooks=parts.hooks)   # parts.hooks follow the hook contract
```

`parts` also carries the system note, a `run` replacement for loop-style SOP control, and `usage()` and `records()` for cost
and audit. With the CUGA agent pass `parts.hooks` to `CugaAgent(hooks=[...])`. For other hosts, and for what a host must
provide, read [docs/hooks.md](docs/hooks.md).

A profile is plain YAML; see [`examples/profiles/`](examples/profiles/) and the field list in
[`policy_config.py`](src/polisys/policy_config.py). An unknown key is an error that names it.

## Compiling a policy

**With the skill (recommended).** Give an agent `skill/SKILL.md`, the policy documents (the original files, not a flattened
text) and `tools.json`. It writes a draft (rules, procedures, obligations, questions); `polisys pack` and `polisys validate`
check it. Decision tests are written by a *separate* agent that has not seen the rules. `validate` checks quotes against the
document text, argument names against the tool schemas, value binding for obligations, and conflicts between rules.

**With the pipeline.** `polisys build` runs an open model (for example gpt-oss-120b) over the statements, with code for tables
and repairs for the known failures. On the policies where both were measured it did not reach the skill, so treat its output as
a draft to validate, not as a replacement.

Either way, an ambiguity in the text goes into `policy_questions` for a person to answer; it is not guessed.

## Evidence

Measured in the private evaluation repo on four benchmarks (an IT service desk, written HR/insurance/pharmacy SOPs,
Zapier-style workflows, and a conversational telecom/airline/retail benchmark). The honest summary:

- **Rules work as a guard.** On the IT service policy, rules written by the skill from the policy text alone caught 106 of 182
  deny tests with 0 false denies on 147 allowed calls (tuning version; the pipeline reached 102 with 1 false deny).
- **Control by agent strength matters.** The same checks helped a small open model (gpt-oss-120b: +18.9 and +8.8 points on two sets, both now seen) and did not help a strong one (Gemini 3.8: −5.8, which could be chance, with more guardrails broken).
- **The pipeline does not replace the skill** on any of the benchmarks measured.
- **Obligations do not send yet.** In replay on saved runs every source stays at level 1 (record, never send): the best case
  has 0 false opens and 27.6% recall against a bar of 30%.
- **No set is truly held-out any more.** Sets that were held out when first run have since been seen and are labelled so.

Every number, its basis (tuning, seen, unseen) and the corrections to earlier claims are in
[docs/history/stability-report.md](docs/history/stability-report.md) and
[docs/history/stabilization-decisions.md](docs/history/stabilization-decisions.md). The design the code was built to is
[docs/design.md](docs/design.md).

## Repository layout

```
src/polisys/
  model.py, package.py, validate.py   the policy model, the package file, the validator
  documents.py, builder.py, handlers/ read documents; the open-model compile pipeline
  skill/SKILL.md                      the compile skill for an agent
  policy_config.py, runtime.py        profiles and build_runtime
  ruleset.py, cedar_engine.py         typed rules, with Cedar for hard rules
  store_gate.py, procedure_*.py       procedures as tools, the judgement gate, the guide
  obligations.py, deliverables.py     the obligation ledger and the request producer
  levels.py, router.py                control levels and WriteChecks; swappable retrieval
  trace.py, dry_run.py                audit lines; wiring description without a model
  hooks.py, adapters/                 the hook contract and host glue
tests/                                 the suite (offline; models are scripted)
examples/                              a made-up policy, two profiles and a quickstart
docs/                                  design, hook contract, components, history
```

## Development

```bash
pip install -e ".[dev,cedar,llm]"
pytest -q                 # offline, no model calls; a few tests need the extras or the CUGA agent and skip without them
ruff check .
```

Tests that run a real CUGA agent loop skip when `cuga` is not installed. Tests that read PDFs skip without `docling`.

## Provenance and licence

polisys is the policy layer of the CUGA agent project, extracted into its own repository on 2026-11-05 from the
`policy/stable-v1` tag of the agent's `policy/stable` branch (it was `cuga.policy_layer`). History was not carried over: the
original history is entangled with the agent's. `polisys.hooks` is a structural copy of `cuga.backend.hooks`; keep the two
in step until the agent depends on this package.

No licence has been chosen. Until one is, the code is private and all rights are reserved. Do not add benchmark policy text or
task data to this repo: the tests use made-up policies only.
