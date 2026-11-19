# Read this first

This page follows one tool call through the code. Read it before you open the folders: after it, each folder has a place
in a story you know.

The example is the one in [`examples/quickstart.py`](../examples/quickstart.py): an agent wants to call
`create_ticket({"priority": "normal"})`, and the policy says that every ticket needs a requester.

## 1. Before the run: a policy becomes a package

`compile/` reads the policy document and writes a **policy package**: typed rules, procedures and obligations, each with
the sentence of the policy it comes from. This happens offline, once per policy. `compile/validate/` checks the result.
The package is a JSON file (`package/files.py` loads it, `package/model.py` says what is in it).

The example package is `examples/facilities_desk/package.json`.

## 2. At the start of a run: a profile becomes hooks

A **profile** is a YAML file that says which components are on (`runtime/profile.py`).

```python
config = load_config("profiles/rules_only.yaml")                         # runtime/profile.py
parts = build_runtime(config, package, RunContext(task_id="demo", tool_specs=tool_specs))    # runtime/assembly.py
```

`build_runtime` does three things, in this order:

| Step | File | What it does |
|---|---|---|
| Level switch | `runtime/levels.py` | Replaces every `auto` in the profile by `on` or `off` |
| Settings check | `runtime/settings_check.py` | Refuses settings that cannot work together, with the reason |
| Components | `runtime/registry.py` | For each component that the profile turns on, calls its `build` |

A **component** is one file or one folder. It holds its own `Settings` (its part of the profile) and its own
`build(a)`, which returns what the component adds to the run: hook objects, tools, notes for the system prompt.
`registry.py` is the list of all components, in the order their hooks run.

The result, `parts.hooks`, is a list of hook objects. For the `rules_only` profile there are two: the typed rules
(`checks/rules/`) and the procedure monitor (`checks/procedure_monitor.py`).

## 3. The host calls the hooks

The **host** is the agent framework. It owns the loop; polisys only answers at nine fixed points (`hooks.py`):
`session_start`, `on_user_message`, `before_llm_call`, `after_llm_call`, `after_plan`, `before_tool_call`,
`after_tool_call`, `before_reply`, `on_task_end`.

```python
registry = HookRegistry(parts.hooks)                     # hooks.py
message = registry.before_tool_call("create_ticket", {"priority": "normal"})
```

`HookRegistry` calls the same hook of every hook object, in order. For `before_tool_call`, a message from any of them
means **the call is blocked**, and the message tells the agent why and what to do instead. No message means the call runs.
A hook that raises is logged and skipped: a broken check must not break the agent.

Glue for real hosts is in `hosts/`: `langchain.py` wraps LangChain tools so that each call passes these hooks, and
`cuga.py` does the same for the CUGA agent.

## 4. Inside the rules: checks, then a decision

`RuleSetHooks.before_tool_call` (`checks/rules/rule_set_hooks.py`) handles the call:

1. **Is it a write?** A read tool is not checked (`is_write_tool` in `tool_kind.py` decides from the tool name).
2. **Checks.** Each rule of this tool is checked by the method for its type, in `checks/rules/code_checks.py`. The rule
   `required_args` finds that `requester` is missing. A check gives a `Check`: the rule, passed or failed, the reason,
   and what to do instead.
3. **Decision.** Each failed check has a level. `deny` blocks the call. `remind` lets the call run and queues a note for
   the agent. `observe` only records. A rule can also block once for each case and then let the agent decide.

Here the level is `deny`, so the agent gets:

```
POLICY: `create_ticket` was not called.
- [] missing mandatory field(s) requester. Do instead: look up the value(s) and pass requester; ...
```

Every decision is also a record. `runtime/trace.py` writes the records of a run as audit lines.

## 5. The other hook points

| Hook | Who uses it | For what |
|---|---|---|
| `on_user_message` | `sources/intake/` | Looks for records that state rules (listing, discovery) |
| `before_llm_call` | `checks/rules/brief.py`, `sources/intake/reply.py` | Tells the model the rules and the queued notes |
| `after_tool_call` | `checks/rules/duties.py`, `sources/intake/discovery.py` | Remembers what the agent has seen; opens and closes duties; finds rules in the result |
| `before_reply` | `guidance/work_method.py`, `finish/`, `sources/intake/reply.py`, `checks/rules/duties.py` | One read-back turn; checks of what is still missing; open duties |

Rules do not only come from the package. **Online intake** (`sources/intake/`) finds rules in the data the agent reads
during the run (a note on a record, a policy page) and checks writes against them.

## 6. Where to look

| Question | Folder |
|---|---|
| What is in a policy package? | `package/`, and `MODEL.md` |
| How is a package made from a document? | `compile/` |
| How is a tool call checked? | `checks/` |
| How are rules found in data during the run? | `sources/` |
| What text does the agent get? | `guidance/` |
| What happens at the final reply? | `finish/` |
| How does a profile become a run? | `runtime/` |
| How does my agent framework connect? | `hosts/`, and [hooks.md](hooks.md) |
| What is not proven yet? | `experimental/`, and the table in [components.md](components.md) |

The rules of this layout are a test: `tests/test_layout.py`.

## 7. Old names

Until November 2026 all modules were in one flat folder. Every old import path still works (`_aliases.py` has the full
table). These modules also changed their name, so older notes and results use the name on the left:

| Old name | Now |
|---|---|
| `ruleset` | `checks/rules/` |
| `cedar_engine` | `checks/cedar.py` |
| `sop_retrieval` | `checks/retrieval.py` |
| `tracker` | `finish/requirements_tracker.py` |
| `runtime` (one file) | `runtime/assembly.py` |
| `policy_config` | `runtime/profile.py` |
| `package` (one file) | `package/files.py` |
| `adapters` | `hosts/` |
| `store_gate` | `experimental/checks/gate.py` (the gate) |
| `obligations` | `experimental/finish/ledger/` (the obligation ledger) |
| `deliverables` | `experimental/finish/request_checklist.py` (the request checklist) |
