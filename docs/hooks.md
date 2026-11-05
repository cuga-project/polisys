# The hook contract

polisys talks to an agent only through `polisys.hooks`. A host agent that can call the nine lifecycle hooks below can run
every component of polisys, and it needs nothing else from this repo.

## The hooks

A hook object is an instance of `AgentHooks` (or any object with the same method names; each method is optional and
returning `None` means "continue unchanged").

| Hook | Called | A returned message means |
|---|---|---|
| `session_start(thread_id)` | once, when a run starts | nothing |
| `on_user_message(message, thread_id)` | when the user message arrives | `HookResult(answer=...)` answers without running the agent |
| `before_llm_call(thread_id)` | before each model call | text to inject into the prompt (system prompt, or the newest message, see `notes.brief`) |
| `after_llm_call(text)` | after each model output | nothing (observe) |
| `after_plan(code)` | after the agent wrote a code block, before it runs | a message blocks the block |
| `before_tool_call(tool, args)` | before every tool call | a message blocks the call and is shown to the model |
| `after_tool_call(tool, args, result)` | after every tool call | a note appended to the result |
| `before_reply(answer)` | when the agent is about to answer | a message sends the agent back to work; `HookResult(answer=...)` rewrites the reply |
| `on_task_end(answer)` | once, at the end | nothing |

`tool_blocked(tool, args, message)` is not a lifecycle hook. When any hook stops a call, every hook is told, to observe
only, because a `before_tool_call` hook cannot see what the others said. The obligation ledger uses it so that a follow-up
never asks for work that a hook has blocked.

A hook that raises does not fail the run: the registry logs it, the run goes on as if the hook had said nothing, and a hook
with an audit sink writes a `hook_failed` line. This is deliberate and it is a fail-open path, so read the audit.

## Running them

```python
from polisys.hooks import HookRegistry, wrap_tools_with_hooks

registry = HookRegistry(parts.hooks)               # parts = build_runtime(...)
tools = wrap_tools_with_hooks(tools, registry)     # LangChain tools: before/after_tool_call around each call
message = registry.before_tool_call("create_ticket", {"priority": "urgent"})   # a str when blocked, else None
```

`HooksCallback(registry)` is a LangChain callback that reports model outputs to `after_llm_call`. For any other
framework, call the registry methods at the matching points of its loop.

## Checking a host

A host may offer less than polisys needs. `polisys.hooks.HOST_FEATURES` names the two extras beyond the nine hooks:

| Feature | What it does | Without it |
|---|---|---|
| `tool_blocked` | Tells every hook that another hook stopped a call | The obligation ledger cannot learn of the block, so a follow-up could ask for work that was blocked. `build_runtime` **refuses** the ledger |
| `failure_report` | Lets a hook that raises leave an audit line | A failed check leaves no line. `build_runtime` **warns** |

`host_features_of(registry)` reads them from a registry class or object (`HookRegistry.FEATURES`, or just a `tool_blocked`
method for an older one). Pass the result as `RunContext(host_features=...)`; `None` means "not told" and skips the check. For
the CUGA agent use `polisys.adapters.cuga.host_features()`, which gives the empty set when the installed agent has no hooks.

## Hosts

| Host | State |
|---|---|
| CUGA agent | Used in every measurement of this system. As of 2026-11-05 no released agent has hooks: they are in the open pull request cuga-project/cuga-agent#1103, which lacks `tool_blocked` and failure reporting. Pass `parts.hooks` to `CugaAgent(hooks=[...])`. `polisys.adapters.cuga` wraps tools that come from a tool provider. `polisys.hooks` is a structural copy of `cuga.backend.hooks`. |
| Other agents | Not tried. Implement the nine calls above, and read the assumptions below. |

Assumptions that are written for the CUGA host and that another host must meet or change:

- **Send-backs.** The obligation ledger assumes the host acts on at most 2 `before_reply` messages per request
  (`obligations.SEND_BACKS`). A host with another limit needs that constant changed.
- **Run loop.** `sop_loop` replaces the host's `invoke` with its own phases (explore, cases, do). It needs a host that can
  run an agent turn on request and stop writes in a phase.
- **Note placement.** `notes.brief: message` needs a host that can show a hook's text as the newest message. The host says so
  through `RunContext.host_note_placement`; the default is `system`.
