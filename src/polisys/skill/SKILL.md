---
name: cuga-policy-compiler
description: Turn company policy documents (PDF, Word, HTML, Markdown; one or several) and the tool schemas of an agent into one checked CUGA policy package (policy-package/1): typed rules, procedures with must-not rules, obligations, policy questions and decision tests. Works in any agent harness that can read files and run shell commands.
---

# CUGA policy compiler

**Hard rule. Read the policy text and the tool schemas only.** Never open or use rubrics, graders, verifiers, test
tasks, grader output, run results, logs, earlier packages or earlier builds of this policy, even if they are in the
working folder. What you write must come from the text alone. A rule you cannot find in the text does not go in. If you
see such a file by accident, stop reading it and say so in your summary.

**Work in a clean folder.** Copy only the input files (the original policy documents and `tools.json`, and `meta.json`
if there is one) into a new empty folder and work there, so that nothing else can be read by mistake. A benchmark or a
company often keeps the rubric or the answers in the same folder as the policy: leave them behind. The agent of step 8
(the test writer) also works in a clean folder, with only the files named there.

You turn policy documents into one **policy package**: one JSON file, `schema: "policy-package/1"`, that the CUGA policy
layer loads and enforces. The model behind it is in `src/cuga/policy_layer/MODEL.md`; read its sections on the package
and on statement types first. You use plain files and the CLI below, in any harness (Claude Code, Bob, Codex, CUGA itself)
with any model. The work is done when `validate` reports `0 errors`. (A build file of the older pipeline format still
validates with `validate build.json --tools tools.json --set set.json`; this skill makes packages.)

## Inputs

- The policy documents (one or more files), **in their original format** (PDF, Word, HTML, Markdown as the company wrote
  them). Do not give `convert` a flattened text copy: it reads headings, lists and tables from the original, and a flat
  copy loses table headers and merges or splits rows, so statements and ids change.
- `tools.json`: the tool schemas of the agent that must follow the policy, a list of `{name, description, inputSchema}`.
- Optional `meta.json`: per document id (file name without extension) `trust` (`trusted` or `untrusted`),
  `effective_date` (ISO date), `authority_rank` (higher wins when dates are equal), `structure` (see step 1).

**Reuse.** The only earlier work you may start from is your own: for a later version of a policy that you already
packaged with this skill, take that base package (made by this skill from the earlier text) and the differences between
the two texts. Copy what did not change, redo what changed, and run every step again on the changed statements. A package
made any other way (a pipeline build, a graph made by hand, an old build) and any run result are never inputs.

## Files you make

| File | Made by | Holds |
|---|---|---|
| `set.json` | `convert` | The documents split into sections and statements, each with an id such as `doc:4.2:3` |
| `draft.json` | you | Everything you write: `modality`, `rules`, `procedures`, `obligations`, `global`, `links`, `coverage`, `policy_questions` |
| `tests.json` | another agent (step 8) | Decision tests, written without seeing `draft.json` |
| `package.json` | `pack` | The result: documents, your draft, `generator`, tests |
| `package.unmarked.txt` | `pack` | The statements you gave no modality, with must-style words first |

Edit `draft.json` and run `pack` and `validate` again after every change. Never edit `package.json` by hand.

**Which copy the runtime reads.** A run reads `package.json` only, never `draft.json`. The procedure tools and the
must-not gate read `procedures` and `global` (as a store graph: steps as text, `decisions`, `must_not`, `deadlines`,
`notes`, `links`, the parts of step 5, and the `global` keys of the shape table). The guide and the SOP loop read
`procedures` as a build (steps with `action`). Rules are read by the rule hooks. `documents`, `tests`, `coverage` and
`policy_questions` are for checking and for the policy owner; a run does not show them to the agent.

## Steps

1. **Convert.** `python -m polisys.cli convert <files...> -o set.json [--meta meta.json]`.
   Read `set.json`: check that sections and statement ids look right (numbers, letters, "Article II", appendices). A
   numbered subsection's title is its statement `:1`; the text follows from `:2`. If a document numbers its parts in a way
   the converter missed, write a structure profile for it (heading and reference patterns, see `structure.py`) into
   `meta.json` as `"structure"` and convert again.
2. **Mark the statements.** Read every statement. In `draft.json` set `"modality": {"<statement id>": "must" | "must_not"
   | "may" | "should"}`. `must` and `must_not` are the statements that need an outcome (step 3), so mark one only if it
   says what the agent has to do or not do. Leave a statement **out** of `modality` when it is:
   - a heading, a line that only opens a list, a column name, an example, a definition, or a fragment of the statement
     before it (for example "Subject: ...");
   - about **another actor** (what a manager, a customer, the system or another team does or must do), not the agent;
   - a list item that is only a condition ("the employee is a contractor") that the next items depend on: its words go to
     `applies_when` or `decisions`;
   - the body of a message or document template: it goes to `global.templates` (step 5).
   `pack` writes every statement you left out to `package.unmarked.txt`, those with must-style words first, and prints the
   first 20 of those. Read them and mark any that really say what the agent must do. `types` of a statement are not used:
   do not write them.
3. **Give every `must` and `must_not` statement an outcome.** `validate` fails on one with none. These count, and only these:

   | Outcome | How it is written |
   |---|---|
   | hard rule, judgment rule or guidance | a rule (step 4) whose `statement_ids` has the statement id |
   | procedure step | a step of a procedure (step 5) whose `id` is the statement id |
   | procedure part | a procedure `preconditions`, `exceptions`, `templates` or `escalations` item whose `quote` copies at least 4 words of the statement |
   | obligation | an obligation (step 6) whose `source.quote` copies at least 4 words of the statement |
   | definition | a `global.definitions` item with `source: {"statement": "<id>"}` |
   | question | a policy question (step 7) whose `statement_ids` has the id |
   | out of scope | a `coverage` item `{"statement_id", "kind": "out_of_scope", "reason": "<why>"}`; a reason is required |

   One statement can have several outcomes (a step that is also a rule). It cannot be `out_of_scope` and have another.
   The text of a procedure's `must_not` and `decisions`, and everything under `global` (`global_rules`,
   `escalation.must_not`, ...), is read at run time but is **not** an outcome: give the same statement a step, or a rule,
   as well. A statement whose meaning is open is a `question`; never hide it as `guidance`. (`cli coverage` reads a build,
   not a package; for a package `validate` is the coverage check, and `draft.json` `coverage` holds only the
   `out_of_scope` items.)
4. **Write the rules.** One rule is one checkable form of one or more statements, in `draft.json` `rules`:

   ```json
   {"id": "r_requester", "class": "hard", "level": "deny", "statement_ids": ["doc:1:1"],
    "quote": "<words copied from the statement>",
    "rule": {"type": "required_args", "tool": "create_ticket", "args": ["requester"]}}
   ```

   `pack` fills `type`, `checker`, the id and the quote inside `rule`. Tool and argument names come from `tools.json`
   only. `quote` is copied word for word from the statement (not needed for `guidance`, `verify`, `grounded_values`,
   `id_provenance`). A `verify` rule has a top-level `why`: what keeps a typed rule from fitting (`validate` warns
   without it).

   **Types** (slots inside `rule`; `when` = `{"<arg>": ["<value>", ...]}` on the tool's own arguments, `"*"` = the
   argument is given; `unless` = `[{"condition": "<text>"}]`). Use only these 15 types. Do not invent a new one: a
   statement that fits none is a `verify` or `guidance` rule or a question.

   | Type | Slots | Says |
   |---|---|---|
   | `required_args` | `tool`, `args`, `when?` | the call must give these arguments |
   | `allowed_values` | `tool`, `arg`, `values`, `when?` | the argument is one of the values |
   | `derived_value` | `tool`, `arg`, `inputs`, `table` | the argument is a lookup of other arguments; `table` keys are the input values joined by `\|` |
   | `transition` | `tool`, `arg`, `id_arg`, `edges` | a status moves only along `edges` (`{"from": ["to", ...]}`) |
   | `reference_status` | `tool`, `arg`, `field`, `allowed` | the record whose id is in `arg` has `field` in `allowed` |
   | `membership` | `tool`, `arg`, `group_arg` | the record in `arg` is a member of the one in `group_arg` |
   | `role_permission` | `tools`, `roles`, `effect: "forbid"` | these roles may not call these tools |
   | `record_condition` | `tools`, `field`, `op`, `value`, `effect: "forbid"`, `when?` | not for a record whose field meets the condition; `op` is `contains`, `equals`, `in`, `gt`, `ge`, `lt`, `le` or `empty`; `gt`, `ge`, `lt` and `le` compare two ISO dates (`2025-02-25`, or with a time) as dates; any other form of date is read by its first number, so write it in ISO form or use a `verify` rule and a policy question (`date_compare`); a `threshold` never compares a date |
   | `threshold` | `tool`, `arg`, `op`, `value`, `then`, `approval_tool?` | a number limit; `op` is `gt`, `ge`, `lt`, `le`; `then` is `forbid` or `requires_approval` |
   | `format` | `tool`, `arg`, `pattern` | the whole value matches a regular expression (a bad expression is `bad_pattern`) |
   | `duty` | `trigger: {tool, when?}`, `then: {tool, when?}`, `unless?` | after the trigger call, the agent must also make the `then` call |
   | `grounded_values` | none | global: optional arguments only from the request or tool results |
   | `id_provenance` | none | global: ids only from tool results or the request |
   | `verify` | `tools`, `statement` | a model reads the statement at run time; use only when no typed rule fits |
   | `guidance` | `text`, `tools?` | advice shown to the agent, never checked |

   **Class and level.** Set both on every rule.
   - `class: hard`: decided by code or Cedar (every type except `verify` and `guidance`). `level: deny` for a clear typed
     rule; `remind` for `duty`, for a role rule that comes from a grant list, and when you doubt the reading;
     `block_once` for a typed rule that reads data the agent may not have read yet; `observe` for `grounded_values`,
     `id_provenance` and a rule with `schema_conflict` (below).
   - `class: judgment`: a `verify` rule, `level: remind`.
   - `class: guidance`: a `guidance` rule, `level: observe`. General statements such as "never assume data" stay here, or
     use `id_provenance` and `grounded_values`.

   **Rules learned for this step** (each one came from a rule that was wrong before):
   - A property that a record must have binds the call that **creates** it, not a partial update. Put `required_args` and
     `allowed_values` on the create tool. Put one on an update tool only when the statement has a condition, and give the
     rule the matching `when`.
   - A statement that lists what a role may do is a grant list. Write `role_permission` rules that forbid that role on
     every other write tool in `tools.json` (tools that create, update, delete, link, send, assign, close, approve and so
     on), `level: remind`, and a policy question about the grant. "Only managers may close" forbids the other named
     roles on the close tool. Use only roles that the policy names (`validate` checks that each role is in the text).
   - Try typed rules before `verify`. Write `verify` only when you can say why no typed rule fits, and say it in `why`.
   - Never put a list of values (`allowed_values`) on an argument that holds a record identifier (name ends in `_id` or
     the description says id). A statement about the status or kind of the record that the id names is
     `reference_status`.
   - If the policy lists a value that the tool's own `enum` does not accept, write the rule as the policy says, add
     `"schema_conflict": true` inside `rule`, set `level: observe`, mention it in your summary and add a question.
   - A rule that holds **only inside one procedure** is not a rule: write it as a `must_not` of that procedure (step 5).
     Rules have no scope; `when` reads the tool's arguments only.
   - Two statements that conflict: link them (`exception_of`, `supersedes`; see below). If nothing decides (same date,
     same rank, no link), keep both rules at `level: remind` and write a question that lists both in `rule_ids`.
   - Cross-references go in `draft.json` `links`: `{"type": "refers_to" | "defines" | "exception_of" | "supersedes" |
     "depends_on" | "source_of_truth", "src": "<statement or section id>", "dst": "<statement or section id>",
     "quote": "<words>"}`. A section id is `doc:4.2`. A reference whose target does not exist is not written as a link.
   - What the agent must know before it acts is not shown by a rule: the rule is only checked when a call is made. Put it
     in `global.global_rules` as well (step 5).
5. **Write the procedures.** When the policy tells how to handle cases in numbered or bulleted steps (an SOP), each such
   section is one procedure card in `draft.json` `procedures`. Keep the steps close to the text, in order, and put what
   an agent must remember on the card. The pattern, in general words:

   | Field | Content |
   |---|---|
   | `id`, `name`, `category`, `src` | the section number and title as the policy writes them; the group it belongs to |
   | `applies_when` | the trigger, in the policy's words |
   | `inputs` | `[{"fact": ..., "source": ...}]`: what the agent must know and where it is read |
   | `decisions` | `[{"id", "rule", "then", "else"}]`: every branch the text states ("if ... then ... otherwise ...") |
   | `steps` | `[{"id": "<statement id>", "text"?, "action"?, "required"?, "condition"?, "quote"?}]` in order, one per `must` statement of the section |
   | `must_not` | strings, see below |
   | `preconditions`, `exceptions`, `templates`, `escalations` | items with a `quote`, see below |
   | `links` | `[{"type": "next", "to": "<procedure id>", "when": "<text>"}]`: only for a procedure that **always** follows this one. A choice between procedures is a `decisions` entry (a `next` link opens that procedure's text with the case: a triage with a dozen `next` links gives a reply of tens of thousands of characters) |
   | `deadlines`, `notes`, `priority` | time limits as written; hints such as which cases are urgent (all three are shown to the agent) |

   **Steps.** `id` is the statement id. `text` is the statement's words; leave it out and `pack` fills it. It may join a
   statement with the one the converter split from it (a sentence and its "Subject: ..." line), and then it must still
   contain the statement. `validate` checks that `text` is in the policy word for word. One more field makes the same
   package work for the guide and the SOP loop: they find the procedure that a write belongs to by the tool of its step,
   then check preconditions and templates for that write (neither checks that a step was skipped). `action` is the tool
   of `tools.json` that the step calls, only when the step is one call of exactly one tool; `"reply"` means a reply to the
   user with no tool (a tool that sends a mail reply is a tool, and its name is the action); leave it out when the step
   names no tool or several. `required: false` for an optional step, `condition` for the words
   of an "if" the step depends on. `quote` is optional (the words, as `text`).

   **Must-nots.** A `must_not` is a prohibition or a hold, and a gate model reads it on each acting call with none of the
   text around it. So each one must **stand alone**: when the prohibition holds only under a condition, write the
   condition first, then the prohibition, both copied from the text (for example "A visitor asks for a spare key. Never
   hand it over." from two sentences of the text, never only "Never hand it over."). A text that says "stop here" under
   several conditions is a step with a `condition` or a `decisions` entry, never a bare must-not. An instruction ("Send
   the invoice", "Stop here") is a step, never a must-not; a description of another actor is not one either. `validate`
   checks that a must-not is the policy's words, or a condition and a prohibition of at least 3 words each, each copied
   word for word, and warns (`bare_must_not`) on one of fewer than 5 words or one that starts with a pronoun. Keep the statement as a step of the procedure as well (step 3).
   `global.escalation.must_not` holds only prohibitions that **one tool call** can break (never send an access code to a
   visitor, never email a card number). General statements never go to a per-call model: write them as `guidance`
   rules.

   **Words of the policy.** the `rule` of each `decision` (its `then` and `else` are flow text and are not checked), `global_rules`, the steps of `triage`, `housekeeping` and `escalation`,
   `definitions`, `notes` and `applies_when` may be shortened, but copy the policy's words where you can: `validate` warns
   (`not_in_text`) on a text that is neither in the policy nor made of up to three runs of at least 3 words that are, and
   on template text, a `literal` value and the fixed part of a `template` value that are not in the policy word for word
   (a `literal` on an argument with an enum, or a boolean, is a value of the tool and is not looked up). Read each
   warning: either copy the words, or the text is a summary and you check it by hand.

   **Table rows.** A table row is one statement, `cell | cell | cell`. The header row is not a statement: the column
   names of a row are in its `source.table_header` in `set.json`. Read each cell with the column name above it, and
   write a rule or a step from the row as a whole.

   **Parts.** Each item has a `quote` (words copied from the statement, at least 4) and the fields below; the procedure
   text shows these to the agent.

   | Part | Fields | Use |
   |---|---|---|
   | `preconditions` | `condition`, `if_not` (`stop`, `hold`, `escalate` or `ask`), `quote` | what must hold before the procedure starts |
   | `exceptions` | `when`, `then`, `quote` | a case where it must not run, must stop, or changes |
   | `templates` | `step` (number), `artifact`, `template`, `quote` | the exact format of one step's output |
   | `escalations` | `when`, `notify`, `channel`, `quote` | who is told, and where |

   **`global`: shapes the runtime reads.** `global` holds what applies to every case. Only these keys are read; another
   key is dropped (`validate` warns), and a value of another shape crashes the procedure store or is lost (`validate`
   reports `invalid_field`). Leave out any key the policy has nothing for.

   | Key | Shape |
   |---|---|
   | `global_rules` | `[{"id", "src", "rule", "attach_when"?}]`: text shown first to the agent; `attach_when` is a regular expression: the rule is also shown with each procedure whose text matches it |
   | `triage`, `housekeeping`, `escalation` | `{"id", "src", "name", "steps": [text], "must_not": [text]}` |
   | `priority` | `{"name", "src", "levels": [text]}` |
   | `routing` | `{"<category>": {"<field>": "<value>"}}`: where each category goes (a procedure's `category` picks its row) |
   | `templates` | `{"<number>": {"number", "name", "slots": [text], "text"}}`, keyed by the number as steps cite it ("Template 7" gives `"7"`, "Template C-2" gives `"C-2"`); `text` copied exactly, with its `[placeholders]` |
   | `calendar` | `{"workdays": [0, 1, 2, 3, 4], "holidays": ["2026-12-25"]}`: weekday numbers (0 is Monday) and ISO dates |
   | `definitions` | `[{"term", "meaning", "source": {"statement": "<id>"}}]` |

   - Copy templates and fixed texts exactly, with their `[placeholders]`.
   - **A step that carries a value that can be checked must not stay only a step**, because a step cannot enforce a
     value. "Set the status to X", "use this subject line", "assign to this queue", "due in 5 business days": route it
     to a typed rule if it holds whenever that tool is called, whatever the case (step 4), or else to an obligation with
     a `literal`, `template` or `computed` value (step 6). Keep the step as well; it is the outcome of the statement.
6. **Write the obligations.** An obligation is work that the policy requires and that the agent might leave undone: a
   write, a file, a message or a reply with content the text prescribes. Write one for each step that does that
   (not for steps that only read). In `draft.json` `obligations`:

   ```json
   {"id": "ob_2_status", "origin": "package", "scope": {"procedure": "2"},
    "effect": {"kind": "write", "tools": ["update_asset"], "match": {"status": {"pattern": "{status}"}}},
    "values": {"status": {"from": "literal", "value": "in_transit"}},
    "check": "code", "level": "observe",
    "source": {"section": "2", "quote": "<at least 4 words copied from the step>"}}
   ```

   - `origin`: always `package` here. (`request`, `data` and `duty` items are made at run time.)
   - `scope`: `{"procedure": "<id>"}` (a procedure of the draft). `{}` (no `procedure`) is an obligation that applies to
     every case in every procedure, such as a housekeeping step done after each case. `"when": {"<key>": "<value>"}`
     is for a step that applies in one branch only: at run time a model answers each key from the case, with quoted
     evidence, and with no answer the item is not sent. So prefer the `id` of a `decisions` entry of the procedure as the
     key (`"when": {"audit_hold": "no"}`), because that answer is the branch that was already chosen.
   - `effect.kind`: `write`, `file`, `message` or `reply`; `tools`: tool names from `tools.json` (required for `write`).
   - **A file obligation lists every tool that can write a file.** The agent can write a file with a tool that is made for it (a
     file or sheet writer) or with code (`executePython`, `executeBash`). Name all of them in `tools`. An obligation that
     names only the code tools stays open when the agent uses the file tool, and the follow-up then asks for work that exists.
     A pattern is allowed only for an argument that every listed tool has (see below); where there is none, write no `match`:
     a successful call of any listed tool then meets the obligation, and its content is not compared.
   - **A tool the agent does not have gets no obligation.** `tools.json` can list tools that a simulated user or the
     customer's own device runs (a `user__` prefix is the usual sign). The agent never calls them, so a step that only they
     do is not work the agent can leave undone: write no obligation for it and keep it as a step. A step that needs both an
     agent tool and such a tool gets the obligation for the agent tool only.
   - **Value matching.** A value is compared with a call only through a tool argument. `effect.match` maps an argument of
     the tool to `{"pattern": text}`; the text holds `{name}` (or `{name:format}`, a date format such as `%d%m%Y`) for the
     values of the obligation, and `{{` `}}` for real braces. A value is **matched** when a pattern names it, directly or
     through a matched `template` value, and only a matched value is compared with the call. A value no pattern names is
     an **unmatched value**: it is shown in the follow-up as information and never compared (`validate` warns,
     `unbound_value`). So for each value you want checked, name the argument that carries it, with the schema's name (an
     argument `channel_id` is not `channel`; if the argument holds an id, the value is a `case_fact` the agent looks up).
     `match` arguments must exist in the schema of **every** tool of the obligation, and every `{name}` must be a key of
     `values`. How the pattern is tested depends on the argument: an argument with an enum or a boolean is a full match;
     a **code argument** (`code` of `executePython`, `command` of `executeBash`) or an argument that holds JSON text
     (a string the tool parses) is a contains test, where "found" counts as met and "not found" as unknown, never wrong, so
     write the fragment that must appear (`"code": "kiln_log.csv"`, `"payload": "\"state\": \"fired\""`); an obligation on both code
     tools has one pattern for `code` and one for `command`; `to`, `cc` and `bcc` match one address of a list or of
     comma-separated text; any other text argument is a contains test after folding case, spaces, dashes and quotes.
   - `values`: each named value has an origin in `from`, and the key that origin needs:

     | `from` | Use when | Other keys |
     |---|---|---|
     | `literal` | the text gives the exact value | `value` |
     | `template` | the text gives a format with placeholders; its `{name}` are other values | `pattern` |
     | `computed` | code can compute it (a date window, a business-day count, a lookup matrix) | `fn` (the calculation in words), `args?` |
     | `case_fact` | it is a field of a record the agent reads for the case | `field`, `record` (which record) |
     | `request` | the request states it | `about` (what the request states) |

     Example of a `template`: a batch label that the text writes as "KILN-[Batch Code]-[Firing Date DDMMYYYY]":
     `"match": {"label": {"pattern": "{label}"}}` and `"values": {"label": {"from": "template", "pattern":
     "KILN-{batch}-{fired_on:%d%m%Y}"}, "batch": {"from": "case_fact", "field": "batch_code", "record": "the firing
     request"}, "fired_on": {"from": "case_fact", "field": "firing_date", "record": "the firing request"}}`.

     "Wrong" is said only of a **matched** `literal`, `template` or `computed` value that the call contradicts; a
     `case_fact` or `request` value is never called wrong. A value you cannot derive from the text is **left out**: the
     item then names the work without a value. Never guess one. A `literal` whose value is open (the question is which
     exact value the policy means) is named in a policy question's `open_values`, and `validate` refuses that literal
     (`literal_open_question`): leave the value out until the owner answers. Only the named value is refused, so write
     a question that names the value and not only the statement.
   - `check: "code"`, `level: "observe"`. Never set `follow_up`: that level is chosen later by measurement.
   - `source.quote`: copied word for word from the statement. It also gives the statement its outcome.
7. **List the policy questions.** Write one for each thing that the owner of the policy must decide and that you could
   not: an ambiguous word or threshold, two statements that tie, a role grant, a value the tool does not accept, a
   statement that depends on something the text does not define. In `draft.json` `policy_questions`:
   `{"statement_ids": [...], "rule_ids": [...] (optional), "open_values": ["<obligation id>.<value name>"] (optional),
   "question": "<one decision, in plain words>", "default_level": "remind"}`. `rule_ids` are rules of the draft;
   `open_values` names each obligation literal that depends on the answer (step 6). `default_level` is the level the related rules keep until the owner
   answers (`observe`, `remind`, `block_once` or `deny`; usually `remind`). Do not ask about things the text states.
8. **Hand off the decision tests to another agent.** Do not write them yourself: tests written by the author of the
   rules repeat the author's mistakes. Start a separate agent (a sub-agent, or a new session when the harness has none)
   that has never seen `draft.json` or any rule, in a **clean folder** that holds only the policy documents,
   `tools.json` and `set.json` (for statement ids). Give it this task:

   > Read only the files in this folder. Do not open, search for or use any file outside it, and never any rubric, grader,
   > verifier, test task, answer file, run result or earlier package, even if you can reach one. For each statement of the
   > policy that a single tool call can break, write up to 4 test cases: one call the assistant could make and whether the
   > policy allows it (`allow`) or the call breaks that statement (`deny`). Use at least one `deny` and one `allow` when
   > the statement can be broken by one call; prefer boundary cases (a value just outside the allowed list, a record in the
   > wrong state, the wrong role). When the decision depends on records read earlier, give them as `reads`. An `allow` case
   > must follow the **whole** policy, not only that statement (every mandatory input given, the record read before it is
   > changed). A `deny` case must break only that statement. Skip a statement that no single call can break (a heading, a
   > definition, system behavior, a missing later step). Use only tool and argument names from `tools.json`. Write a JSON
   > list to `tests.json`.

   Each case has exactly these fields (the separate audit tool of the evaluation repository, `rule_audit.py tests-run`,
   reads them; it is not part of this repository):

   | Field | Content |
   |---|---|
   | `tool` | tool name (required) |
   | `args` | arguments of the call, an object |
   | `reads` | `[{"tool", "args", "result"}]`: calls made before, with the full record as `result`; the record holds its own id under the argument name the tools use |
   | `role` | the logged-in user's role as the policy names it; leave out if the policy has no roles |
   | `expected` | `"allow"` or `"deny"` (required) |
   | `why` | a short reason |
   | `statement_id` | the id from `set.json` of the statement tested |
   | `by` | the model that wrote the case |

   Use a strong model for the writer. Do not change a test to make the package pass it. Pass the file to `pack` with
   `--tests tests.json`; the tests are stored in the package's `tests` and never change its rules.
9. **Pack and validate.**
   ```
   python -m polisys.cli pack set.json draft.json -o package.json --harness <your harness> --model <your model> --tools tools.json [--tests tests.json]
   python -m polisys.cli validate package.json --tools tools.json
   ```
   `pack` stops with `ERROR pack` and writes nothing if an id does not exist (statements, steps, `scope.procedure`,
   `links.to`, `rule_ids`, definition sources), a step `text` does not contain its statement, or a test has no tool or no
   `allow`/`deny`. It records `generator` as `{"kind": "skill", "harness", "model"}`. `version` and `policy_hash` are one
   hash of the statements' texts as the converter split them, so a change of the converter can move it for the same
   documents. `validate` exits with code 1 while errors remain: fix `draft.json`, `pack` and `validate` again until it
   prints `0 errors`. Never use `--legacy`. The codes you will meet:

   | Code | Means | Fix |
   |---|---|---|
   | `coverage_incomplete` | a `must` or `must_not` statement has no outcome | step 3 |
   | `quote_not_verbatim` | a quote, a step `text` or a must-not is not the policy's words | copy again from the statement (steps 4, 5, 6) |
   | `rule_no_trigger` | a rule names no tool | add `tool` or `tools`, or make it `guidance` |
   | `unfilled_slot`, `invalid_slots` | an empty, placeholder or wrong-typed slot | fill it |
   | `invalid_field` | a field of the wrong type, or a `global` value of the wrong shape | the shape table of step 5 |
   | `unknown_tool`, `unknown_arg`, `value_not_in_enum` | a name or value not in `tools.json` (also a step `action` and a `match` argument) | use the schema's names |
   | `bad_pattern` | a `format` rule with no pattern or a regular expression that does not compile | fix the expression |
   | `identifier_values` | a value list on a record-id argument | `reference_status` |
   | `unknown_class`, `unknown_level` | `class` or `level` not allowed | classes `hard`, `judgment`, `guidance`; levels `observe`, `remind`, `block_once`, `deny` |
   | `unknown_role` | a role of a role rule that the policy never names | use the policy's role names |
   | `unknown_value_name`, `literal_open_question`, `unknown_open_value` | a `{name}` that is not a value; a literal that a question lists as open; an `open_values` entry that is no `<obligation id>.<value name>` | step 6, step 7 |
   | `rule_conflict` | two rules disagree | step 4, conflicts |
   | `link_target_missing` | a link end does not exist | fix the id or remove the link |
   | `unknown_field` (warning) | a key the model does not have, or a `global` key the runtime drops | remove it |
   | `unbound_value`, `verify_without_why` (warning) | an unmatched value (no match pattern names it); a `verify` rule with no `why` | step 6; step 4 |
   | `not_in_text`, `bare_must_not` (warning) | a global rule, decision, template text, literal, definition, note or `applies_when` not in the policy's words; a must-not that cannot be read alone | "Words of the policy", "Must-nots" in step 5 |
   | `date_compare` (warning) | a date in a `threshold`, or a date that is not in ISO form in a `record_condition` comparison: it is read by its first number | write the date in ISO form in a `record_condition`; otherwise use `verify` and a policy question |
   | `schema_conflict` (warning) | the policy lists a value the tool does not accept | keep it, mention it in the summary |

## Rules for you

- Use only what the documents say. Do not add rules, values, tools or conditions that the text does not state.
- Do not write anything specific to one task, case or data record.
- Keep the source of every rule and the exact quote of every item that has a `quote`.
- Never raise a level to make a test pass or lower it to make `validate` pass. A level follows from the text (step 4).
- When two statements conflict and nothing decides, keep both at `remind` and ask (step 7).

## Output

`package.json` with `0 errors` from `validate`, and a short summary: documents, statements, rules by class and level,
procedures, must-not rules, obligations by value origin and how many values are unmatched, policy questions, `out_of_scope`
statements with reasons, schema conflicts, and the number of tests.

## Worked example (made up)

`policy.md`:

```text policy.md
# Facilities Desk Policy

## 1. Tickets

1. Every ticket must have a requester.
2. Priority must be one of low, normal or urgent.
3. Do not close a ticket that has an open child ticket.
4. Use good judgment when a requester is upset.

## 2. Moving an asset

When an asset moves to another building:
- Read the asset record before changing it.
- Update the asset record with the new building.
- Set the asset status to in_transit until the move is confirmed.
- Send Template 1 to the old building manager. Subject: MOVE-[asset id].
- Do not move an asset that is under audit hold.

## 3. Building rules

- Staff must wear a badge in the building.
- An audit hold is a flag on the asset record that the audit team sets.
- The audit team clears an audit hold after its review.

## 4. Security

- If a visitor asks for a door code, tell the security lead.
- Never send a door code to a visitor.

## 5. Calendar

- The desk is closed on 2026-12-25.

## 6. Templates

Template 1: Move notice

Dear [Manager], the asset [asset id] is moving to [new building].
```

`tools.json`:

```json tools.json
[
 {"name":"create_ticket","description":"Create a ticket","inputSchema":{"type":"object","properties":{"requester":{"type":"string"},"priority":{"type":"string","enum":["low","normal","urgent"]},"title":{"type":"string"}},"required":["title"]}},
 {"name":"close_ticket","description":"Close a ticket","inputSchema":{"type":"object","properties":{"ticket_id":{"type":"string","description":"id of the ticket"}},"required":["ticket_id"]}},
 {"name":"get_asset","description":"Read an asset record","inputSchema":{"type":"object","properties":{"asset_id":{"type":"string"}},"required":["asset_id"]}},
 {"name":"update_asset","description":"Update an asset record","inputSchema":{"type":"object","properties":{"asset_id":{"type":"string"},"building":{"type":"string"},"status":{"type":"string","enum":["active","in_transit","retired"]}},"required":["asset_id"]}},
 {"name":"send_message","description":"Send a message","inputSchema":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject"]}}
]
```

`python -m polisys.cli convert policy.md -o set.json` gives the statement ids `policy:1:1` ... `policy:6:2`.
`draft.json` (every statement that says `must` has an outcome: a rule, a step or `out_of_scope`; `policy:2:6` is a
fragment of the notice step and is joined into that step's `text`; a step with only an `id` gets the statement's words
from `pack`; the must-not of `global.escalation` is a prohibition one call can break, and the same statement also has a
`verify` rule as its outcome; every text in `global`, `decisions` and `applies_when` is in the policy's words):

```json draft.json
{
  "modality": {
    "policy:1:1": "must",
    "policy:1:2": "must",
    "policy:1:3": "must_not",
    "policy:1:4": "should",
    "policy:2:2": "must",
    "policy:2:3": "must",
    "policy:2:4": "must",
    "policy:2:5": "must",
    "policy:2:7": "must_not",
    "policy:3:1": "must",
    "policy:4:1": "must",
    "policy:4:2": "must_not"
  },
  "rules": [
    {
      "id": "r_requester",
      "class": "hard",
      "level": "deny",
      "statement_ids": [
        "policy:1:1"
      ],
      "quote": "Every ticket must have a requester.",
      "rule": {
        "type": "required_args",
        "tool": "create_ticket",
        "args": [
          "requester"
        ]
      }
    },
    {
      "id": "r_priority",
      "class": "hard",
      "level": "deny",
      "statement_ids": [
        "policy:1:2"
      ],
      "quote": "Priority must be one of low, normal or urgent.",
      "rule": {
        "type": "allowed_values",
        "tool": "create_ticket",
        "arg": "priority",
        "values": [
          "low",
          "normal",
          "urgent"
        ]
      }
    },
    {
      "id": "r_child",
      "class": "judgment",
      "level": "remind",
      "statement_ids": [
        "policy:1:3"
      ],
      "why": "Needs the child tickets of the ticket, which no argument of close_ticket carries.",
      "rule": {
        "type": "verify",
        "tools": [
          "close_ticket"
        ],
        "statement": "Do not close a ticket that has an open child ticket."
      }
    },
    {
      "id": "r_door",
      "class": "judgment",
      "level": "remind",
      "statement_ids": [
        "policy:4:2"
      ],
      "why": "Needs the meaning of the message body, which no typed rule can test.",
      "rule": {
        "type": "verify",
        "tools": [
          "send_message"
        ],
        "statement": "Never send a door code to a visitor."
      }
    },
    {
      "id": "r_upset",
      "class": "guidance",
      "level": "observe",
      "statement_ids": [
        "policy:1:4"
      ],
      "rule": {
        "type": "guidance",
        "text": "Use good judgment when a requester is upset."
      }
    },
    {
      "id": "r_security",
      "class": "guidance",
      "level": "observe",
      "statement_ids": [
        "policy:4:1"
      ],
      "rule": {
        "type": "guidance",
        "text": "If a visitor asks for a door code, tell the security lead."
      }
    }
  ],
  "procedures": [
    {
      "id": "2",
      "name": "Moving an asset",
      "category": "Assets",
      "src": "§2",
      "applies_when": "When an asset moves to another building.",
      "inputs": [
        {
          "fact": "whether the asset is under audit hold",
          "source": "the asset record (get_asset)"
        }
      ],
      "decisions": [
        {
          "id": "audit_hold",
          "rule": "an asset that is under audit hold",
          "then": "Do not move an asset",
          "else": "Update the asset record with the new building"
        }
      ],
      "steps": [
        {
          "id": "policy:2:2",
          "action": "get_asset"
        },
        {
          "id": "policy:2:3",
          "action": "update_asset"
        },
        {
          "id": "policy:2:4",
          "action": "update_asset"
        },
        {
          "id": "policy:2:5",
          "action": "send_message",
          "text": "Send Template 1 to the old building manager. Subject: MOVE-[asset id]."
        },
        {
          "id": "policy:2:7"
        }
      ],
      "must_not": [
        "Do not move an asset that is under audit hold."
      ],
      "templates": [
        {
          "step": 4,
          "artifact": "subject of the move notice",
          "template": "MOVE-[asset id]",
          "quote": "Subject: MOVE-[asset id]."
        }
      ],
      "links": []
    }
  ],
  "obligations": [
    {
      "id": "ob_2_status",
      "origin": "package",
      "scope": {
        "procedure": "2"
      },
      "effect": {
        "kind": "write",
        "tools": [
          "update_asset"
        ],
        "match": {
          "status": {
            "pattern": "{status}"
          }
        }
      },
      "values": {
        "status": {
          "from": "literal",
          "value": "in_transit"
        }
      },
      "check": "code",
      "level": "observe",
      "source": {
        "section": "2",
        "quote": "Set the asset status to in_transit until the move is confirmed."
      }
    },
    {
      "id": "ob_2_notice",
      "origin": "package",
      "scope": {
        "procedure": "2"
      },
      "effect": {
        "kind": "message",
        "tools": [
          "send_message"
        ],
        "match": {
          "subject": {
            "pattern": "{subject}"
          }
        }
      },
      "values": {
        "subject": {
          "from": "template",
          "pattern": "MOVE-{asset_id}"
        },
        "asset_id": {
          "from": "case_fact",
          "field": "asset_id",
          "record": "the asset record"
        }
      },
      "check": "code",
      "level": "observe",
      "source": {
        "section": "2",
        "quote": "Send Template 1 to the old building manager."
      }
    }
  ],
  "global": {
    "global_rules": [
      {
        "id": "G-hold",
        "src": "§3",
        "rule": "An audit hold is a flag on the asset record that the audit team sets."
      }
    ],
    "escalation": {
      "id": "escalation",
      "src": "§4",
      "name": "Security",
      "steps": [
        "If a visitor asks for a door code, tell the security lead."
      ],
      "must_not": [
        "Never send a door code to a visitor."
      ]
    },
    "calendar": {
      "workdays": [
        0,
        1,
        2,
        3,
        4
      ],
      "holidays": [
        "2026-12-25"
      ]
    },
    "templates": {
      "1": {
        "number": "1",
        "name": "Move notice",
        "slots": [
          "Manager",
          "asset id",
          "new building"
        ],
        "text": "Dear [Manager], the asset [asset id] is moving to [new building]."
      }
    },
    "definitions": [
      {
        "term": "audit hold",
        "meaning": "a flag on the asset record that the audit team sets",
        "source": {
          "statement": "policy:3:2"
        }
      }
    ]
  },
  "links": [],
  "coverage": [
    {
      "statement_id": "policy:3:1",
      "kind": "out_of_scope",
      "reason": "a behavior in the building: no tool call shows it"
    }
  ],
  "policy_questions": [
    {
      "statement_ids": [
        "policy:1:3"
      ],
      "rule_ids": [
        "r_child"
      ],
      "question": "Does a child ticket that is on hold count as open?",
      "default_level": "remind"
    }
  ]
}
```

`tests.json`, as the separate agent would write it:

```json tests.json
[
 {"tool":"create_ticket","args":{"title":"Leaking tap","priority":"normal"},"expected":"deny","why":"no requester","statement_id":"policy:1:1","by":"test-writer"},
 {"tool":"create_ticket","args":{"title":"Leaking tap","priority":"normal","requester":"Ana"},"expected":"allow","why":"requester given","statement_id":"policy:1:1","by":"test-writer"}
]
```

```
python -m polisys.cli pack set.json draft.json -o package.json --harness claude-code --model <model> --tools tools.json --tests tests.json
python -m polisys.cli validate package.json --tools tools.json     # 0 errors, 0 warnings
```
