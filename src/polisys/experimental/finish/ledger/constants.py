"""The names the parts of the ledger share: the statuses of an item, the run types, the limits on text, and the patterns
that sort the arguments of a call (code, address, free text, secret)."""

from __future__ import annotations

import re

from polisys.tool_kind import ID_ARG

UNBOUND, OPEN, MET, NOT_APPLICABLE, HELD, UNKNOWN, WRONG = (
    "unbound",
    "open",
    "met",
    "not_applicable",
    "held",
    "unknown",
    "wrong",
)
ONE_REQUEST, SOP_LOOP, CONVERSATION = "one_request", "sop_loop", "conversation"
SEND_BACKS = 2  # CUGA acts on this many before_reply messages per request (``CugaAgent._invoke_with_hooks``)
FOLLOW_UPS_BY_LEVEL = {0: 0, 1: 0, 2: 1, 3: 2}  # design section 5, the table of levels
# a whole SOP file, as the requirements tracker reads them
READ_CHARS, DATA_CHARS = 80_000, 160_000
MAX_CASES = 12  # as the loop
MAX_LINES = 40  # lines of one follow-up; the audit has all items
QUOTE_CHARS = 160
MIN_EVIDENCE_CHARS = 12  # a shorter quote about a case is found almost anywhere
MIN_VALUE_CHARS = 2  # a value and the quote that holds it: an id or a last name can be short
MIN_KEY_CHARS = 3
MAX_KEY_CHARS, MAX_KEY_WORDS = 40, 4  # a key identifies a case: a name, an id, an address
MAX_VALUE_CHARS = 160  # a value from a record is one short line
SHOWN_REASON_CHARS = 240

CODE_TOOL = re.compile(r"execute_?(python|bash)", re.I)
CODE_ARGS = ("code", "command")
ADDRESS_ARGS = ("to", "cc", "bcc")
# Arguments that say what a call does and not what it is about. The rest identify the target of the call.
FREE_TEXT_ARG = re.compile(
    r"body|text|comment|description|message|content|notes?$|reason|workaround|problem|statement|details|fields",
    re.I,
)
# problem_id is an id, whatever else its name says. A block compares every other argument, so that a new name cannot pass.
IDENTITY_CHARS = 200
FILE_TOOL = re.compile(r"file|sheet|workbook|document", re.I)  # with a code tool: what can write a file item
SECRET_ARG = re.compile(
    r"pass(word|wd|phrase)?$|pwd|secret|token|api_?key|access_?key|private_?key|credential|authorization",
    re.I,
)
REDACTED = "[redacted]"
# A sign in code that it writes. A limit, named in design section 5: code that writes through its own helper shows no sign and
# gives ``unknown``, and code that writes something else and only prints the literal counts as a sign.
WRITES_IN_CODE = re.compile(
    r"\.to_(excel|csv|json|parquet|pickle|sql|html|xml)\s*\(|ExcelWriter|\.save\s*\(|\.write(lines|rows?)?\s*\("
    r"|write_(text|bytes)\s*\(|\.dump\s*\(|\.commit\s*\(|\bopen\s*\([^)]*[\"'][wax][bt+]*[\"']"
    r"|\bshutil\.(copy|move)|\bos\.(rename|replace|remove)|\.mkdir\s*\(",
    re.I,
)
WRITES_IN_SHELL = re.compile(
    r"(?<![0-9&<>=-])>>?\s*(?!/dev/null|&)[\w./~\"'$]|\btee\s|\bsed\s+-i|(^|[;&|\s])(cp|mv|touch|rm|mkdir)\s",
    re.I,
)
TEMPLATE_REF = re.compile(r"\btemplate\s+#?\s*([\w.-]*\d[\w.-]*)", re.I)
POLICY_TOOL_PREFIX = "policy__"  # the procedure store's own tools: they read the SOP, not the case
NOTE_PREFIX = "[Policy note"

_QUOTES = dict.fromkeys(map(ord, "\"`‘’‚‛“”„‟′″"), "'")


def _is_free_text(arg: str) -> bool:
    return bool(FREE_TEXT_ARG.search(arg)) and not ID_ARG.search(arg)
