"""The rule types and who checks each one (``RULE_TYPES``, ``CHECKER``), the result of one check (``Check``), and the
limits the rule hooks share."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

RULE_TYPES = (
    "required_args",
    "allowed_values",
    "derived_value",
    "transition",
    "reference_status",
    "membership",
    "role_permission",
    "grounded_values",
    "id_provenance",
    "duty",
    "guidance",
    "record_condition",
    "threshold",
    "format",
    "verify",
)
CHECKER = {"role_permission": "cedar", "duty": "monitor", "guidance": "brief"}
FREE_TEXT = re.compile(r"description|notes?$|comment|message|subject|body|text|summary|title|worknote", re.I)
ID_VALUE = re.compile(r"^(?=.*\d)[A-Za-z]{2,}[A-Za-z0-9]*([_-][A-Za-z0-9]+)+$|^[A-Z]{2,}\d{3,}$")
SCOPE_MIN = 0.35  # decision-model answer below which a code rule is taken as not applying to the call
MAX_SAME_DENY = 3  # repeated-denial cap: the 4th identical denial is let through and recorded


@dataclass
class Check:
    rule: Dict[str, Any]
    ok: bool
    reason: str
    remedy: str = ""
