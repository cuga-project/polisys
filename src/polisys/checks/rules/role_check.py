"""The role check (part of ``RuleSetHooks``)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from polisys.checks.rules.rule_types import Check


class _RoleCheck:
    """``role_permission`` rules as Cedar policies (``_cedar``), decided for the acting user (``_role_check``)."""

    def _role_check(self, tool: str) -> Optional[Check]:
        rules = [r for r in self.by_tool.get(tool, []) if r["type"] == "role_permission"]
        if not rules or self.cedar is None:
            return None
        role = self.principal.get("role")
        if not role:
            return Check(rules[0], False, "the user's role is not known", "check the user information first")
        principal = f'User::"{self.principal.get("id", "me")}"'
        dec = self.cedar.is_authorized(
            principal,
            f'Action::"{tool}"',
            'Resource::"any"',
            [
                {
                    "uid": {"type": "User", "id": self.principal.get("id", "me")},
                    "attrs": {"role": role},
                    "parents": [],
                }
            ],
        )
        decided = next((r for r in rules if r["id"] in dec.policy_ids), rules[0])
        if not dec.allowed:
            return Check(
                decided,
                False,
                f"role `{role}` may not call `{tool}`",
                "stop and state the policy reason; do not try another way to make this change",
            )
        return Check(decided, True, f"role {role} allowed")

    @staticmethod
    def _cedar(rules: List[Dict[str, Any]]):
        if not rules:
            return None
        try:
            from polisys.checks.cedar import CedarEngine
        except ImportError:  # pragma: no cover
            return None
        text = ['@id("base_permit")\npermit(principal, action, resource);']
        for r in rules:
            acts = ", ".join(f'Action::"{t}"' for t in r.get("tools", []))
            roles = ", ".join(json.dumps(x) for x in r.get("roles", []))
            text.append(
                f'@id("{r["id"]}")\nforbid(principal, action in [{acts}], resource) '
                f"when {{ [{roles}].contains(principal.role) }};"
            )
        try:
            return CedarEngine("\n".join(text))
        except ImportError:  # pragma: no cover
            return None
