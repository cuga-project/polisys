"""Cedar evaluation for compiled permit/forbid rules (needs the ``policy-gate`` extra: ``cedarpy``)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class CedarDecision:
    allowed: bool
    policy_ids: List[str] = field(default_factory=list)  # @id annotations of the policies that decided
    errors: List[str] = field(default_factory=list)


class CedarEngine:
    """Thin wrapper over cedarpy that reports decisions by the ``@id("...")`` annotation of each policy."""

    def __init__(self, policies: str):
        try:
            import cedarpy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("CedarEngine needs cedarpy: pip install 'polisys[cedar]'") from exc
        self.policies = policies
        # cedarpy names policies policy0..N in statement order; map those to the @id annotations.
        text = re.sub(r"//[^\n]*", "", policies)
        self._ids = [
            m.group(1) or f"policy{i}"
            for i, m in enumerate(re.finditer(r'(?:@id\("([^"]+)"\)\s*)?\b(?:permit|forbid)\s*\(', text))
        ]

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "CedarEngine":
        return cls(Path(path).read_text())

    def is_authorized(
        self,
        principal: str,
        action: str,
        resource: str,
        entities: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> CedarDecision:
        import cedarpy

        request = {"principal": principal, "action": action, "resource": resource, "context": context or {}}
        result = cedarpy.is_authorized(request, self.policies, entities)
        ids = []
        for reason in result.diagnostics.reasons:
            m = re.fullmatch(r"policy(\d+)", reason)
            ids.append(self._ids[int(m.group(1))] if m and int(m.group(1)) < len(self._ids) else reason)
        return CedarDecision(
            allowed=result.decision == cedarpy.Decision.Allow,
            policy_ids=ids,
            errors=[str(e) for e in result.diagnostics.errors],
        )
