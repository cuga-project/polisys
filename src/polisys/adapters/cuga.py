"""Glue for the CUGA agent (https://github.com/cuga-project): hooks on tools that come from a tool provider.

Needs the ``cuga`` package. Hooks themselves need none: pass ``polisys`` hook objects to ``CugaAgent(hooks=[...])``."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import BaseTool

from polisys.hooks import HookRegistry, wrap_tools_with_hooks


def wrap_provider_with_hooks(provider: Any, registry: HookRegistry) -> Any:
    """The same before/after_tool_call wrapping for tools that come from a tool provider
    (``CugaAgent(tool_provider=...)``), which ``wrap_tools_with_hooks`` on ``tools=`` never sees."""
    from cuga.backend.cuga_graph.nodes.cuga_lite.providers.base import ToolProviderInterface

    class HookedToolProvider(ToolProviderInterface):
        def __init__(self, inner: Any):
            self._inner = inner
            self._wrapped: Dict[int, BaseTool] = {}

        def __getattr__(self, name: str) -> Any:  # initialized, add_tools, ... of the inner provider
            return getattr(self._inner, name)

        def _wrap(self, tools: List[BaseTool]) -> List[BaseTool]:
            out = []
            for t in tools:
                if (t.metadata or {}).get("polisys_hooks"):
                    out.append(t)
                    continue
                if id(t) not in self._wrapped:
                    self._wrapped[id(t)] = wrap_tools_with_hooks([t], registry)[0]
                out.append(self._wrapped[id(t)])
            return out

        async def initialize(self):
            return await self._inner.initialize()

        async def get_apps(self):
            return await self._inner.get_apps()

        async def get_tools(self, app_name: str):
            return self._wrap(await self._inner.get_tools(app_name))

        async def get_all_tools(self):
            return self._wrap(await self._inner.get_all_tools())

    return HookedToolProvider(provider)
