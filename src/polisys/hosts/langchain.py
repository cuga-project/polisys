"""Glue for LangChain: ``wrap_tools_with_hooks`` runs ``before_tool_call`` and ``after_tool_call`` around each LangChain tool,
and ``HooksCallback`` reports each model output to ``after_llm_call``. A host that is not built on LangChain calls the
methods of ``HookRegistry`` at its own points and needs nothing from this file."""

from __future__ import annotations

from typing import Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool, StructuredTool

from polisys.hooks import HookBlocked, HookRegistry


def wrap_tools_with_hooks(tools: List[BaseTool], registry: HookRegistry) -> List[BaseTool]:
    """before_tool_call / after_tool_call around each tool. A block raises HookBlocked with the hook's message."""

    def wrap(tool: BaseTool) -> BaseTool:
        schema = tool.args_schema
        order = list(getattr(schema, "model_fields", {}) or {}) if isinstance(schema, type) else []

        async def _hooked(*pos: Any, **kwargs: Any) -> Any:
            # CodeAct calls the coroutine directly, so positional arguments arrive here; map them to
            # the schema fields in signature order (an unwrapped tool accepts them the same way).
            if len(pos) > len(order):
                raise TypeError(
                    f"{tool.name}() takes at most {len(order)} positional arguments ({len(pos)} given)"
                )
            kwargs = {**dict(zip(order, pos)), **kwargs}
            args = {
                k: v for k, v in kwargs.items() if v is not None
            }  # schema defaults are not part of the call
            blocked = registry.before_tool_call(tool.name, args)
            if blocked:
                raise HookBlocked(blocked)
            result = await tool.ainvoke(args)
            note = registry.after_tool_call(tool.name, args, result)
            return registry.attach_note(result, note) if note else result

        return StructuredTool.from_function(
            coroutine=_hooked,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            metadata={**(tool.metadata or {}), "polisys_hooks": True},
        )

    return [wrap(t) for t in tools]


class HooksCallback(BaseCallbackHandler):
    """Reports each model output to after_llm_call (observe only)."""

    def __init__(self, registry: HookRegistry):
        self.registry = registry

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            gen = response.generations[0][0]
            text = getattr(getattr(gen, "message", None), "content", None) or getattr(gen, "text", "")
        except Exception:
            return
        if isinstance(text, str) and text.strip():
            self.registry.after_llm_call(text)
