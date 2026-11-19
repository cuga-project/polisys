"""Run time: what one run is made of.

- ``profile``: the profile file and its settings. ``setting_types``: the types settings share.
- ``context``: what a task gives (``RunContext``), what the runner gets (``RuntimeParts``), what a component adds (``Piece``).
- ``registry``: the list of components, in hook order.
- ``assembly``: ``build_runtime`` and what the components of one run share (``Assembly``).
- ``settings_check``: settings that cannot work together.
- ``levels``: control levels. ``trace``: the audit. ``dry_run``: the wiring description without a model.

    from polisys.runtime import RunContext, build_runtime
"""


def __getattr__(name: str):
    """The names of ``assembly`` and ``settings_check``, read on first use: ``polisys.runtime`` was one module that held
    both before the folders existed."""
    import importlib

    for module in ("assembly", "settings_check"):
        try:
            return getattr(importlib.import_module(f"polisys.runtime.{module}"), name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
