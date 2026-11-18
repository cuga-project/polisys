"""Run time: the profile (``profile``), the assembly of one run (``assembly``), control levels (``levels``), the audit
(``trace``) and the dry run (``dry_run``).

    from polisys.runtime import RunContext, build_runtime
"""


def __getattr__(name: str):
    """The names of ``assembly``, read on first use: ``polisys.runtime`` was that module before the folders existed."""
    import importlib

    try:
        return getattr(importlib.import_module("polisys.runtime.assembly"), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
