"""The policy model (``model``) and the policy package file (``files``): load, save, convert.

    from polisys.package import load_package
"""


def __getattr__(name: str):
    """The names of ``files``, read on first use: ``polisys.package`` was that module before the folders existed."""
    import importlib

    try:
        return getattr(importlib.import_module("polisys.package.files"), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
