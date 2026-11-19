"""The policy model (``model``) and the policy package file (``files``): load, save, convert.

    from polisys.package import load_package
"""

from polisys._aliases import names_of

# The names of ``files``, read on first use: ``polisys.package`` was that module before the folders existed.
__getattr__ = names_of(__name__, "polisys.package.files")
