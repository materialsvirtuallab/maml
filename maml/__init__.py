"""maml - materials machine learning."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matgl")
except PackageNotFoundError:
    pass  # package not installed
