"""Notebook-friendly facade for :mod:`zhong2025.plot`.

Team notebooks may simply ``import plot`` from the shared workspace.  Package
users can use the identical API through ``from zhong2025 import plot``.
"""

from zhong2025.plot import *  # noqa: F401,F403
from zhong2025.plot import __all__

