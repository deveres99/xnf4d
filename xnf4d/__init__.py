"""
4D normal forms of one-turn maps for non-linear beam dynamics.
"""

from importlib import metadata

try:
    __version__ = metadata.version("xnf4d")
except metadata.PackageNotFoundError:  # not installed, e.g. run from the source tree
    __version__ = "unknown"
