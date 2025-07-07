from importlib.metadata import version as get_version

try:
    __version__ = get_version("simple")
except Exception:
    __version__ = "unknown"