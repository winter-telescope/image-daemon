try:
    from importlib.metadata import PackageNotFoundError, version  # Python 3.8+
except ImportError:
    # For older versions of Python, use the `importlib_metadata` backport
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("image-daemon")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"
