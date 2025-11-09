try:
    import importlib.metadata as importlib_metadata  # type: ignore
except Exception:
    import importlib_metadata  # type: ignore

__all__ = ("importlib_metadata",)
