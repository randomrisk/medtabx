r"""Utility package."""

__all__ = [
    "save",
    "load",
    "cat",
    "generate_random_split",
    "infer_series_stype",
    "infer_df_stype",
    "num_bytes",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "save" or name == "load":
        from .io import save, load
        return save if name == "save" else load
    elif name == "cat":
        from .concat import cat
        return cat
    elif name == "generate_random_split":
        from .split import generate_random_split
        return generate_random_split
    elif name == "infer_series_stype" or name == "infer_df_stype":
        from .infer_stype import infer_series_stype, infer_df_stype
        return infer_series_stype if name == "infer_series_stype" else infer_df_stype
    elif name == "num_bytes":
        from .memory import num_bytes
        return num_bytes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
