try:
    import matplotlib  # noqa: F401
except ImportError as e:
    raise ImportError(
        "dise.plotting requires matplotlib. "
        "Install it with:  pip install dise[plotting]"
    ) from e

from .plotter import ExplainerPlotter

__all__ = ["ExplainerPlotter"]
