import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from .core import (
    CompiledSystem,
    Input,
    Noise,
    Output,
    Param,
    Signal,
    State,
    StateDerivative,
    System,
    equation,
)
from .systems import DynamicSystem, StaticSystem

try:
    __version__ = version("synaxis")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

_SUBMODULES = frozenset({"blocks", "controller", "core", "diagram", "dynamics", "solvers", "systems"})

if TYPE_CHECKING:
    from . import blocks, controller, core, diagram, dynamics, solvers, systems


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULES)


__all__ = [
    # Signals
    "Input",
    "Noise",
    "Output",
    "Param",
    "Signal",
    "State",
    "StateDerivative",
    # Systems
    "DynamicSystem",
    "StaticSystem",
    "System",
    "equation",
    # Runtime
    "CompiledSystem",
    # Subpackages
    "blocks",
    "controller",
    "core",
    "diagram",
    "dynamics",
    "solvers",
    "systems",
    "__version__",
]
