"""Linear time-invariant dynamics."""

from __future__ import annotations

import numbers
from typing import Any, Self

from synaxis.core import Input, Output, Param, State, System


def _ndim(value: Any) -> int:
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        return ndim
    if isinstance(value, numbers.Number):
        return 0

    raise TypeError("value must be a numeric scalar or an array")


def _matvec(matrix: Any, vector: Any) -> Any:
    """Multiply a matrix by a vector without selecting an array backend."""
    matrix_ndim = _ndim(matrix)
    vector_ndim = _ndim(vector)

    if matrix_ndim == 0:
        return matrix * vector

    if vector_ndim == 0:
        return matrix[..., 0] * vector

    return (matrix @ vector[..., None])[..., 0]


class LTISystem(System):
    """Continuous linear time-invariant state-space system."""

    default_name = "lti_system"

    A = Param()
    B = Param()
    C = Param()
    D = Param()

    x = State()

    u = Input()

    def __init__(self, name: str | None = None, *, direct_feedthrough: bool = True) -> None:
        super().__init__(name=name)
        self._direct_feedthrough = direct_feedthrough

    @property
    def direct_feedthrough(self) -> bool:
        """Whether the output equation includes the direct term ``D @ u``."""
        return self._direct_feedthrough

    def clone(self) -> Self:
        return type(self)(name=self.name, direct_feedthrough=self.direct_feedthrough)

    @Output
    def y(self):
        y = _matvec(self.C, self.x)
        return y + _matvec(self.D, self.u) if self.direct_feedthrough else y

    @x.derivative
    def dx(self):
        return _matvec(self.A, self.x) + _matvec(self.B, self.u)
