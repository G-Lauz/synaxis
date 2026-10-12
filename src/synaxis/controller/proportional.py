import numbers
from typing import Any

from synaxis.core import Input, Output, Param, System


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


class ProportionalController(System):
    default_name = "p_controller"

    kp = Param()

    observation = Input()
    reference = Input()

    @Output
    def u(self):
        return _matvec(-self.kp, self.observation - self.reference)
