import numbers
from typing import Any

from synaxis.core import Input, Output, Param
from synaxis.systems import StaticSystem


def _ndim(value: Any) -> int:
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        return ndim
    if isinstance(value, numbers.Number):
        return 0

    print(type(value), type(value.get_value()), value.get_value())

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


class ProportionalController(StaticSystem):
    kp = Param()

    observation = Input()
    reference = Input()

    u = Output()

    def __init__(self, name=None) -> None:
        name = name if name is not None else "p_controller"
        super().__init__(name=name)

    def compute_outputs(self):
        return _matvec(-self.kp, self.observation - self.reference)
