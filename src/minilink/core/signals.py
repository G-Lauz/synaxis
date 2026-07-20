from __future__ import annotations

import abc
import contextlib
import contextvars
import enum
import uuid
from typing import Any, Callable, Generic, Iterator, List, Optional, Set, Tuple, TypeVar, Union, cast, overload

import jax
import numpy as np

from .components import ComponentDescriptor, ComponentKind

T = TypeVar("T")
A = TypeVar("A")


class NonePlaceholder:
    """
    Replace None value when None is usefull
    """


DUMMY = NonePlaceholder()


class SignalKind(enum.Enum):
    PARAM = enum.auto()
    STATE = enum.auto()
    STATE_DERIVATIVE = enum.auto()
    INPUT = enum.auto()
    OUTPUT = enum.auto()


class Signal(ComponentDescriptor, Generic[T], abc.ABC):
    kind = ComponentKind.SIGNAL
    signal_kind: SignalKind
    value: T
    dim: int
    lower_bounds: Optional[jax.typing.ArrayLike] = None
    upper_bounds: Optional[jax.typing.ArrayLike] = None
    nominal_value: Optional[T] = None

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nominal_value: Optional[T] = None,
        lower_bounds: Optional[jax.typing.ArrayLike] = None,
        upper_bounds: Optional[jax.typing.ArrayLike] = None,
    ):
        self._user_defined_name = name

        self.nominal_value = nominal_value
        self.value = nominal_value if nominal_value is not None else 0.0

        if nominal_value is not None:
            self.dim = 1  # TODO

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def __repr__(self) -> str:
        owner_id = None if self.owner_id is None else self.owner_id.hex[:5]
        return f'Signal(name="{self.name}", id="{self._uuid.hex[:5]}", owner={self.owner_cls}("id={owner_id}"), value={self.value})'

    def __getattribute__(self, name: str) -> Any:
        # Intercept signal value access to trace reads if within a _trace_signals context
        if name == "value":
            signal_reads = _SIGNAL_READS.get()
            if signal_reads is not None:
                traced, seen_ids = signal_reads
                if self._uuid not in seen_ids:
                    traced.append(self)
                    seen_ids.add(self._uuid)

        return super().__getattribute__(name)

    def __jax_array__(self) -> jax.Array:
        return jax.numpy.asarray(self.get_value())

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        if copy is None:
            return np.asarray(self.get_value(), dtype=dtype)

        return np.asarray(self.get_value(), dtype=dtype, copy=copy)

    def get_value(self, *, index: Any = DUMMY) -> T:
        value = jax.tree.map(lambda x: x, self.value)  # shallow copy to prevent mutation of value

        if not isinstance(index, NonePlaceholder):
            value = value[index]

        return value

    def set_value(self, value: T, *, index: Any = DUMMY) -> None:
        value = jax.tree.map(lambda x: x, value)  # shallow copy to prevent mutation of value

        if isinstance(index, NonePlaceholder):
            self.value = value
            return

        current = self.value

        if isinstance(current, jax.Array):
            self.value = cast(T, current.at[index].set(value))
            return

        cast(Any, current)[index] = value

    def clone(self):
        signal_type = type(self)
        return signal_type(
            name=self.name,
            nominal_value=self.nominal_value,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

    # =================================================================================
    # Proxy methods
    # =================================================================================

    @staticmethod
    def _operator_on_value(name: str) -> Callable[[Signal[T], Any], T]:
        """
        Translate operator function on the Signal object into operation on Signal.value

        from `flax.nnx`:
            https://github.com/google/flax/blob/0f128b5141a7ab40df430d280bc86d3929ccb1ce/flax/nnx/variablelib.py#L1126
        """

        def op_func(self, other):
            value = self.get_value()
            if isinstance(other, Signal):
                other = other.get_value()
            return getattr(value, name)(other)

        op_func.__name__ = name
        return op_func

    @staticmethod
    def _unary_operator_on_value(name: str):
        """
        Translate unary operator function on the Signal object into operation on Signal.value

        from `flax.nnx`:
            https://github.com/google/flax/blob/0f128b5141a7ab40df430d280bc86d3929ccb1ce/flax/nnx/variablelib.py#L1137
        """

        def op_func(self):
            value = self.get_value()
            return getattr(value, name)()

        op_func.__name__ = name
        return op_func

    @overload
    def __getitem__(self: Signal[jax.Array], key) -> jax.Array: ...

    @overload
    def __getitem__(self: Signal[dict[Any, A]], key) -> A: ...

    @overload
    def __getitem__(self: Signal[list[A]], key: int) -> A: ...

    @overload
    def __getitem__(self: Signal[tuple[A, ...]], key: int) -> A: ...

    @overload
    def __getitem__(self, key) -> Any: ...

    def __getitem__(self, key) -> Any:
        return self.get_value(index=key)

    __add__ = _operator_on_value("__add__")
    __sub__ = _operator_on_value("__sub__")
    __mul__ = _operator_on_value("__mul__")
    __matmul__ = _operator_on_value("__matmul__")
    __truediv__ = _operator_on_value("__truediv__")
    __floordiv__ = _operator_on_value("__floordiv__")
    __mod__ = _operator_on_value("__mod__")
    __pow__ = _operator_on_value("__pow__")
    __lshift__ = _operator_on_value("__lshift__")
    __rshift__ = _operator_on_value("__rshift__")
    __and__ = _operator_on_value("__and__")
    __xor__ = _operator_on_value("__xor__")
    __or__ = _operator_on_value("__or__")
    __radd__ = _operator_on_value("__radd__")
    __rsub__ = _operator_on_value("__rsub__")
    __rmul__ = _operator_on_value("__rmul__")
    __rmatmul__ = _operator_on_value("__rmatmul__")
    __rtruediv__ = _operator_on_value("__rtruediv__")
    __rfloordiv__ = _operator_on_value("__rfloordiv__")
    __rmod__ = _operator_on_value("__rmod__")
    __rpow__ = _operator_on_value("__rpow__")
    __rlshift__ = _operator_on_value("__rlshift__")
    __rrshift__ = _operator_on_value("__rrshift__")
    __rand__ = _operator_on_value("__rand__")
    __rxor__ = _operator_on_value("__rxor__")
    __ror__ = _operator_on_value("__ror__")

    __neg__ = _unary_operator_on_value("__neg__")
    __pos__ = _unary_operator_on_value("__pos__")
    __abs__ = _unary_operator_on_value("__abs__")
    __invert__ = _unary_operator_on_value("__invert__")
    __complex__ = _unary_operator_on_value("__complex__")
    __int__ = _unary_operator_on_value("__int__")
    __float__ = _unary_operator_on_value("__float__")
    __index__ = _unary_operator_on_value("__index__")
    __trunc__ = _unary_operator_on_value("__trunc__")
    __floor__ = _unary_operator_on_value("__floor__")
    __ceil__ = _unary_operator_on_value("__ceil__")


class Param(Signal[T]):
    signal_kind: SignalKind = SignalKind.PARAM


class State(Signal[T]):
    signal_kind: SignalKind = SignalKind.STATE


class StateDerivative(Signal[T]):
    signal_kind: SignalKind = SignalKind.STATE_DERIVATIVE

    @classmethod
    def from_state(cls, state: State[T]) -> StateDerivative[T]:
        return cls(name=f"d{state.name}")


class Input(Signal[T]):
    signal_kind: SignalKind = SignalKind.INPUT


class Output(Signal[T]):
    signal_kind: SignalKind = SignalKind.OUTPUT


# tracing and evaluating signals
SignalLike = Union[T, Signal[T]]


_SignalReadState = Tuple[List[Signal[Any]], Set[uuid.UUID]]
_SIGNAL_READS: contextvars.ContextVar[Optional[_SignalReadState]] = contextvars.ContextVar("_SIGNAL_READS", default=None)
_SIGNAL_USES: contextvars.ContextVar[Optional[dict[uuid.UUID, Any]]] = contextvars.ContextVar("_SIGNAL_USES", default=None)


@contextlib.contextmanager
def _trace_signals() -> Iterator[List[Signal[Any]]]:
    traced: List[Signal[Any]] = []
    token = _SIGNAL_READS.set((traced, set()))
    try:
        yield traced
    finally:
        _SIGNAL_READS.reset(token)


@contextlib.contextmanager
def _evaluate_signals(values: dict[uuid.UUID, Any]) -> Iterator[None]:
    token = _SIGNAL_USES.set(values)
    try:
        yield
    finally:
        _SIGNAL_USES.reset(token)
