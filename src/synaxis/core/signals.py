from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Self, TypeVar, cast, overload

from .declaration import Declaration
from .equations import Equation

R = TypeVar("R")  # return type of the equation function
SignalType = TypeVar("SignalType")
ContainerType = TypeVar("ContainerType")


_NonePlaceholder = object()


class Signal(Declaration, Generic[SignalType]):
    _nominal_value: SignalType | None
    _lower_bounds: SignalType | None
    _upper_bounds: SignalType | None

    def __init__(
        self,
        *,
        name: str | None = None,
        nominal_value: SignalType | None = None,
        lower_bounds: SignalType | None = None,
        upper_bounds: SignalType | None = None,
    ):
        super().__init__(name=name)

        self._nominal_value = nominal_value
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    @property
    def nominal_value(self) -> SignalType | None:
        return self._nominal_value

    @property
    def lower_bounds(self) -> SignalType | None:
        return self._lower_bounds

    @property
    def upper_bounds(self) -> SignalType | None:
        return self._upper_bounds

    def clone(self) -> Self:
        signal_type = type(self)
        return signal_type(
            name=self._declared_name,
            nominal_value=self._nominal_value,
            lower_bounds=self._lower_bounds,
            upper_bounds=self._upper_bounds,
        )

    def get_value(self, *, index: Any = _NonePlaceholder) -> SignalType:
        """
        Signal doesn't hold any value, it is just a declaration used for compilation.
        However, a value is required for the signal to be used in equations.
        Therefore, we return the nominal value if it is defined, otherwise we return 0.
        """
        value = self._nominal_value if self._nominal_value is not None else cast(SignalType, 0)

        if index is not _NonePlaceholder:
            return cast(Any, value)[index]
        return value

    # ===========================================================================
    # Python array API standard
    # ===========================================================================
    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        namespace = getattr(self.get_value(), "__array_namespace__", None)
        if namespace is None:
            raise TypeError(f"{type(self.get_value()).__name__} has no array namespace")
        return namespace(api_version=api_version)

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        return getattr(ufunc, method)(
            *(_unwrap(item) for item in inputs),
            **_unwrap_tree(kwargs),
        )

    def __array_function__(
        self,
        function: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        del types
        return function(*_unwrap_tree(args), **_unwrap_tree(kwargs))

    # ===========================================================================
    # Proxy methods to allow the signal to be used as a value directly
    # ===========================================================================

    @overload
    def __getitem__(self: Signal[dict[Any, ContainerType]], key: Any) -> ContainerType: ...

    @overload
    def __getitem__(self: Signal[list[ContainerType]], key: int) -> ContainerType: ...

    @overload
    def __getitem__(self: Signal[tuple[ContainerType, ...]], key: int) -> ContainerType: ...

    def __getitem__(self, key: Any) -> Any:
        return self.get_value(index=key)

    @staticmethod
    def _operator_on_value(name: str) -> Callable[[Signal[SignalType], Any], SignalType]:
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


def _unwrap(value: Any) -> Any:
    return value.get_value() if isinstance(value, Signal) else value


def _unwrap_tree(value: Any) -> Any:
    """Unwrap signals nested in the containers used by NumPy arguments."""
    if isinstance(value, Signal):
        return value.get_value()
    if isinstance(value, tuple):
        return tuple(_unwrap_tree(item) for item in value)
    if isinstance(value, list):
        return [_unwrap_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap_tree(item) for key, item in value.items()}
    return value


class _SourceSignal(Signal[SignalType]):
    """
    A source signal is a signal that can be used to compute a system equations.

    It is used here as a namespace to distinguish signals via inheritance with `isinstance(obj, _SourceSignal)`.
    """


class _ComputedSignal(Signal[SignalType]):
    """
    An output signal is a signal that can be computed by a system equations or given by a connection.

    It is used here as a namespace to distinguish signals via inheritance with `isinstance(obj, _ComputedSignal)`.
    """

    _equation: Equation[SignalType] | None
    _isabstractmethod: bool

    def __init__(
        self,
        function: Callable[[Any], SignalType] | None = None,
        /,
        *,
        name: str | None = None,
        nominal_value: SignalType | None = None,
        lower_bounds: SignalType | None = None,
        upper_bounds: SignalType | None = None,
    ) -> None:
        super().__init__(name=name, nominal_value=nominal_value, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

        self._equation = None
        self._isabstractmethod = False

        if function is not None:
            self._attach(function)

    @property
    def equation(self) -> Equation[SignalType] | None:
        return self._equation

    def _attach(self, function: Callable[[Any], SignalType]) -> Self:
        label = getattr(function, "__name__", None) or self._declared_name or "<unnamed>"

        if self._equation is not None:
            raise TypeError(f"{type(self).__name__} {label!r} already has a defining equation")

        # TODO: this guard require the user to re-declare the signal including its metadata in subclasses if the signal is declared in a base class.
        # we may avoid this issue by using a fresh copy of the signal at decoration time (__call__)
        if self._attribute_name is not None:
            raise TypeError(
                f"{type(self).__name__} {label!r} is already bound to attribute {self._attribute_name!r}; "
                "apply the decorator before assigning the signal to a class or system"
            )

        if not callable(function):
            raise TypeError(f"{type(self).__name__} {label!r} expects a callable, got {type(function).__name__}")

        self._equation = Equation(function)

        # order: @Output above @abstractmethod
        self._isabstractmethod = bool(getattr(function, "__isabstractmethod__", False))

        self.__doc__ = getattr(function, "__doc__", None)
        return self  # TODO: why return self?

    def bind_instance(self, instance: object) -> None:
        if self._equation is not None:
            self._equation.bind(instance)

    def clone(self) -> Self:
        function = self._equation.unbound_function if self._equation is not None else None

        signal_type = type(self)
        clone = signal_type(
            function,
            name=self._declared_name,
            nominal_value=self._nominal_value,
            lower_bounds=self._lower_bounds,
            upper_bounds=self._upper_bounds,
        )
        clone._isabstractmethod = self._isabstractmethod
        return clone

    # ===========================================================================
    # Keep @abc.abstractmethod working in either decorator order
    # See earlier comment in _attach() that handles the case where @Output is applied above @abstractmethod
    # ===========================================================================
    @property
    def __isabstractmethod__(self) -> bool:
        return self._isabstractmethod

    @__isabstractmethod__.setter
    def __isabstractmethod__(self, value: bool) -> None:
        self._isabstractmethod = value  # order: @abstractmethod above @Output sets it on the signal


# ===========================================================================
# Source Signals
# ===========================================================================
class Param(_SourceSignal[SignalType]):
    pass


class Input(_SourceSignal[SignalType]):
    pass


class Noise(_SourceSignal[SignalType]):
    pass


class Time(_SourceSignal[SignalType]):
    pass


class State(_SourceSignal[SignalType]):
    def derivative(self, function: Callable[[Any], SignalType]) -> StateDerivative[SignalType]:
        return StateDerivative(function, of=self)


# ===========================================================================
# Computed Signals
# ===========================================================================
class Output(_ComputedSignal[SignalType]):
    @overload
    def __init__(
        self: Output[R],
        function: Callable[[Any], R],
        /,
        *,
        name: str | None = None,
        nominal_value: R | None = None,
        lower_bounds: R | None = None,
        upper_bounds: R | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        function: None = None,
        /,
        *,
        name: str | None = None,
        nominal_value: SignalType | None = None,
        lower_bounds: SignalType | None = None,
        upper_bounds: SignalType | None = None,
    ) -> None: ...

    def __init__(
        self, function=None, /, *, name=None, nominal_value=None, lower_bounds=None, upper_bounds=None
    ) -> None:
        super().__init__(
            function, name=name, nominal_value=nominal_value, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )

    def __call__(self: Output[Any], function: Callable[[Any], R]) -> Output[R]:
        return cast("Output[R]", self._attach(function))


class Internal(_ComputedSignal[SignalType]):
    @overload
    def __init__(
        self: Internal[R],
        function: Callable[[Any], R],
        /,
        *,
        name: str | None = None,
        nominal_value: R | None = None,
        lower_bounds: R | None = None,
        upper_bounds: R | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        function: None = None,
        /,
        *,
        name: str | None = None,
        nominal_value: SignalType | None = None,
        lower_bounds: SignalType | None = None,
        upper_bounds: SignalType | None = None,
    ) -> None: ...

    def __init__(
        self, function=None, /, *, name=None, nominal_value=None, lower_bounds=None, upper_bounds=None
    ) -> None:
        super().__init__(
            function, name=name, nominal_value=nominal_value, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )

    def __call__(self: Internal[Any], function: Callable[[Any], R]) -> Internal[R]:
        return cast("Internal[R]", self._attach(function))

    def bind_name(self, name: str) -> None:
        # Here, an Internal signal must be defined by an equation
        # TODO: we should refactor this in a way that will make this guard an explicit check at dectoration time, rather
        # then delegating the responsibility to the bind_name() method, which is called at class creation time.
        if self._equation is None:
            raise TypeError(f"Local signal {name!r} must be defined by an equation; use `@Local` as a decorator")
        super().bind_name(name)


class StateDerivative(_ComputedSignal[SignalType]):
    """
    State derivative is a special kind of signal since it is paired to a state signal.
    """

    @overload
    def __init__(
        self: StateDerivative[R],
        function: Callable[[Any], R],
        /,
        *,
        of: State[R] | None = None,
        name: str | None = None,
        nominal_value: R | None = None,
        lower_bounds: R | None = None,
        upper_bounds: R | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        function: None = None,
        /,
        *,
        of: State[SignalType] | None = None,
        name: str | None = None,
        nominal_value: SignalType | None = None,
        lower_bounds: SignalType | None = None,
        upper_bounds: SignalType | None = None,
    ) -> None: ...

    def __init__(
        self,
        function=None,
        /,
        *,
        of: State[SignalType] | None = None,
        name=None,
        nominal_value=None,
        lower_bounds=None,
        upper_bounds=None,
    ) -> None:
        super().__init__(
            function, name=name, nominal_value=nominal_value, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )

        self._state = of
        self._state_attribute_name: str | None = None

    def __call__(self: StateDerivative[Any], function: Callable[[Any], R]) -> StateDerivative[R]:
        return cast("StateDerivative[R]", self._attach(function))

    @property
    def state(self) -> State[SignalType] | None:
        return self._state

    def clone(self) -> Self:
        clone = super().clone()
        clone._state = self._state
        clone._state_attribute_name = self._state_attribute_name
        return clone

    # ===========================================================================
    # Descriptor protocol methods
    # ===========================================================================
    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)

        if self._state is None or self._state._attribute_name is None:
            return  # The state signal is not yet bound to a class attribute

        if getattr(owner, self._state._attribute_name, None) is self._state:
            # The state signal is bound to a class attribute, so we can check for the relationship
            # we store the attribute name of the state signal for later use in the __get__ method to retrieve the
            # instance-level state signal
            self._state_attribute_name = self._state._attribute_name
            self._state = None  # clear the reference to the state signal because it is a class-level declaration

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> Self: ...

    def __get__(self, instance: object | None, owner: type | None = None) -> Self:
        derivative = super().__get__(instance, owner)

        if instance is None or derivative._state_attribute_name is None:
            return derivative  # Return the descriptor itself when accessed through the class (MyClass.signal)

        # Assign the instance-level state signal to the derivative signal based on the attribute name stored earlier in
        # __set_name__
        state_instance = getattr(type(instance), derivative._state_attribute_name, None)
        if isinstance(state_instance, State):
            state = state_instance.__get__(instance, type(instance))
            if isinstance(state, State):
                derivative._state = state

        return derivative
