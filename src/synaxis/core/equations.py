import functools
from collections.abc import Callable
from typing import Generic, Self, TypeVar, cast, overload

from .declaration import Declaration
from .signals import Output

OwnerType = TypeVar("OwnerType")
R = TypeVar("R")  # method return type # TODO: we already know that by design the return should either be an output
# signal or a mapping of output signals, we should enforce that in the type hinting.


class Equation(Declaration, Generic[OwnerType, R]):
    name: str

    _function: Callable[[OwnerType], R]
    _bound_function: Callable[[], R] | None

    _output_type: type

    _isabstractmethod: bool

    _attribute_name: str | None  # The name of the attribute in the class that owns this declaration
    _declared_name: str | None  # The name of the declaration as declared by the user, if any.

    def __init__(self, function: Callable[[OwnerType], R], *, name: str | None = None, otype: type = Output) -> None:
        """
        Args:
            function: The function that defines the equation.
            name: The name of the equation, if any.
            otype: The output type of the equation, if any. Defaults to Output.
        """
        self._declared_name = name

        # the unbound function as defined in the class, before it is bound to an instance of the class.
        self._function = function
        self._isabstractmethod = bool(getattr(function, "__isabstractmethod__", False))

        self._bound_function = None

        self._output_type = otype

    @property
    def function(self) -> Callable[[], R]:
        if self._bound_function is None:
            raise RuntimeError("equation is not bound to an instance")
        return self._bound_function

    @property
    def output_type(self) -> type:
        return self._output_type

    def clone(self) -> Self:
        equation_type = type(self)
        clone = equation_type(self._function, name=self.name, otype=self.output_type)
        clone._isabstractmethod = self._isabstractmethod
        clone._bound_function = self._bound_function
        return clone

    # ===========================================================================
    # Keep @abc.abstractmethod working in either decorator order
    # ===========================================================================
    @property
    def __isabstractmethod__(self) -> bool:
        return self._isabstractmethod

    @__isabstractmethod__.setter
    def __isabstractmethod__(self, value: bool) -> None:
        self._isabstractmethod = value

    # ===========================================================================
    # Descriptor protocol methods
    # ===========================================================================
    def __get__(self, instance: object | None, owner: type | None = None) -> Self:
        if instance is None:
            return self  # Return the descriptor itself when accessed through the class (MyClass.equation)

        instance_equation = super().__get__(instance, owner)

        # we need to cast the instance to the correct type, because of the overloading of the generic __get__ method.
        # only really useful for type hinting.
        casted_instance = cast(OwnerType, instance)

        # bind the function to the instance, so that it can access the instance's attributes and methods.
        instance_equation._bound_function = functools.partial(self._function, casted_instance)

        return instance_equation


@overload
def equation(
    function: Callable[[OwnerType], R],
    /,
    *,
    name: str | None = None,
    otype: type = Output,
) -> Equation[OwnerType, R]: ...


@overload
def equation(
    function: None = None,
    /,
    *,
    name: str | None = None,
    otype: type = Output,
) -> Callable[
    [Callable[[OwnerType], R]],
    Equation[OwnerType, R],
]: ...


def equation(
    function: Callable[[OwnerType], R] | None = None, /, *, name: str | None = None, otype: type = Output
) -> Callable[[Callable[[OwnerType], R]], Equation[OwnerType, R]] | Equation[OwnerType, R]:
    """
    Decorator to mark a method as an equation in a System.

    Args:
        function: The function that defines the equation.
        name: The name of the equation, if any.
        otype: The output type of the equation, if any. This is used to validate the return type of the function.
    """

    def decorate(func: Callable[[OwnerType], R]) -> Equation[OwnerType, R]:
        return Equation(func, name=name, otype=otype)

    return decorate if function is None else decorate(function)
