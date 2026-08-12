import abc
import uuid
from typing import Self, cast, overload


class Declaration(abc.ABC):
    """
    A declaration is a statement that defines a component of a system, such as a signal, a subsystem, or an equation.

    Example:
    ```Python
        class MySystem(System):
            subsystem = OtherSystem()   # Declaration of subsystem
            x = Signal()                # Declaration of signal

            def equation(self):         # Declaration of equation
                return self.x
    ```
    """

    name: str

    _uuid: uuid.UUID  # UUID are used to help user to identify the declaration in a system, especially for debugging
    _attribute_name: str | None  # The name of the attribute in the class that owns this declaration
    _declared_name: str | None  # The name of the declaration as declared by the user, if any.

    def __new__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)
        instance._uuid = uuid.uuid4()  # Generate a unique identifier for the declaration
        return instance

    def __init__(self, *, name: str | None = None) -> None:
        self._declared_name = name

    @property
    def id(self) -> uuid.UUID:
        """
        Get the unique identifier of the component.
        """
        return self._uuid

    @abc.abstractmethod
    def clone(self) -> Self:
        """Return an independent declaration instance"""

    def bind_name(self, name: str) -> None:
        """
        Bind the declaration to a name in the class that owns it.
        This is called when a declaration is assigned to an attribute of a class.
        """
        self._attribute_name = name
        self.name = self._declared_name or name

    # ===========================================================================
    # Descriptor protocol methods
    # ===========================================================================
    def __set_name__(self, owner: type, name: str) -> None:
        # Here, `owner` is the class that owns this declaration, and `name` is the name of the attribute in that class.
        self._attribute_name = name

        # bind appropriate name to the declaration, either the user-defined name or the attribute name.
        self.bind_name(name)

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> Self: ...

    def __get__(self, instance: object | None, owner: type | None = None) -> Self:
        # self:     is the descriptor object
        # instance: is an instance of the class that owns this descriptor
        # owner:    is the owner's class itself.
        if instance is None:
            return self  # Return the descriptor itself when accessed through the class (MyClass.signal)

        attribute_name = self._attribute_name or self.name
        instance_declaration = vars(instance).get(attribute_name, None)
        if instance_declaration is None:
            # class-level declaration is shared across all instances, so we clone it for independency of instances.
            instance_declaration = self.clone()
            instance_declaration._attribute_name = attribute_name
            instance_declaration.name = instance_declaration._declared_name or attribute_name
            setattr(instance, attribute_name, instance_declaration)

        return cast(Self, instance_declaration)
