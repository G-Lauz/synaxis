from __future__ import annotations

import abc
import enum
import uuid
from typing import Any, ClassVar, Optional, TypeVar, cast, overload


T = TypeVar("T", bound="ComponentDescriptor")


_MISSING = object()


class ComponentKind(enum.Enum):
    EQUATION = enum.auto()
    SIGNAL = enum.auto()
    SYSTEM = enum.auto()


class ComponentDescriptor(abc.ABC):
    _uuid: uuid.UUID
    name: str
    _user_defined_name: Optional[str] = None
    _attribute_name: Optional[str] = None
    owner_id: Optional[uuid.UUID] = None
    owner_cls: Optional[str] = None
    kind: ClassVar[ComponentKind]

    def __new__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_uuid", uuid.uuid4())
        return cast(T, instance)

    def __set_name__(self, owner: type, name: str) -> None:
        self._attribute_name = name
        self.name = self._user_defined_name if self._user_defined_name is not None else name
        self.owner_cls = owner.__name__

    @overload
    def __get__(self: T, instance: None, owner: Optional[type] = None) -> T: ...

    @overload
    def __get__(self: T, instance: object, owner: Optional[type] = None) -> T: ...

    def __get__(self: T, instance: Optional[object], owner: Optional[type] = None) -> T:
        if instance is None:
            return self

        return self._materialize(instance)

    def _materialize(self: T, instance: object) -> T:
        # if obj as been defined as class attribute of a system it should be cloned to avoid
        # shared obj for multiple instance of the system
        instance_dict = cast(Any, instance).__dict__
        attribute_name = self._attribute_name if self._attribute_name is not None else self.name
        obj = instance_dict.get(attribute_name, _MISSING)
        if obj is _MISSING:
            obj = self.clone()
            setattr(instance, attribute_name, obj)

        return cast(T, obj)

    def bind_owner(
        self, *, name: str, owner: Optional[type] = None, owner_id: Optional[uuid.UUID] = None
    ) -> None:
        self.name = self._user_defined_name if self._user_defined_name is not None else name
        self.owner_id = owner_id
        self.owner_cls = owner.__name__ if owner is not None else None

    @abc.abstractmethod
    def clone(self) -> ComponentDescriptor:
        pass
