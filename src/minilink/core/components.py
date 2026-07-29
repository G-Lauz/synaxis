from __future__ import annotations

import abc
import enum
import uuid
from typing import Any, ClassVar, Self, cast, overload

_MISSING = object()


class ComponentKind(enum.Enum):
    EQUATION = enum.auto()
    SIGNAL = enum.auto()
    SYSTEM = enum.auto()


class ComponentDescriptor(abc.ABC):
    _uuid: uuid.UUID
    name: str
    _user_defined_name: str | None = None
    _attribute_name: str | None = None
    owner_id: uuid.UUID | None = None
    owner_cls: str | None = None
    kind: ClassVar[ComponentKind]

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_uuid", uuid.uuid4())
        return instance

    def __set_name__(self, owner: type, name: str) -> None:
        self._attribute_name = name
        self.name = self._user_defined_name if self._user_defined_name is not None else name
        self.owner_cls = owner.__name__

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> Self: ...

    def __get__(self, instance: object | None, owner: type | None = None) -> Self:
        if instance is None:
            return self

        return self._materialize(instance)

    def _materialize(self, instance: object) -> Self:
        # if obj as been defined as class attribute of a system it should be cloned to avoid
        # shared obj for multiple instance of the system
        instance_dict = cast(Any, instance).__dict__
        attribute_name = self._attribute_name if self._attribute_name is not None else self.name
        obj = instance_dict.get(attribute_name, _MISSING)
        if obj is _MISSING:
            obj = self.clone()
            setattr(instance, attribute_name, obj)

        return cast(Self, obj)

    def bind_owner(self, *, name: str, owner: type | None = None, owner_id: uuid.UUID | None = None) -> None:
        self.name = self._user_defined_name if self._user_defined_name is not None else name
        self.owner_id = owner_id
        self.owner_cls = owner.__name__ if owner is not None else None

    @abc.abstractmethod
    def clone(self) -> ComponentDescriptor:
        pass
