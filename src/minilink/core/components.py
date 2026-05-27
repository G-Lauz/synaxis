from __future__ import annotations

import abc
import uuid
from typing import Any, Optional, TypeVar, cast, overload

T = TypeVar("T", bound="ComponentDescriptor")

_MISSING = object()


class ComponentDescriptor(abc.ABC):
    name: str
    _user_defined_name: Optional[str] = None
    _attribute_name: Optional[str] = None
    owner_id: Optional[uuid.UUID] = None
    owner_cls: Optional[str] = None

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

        # if obj as been defined as class attribute of a system it should be cloned to avoid
        # shared obj for multiple instance of the system
        instance_dict = cast(Any, instance).__dict__
        attribute_name = self._attribute_name if self._attribute_name is not None else self.name
        obj = instance_dict.get(attribute_name, _MISSING)
        if obj is _MISSING:
            obj = self.clone()
            obj.bind_owner(name=self.name, owner=owner, owner_id=cast(Any, instance)._uuid)
            instance_dict[attribute_name] = obj

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
