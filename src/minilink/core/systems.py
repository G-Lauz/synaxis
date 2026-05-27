from __future__ import annotations

import abc
import uuid
from typing import List, Optional, Tuple

from .components import ComponentDescriptor
from .signals import Signal


class System(ComponentDescriptor, abc.ABC):
    _uuid: uuid.UUID

    _blocks: Optional[List[System]] = None
    _signals: Optional[List[Signal]] = None
    _connections: Optional[List[Tuple[Signal, Signal]]] = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._uuid = uuid.uuid4()
        instance.name = cls.__name__
        return instance

    def __init__(self, name: Optional[str] = None):
        """
        Caution: __init__ could not be called by the user, hence everything here should be optional
        """
        super().__init__()

        self._user_defined_name = name
        self.name = name if name is not None else self.name

    def __repr__(self) -> str:
        return f'System(name="{self.name}", id="{self._uuid.hex[:5]}")'

    def __setattr__(self, name: str, value: object) -> None:
        # call __set_name__ at instantiation to enable descriptor
        # bind owner if attribute has been defined within the init constructor
        if isinstance(value, ComponentDescriptor):
            usage_name = value._user_defined_name if value._user_defined_name is not None else name
            value.bind_owner(name=usage_name, owner=type(self), owner_id=self._uuid)

        super().__setattr__(name, value)

    def clone(self):
        system_type = type(self)
        return system_type(name=self.name)

    def connect(self, source: Signal, target: Signal) -> None:
        if self._connections is None:
            self._connections = []

        self._connections.append((source, target))

    def compile(self):
        # discover child blocks and signals
        self.attr_discovery()
        print(self._blocks)
        print(self._signals)

    def _get_owned_members(self, kind: type):
        members = []
        seen = set()

        # discover class attributes
        # required if attributes hasn't been access yet (__get__)
        system_type = type(self)
        for cls in reversed(system_type.__mro__):
            for name, template in vars(cls).items():
                if isinstance(template, kind):
                    seen.add(name)
                    value = getattr(self, name)
                    if isinstance(value, kind):
                        members.append(value)

        # discover runtime assigned attributes
        for name, value in self.__dict__.items():
            if name in seen:
                continue

            if isinstance(value, kind):
                members.append(value)

        return members

    def attr_discovery(self):
        self._blocks = self._get_owned_members(System)
        self._signals = self._get_owned_members(Signal)


class StaticSystem(System):
    @abc.abstractmethod
    def compute_outputs(self):
        pass


class DynamicSystem(System):
    @abc.abstractmethod
    def compute_outputs(self):
        pass

    @abc.abstractmethod
    def compute_dynamics(self):
        pass
