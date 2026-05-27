import abc
import uuid
from typing import Optional

from .signals import Param, Signal, State


class System(abc.ABC):
    name: str
    _uuid: uuid.UUID

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._uuid = uuid.uuid4()
        instance.name = cls.__name__
        return instance

    def __init__(self, name: Optional[str] = None):
        super().__init__()

        self.name = name if name is not None else self.name

    def __repr__(self) -> str:
        return f'System(name="{self.name}", id="{self._uuid.hex[:5]}")'

    def __setattr__(self, name: str, value: object) -> None:
        # call __set_name__ at instantiation to enable descriptor
        # bind owner if attribute has been defined within the init constructor
        if isinstance(value, Signal):
            value.bind_owner(name=name, owner=type(self), owner_id=self._uuid)

        super().__setattr__(name, value)


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
