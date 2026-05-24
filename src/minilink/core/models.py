from typing import List, Optional, Tuple

from .signals import Signal
from .systems import System


class Model(System):
    _connections: List[Tuple[Signal, Signal]]

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()

        self._connections = []

    def connect(self, source: Signal, target: Signal) -> None:
        self._connections.append((source, target))

    def compile(self):
        pass
