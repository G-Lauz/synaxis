import dataclasses
from typing import List, Optional, Tuple

from .signals import Input, Output, Param, Signal, State
from .systems import System


class Model(System):
    _blocks: List[System]
    # _signals: List[Tuple[Signal, type]]
    _connections: List[Tuple[Signal, Signal]]

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()

        self._blocks = []
        # self._signals = []
        self._connections = []

    def attr_discovery(self):
        for _, obj in self.__dict__.items():
            if isinstance(obj, System):
                self._blocks.append(obj)

            # if isinstance(obj, Signal):
            #     signal_type = type(obj)
            #     self._signals.append((obj, signal_type))

    def connect(self, source: Signal, target: Signal) -> None:
        self._connections.append((source, target))

    def compile(self):
        # discover child blocks and signals
        self.attr_discovery()
        print(self._blocks)
        # print(self._signals)
